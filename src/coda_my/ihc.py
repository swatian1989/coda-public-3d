"""Quantification and spatial analysis of DAB IHC: ER, PR, HER2, Ki67.

This is the part TCGA cannot give you. TCGA carries categorical receptor status
transcribed from pathology reports. You have the sections and can measure
continuous, spatially resolved expression yourself.

The scientific opening is Ki67. Its scoring is irreproducible because observers
disagree on hotspot versus average assessment, and the 20 percent cutoff that
drives chemotherapy decisions sits exactly where reproducibility is worst.
Nobody routinely quantifies HOW Ki67-positive nuclei are ARRANGED, only what
percentage they are. A clustered 18 percent and a dispersed 18 percent get the
same score and the same treatment decision.

The spatial statistics for answering that already exist in the canvas-brca
package (Ripley K and L with border correction, Donnelly-corrected Clark-Evans,
quadrat variance-to-mean, KDE hotspot coefficient of variation). This module
produces the labelled point pattern they consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from .deconv import deconvolve_dab

logger = logging.getLogger(__name__)

MARKERS = ("ER", "PR", "HER2", "Ki67")

# HER2 is membranous, not nuclear. Nuclear DAB thresholding is the wrong
# operation for it and will give a meaningless number. Treat HER2 separately.
NUCLEAR_MARKERS = ("ER", "PR", "Ki67")


@dataclass
class IHCConfig:
    mpp: float = 0.5
    min_nucleus_um2: float = 10.0
    max_nucleus_um2: float = 200.0
    dab_threshold: float | None = None   # None = Otsu on the DAB channel
    hematoxylin_threshold: float = 0.15
    seed: int = 42


def detect_nuclei(
    hematoxylin: np.ndarray, cfg: IHCConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Detect nuclei from the hematoxylin concentration map.

    CODA smoothed the hematoxylin channel and took 2D intensity minima of a
    given size and separation as nuclei, reaching >90 percent precision and
    recall against two manual annotators and running about 3-fold faster than
    HoVer-Net or QuPath. Here the equivalent is a distance-transform watershed,
    which is the same idea implemented with standard tooling.

    Returns the label image and a table of centroid coordinates in microns.
    """
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops_table
    from skimage.segmentation import watershed

    smooth = ndimage.gaussian_filter(hematoxylin, 1.0)
    mask = smooth > cfg.hematoxylin_threshold
    mask = ndimage.binary_opening(mask, np.ones((3, 3)))

    dist = ndimage.distance_transform_edt(mask)
    min_radius_px = np.sqrt(cfg.min_nucleus_um2 / np.pi) / cfg.mpp
    coords = peak_local_max(dist, min_distance=max(int(min_radius_px), 2),
                            labels=mask)
    markers = np.zeros_like(dist, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = watershed(-dist, markers, mask=mask)

    if labels.max() == 0:
        return labels, pd.DataFrame(columns=["label", "x_um", "y_um",
                                             "area_um2", "aspect_ratio"])

    props = pd.DataFrame(regionprops_table(
        labels, properties=("label", "centroid", "area",
                            "major_axis_length", "minor_axis_length")))
    props = props.rename(columns={"centroid-0": "y_px", "centroid-1": "x_px"})
    props["area_um2"] = props["area"] * cfg.mpp ** 2
    props["x_um"] = props["x_px"] * cfg.mpp
    props["y_um"] = props["y_px"] * cfg.mpp
    props["aspect_ratio"] = props["major_axis_length"] / np.maximum(
        props["minor_axis_length"], 1e-6)

    keep = props["area_um2"].between(cfg.min_nucleus_um2, cfg.max_nucleus_um2)
    logger.info("detected %d nuclei, kept %d after size filter",
                len(props), int(keep.sum()))
    return labels, props[keep].reset_index(drop=True)


def score_marker(
    rgb: np.ndarray, marker: str, cfg: IHCConfig | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Per-nucleus DAB positivity plus a slide-level summary.

    Returns
    -------
    nuclei
        Table with x_um, y_um, dab_mean, positive (bool), area_um2,
        aspect_ratio. This is the labelled point pattern for spatial analysis.
    summary
        n_nuclei, n_positive, percent_positive, dab_threshold_used.
    """
    cfg = cfg or IHCConfig()
    if marker not in MARKERS:
        raise ValueError(f"marker must be one of {MARKERS}")
    if marker == "HER2":
        raise ValueError(
            "HER2 is membranous. Per-nucleus DAB scoring is the wrong operation "
            "and gives a meaningless number. Use membrane_completeness() instead."
        )

    ch = deconvolve_dab(rgb)
    labels, nuclei = detect_nuclei(ch["hematoxylin"], cfg)
    if nuclei.empty:
        return nuclei, {"n_nuclei": 0, "n_positive": 0,
                        "percent_positive": np.nan, "dab_threshold_used": np.nan}

    dab = ch["dab"]
    means = ndimage.mean(dab, labels, index=nuclei["label"].to_numpy())
    nuclei = nuclei.assign(dab_mean=means)

    if cfg.dab_threshold is not None:
        thr = cfg.dab_threshold
    else:
        from skimage.filters import threshold_otsu
        try:
            thr = float(threshold_otsu(means))
        except ValueError:
            thr = float(np.median(means))
        # Otsu on a nearly all-negative slide splits noise into two halves and
        # invents positivity. Floor it.
        thr = max(thr, 0.10)

    nuclei = nuclei.assign(positive=nuclei["dab_mean"] > thr)
    summary = {
        "n_nuclei": int(len(nuclei)),
        "n_positive": int(nuclei["positive"].sum()),
        "percent_positive": float(100 * nuclei["positive"].mean()),
        "dab_threshold_used": float(thr),
    }
    logger.info("%s: %d/%d nuclei positive (%.1f%%), threshold %.3f", marker,
                summary["n_positive"], summary["n_nuclei"],
                summary["percent_positive"], thr)
    return nuclei, summary


def to_point_pattern(nuclei: pd.DataFrame) -> pd.DataFrame:
    """Convert scored nuclei into the two-class point pattern used downstream.

    Output columns x_um, y_um, habitat where 0 is negative and 1 is positive.
    This feeds straight into the canvas-brca spatial feature functions, which
    then give border-corrected Ripley K and L, Donnelly-corrected Clark-Evans,
    quadrat dispersion and KDE hotspot statistics for the positive population.
    """
    return pd.DataFrame({
        "x_um": nuclei["x_um"].to_numpy(),
        "y_um": nuclei["y_um"].to_numpy(),
        "habitat": nuclei["positive"].astype(int).to_numpy(),
    })


def hotspot_vs_average(
    nuclei: pd.DataFrame, window_um: float = 500.0
) -> dict[str, float]:
    """Quantify the hotspot versus average scoring discrepancy directly.

    Slides the reporting window across the section and reports the average
    positivity, the maximum window positivity (the hotspot score a pathologist
    would give), and their difference. A large gap is precisely the situation in
    which two observers reach different scores and, near the 20 percent cutoff,
    different treatment decisions.
    """
    if nuclei.empty:
        return {k: np.nan for k in
                ("average_percent", "hotspot_percent", "hotspot_minus_average",
                 "n_windows")}

    x, y = nuclei["x_um"].to_numpy(), nuclei["y_um"].to_numpy()
    pos = nuclei["positive"].to_numpy()

    xs = np.arange(x.min(), x.max(), window_um / 2)
    ys = np.arange(y.min(), y.max(), window_um / 2)
    scores = []
    for x0 in xs:
        for y0 in ys:
            sel = (x >= x0) & (x < x0 + window_um) & (y >= y0) & (y < y0 + window_um)
            if sel.sum() >= 100:                # a scorable field
                scores.append(100 * pos[sel].mean())

    if not scores:
        return {"average_percent": float(100 * pos.mean()),
                "hotspot_percent": np.nan, "hotspot_minus_average": np.nan,
                "n_windows": 0}

    avg = float(100 * pos.mean())
    hot = float(np.max(scores))
    return {"average_percent": avg, "hotspot_percent": hot,
            "hotspot_minus_average": hot - avg, "n_windows": len(scores)}

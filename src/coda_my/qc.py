"""CODA validation: registration quality, z-resolution, and cell detection.

Every check here is one CODA performed and reported. Reproducing them on a
public dataset is what turns "I ran some code" into "I validated the method",
and each has a published number to compare against.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def axial_vs_lateral_correlation(
    stack: np.ndarray, max_distance_um: float = 300.0,
    mpp: float = 8.0, section_um: float = 4.0, n_sample: int = 20_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare z-direction pixel correlation to within-section xy correlation.

    CODA's logic: xy correlation measures how pixel intensity varies across
    INTACT tissue, so it is the ceiling. Perfect registration would make the
    z correlation match it. The gap between them is registration error.

    They reported >95% correlation retained when skipping up to four serial
    sections (20 um). Reproduce that number here.

    Returns correlation versus separation distance for both axes.
    """
    rng = np.random.default_rng(seed)
    n_z, h, w = stack.shape
    rows = []

    lat_steps = [int(d / mpp) for d in np.arange(0, max_distance_um, mpp * 4)]
    for step in lat_steps:
        if step >= w - 1:
            break
        ys = rng.integers(0, h, n_sample)
        xs = rng.integers(0, w - step, n_sample)
        zs = rng.integers(0, n_z, n_sample)
        a, b = stack[zs, ys, xs], stack[zs, ys, xs + step]
        r, _ = spearmanr(a, b)
        rows.append({"axis": "xy", "distance_um": step * mpp,
                     "correlation": float(r) if np.isfinite(r) else np.nan})

    for dz in range(0, min(n_z, int(max_distance_um / section_um))):
        if dz >= n_z:
            break
        zs = rng.integers(0, n_z - dz, n_sample) if dz else rng.integers(0, n_z, n_sample)
        ys, xs = rng.integers(0, h, n_sample), rng.integers(0, w, n_sample)
        a, b = stack[zs, ys, xs], stack[zs + dz, ys, xs]
        r, _ = spearmanr(a, b)
        rows.append({"axis": "z", "distance_um": dz * section_um,
                     "correlation": float(r) if np.isfinite(r) else np.nan})

    return pd.DataFrame(rows)


def z_skip_validation(
    stack: np.ndarray, labels: np.ndarray | None = None,
    skips: tuple[int, ...] = (1, 2, 3, 4, 5), section_um: float = 4.0,
) -> pd.DataFrame:
    """Quantify what is lost by processing only every Nth section. [PAPER]

    CODA found <5% error in 3D cell count and tissue composition when skipping
    up to two sections (12 um), which is why they processed one section in
    three and cut the workload by two thirds. Run this on your own data before
    adopting the same shortcut; the answer depends on how fast your tissue
    changes through z, and breast is not pancreas.

    Returns percent change in composition relative to the full stack.
    """
    if labels is None:
        labels = stack
    full = _composition_vector(labels)
    rows = []
    for s in skips:
        sub = labels[::s]
        comp = _composition_vector(sub)
        common = set(full) & set(comp)
        err = np.mean([abs(comp[k] - full[k]) / max(full[k], 1e-9) * 100
                       for k in common]) if common else np.nan
        rows.append({"skip": s, "spacing_um": s * section_um,
                     "n_sections_used": len(sub),
                     "percent_composition_error": float(err)})
    df = pd.DataFrame(rows)
    logger.info("z-skip validation:\n%s", df.to_string(index=False))
    return df


def _composition_vector(volume: np.ndarray) -> dict[int, float]:
    vals, counts = np.unique(volume[volume > 0], return_counts=True)
    total = counts.sum()
    return {int(v): float(c / total) for v, c in zip(vals, counts)} if total else {}


def target_registration_error(
    fixed_landmarks: np.ndarray, moving_landmarks: np.ndarray, mpp: float = 0.46,
) -> dict[str, float]:
    """TRE between paired fiducials, in microns. [PAPER]

    The Kartasalo serial datasets ship fiducial points marked by human
    operators on structures visible in both sections, preferentially nuclei
    split by the sectioning blade. That gives an objective accuracy number with
    no annotation work of your own, and lets you compare directly against the
    seven registration methods in the published benchmark.
    """
    if fixed_landmarks.shape != moving_landmarks.shape:
        raise ValueError("landmark arrays must have matching shapes")
    d = np.linalg.norm(fixed_landmarks - moving_landmarks, axis=1) * mpp
    return {"tre_mean_um": float(d.mean()), "tre_median_um": float(np.median(d)),
            "tre_p95_um": float(np.percentile(d, 95)), "n_landmarks": int(len(d))}


def accumulated_tre(
    landmark_series: list[np.ndarray], reference_index: int | None = None,
    mpp: float = 0.46,
) -> pd.DataFrame:
    """ATRE: drift of landmarks relative to a fixed reference section. [PAPER]

    This is the metric CODA outperformed all seven competitors on, and it is
    the one that matters for 3D. Pairwise error can be excellent while the
    stack still banana-bends, because small consistent errors accumulate.
    Rising ATRE with distance from the reference means the stack is drifting.
    """
    ref = reference_index if reference_index is not None else len(landmark_series) // 2
    base = landmark_series[ref]
    rows = []
    for i, lm in enumerate(landmark_series):
        if lm.shape != base.shape:
            continue
        d = np.linalg.norm(lm - base, axis=1) * mpp
        rows.append({"section": i, "distance_from_reference": abs(i - ref),
                     "atre_mean_um": float(d.mean()),
                     "atre_max_um": float(d.max())})
    return pd.DataFrame(rows)


def cell_detection_metrics(
    detected: np.ndarray, manual: np.ndarray, tolerance_um: float = 2.0,
    mpp: float = 0.5,
) -> dict[str, float]:
    """Precision and recall of cell detection against manual annotation. [PAPER]

    CODA matched detections to manual points within 2 um, the average nuclear
    radius in their images, with one-to-one assignment so a single manual point
    cannot absorb several detections. They reported >90% precision and recall,
    beating HoVer-Net and QuPath while running about 3-fold faster.

    Greedy nearest-neighbour matching is used here, which is what the paper
    describes.
    """
    from scipy.spatial import cKDTree

    if len(detected) == 0 or len(manual) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": len(detected), "fn": len(manual)}

    tol_px = tolerance_um / mpp
    tree = cKDTree(manual)
    used = np.zeros(len(manual), dtype=bool)
    tp = 0
    for point in detected:
        candidates = tree.query_ball_point(point, tol_px)
        free = [c for c in candidates if not used[c]]
        if free:
            nearest = min(free, key=lambda c: np.linalg.norm(manual[c] - point))
            used[nearest] = True
            tp += 1

    fp, fn = len(detected) - tp, len(manual) - tp
    precision = tp / max(len(detected), 1)
    recall = tp / max(len(manual), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn}

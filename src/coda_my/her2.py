"""HER2 membrane scoring. Separate from nuclear markers, by necessity.

HER2 is a membrane protein. Per-nucleus DAB scoring, which is correct for ER,
PR and Ki67, produces a confident and meaningless number for HER2 because the
signal is not in the nucleus. `ihc.score_marker` refuses HER2 for that reason.

ASCO/CAP scores HER2 on two axes: intensity of membrane staining, and
COMPLETENESS of the membrane outline around each cell. 3+ requires complete,
intense circumferential staining in >10% of tumour cells; 2+ is weak-to-moderate
complete, or intense but incomplete; 1+ is faint and incomplete.

This module measures both axes directly:

  - membrane_fraction: how much tissue carries membrane-pattern signal
  - completeness: for each enclosed cell region, what fraction of its perimeter
    is stained, which is the "complete versus incomplete" axis
  - a chicken-wire index from the skeleton of the membrane mask, since complete
    circumferential staining across a sheet of cells produces a connected
    lattice while incomplete staining produces broken fragments

This is a quantitative descriptor, NOT a clinical score. It does not replace
pathologist assessment and must never be reported as an ASCO/CAP category.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .deconv import deconvolve_dab
from .scalebar import mask_overlay_region

logger = logging.getLogger(__name__)


@dataclass
class HER2Config:
    mpp: float = 0.5
    dab_threshold: float | None = None      # None = Otsu
    min_cell_area_um2: float = 25.0
    max_cell_area_um2: float = 600.0
    membrane_width_um: float = 2.0


def membrane_mask(rgb: np.ndarray, cfg: HER2Config) -> np.ndarray:
    """Binary mask of DAB-positive membrane-like structure."""
    from skimage.filters import threshold_otsu

    dab = deconvolve_dab(rgb)["dab"]
    dab[mask_overlay_region(rgb)] = 0.0

    if cfg.dab_threshold is not None:
        thr = cfg.dab_threshold
    else:
        try:
            thr = float(threshold_otsu(dab[dab > 0.02]))
        except ValueError:
            thr = 0.15
        thr = max(thr, 0.10)

    mask = dab > thr
    return ndimage.binary_opening(mask, np.ones((3, 3)))


def membrane_completeness(rgb: np.ndarray, cfg: HER2Config | None = None) -> dict:
    """Quantify membrane staining completeness. The ASCO/CAP axis that matters.

    Enclosed background regions inside the membrane mask are treated as cell
    interiors. For each, the fraction of its boundary that is stained gives a
    per-cell completeness. A sheet of cells with complete circumferential
    staining yields many enclosed regions with high completeness; incomplete
    staining yields few enclosed regions and low values.
    """
    cfg = cfg or HER2Config()
    mask = membrane_mask(rgb, cfg)
    valid = ~mask_overlay_region(rgb)

    filled = ndimage.binary_fill_holes(mask)
    interiors = filled & ~mask
    labels, n = ndimage.label(interiors)

    px_area = cfg.mpp ** 2
    lo = cfg.min_cell_area_um2 / px_area
    hi = cfg.max_cell_area_um2 / px_area

    # Areas via bincount rather than per-object bounding boxes: faster, and it
    # avoids the label-index-versus-find_objects-order mismatch that silently
    # drops every object when some labels are absent from the sequence.
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep_ids = np.flatnonzero((counts >= lo) & (counts <= hi))

    completeness, areas = [], []
    for obj_id in keep_ids:
        region = labels == obj_id
        border = ndimage.binary_dilation(region, np.ones((3, 3))) & ~region
        b = int(border.sum())
        if b:
            completeness.append(float((border & mask).sum() / b))
            areas.append(float(counts[obj_id] * px_area))

    dilated = ndimage.binary_dilation(mask, np.ones((3, 3)))
    out = {
        "membrane_area_fraction": float(mask[valid].mean()),
        "n_enclosed_cells": len(completeness),
        "mean_completeness": float(np.mean(completeness)) if completeness else np.nan,
        "frac_complete_gt80": float(np.mean(np.array(completeness) > 0.8))
        if completeness else np.nan,
        "median_cell_area_um2": float(np.median(areas)) if areas else np.nan,
        "chicken_wire_index": _chicken_wire(mask, dilated),
    }
    logger.info("HER2 membrane: %.1f%% area, %d enclosed cells, mean completeness %.2f",
                out["membrane_area_fraction"] * 100, out["n_enclosed_cells"],
                out["mean_completeness"])
    logger.info("This is a quantitative descriptor. It is NOT an ASCO/CAP score "
                "and must not be reported as 0/1+/2+/3+.")
    return out


def _chicken_wire(mask: np.ndarray, dilated: np.ndarray) -> float:
    """How lattice-like the membrane mask is.

    A connected honeycomb has few components relative to its area; broken
    fragments have many. Reported as 1 - (components / area in kilopixels),
    clipped to [0, 1], so higher means more lattice-like.
    """
    _, n_components = ndimage.label(mask)
    area_kpx = mask.sum() / 1000.0
    if area_kpx < 1:
        return np.nan
    return float(np.clip(1.0 - n_components / area_kpx, 0.0, 1.0))

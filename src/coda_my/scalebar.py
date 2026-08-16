"""Read the burned-in scale bar to recover microns per pixel, and mask it out.

Field-of-view captures from a microscope camera carry a red scale bar and a
label in one corner. Two consequences that must both be handled before any
measurement:

  1. The bar IS the calibration. Without it there is no mpp, and every distance,
     area and density is in pixels rather than microns and cannot be compared
     across images taken at different objectives.
  2. The bar and its text are high-contrast objects sitting on the tissue. Any
     nuclei detector will happily segment them, any spatial statistic will treat
     them as a dense cluster in the corner, and the resulting hotspot is pure
     artefact. Mask before measuring.

The bar is found as the longest contiguous run of saturated red pixels within a
single image row, which separates it from the label text: text is red too, but
broken into short runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScaleBar:
    length_px: int
    length_um: float | None
    mpp: float | None
    row: int
    col_start: int
    col_end: int

    def __repr__(self) -> str:
        mpp = f"{self.mpp:.4f}" if self.mpp else "unknown"
        return (f"ScaleBar({self.length_px}px = {self.length_um}um, "
                f"mpp={mpp})")


def find_red_mask(rgb: np.ndarray, corner_frac: float = 0.30) -> np.ndarray:
    """Saturated-red mask restricted to the bottom-left corner.

    Restricting to the corner avoids picking up genuinely red tissue, which
    matters for stains other than DAB and for any slide with haemorrhage.
    """
    a = np.asarray(rgb)
    if a.ndim != 3:
        raise ValueError("expected an RGB image")
    h, w = a.shape[:2]
    r = a[..., 0].astype(int)
    g = a[..., 1].astype(int)
    b = a[..., 2].astype(int)

    red = (r > 110) & (r - g > 50) & (r - b > 50)
    corner = np.zeros((h, w), dtype=bool)
    corner[int(h * (1 - corner_frac)):, :int(w * corner_frac)] = True
    return red & corner


def detect_scale_bar(rgb: np.ndarray, label_um: float | None = None) -> ScaleBar:
    """Locate the bar and compute microns per pixel.

    ``label_um`` is the number printed beside the bar (30, 50, 80 ...). It
    cannot be read reliably without OCR, so pass it in. If you have it in a
    filename or a lab log, use that; guessing it silently mis-scales every
    downstream measurement by a constant factor, which is the kind of error that
    survives all the way to a figure.
    """
    mask = find_red_mask(rgb)
    if not mask.any():
        raise ValueError("no red scale bar found in the bottom-left corner")

    best = (0, -1, -1, -1)      # length, row, start, end
    for row in np.unique(np.nonzero(mask)[0]):
        cols = np.nonzero(mask[row])[0]
        if len(cols) < 5:
            continue
        splits = np.split(cols, np.nonzero(np.diff(cols) > 3)[0] + 1)
        for run in splits:
            if len(run) > best[0]:
                best = (len(run), int(row), int(run[0]), int(run[-1]))

    length_px, row, c0, c1 = best
    mpp = (label_um / length_px) if (label_um and length_px) else None
    bar = ScaleBar(length_px, label_um, mpp, row, c0, c1)
    logger.info("%s", bar)
    return bar


def parse_label_from_filename(name: str) -> float | None:
    """Pull a micron value out of a filename if one is embedded."""
    m = re.search(r"(\d+)\s*(?:um|µm)", name, flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


def mask_overlay_region(
    rgb: np.ndarray, corner_frac: float = 0.30, pad: int = 12
) -> np.ndarray:
    """Boolean mask of the burned-in overlay, to EXCLUDE from all measurements.

    Covers the bounding box of every red pixel in the corner plus padding, which
    catches the bar and its text together.
    """
    red = find_red_mask(rgb, corner_frac)
    out = np.zeros(red.shape, dtype=bool)
    if not red.any():
        return out
    ys, xs = np.nonzero(red)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, red.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, red.shape[1])
    out[y0:y1, x0:x1] = True
    logger.info("masking overlay region rows %d-%d cols %d-%d (%.2f%% of image)",
                y0, y1, x0, x1, 100 * out.mean())
    return out


def has_counterstain(rgb: np.ndarray, marginal: float = 0.01,
                     adequate: float = 0.05) -> tuple[str, float]:
    """Grade counterstain adequacy. This gate decides what you can report.

    Measured by deconvolution, not an RGB heuristic: a pixel counts as
    counterstained nucleus if hematoxylin concentration exceeds 0.15 AND
    exceeds DAB, i.e. it is blue rather than brown. RGB thresholds fail here
    because pale lavender nuclei on a beige background are not "blue" by any
    simple channel rule.

    Returns (grade, fraction) where grade is one of:

      "adequate"  >=5% counterstained pixels. Negative nuclei are countable, so
                  percent-positive indices (Ki67 index, Allred) are available.
      "marginal"  1-5%. Some negative nuclei visible. A percentage computed
                  here is biased upward because faintly stained negatives are
                  missed. Report it only with the fraction quoted alongside.
      "absent"    <1%. There is NO denominator. Positive-cell density and
                  spatial pattern remain valid; percent-positive does not and
                  must not be back-calculated from DAB area.

    The last case is common in DAB-only captures taken to photograph positive
    staining. It is not a defect in the slide, but it does constrain the
    analysis, and the constraint has to be stated rather than worked around.
    """
    from .deconv import deconvolve_dab

    ch = deconvolve_dab(rgb)
    h, d = ch["hematoxylin"], ch["dab"]
    valid = ~mask_overlay_region(rgb)
    blue_nuclei = ((h > 0.15) & (h > d * 1.15))[valid]
    frac = float(blue_nuclei.mean())

    if frac >= adequate:
        grade = "adequate"
    elif frac >= marginal:
        grade = "marginal"
        logger.warning(
            "counterstain %.2f%% is marginal. Percent-positive computed from "
            "this image is biased upward, because faint negative nuclei are "
            "missed. Quote the fraction alongside any index you report.", frac * 100)
    else:
        grade = "absent"
        logger.warning(
            "counterstain %.2f%% is effectively absent. There is no denominator: "
            "percent-positive CANNOT be computed from this image. Positive-cell "
            "density and spatial pattern are still valid. Do not back-calculate "
            "an index from DAB area.", frac * 100)
    return grade, frac

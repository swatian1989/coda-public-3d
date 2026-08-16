"""Collagen and nerve fiber alignment from the eosin channel.

This is the CODA measurement that transfers directly to breast. CODA isolated
collagen using the eosin channel and computed a fiber anisotropy index within
small windows, where 1 means perfectly aligned fibers and 0 means an isotropic
matrix. They found alignment 2.2 to 2.5-fold higher in longitudinally versus
axially sectioned ducts, vessels and nerves.

Why it matters in breast specifically. Stromal collagen orientation at the
tumour boundary is an established prognostic axis: fibers running parallel to
the boundary versus perpendicular to it distinguish contained from invasive
behaviour. Pathologists assess this by eye. Quantifying it is straightforward
and nobody does it routinely.

Method here is the structure tensor. For each pixel, the local gradient
covariance matrix is smoothed over a window; its eigenvalues describe how
directional the local texture is. Coherence = (L1 - L2) / (L1 + L2), which is 0
for isotropic texture and 1 for a perfectly oriented one.

CRITICAL CAVEAT, carried over from CODA. Fiber alignment measured on a 2D
section depends on the angle at which the structure was cut. A duct sectioned
along its length shows aligned periductal collagen; the same duct sectioned
across shows apparently isotropic collagen. CODA could correct for this because
it had the 3D volume and could pick sectioning angle deliberately. On single
sections you cannot. Either restrict measurement to a defined anatomical
context (for example, the invasive front, where orientation is defined relative
to the tumour boundary rather than to a tube), or report the distribution across
many regions and treat sectioning angle as noise. Do not report a single
per-slide alignment number as if it were a property of the patient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class FiberConfig:
    gradient_sigma: float = 1.0     # derivative smoothing, in pixels
    tensor_sigma: float = 4.0       # integration scale: the "window"
    window_um: float = 50.0         # CODA used 2500 um^2 windows, so 50 x 50 um
    mpp: float = 0.5                # microns per pixel, 20x
    min_eosin: float = 0.15         # ignore pixels with little collagen signal


def structure_tensor(
    image: np.ndarray, cfg: FiberConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed gradient covariance components Jxx, Jxy, Jyy."""
    img = np.asarray(image, dtype=np.float64)
    gy, gx = np.gradient(ndimage.gaussian_filter(img, cfg.gradient_sigma))
    s = cfg.tensor_sigma
    return (
        ndimage.gaussian_filter(gx * gx, s),
        ndimage.gaussian_filter(gx * gy, s),
        ndimage.gaussian_filter(gy * gy, s),
    )


def fiber_orientation_map(
    eosin: np.ndarray, cfg: FiberConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel fiber orientation (radians) and coherence (0 to 1).

    Orientation is the direction of the minor eigenvector, which runs ALONG the
    fiber rather than across it. Values are in [0, pi) because fibers have no
    head or tail: 10 degrees and 190 degrees are the same fiber.
    """
    cfg = cfg or FiberConfig()
    jxx, jxy, jyy = structure_tensor(eosin, cfg)

    diff = jxx - jyy
    root = np.sqrt(diff ** 2 + 4 * jxy ** 2)
    l1 = 0.5 * (jxx + jyy + root)
    l2 = 0.5 * (jxx + jyy - root)

    coherence = np.divide(l1 - l2, l1 + l2,
                          out=np.zeros_like(l1), where=(l1 + l2) > 1e-12)
    orientation = 0.5 * np.arctan2(2 * jxy, diff)
    return np.mod(orientation, np.pi), np.clip(coherence, 0.0, 1.0)


def anisotropy_index(
    eosin: np.ndarray, cfg: FiberConfig | None = None
) -> float:
    """Fiber anisotropy index for one window. 0 isotropic, 1 fully aligned.

    Computed as the resultant vector length of the doubled orientation angles,
    weighted by coherence. Doubling handles the pi-periodicity of orientation:
    without it, fibers at 5 and 175 degrees would appear opposed rather than
    nearly parallel.

    Weighting by coherence means textureless regions contribute nothing rather
    than contributing a random angle, which would drag every measurement toward
    zero regardless of the tissue.
    """
    cfg = cfg or FiberConfig()
    orientation, coherence = fiber_orientation_map(eosin, cfg)

    mask = (eosin > cfg.min_eosin) & (coherence > 0.05)
    if mask.sum() < 100:
        return np.nan

    theta = orientation[mask] * 2.0
    w = coherence[mask]
    r = np.sqrt((w * np.cos(theta)).sum() ** 2 + (w * np.sin(theta)).sum() ** 2)
    return float(r / w.sum())


def tiled_anisotropy(
    eosin: np.ndarray, cfg: FiberConfig | None = None
) -> np.ndarray:
    """Anisotropy index per non-overlapping window across an image.

    Returns the per-window values so you can report a distribution rather than a
    single number. Given the sectioning-angle caveat, the distribution is the
    honest summary; the mean alone is not.
    """
    cfg = cfg or FiberConfig()
    side = max(int(round(cfg.window_um / cfg.mpp)), 16)
    h, w = eosin.shape
    out = []
    for y in range(0, h - side + 1, side):
        for x in range(0, w - side + 1, side):
            out.append(anisotropy_index(eosin[y:y + side, x:x + side], cfg))
    return np.array(out, dtype=float)


def boundary_relative_orientation(
    eosin: np.ndarray,
    tumour_mask: np.ndarray,
    cfg: FiberConfig | None = None,
) -> dict[str, float]:
    """Fiber orientation relative to the tumour boundary. Breast-specific.

    This is the measurement worth making in breast, and it sidesteps the
    sectioning-angle problem: orientation is defined relative to the local
    tumour boundary normal, not to an arbitrary image axis. Fibers running
    parallel to the boundary versus perpendicular to it is the distinction that
    carries prognostic weight in the tumour-associated collagen literature.

    Returns mean absolute angle to the boundary in degrees (0 parallel, 90
    perpendicular), the fraction of windows that are perpendicular-dominant, and
    the mean coherence.
    """
    cfg = cfg or FiberConfig()
    orientation, coherence = fiber_orientation_map(eosin, cfg)

    dist = ndimage.distance_transform_edt(~tumour_mask.astype(bool))
    gy, gx = np.gradient(ndimage.gaussian_filter(dist, cfg.tensor_sigma))
    boundary_normal = np.mod(np.arctan2(gy, gx), np.pi)

    band = (dist > 0) & (dist < (200.0 / cfg.mpp)) & (eosin > cfg.min_eosin) \
        & (coherence > 0.05)
    if band.sum() < 100:
        return {"mean_angle_deg": np.nan, "frac_perpendicular": np.nan,
                "mean_coherence": np.nan}

    delta = np.abs(orientation[band] - boundary_normal[band])
    delta = np.minimum(delta, np.pi - delta)          # fold to [0, pi/2]
    deg = np.degrees(delta)
    return {
        "mean_angle_deg": float(np.average(deg, weights=coherence[band])),
        "frac_perpendicular": float((deg > 45).mean()),
        "mean_coherence": float(coherence[band].mean()),
    }

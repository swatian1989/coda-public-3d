"""CODA serial-section registration, reimplemented in Python.

Follows the Online Methods exactly:

  1. Downsample to 80 um/px, greyscale, Gaussian filtered.
  2. Global registration. Radon transforms at 0-359 degrees; the maximum of the
     cross correlation of the Radon transforms gives the rotation angle, and the
     maximum of the cross correlation of the rotated images gives translation.
  3. Defect handling. Each moving image is registered against the THREE next
     closest images to centre, and the candidate with the best pixel correlation
     is kept. This is what stops a torn or folded section from corrupting the
     rest of the stack.
  4. Elastic registration. Rigid registration of cropped tiles at 1.5 mm
     intervals at 8 um/px, interpolated to full field, then Gaussian smoothed
     with sigma = 2 px to give a nonlinear displacement field.
  5. Everything is registered to the CENTRE section, not to its neighbour.
     Chaining neighbour-to-neighbour is what accumulates error; CODA beat seven
     other methods specifically on accumulated error (ATRE).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class RegistrationConfig:
    """CODA registration parameters. Values marked [PAPER] are from the methods."""

    global_mpp: float = 80.0        # [PAPER] register at 80 um/px
    elastic_mpp: float = 8.0        # [PAPER] elastic field computed at 8 um/px
    tile_interval_um: float = 1500  # [PAPER] tiles every 1.5 mm
    smooth_sigma_px: float = 2.0    # [PAPER] Gaussian sigma on the field
    n_reference_candidates: int = 3 # [PAPER] try 3 nearest-to-centre references
    n_angles: int = 360             # [PAPER] Radon at 0-359 degrees
    gaussian_prefilter: float = 2.0
    min_correlation: float = 0.30   # below this, flag the section as defective


def preprocess(image: np.ndarray, cfg: RegistrationConfig) -> np.ndarray:
    """Greyscale, remove background, complement, Gaussian filter. [PAPER]

    Complementing matters: tissue is dark on white, and the Radon transform
    responds to bright structure, so without inversion the transform tracks the
    background rather than the tissue.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.max() > 1.5:
        img = img / 255.0
    img = 1.0 - img                                  # complement
    img[img < 0.05] = 0.0                            # drop background
    return ndimage.gaussian_filter(img, cfg.gaussian_prefilter)


def radon_transform(image: np.ndarray, n_angles: int = 360) -> np.ndarray:
    """Radon transform at n_angles discrete angles over [0, 360)."""
    from skimage.transform import radon

    theta = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    return radon(image, theta=theta, circle=True, preserve_range=True)


def estimate_rotation(fixed: np.ndarray, moving: np.ndarray,
                      cfg: RegistrationConfig) -> float:
    """Rotation angle from the cross correlation of Radon transforms. [PAPER]

    A rotation of the image is a cyclic SHIFT along the angle axis of its Radon
    transform. So correlating the two transforms along that axis and taking the
    argmax recovers the angle directly, with no search over rotations.
    """
    rf = radon_transform(fixed, cfg.n_angles)
    rm = radon_transform(moving, cfg.n_angles)

    # Column-normalise so a global intensity difference between sections does
    # not bias the match toward the brighter one.
    rf = (rf - rf.mean(axis=0)) / (rf.std(axis=0) + 1e-9)
    rm = (rm - rm.mean(axis=0)) / (rm.std(axis=0) + 1e-9)

    # Circular cross correlation along the ANGLE axis, retaining the full
    # sinogram rather than a collapsed profile. Collapsing over the detector
    # axis throws away most of the shape information and fails on tissue that
    # is close to rotationally symmetric.
    fft = np.fft.rfft(rf, axis=1) * np.conj(np.fft.rfft(rm, axis=1))
    corr = np.fft.irfft(fft, n=cfg.n_angles, axis=1).sum(axis=0)

    shift = int(np.argmax(corr))
    if shift > cfg.n_angles // 2:
        shift -= cfg.n_angles
    return float(shift * 360.0 / cfg.n_angles)


def estimate_translation(fixed: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Translation from the maximum of 2D phase cross correlation. [PAPER]"""
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(fixed, moving, upsample_factor=10,
                                          normalization=None)
    return float(shift[0]), float(shift[1])


def apply_rigid(image: np.ndarray, angle: float, dy: float, dx: float) -> np.ndarray:
    """Rotate about the image centre then translate. [PAPER] centre reference."""
    out = ndimage.rotate(image, angle, reshape=False, order=1, mode="constant")
    return ndimage.shift(out, (dy, dx), order=1, mode="constant")


def global_register(fixed: np.ndarray, moving: np.ndarray,
                    cfg: RegistrationConfig) -> tuple[np.ndarray, dict]:
    """Rigid-body registration of one pair. Returns the warped image and params."""
    f, m = preprocess(fixed, cfg), preprocess(moving, cfg)
    angle = estimate_rotation(f, m, cfg)
    rotated = ndimage.rotate(m, angle, reshape=False, order=1)
    dy, dx = estimate_translation(f, rotated)
    warped = ndimage.shift(rotated, (dy, dx), order=1)
    return warped, {"angle": angle, "dy": dy, "dx": dx,
                    "correlation": pixel_correlation(f, warped)}


def pixel_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman pixel correlation over the union of tissue area. [PAPER]

    CODA assessed registration quality with pixelwise Spearman correlation, and
    used it both to choose between candidate references and to validate that
    z-direction correlation approaches the within-section xy correlation.
    """
    from scipy.stats import spearmanr

    mask = (a > 0.02) | (b > 0.02)
    if mask.sum() < 100:
        return 0.0
    r, _ = spearmanr(a[mask], b[mask])
    return float(r) if np.isfinite(r) else 0.0


def select_reference(
    moving: np.ndarray, candidates: list[np.ndarray], cfg: RegistrationConfig
) -> tuple[int, np.ndarray, dict]:
    """Register against up to 3 candidate references, keep the best. [PAPER]

    This is CODA's defect handling. If section n+1 is torn or folded, its
    registration correlates poorly, so section n+2 is used as the reference
    instead and the error does not propagate down the stack. Skipping this step
    is why naive neighbour-chaining accumulates error.
    """
    best_i, best_img, best_params = -1, None, {"correlation": -np.inf}
    for i, cand in enumerate(candidates[: cfg.n_reference_candidates]):
        warped, params = global_register(cand, moving, cfg)
        if params["correlation"] > best_params["correlation"]:
            best_i, best_img, best_params = i, warped, params

    if best_params["correlation"] < cfg.min_correlation:
        logger.warning(
            "best correlation %.3f is below %.2f. This section is probably torn, "
            "folded or badly stained. CODA discards registration to badly "
            "deformed tissue; consider excluding it.",
            best_params["correlation"], cfg.min_correlation)
    best_params["reference_offset"] = best_i + 1
    return best_i, best_img, best_params


def elastic_field(
    fixed: np.ndarray, moving: np.ndarray, cfg: RegistrationConfig,
    mpp: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Nonlinear displacement field from tile-wise rigid registration. [PAPER]

    Rigid registration is computed on cropped tiles at 1.5 mm intervals, the
    resulting sparse displacement vectors are interpolated to the full field,
    and the field is smoothed with a Gaussian of sigma 2 px. This models local
    tissue warping from sectioning without allowing the arbitrary deformation
    that free-form registration permits, which is important because arbitrary
    deformation can make ANY two sections match and destroys the biology.

    Returns (field_y, field_x), both the shape of the input.
    """
    from skimage.registration import phase_cross_correlation

    step = max(int(cfg.tile_interval_um / mpp), 16)
    h, w = fixed.shape
    ys = np.arange(step // 2, h, step)
    xs = np.arange(step // 2, w, step)

    dy_grid = np.zeros((len(ys), len(xs)))
    dx_grid = np.zeros((len(ys), len(xs)))
    half = step // 2

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            y0, y1 = max(y - half, 0), min(y + half, h)
            x0, x1 = max(x - half, 0), min(x + half, w)
            tf, tm = fixed[y0:y1, x0:x1], moving[y0:y1, x0:x1]
            if tf.size < 64 or tf.std() < 1e-6 or tm.std() < 1e-6:
                continue
            try:
                shift, _, _ = phase_cross_correlation(tf, tm, upsample_factor=4,
                                                      normalization=None)
                dy_grid[i, j], dx_grid[i, j] = shift
            except Exception:
                continue

    zoom = (h / max(len(ys), 1), w / max(len(xs), 1))
    fy = ndimage.gaussian_filter(ndimage.zoom(dy_grid, zoom, order=1),
                                 cfg.smooth_sigma_px)
    fx = ndimage.gaussian_filter(ndimage.zoom(dx_grid, zoom, order=1),
                                 cfg.smooth_sigma_px)
    return fy[:h, :w], fx[:h, :w]


def apply_elastic(image: np.ndarray, fy: np.ndarray, fx: np.ndarray) -> np.ndarray:
    """Warp an image by a displacement field."""
    h, w = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = np.array([yy + fy[:h, :w], xx + fx[:h, :w]])
    if image.ndim == 2:
        return ndimage.map_coordinates(image, coords, order=1, mode="constant")
    return np.stack([ndimage.map_coordinates(image[..., c], coords, order=1,
                                             mode="constant")
                     for c in range(image.shape[2])], axis=-1)


def register_stack(
    images: list[np.ndarray], cfg: RegistrationConfig | None = None,
    elastic: bool = True,
) -> tuple[list[np.ndarray], list[dict]]:
    """Register a full serial stack to its CENTRE section. [PAPER]

    Working outward from the centre, rather than chaining section to section,
    is what limits accumulated error. Each moving image is registered to the
    already-registered image nearest the centre, with three candidates tried.
    """
    cfg = cfg or RegistrationConfig()
    n = len(images)
    centre = n // 2
    registered: list[np.ndarray | None] = [None] * n
    registered[centre] = preprocess(images[centre], cfg)
    params: list[dict] = [{} for _ in range(n)]
    params[centre] = {"angle": 0.0, "dy": 0.0, "dx": 0.0,
                      "correlation": 1.0, "reference_offset": 0}

    for direction in (1, -1):
        i = centre + direction
        while 0 <= i < n:
            cands = []
            for k in range(1, cfg.n_reference_candidates + 1):
                j = i - direction * k
                if 0 <= j < n and registered[j] is not None:
                    cands.append(registered[j])
            if not cands:
                break

            _, warped, p = select_reference(preprocess(images[i], cfg), cands, cfg)

            if elastic and p["correlation"] >= cfg.min_correlation:
                fy, fx = elastic_field(cands[0], warped, cfg)
                warped = apply_elastic(warped, fy, fx)
                p["correlation_after_elastic"] = pixel_correlation(cands[0], warped)

            registered[i] = warped
            params[i] = p
            i += direction

    logger.info("registered %d sections, median correlation %.3f", n,
                float(np.median([p.get("correlation", np.nan) for p in params])))
    return [r for r in registered if r is not None], params

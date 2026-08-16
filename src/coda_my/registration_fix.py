"""Replacement rigid estimator for serial registration, and a two-scale driver.

registration.py is a tested module and is NOT modified. This sits beside it,
reuses its public helpers (preprocess, apply_rigid, pixel_correlation,
elastic_field, apply_elastic), and replaces only the parts measured to fail.

THREE DEFECTS THIS ADDRESSES, ALL MEASURED RATHER THAN ASSUMED

1. Rotation estimation. registration.estimate_rotation infers the angle from
   the Radon transform. On the Kartasalo mouse liver series it averages 37.5
   degrees of error against the rotation implied by the operator fiducials and
   lands within five degrees on 3 of 15 consecutive pairs, where a uniform
   guess averages about ninety. Liver is a compact, near-convex, texturally
   homogeneous object, so its Radon transform carries little orientation
   signal. The consequence is severe: registering the stack leaves landmarks
   2544 um apart against 727 um for applying no transform at all.

   Replaced here by a direct search. Rotation and translation are coupled, so
   they are not estimated separately: for each candidate angle the moving image
   is rotated, translation is recovered by phase correlation, and the pair is
   scored by the same pixel correlation the pipeline uses to judge quality. The
   angle that maximises the score wins. Searching the objective you actually
   care about avoids relying on a proxy that this tissue does not satisfy.

2. Dead configuration. RegistrationConfig declares global_mpp = 80.0 marked
   [PAPER] and nothing reads it, so the rigid stage inherits whatever
   resolution the caller supplies, which must be the fine scale the elastic
   stage needs. Here the two stages genuinely run at their own scales: rigid on
   a coarse copy, elastic on the fine one, with the recovered translation
   rescaled between them.

3. Discarded elastic field. register_stack applies the elastic displacement to
   the image and drops it, so it cannot be replayed onto point coordinates and
   landmark error can only ever describe the rigid stage. The driver here
   returns the fields, so target registration error can be evaluated for the
   full transform.

Nothing here is asserted to be better. run_kartasalo_rotation_fix.py measures
this estimator against the same fiducial ground truth that condemned the
original, and the comparison is the justification.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .registration import (
    RegistrationConfig, apply_elastic, apply_rigid, elastic_field,
    pixel_correlation, preprocess,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Angular search settings. Coarse pass, then refinement around the winner."""

    coarse_step_deg: float = 4.0
    fine_half_width_deg: float = 4.0
    fine_step_deg: float = 0.5
    upsample: int = 10                 # phase-correlation subpixel factor
    max_abs_deg: float = 180.0
    """Restrict the search to plus/minus this rotation.

    Serial sections are placed on slides by hand and differ by modest
    rotations, so a candidate 130 degrees away is a spurious maximum rather
    than a plausible alignment. Unrestricted search on this liver series found
    exactly such maxima on 3 of 20 pairs, each 128 to 141 degrees off, and each
    carrying a visibly lower correlation than any correct match.

    Left at 180 the search is unrestricted. Any narrower bound is a prior about
    the specimen and must be stated, and checked against the data so that it
    cannot clip a real rotation: across all 46 pairs of this series the largest
    fiducial-implied rotation is 34.1 degrees, so a 45 degree bound excludes no
    true value here.
    """


def _translate_and_score(fixed: np.ndarray, rotated: np.ndarray,
                         upsample: int) -> tuple[float, float, float]:
    """Phase-correlate for translation, then score with pixel correlation."""
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(fixed, rotated, upsample_factor=upsample,
                                          normalization=None)
    dy, dx = float(shift[0]), float(shift[1])
    moved = ndimage.shift(rotated, (dy, dx), order=1, mode="constant")
    return dy, dx, pixel_correlation(fixed, moved)


def estimate_rigid_search(
    fixed: np.ndarray, moving: np.ndarray,
    cfg: RegistrationConfig | None = None, search: SearchConfig | None = None,
) -> dict:
    """Rigid transform by direct search over rotation, preprocessed inputs assumed.

    Parameters
    ----------
    fixed, moving
        Already passed through ``preprocess``.

    Returns
    -------
    dict
        ``angle``, ``dy``, ``dx``, ``correlation``, and ``n_angles_tried``.
        Angle and translation follow the convention of ``apply_rigid``.
    """
    cfg = cfg or RegistrationConfig()
    search = search or SearchConfig()

    best = {"angle": 0.0, "dy": 0.0, "dx": 0.0, "correlation": -np.inf}
    tried = 0

    if search.max_abs_deg >= 180.0:
        angles = np.arange(0.0, 360.0, search.coarse_step_deg)
    else:
        angles = np.arange(-search.max_abs_deg, search.max_abs_deg + 1e-9,
                           search.coarse_step_deg)
    for a in angles:
        rot = ndimage.rotate(moving, a, reshape=False, order=1, mode="constant")
        dy, dx, c = _translate_and_score(fixed, rot, search.upsample)
        tried += 1
        if c > best["correlation"]:
            best = {"angle": float(a), "dy": dy, "dx": dx, "correlation": float(c)}

    lo = best["angle"] - search.fine_half_width_deg
    hi = best["angle"] + search.fine_half_width_deg
    for a in np.arange(lo, hi + 1e-9, search.fine_step_deg):
        rot = ndimage.rotate(moving, a, reshape=False, order=1, mode="constant")
        dy, dx, c = _translate_and_score(fixed, rot, search.upsample)
        tried += 1
        if c > best["correlation"]:
            best = {"angle": float(a % 360.0), "dy": dy, "dx": dx,
                    "correlation": float(c)}

    best["n_angles_tried"] = tried
    return best


def select_reference_search(
    moving: np.ndarray, candidates: list[np.ndarray],
    cfg: RegistrationConfig | None = None, search: SearchConfig | None = None,
) -> tuple[int, dict]:
    """Try each already-registered candidate, keep the best-scoring alignment."""
    best_i, best_p = 0, None
    for i, cand in enumerate(candidates):
        p = estimate_rigid_search(cand, moving, cfg, search)
        if best_p is None or p["correlation"] > best_p["correlation"]:
            best_i, best_p = i, p
    best_p["reference_offset"] = best_i
    return best_i, best_p


def register_stack_two_scale(
    coarse: np.ndarray, fine: np.ndarray, coarse_mpp: float, fine_mpp: float,
    cfg: RegistrationConfig | None = None, search: SearchConfig | None = None,
    elastic: bool = True,
) -> tuple[np.ndarray, list[dict], list[tuple[np.ndarray, np.ndarray] | None]]:
    """Register a stack centre-out, rigid at the coarse scale, elastic at the fine one.

    Parameters
    ----------
    coarse, fine
        The same sections at two resolutions, (n, H, W) each.
    coarse_mpp, fine_mpp
        Microns per pixel for each, used to rescale the recovered translation.

    Returns
    -------
    (registered_fine, params, fields)
        ``fields[i]`` is the elastic displacement (fy, fx) applied to section i,
        or None. Returning it is the point: it lets point coordinates receive
        the same transform the image received.
    """
    cfg = cfg or RegistrationConfig()
    search = search or SearchConfig()
    n = len(coarse)
    scale = coarse_mpp / fine_mpp          # coarse pixel -> fine pixel

    reg_c: list[np.ndarray | None] = [None] * n
    reg_f: list[np.ndarray | None] = [None] * n
    params: list[dict] = [{} for _ in range(n)]
    fields: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n

    centre = n // 2
    reg_c[centre] = preprocess(coarse[centre], cfg)
    reg_f[centre] = preprocess(fine[centre], cfg)
    params[centre] = {"angle": 0.0, "dy": 0.0, "dx": 0.0, "correlation": 1.0,
                      "reference_offset": 0, "dy_fine": 0.0, "dx_fine": 0.0}

    for direction in (1, -1):
        i = centre + direction
        while 0 <= i < n:
            cands_c, cands_f = [], []
            for k in range(1, cfg.n_reference_candidates + 1):
                j = i - direction * k
                if 0 <= j < n and reg_c[j] is not None:
                    cands_c.append(reg_c[j]); cands_f.append(reg_f[j])
            if not cands_c:
                break

            mov_c = preprocess(coarse[i], cfg)
            _, p = select_reference_search(mov_c, cands_c, cfg, search)

            # rigid solved coarse, applied at both scales
            dy_f, dx_f = p["dy"] * scale, p["dx"] * scale
            p["dy_fine"], p["dx_fine"] = dy_f, dx_f
            warped_c = apply_rigid(mov_c, p["angle"], p["dy"], p["dx"])
            warped_f = apply_rigid(preprocess(fine[i], cfg), p["angle"], dy_f, dx_f)
            ref_f = cands_f[p["reference_offset"]]
            p["correlation_fine"] = pixel_correlation(ref_f, warped_f)

            if elastic and p["correlation"] >= cfg.min_correlation:
                fy, fx = elastic_field(ref_f, warped_f, cfg, mpp=fine_mpp)
                warped_f = apply_elastic(warped_f, fy, fx)
                fields[i] = (fy, fx)
                p["correlation_after_elastic"] = pixel_correlation(ref_f, warped_f)

            reg_c[i], reg_f[i] = warped_c, warped_f
            params[i] = p
            i += direction

    out = np.stack([r if r is not None else np.zeros_like(reg_f[centre])
                    for r in reg_f])
    logger.info("two-scale registration: rigid at %.1f um/px, elastic at %.2f um/px",
                coarse_mpp, fine_mpp)
    return out, params, fields


def transform_points(points: np.ndarray, shape: tuple[int, int], angle: float,
                     dy: float, dx: float,
                     field: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Apply the same rigid (and optional elastic) transform to (y, x) points.

    The elastic part is a displacement field defined on the output grid, so a
    point is moved by sampling the field at its rigid-transformed location.
    """
    h, w = shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    t = np.deg2rad(angle)
    c, s = np.cos(t), np.sin(t)
    y, x = points[:, 0] - cy, points[:, 1] - cx
    out = np.column_stack([cy + y * c - x * s + dy, cx + y * s + x * c + dx])
    if field is not None:
        fy, fx = field
        yy = np.clip(out[:, 0], 0, fy.shape[0] - 1)
        xx = np.clip(out[:, 1], 0, fy.shape[1] - 1)
        out[:, 0] += ndimage.map_coordinates(fy, [yy, xx], order=1, mode="nearest")
        out[:, 1] += ndimage.map_coordinates(fx, [yy, xx], order=1, mode="nearest")
    return out


def procrustes_rigid(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """Best rigid alignment of B onto A. Returns (mean residual, angle in degrees).

    In two dimensions the optimal rotation has a closed form and there is no
    reason to route it through an SVD whose convention is easy to invert. The
    inverted form is not a harmless slip: it applies the opposite rotation,
    still returns a plausible residual, and on this liver series reported a
    518 um rigid floor where the true value is 74 um. That single number
    reversed the conclusion about whether the limit was tissue deformation or
    the algorithm, so the closed form is used and checked against a brute force
    angular sweep.

    Parameters
    ----------
    A, B
        (n, 2) point sets in the same units. B is aligned onto A.

    Returns
    -------
    (residual, angle_deg)
        Mean Euclidean residual after the optimal rotation and translation, in
        the input units, and the rotation applied.
    """
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    # theta = atan2( sum b x a , sum b . a )
    cross = float(np.sum(Bc[:, 0] * Ac[:, 1] - Bc[:, 1] * Ac[:, 0]))
    dot = float(np.sum(Bc[:, 0] * Ac[:, 0] + Bc[:, 1] * Ac[:, 1]))
    theta = np.arctan2(cross, dot)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    resid = float(np.linalg.norm(Ac - Bc @ R.T, axis=1).mean())
    return resid, float(np.degrees(theta))

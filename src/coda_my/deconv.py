"""Colour deconvolution for H&E and DAB-IHC.

Ruifrok and Johnston optical density deconvolution. This is the entry point for
every downstream CODA-style measurement: fiber alignment reads the eosin
channel, cell detection reads the hematoxylin channel, and IHC quantification
reads the DAB channel.

CODA estimated stain vectors per image by k-means over optical densities rather
than using fixed textbook vectors. That matters across cohorts: TCGA slides and
USM slides were stained in different labs with different reagents, and fixed
vectors bake that difference into every downstream measurement. Per-image
estimation removes a large part of it before any comparison is made.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Reference stain vectors (Ruifrok and Johnston). Fallback only, and used to
# match estimated vectors to stain identity.
STAIN_VECTORS = {
    "hematoxylin": np.array([0.650, 0.704, 0.286]),
    "eosin": np.array([0.072, 0.990, 0.105]),
    "dab": np.array([0.268, 0.570, 0.776]),
}


def rgb_to_od(rgb: np.ndarray, background: int = 255) -> np.ndarray:
    """Convert RGB to optical density. OD = -log10(I / I0)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() <= 1.0:
        rgb = rgb * 255.0
    rgb = np.maximum(rgb, 1.0)
    return -np.log10(rgb / background)


def estimate_stain_vectors(
    rgb: np.ndarray,
    n_clusters: int = 100,
    od_threshold: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Estimate per-image stain vectors by k-means over optical densities.

    Follows CODA: cluster the optical densities of the image, take the most
    common blue-favoured cluster as hematoxylin and the most common red-favoured
    cluster as eosin.

    ``od_threshold`` excludes background. Without it the white background
    dominates the clustering and the estimated vectors are meaningless.
    """
    from sklearn.cluster import MiniBatchKMeans

    od = rgb_to_od(rgb).reshape(-1, 3)
    tissue = od[od.sum(axis=1) > od_threshold]
    if len(tissue) < n_clusters * 10:
        logger.warning("only %d tissue pixels, using reference vectors", len(tissue))
        return {k: STAIN_VECTORS[k].copy() for k in ("hematoxylin", "eosin")}

    rng = np.random.default_rng(seed)
    sample = tissue[rng.choice(len(tissue), min(200_000, len(tissue)), replace=False)]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init=3).fit(sample)

    centres = km.cluster_centers_
    unit = centres / np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-9)
    sizes = np.bincount(km.labels_, minlength=n_clusters).astype(float)
    common = (sizes / sizes.sum()) > 0.002

    blue = np.where(common, unit[:, 2] - unit[:, 0], -np.inf)
    red = np.where(common, unit[:, 0] - unit[:, 2], -np.inf)
    return {"hematoxylin": unit[int(np.argmax(blue))],
            "eosin": unit[int(np.argmax(red))]}


def deconvolve(
    rgb: np.ndarray,
    stains: dict[str, np.ndarray] | None = None,
    stain_names: tuple[str, ...] = ("hematoxylin", "eosin"),
) -> dict[str, np.ndarray]:
    """Separate an RGB image into per-stain concentration maps.

    The third basis vector is the cross product of the first two, the standard
    construction for a two-stain image, which makes the 3x3 system invertible.

    Returns concentration maps: higher values mean more stain, unlike the raw
    RGB channels where more stain means lower intensity.
    """
    stains = stains or {k: STAIN_VECTORS[k] for k in stain_names}
    v = np.stack([stains[n] / np.linalg.norm(stains[n]) for n in stain_names])
    if len(v) == 2:
        third = np.cross(v[0], v[1])
        v = np.vstack([v, third / np.linalg.norm(third)])

    od = rgb_to_od(rgb)
    h, w = od.shape[:2]
    conc = np.linalg.solve(v.T, od.reshape(-1, 3).T).T.reshape(h, w, 3)
    return {name: conc[:, :, i] for i, name in enumerate(stain_names)}


def deconvolve_dab(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Separate hematoxylin counterstain from DAB chromogen for IHC.

    For ER, PR, HER2 and Ki67. DAB is the readout; hematoxylin gives the
    denominator (all nuclei) so positivity is a fraction of cells rather than an
    area, which is what pathologists actually report.
    """
    return deconvolve(rgb, stain_names=("hematoxylin", "dab"))

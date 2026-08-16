"""Loader for the Kartasalo PROSTATE series, whose landmarks work differently.

The prostate and liver series ship in the same archive and look superficially
alike, and treating them alike produces wrong numbers silently. The difference
is in the landmarks.

LIVER
    Four laser-cut holes were driven through the block before embedding, so the
    same four physical objects appear on every section. The table has one row
    per section and eight coordinate columns, and accumulated error is the
    residual scatter of each hole about a straight line fitted down z.

PROSTATE
    No holes. An operator marked corresponding anatomical points on ADJACENT
    PAIRS of sections. The table has one row per PAIR, 259 rows for 260
    sections, and sixteen coordinate columns: Y1/X1 is a point on section n and
    Y2/X2 is that same point located again on section n+1. A landmark exists
    only for the pair it was drawn on; it does not persist through the stack.

    Accumulated error therefore cannot be the residual about a line through z,
    because no landmark spans z. The benchmark defines it instead as the
    cumulative magnitude of the mean pairwise displacement vector: each pair
    contributes a mean vector, the vectors are summed along the stack, and the
    length of the running sum is the drift accumulated to that point.

    The distinction matters. Residual-about-a-line and cumulative-vector answer
    different questions, and the cumulative form is the one that detects a stack
    bending steadily in one direction, which is the failure that ruins a
    reconstruction while leaving every individual pair looking well aligned.

THE TWO OBSERVERS ARE NOT REPEATED MEASUREMENTS

For the liver the two observers annotated the SAME four laser-cut holes, which
are objective physical objects, so the distance between them is an annotation
noise floor and was reported as one: 6.8 um median.

For the prostate there are no holes and each observer chose their own
anatomical features. Measured across the first 60 pairs, observer 1's point k
sits a median 5750 um from observer 2's point k, and 1286 um from the NEAREST
of observer 2's points, so they are not even the same features in a different
order. These are two independent landmark sets.

Inter-observer distance therefore does NOT bound annotation precision here and
must not be quoted as a floor. The published table reflects this by reporting
TRE1 and TRE2 separately and fitting LS 1 and LS 2 to each observer alone. Use
one observer's landmarks throughout an analysis and report which.

`coda_my.qc.accumulated_tre` implements the liver form. It is a tested module
and is not modified; the pairwise form lives here instead.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# [PAPER, NOT METADATA] Kartasalo et al. Bioinformatics 2018;34:3013-21.
NATIVE_MPP_UM = 0.46
SECTION_THICKNESS_UM = 5.0
N_PROSTATE_SECTIONS = 260
N_FIDUCIALS = 4


def load_pairwise_fiducials(path: str | Path) -> pd.DataFrame:
    """Read one observer's pairwise landmark table into long form.

    Returns
    -------
    pandas.DataFrame
        Columns ``pair`` (1-based index of the section pair), ``section_a``,
        ``section_b``, ``fiducial``, ``y_a``, ``x_a``, ``y_b``, ``x_b``. Pixel
        units in the native grid.
    """
    df = pd.read_csv(path, sep="\t")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    rows = []
    for i, r in df.iterrows():
        fn = str(r["Filename"])
        m = re.match(r"0*(\d+)", fn)
        a = int(m.group(1)) if m else i + 1
        for k in range(1, N_FIDUCIALS + 1):
            ka = (f"Fiducial{k} Y1", f"Fiducial{k} X1")
            kb = (f"Fiducial{k} Y2", f"Fiducial{k} X2")
            if ka[0] in df.columns and kb[0] in df.columns:
                rows.append({"pair": a, "section_a": a, "section_b": a + 1,
                             "fiducial": k,
                             "y_a": float(r[ka[0]]), "x_a": float(r[ka[1]]),
                             "y_b": float(r[kb[0]]), "x_b": float(r[kb[1]])})
    out = pd.DataFrame(rows)
    logger.info("prostate fiducials %s: %d pairs x %d points",
                Path(path).name, out["pair"].nunique(), N_FIDUCIALS)
    return out


def pair_arrays(df: pd.DataFrame, pair: int,
                downsample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """The four matched points for one pair, as (a, b), each (4, 2) in (y, x)."""
    g = df[df["pair"] == pair].sort_values("fiducial")
    a = g[["y_a", "x_a"]].to_numpy(float) / downsample
    b = g[["y_b", "x_b"]].to_numpy(float) / downsample
    return a, b


def pairwise_tre(df: pd.DataFrame, transforms: list | None = None,
                 shape: tuple[int, int] | None = None,
                 mpp: float = NATIVE_MPP_UM,
                 downsample: int = 1) -> pd.DataFrame:
    """Target registration error for every adjacent pair, in microns.

    Parameters
    ----------
    transforms
        Optional list of per-section ``(angle, dy, dx, field)`` tuples in the
        registered frame. If omitted, the unregistered error is returned, which
        is the baseline every method must beat.
    """
    from coda_my.registration_fix import transform_points

    rows = []
    for pair in sorted(df["pair"].unique()):
        a, b = pair_arrays(df, pair, downsample)
        if transforms is not None and shape is not None:
            ia, ib = pair - 1, pair
            if ib >= len(transforms):
                continue
            ta, tb = transforms[ia], transforms[ib]
            a = transform_points(a, shape, *ta)
            b = transform_points(b, shape, *tb)
        d = np.linalg.norm(a - b, axis=1) * mpp
        vec = (b - a).mean(axis=0) * mpp
        rows.append({"pair": int(pair), "tre_mean_um": float(d.mean()),
                     "tre_median_um": float(np.median(d)),
                     "tre_max_um": float(d.max()),
                     "vec_y_um": float(vec[0]), "vec_x_um": float(vec[1]),
                     "n_landmarks": int(len(d))})
    return pd.DataFrame(rows)


def accumulated_tre_pairwise(tre: pd.DataFrame,
                             reference_pair: int | None = None) -> pd.DataFrame:
    """Cumulative resultant of the mean pairwise displacement vectors.

    This is the accumulated error definition the benchmark specifies for
    landmarks that exist only on pairs. Each pair contributes a mean
    displacement vector; the vectors are summed outward from a reference pair,
    and the magnitude of the running sum is the drift accumulated to that point.

    Summing vectors rather than magnitudes is the whole point. Errors that
    alternate in direction cancel and do not accumulate, while errors that share
    a direction add, which is exactly the bending that destroys a reconstruction
    while every individual pair still looks well aligned.
    """
    tre = tre.sort_values("pair").reset_index(drop=True)
    ref = reference_pair if reference_pair is not None else len(tre) // 2
    rows = []
    for direction in (1, -1):
        run = np.zeros(2)
        i = ref
        while 0 <= i < len(tre):
            if i != ref:
                r = tre.iloc[i]
                run = run + direction * np.array([r.vec_y_um, r.vec_x_um])
            rows.append({"pair": int(tre.iloc[i]["pair"]),
                         "distance_from_reference": abs(i - ref),
                         "atre_um": float(np.hypot(*run))})
            i += direction
    out = pd.DataFrame(rows).drop_duplicates("pair").sort_values("pair")
    return out.reset_index(drop=True)


def section_paths(image_dir: str | Path) -> list[Path]:
    """TIFF paths sorted by the numeric z index in the filename."""
    return sorted(Path(image_dir).glob("*.tif"),
                  key=lambda p: int(re.match(r"0*(\d+)", p.stem).group(1)))

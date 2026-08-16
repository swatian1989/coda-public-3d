"""Loader for the Kartasalo 3D-histology benchmark data (liver and prostate).

Deliberately OUTSIDE the tested modules. `registration.py`, `qc.py` and the
rest stay untouched; this only turns files on disk into the arrays they expect.

SOURCE
    Etsin c76335fa-cdcf-4ddc-ab1c-1882bad82861, "Supplementary Data: A
    Comparison of Reconstruction Algorithms for 3D Histology", Tampere
    University, CC BY 4.0, access type Open. Published as one 63.79 GB zip.

LAYOUT (observed by streaming the archive, not assumed)
    Data_to_IDA/fiducialcoordinates_liver_observer1.txt
    Data_to_IDA/fiducialcoordinates_liver_observer2.txt
    Data_to_IDA/fiducialcoordinates_prostate_observer1.txt
    Data_to_IDA/fiducialcoordinates_prostate_observer2.txt
    Data_to_IDA/liver/001.tif ... 047.tif

    Liver sections are 19457 x 21249 RGB, PackBits, ~345 MB each, ~16 GB for
    the stack. Z order is the zero-padded filename, which is also the key used
    in the fiducial tables.

COORDINATES AND UNITS, THE PART THAT IS EASY TO GET WRONG
    Fiducial tables are tab-separated with a header row and columns in the
    order Filename, then Y then X for each of four fiducials. Y BEFORE X. They
    are floating point PIXEL coordinates in the native grid of the matching
    TIFF, so any downsampling applied to the images must be applied to them
    too.

    The four liver fiducials are the laser-cut holes driven through the block
    before embedding, so they run through the whole series and give ground
    truth that is independent of tissue appearance. Two observers annotated
    them independently, which also bounds annotation error.

    The TIFFs carry NO physical calibration: XResolution and YResolution are
    72 with ResolutionUnit=inch, the generic placeholder, not a microscope
    value. Pixel size therefore cannot be confirmed from the file metadata and
    is supplied by the caller. The published values for this material are
    0.46 um/px at 20x and 5 um sections; both are recorded here as constants
    with their provenance, and neither is treated as verified from the data.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None          # these are 413 Mpx; the bomb guard is wrong here

logger = logging.getLogger(__name__)

# [PAPER, NOT METADATA] Kartasalo et al. Bioinformatics 2018;34:3013-21.
# The TIFFs do not carry calibration, so these come from the article text.
NATIVE_MPP_UM = 0.46
SECTION_THICKNESS_UM = 5.0
N_LIVER_SECTIONS = 47
N_FIDUCIALS = 4


@dataclass
class StackMeta:
    """What was loaded and at what scale, so downstream units stay honest."""

    n_sections: int
    downsample: int
    mpp_um: float
    section_um: float
    native_shape: tuple[int, int]
    loaded_shape: tuple[int, int]
    mpp_provenance: str = "paper text, not file metadata"


def load_fiducials(path: str | Path) -> pd.DataFrame:
    """Read one observer's fiducial table into long form.

    Parameters
    ----------
    path
        A ``fiducialcoordinates_*.txt`` file.

    Returns
    -------
    pandas.DataFrame
        Columns ``section`` (int, from the filename), ``filename``,
        ``fiducial`` (1-based), ``y``, ``x``. Pixel units in the native grid.
    """
    df = pd.read_csv(path, sep="\t")
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    rows = []
    for _, r in df.iterrows():
        fn = str(r["Filename"])
        m = re.match(r"0*(\d+)", fn)
        sec = int(m.group(1)) if m else -1
        for k in range(1, N_FIDUCIALS + 1):
            ky, kx = f"Fiducial{k} Y", f"Fiducial{k} X"
            if ky in df.columns and kx in df.columns:
                rows.append({"section": sec, "filename": fn, "fiducial": k,
                             "y": float(r[ky]), "x": float(r[kx])})
    out = pd.DataFrame(rows)
    logger.info("fiducials %s: %d sections x %d points",
                Path(path).name, out["section"].nunique(), N_FIDUCIALS)
    return out


def fiducial_array(df: pd.DataFrame, section: int, downsample: int = 1) -> np.ndarray:
    """Return one section's fiducials as an (N_FIDUCIALS, 2) array of (y, x)."""
    g = df[df["section"] == section].sort_values("fiducial")
    return g[["y", "x"]].to_numpy(float) / downsample


def fiducial_series(df: pd.DataFrame, downsample: int = 1) -> list[np.ndarray]:
    """All sections' fiducials in z order, for ``accumulated_tre``."""
    return [fiducial_array(df, s, downsample)
            for s in sorted(df["section"].unique())]


def section_paths(image_dir: str | Path) -> list[Path]:
    """TIFF paths sorted by the numeric z index in the filename."""
    paths = sorted(Path(image_dir).glob("*.tif"),
                   key=lambda p: int(re.match(r"0*(\d+)", p.stem).group(1)))
    return paths


def load_section(path: str | Path, downsample: int = 16,
                 grayscale: bool = True) -> np.ndarray:
    """Load a single section, downsampled, without ever holding 47 of them.

    One native RGB section is 19457 x 21249 x 3, about 1.24 GB in memory, so
    the stack cannot be held at full size on a 16 GB machine. Conversion to
    grayscale happens before reduction to keep the peak down.
    """
    with Image.open(path) as im:
        if grayscale:
            im = im.convert("L")
        if downsample > 1:
            im = im.reduce(downsample)
        return np.asarray(im)


def load_stack(
    image_dir: str | Path,
    downsample: int = 16,
    limit: int | None = None,
    cache: str | Path | None = None,
    grayscale: bool = True,
) -> tuple[np.ndarray, StackMeta]:
    """Load the serial stack at a working resolution.

    Parameters
    ----------
    image_dir
        Directory of ``001.tif`` ... ``NNN.tif``.
    downsample
        Integer reduction factor. 16 gives 7.36 um/px and 8 gives 3.68 um/px
        from a 0.46 um/px native grid; those are the two working resolutions
        used in the source publication.
    limit
        Load only the first N sections. For smoke tests.
    cache
        ``.npy`` path. Written after a successful load and reused thereafter,
        because decoding 16 GB of PackBits TIFF takes minutes.

    Returns
    -------
    (stack, meta)
        ``stack`` is (n_sections, H, W), uint8 if grayscale.
    """
    cache = Path(cache) if cache else None
    paths = section_paths(image_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no .tif sections under {image_dir}")

    with Image.open(paths[0]) as probe:
        native = (probe.height, probe.width)

    if cache and cache.exists():
        stack = np.load(cache)
        logger.info("loaded cached stack %s %s", cache.name, stack.shape)
    else:
        first = load_section(paths[0], downsample, grayscale)
        stack = np.zeros((len(paths), *first.shape), dtype=first.dtype)
        stack[0] = first
        for i, p in enumerate(paths[1:], start=1):
            a = load_section(p, downsample, grayscale)
            if a.shape != first.shape:
                # sections differ in size; pad or crop to the first section's grid
                h = min(a.shape[0], first.shape[0])
                w = min(a.shape[1], first.shape[1])
                stack[i, :h, :w] = a[:h, :w]
                logger.warning("section %s is %s, not %s; cropped to overlap",
                               p.name, a.shape, first.shape)
            else:
                stack[i] = a
            if (i + 1) % 10 == 0:
                logger.info("loaded %d/%d sections", i + 1, len(paths))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, stack)

    meta = StackMeta(
        n_sections=len(paths), downsample=downsample,
        mpp_um=NATIVE_MPP_UM * downsample, section_um=SECTION_THICKNESS_UM,
        native_shape=native, loaded_shape=stack.shape[1:],
    )
    logger.info("stack %s at %.2f um/px (native %.2f x %d)", stack.shape,
                meta.mpp_um, NATIVE_MPP_UM, downsample)
    return stack, meta


def tissue_masks(stack: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    """Binary tissue masks by intensity, since the archive ships no masks.

    Background on these slides is bright and near-uniform; tissue is darker.
    The threshold is a per-section percentile rather than a fixed grey value so
    that staining differences between sections do not change the tissue area,
    which would otherwise show up as spurious shrinkage.
    """
    thr = np.percentile(stack.reshape(len(stack), -1), 100 - percentile, axis=1)
    return stack < thr[:, None, None]

#!/usr/bin/env python
"""Arm 2: the stages TCGA can support, on human breast at cohort scale.

    python scripts/run_tcga_analysis.py [--max-slides 10]

Stage 3 (cell detection), stage 7 (fibre alignment) and stereology. Stages 1, 2,
5 and 6 are not attempted: TCGA has no consecutive sections, verified against
the GDC API, where the maximum for any patient is seven slides taken from
different blocks and labelled TOP or BOTTOM precisely because they sample
different regions.

READING THE SLIDES

Aperio SVS files are pyramidal TIFFs, so tifffile reads them without OpenSlide,
which has no wheel on this platform. The pyramid is used rather than ignored: a
low level is read to find tissue, and analysis tiles are pulled from a high
level only where tissue exists. A whole slide is never held in memory.

THE SECTIONING-ANGLE CAVEAT, HANDLED HONESTLY

Fibre alignment on a single section depends on the angle the structure was cut
at, and on a single section that angle is unknown and uncorrectable. The Arm 1
prostate volume can choose its plane; this arm cannot. Anisotropy is therefore
reported as a DISTRIBUTION across many windows and many slides, never as a
per-patient property, which is the response the protocol specifies when the
angle cannot be controlled.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda_my.deconv import deconvolve  # noqa: E402
from coda_my.fibers import FiberConfig, tiled_anisotropy  # noqa: E402

DATA = ROOT / "data/raw/tcga_brca"
OUT = ROOT / "results/tcga"
TILE = 1024
FULLMAN = 4.0 / np.pi
logger = logging.getLogger("tcga")


def read_level(path: Path, level: int) -> np.ndarray:
    """Read one pyramid level of an SVS as RGB."""
    import tifffile
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        levels = series.levels if hasattr(series, "levels") else [series]
        lv = levels[min(level, len(levels) - 1)]
        a = lv.asarray()
    if a.ndim == 3 and a.shape[0] in (3, 4):
        a = np.moveaxis(a, 0, -1)
    return a[..., :3]


def slide_mpp(path: Path) -> float | None:
    """Microns per pixel from the Aperio description, or None if unreadable.

    Returns None rather than raising on a damaged file. One truncated download
    in this cohort crashed the whole run here, several steps away from the
    actual problem; a slide that cannot be read is a slide to skip and report,
    not a reason to lose the other nine.
    """
    import tifffile
    try:
        with tifffile.TiffFile(str(path)) as tf:
            if not len(tf.pages):
                return None
            desc = tf.pages[0].description or ""
    except Exception:
        return None
    m = [p for p in desc.replace("|", "\n").split("\n") if "MPP" in p.upper()]
    for p in m:
        try:
            return float(p.split("=")[1].strip())
        except Exception:
            continue
    return None


def tissue_tiles(low: np.ndarray, scale: int, n: int, rng) -> list[tuple[int, int]]:
    """Pick tile origins at full resolution where the thumbnail shows tissue."""
    g = low.mean(axis=2)
    tissue = g < np.percentile(g, 75)
    tissue = ndimage.binary_opening(tissue, np.ones((3, 3)))
    ys, xs = np.nonzero(tissue)
    if not len(ys):
        return []
    idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(int(ys[i] * scale), int(xs[i] * scale)) for i in idx]


def detect_watershed(h: np.ndarray, mpp: float,
                     min_um: float = 2.5, max_um: float = 15.0) -> int:
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    sm = ndimage.gaussian_filter(h, max(0.5 / mpp, 0.6))
    try:
        thr = threshold_otsu(sm)
    except ValueError:
        return 0
    mask = ndimage.binary_opening(sm > thr, np.ones((3, 3)))
    if mask.sum() < 10:
        return 0
    dist = ndimage.distance_transform_edt(mask)
    peaks = peak_local_max(dist, min_distance=max(int(min_um / mpp), 2), labels=mask)
    if not len(peaks):
        return 0
    markers = np.zeros(h.shape, int)
    for i, (y, x) in enumerate(peaks, 1):
        markers[y, x] = i
    lab = watershed(-dist, markers, mask=mask)
    lo = np.pi * (min_um / 2 / mpp) ** 2
    hi = np.pi * (max_um / 2 / mpp) ** 2
    sizes = ndimage.sum(mask, lab, range(1, lab.max() + 1))
    return int(((sizes >= lo) & (sizes <= hi)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-slides", type=int, default=10)
    ap.add_argument("--tiles-per-slide", type=int, default=12)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs/tcga_analysis.log"),
                                  logging.StreamHandler(sys.stdout)])

    slides = sorted(DATA.glob("*.svs"))[:args.max_slides]
    if not slides:
        raise SystemExit(f"no SVS under {DATA}; run scripts/fetch_tcga_brca.py first")
    logger.info("TCGA-BRCA, %d diagnostic slides", len(slides))

    rng = np.random.default_rng(0)
    rows, aniso_rows, diam = [], [], []
    for si, p in enumerate(slides, 1):
        mpp = slide_mpp(p)
        if mpp is None:
            logger.warning("[%d/%d] %s: unreadable or no MPP in the header, "
                           "skipped. Scale cannot be guessed and every distance "
                           "would be wrong.", si, len(slides), p.name[:40])
            continue
        try:
            low = read_level(p, 2)
            full_shape = read_level(p, 0).shape[:2] if False else None
        except Exception as exc:
            logger.warning("[%d/%d] %s: unreadable (%s)", si, len(slides),
                           p.name[:40], str(exc)[:50])
            continue

        import tifffile
        with tifffile.TiffFile(str(p)) as tf:
            lv = tf.series[0].levels if hasattr(tf.series[0], "levels") else [tf.series[0]]
            shapes = [l.shape[:2] for l in lv]
        scale = shapes[0][0] / low.shape[0]
        origins = tissue_tiles(low, scale, args.tiles_per_slide, rng)
        if not origins:
            logger.warning("[%d/%d] %s: no tissue found in the thumbnail",
                           si, len(slides), p.name[:34])
            continue

        cfg = FiberConfig(mpp=mpp, window_um=50.0)
        n_nuc, n_tiles = 0, 0
        # Level 0 of these slides is tens of gigapixels, so it is never
        # materialised. The pages are tiled, so tifffile's zarr view reads only
        # the requested window from disk. An earlier version guarded on total
        # size and silently skipped every large slide, which left 1 of 9 slides
        # analysed and said nothing about the other 8.
        import zarr
        with tifffile.TiffFile(str(p)) as tf:
            lv0 = tf.series[0].levels[0] if hasattr(tf.series[0], "levels") else tf.series[0]
            z = zarr.open(lv0.aszarr(), mode="r")
            if not hasattr(z, "shape"):
                # zarr 3 returns a Group for a multiscale store, and array_keys
                # comes back UNSORTED: on these slides it is ['2','3','0','1'].
                # Taking the first key silently reads a downsampled level while
                # the tile origins were computed for level 0, so nearly every
                # tile lands off the edge and the slide reports no usable tiles.
                # Select level 0 by name, and verify the shape matches.
                z = z["0"]
            if z.shape[:2] != tuple(lv0.shape[:2]):
                logger.warning("[%d/%d] %s: zarr level shape %s does not match "
                               "tifffile level 0 %s, skipping",
                               si, len(slides), p.name[:30], z.shape[:2],
                               lv0.shape[:2])
                continue
            for (y, x) in origins:
                if y + TILE > z.shape[0] or x + TILE > z.shape[1]:
                    continue
                tile = np.asarray(z[y:y + TILE, x:x + TILE])
                if tile.shape[:2] != (TILE, TILE):
                    continue
                if tile.ndim == 3 and tile.shape[2] > 3:
                    tile = tile[..., :3]
                if tile.mean() > 235:            # blank
                    continue
                ch = deconvolve(tile.astype(np.uint8))
                h = ch["hematoxylin"]
                h = np.clip((h - np.percentile(h, 1)) /
                            max(np.percentile(h, 99.5) - np.percentile(h, 1), 1e-6), 0, 1)
                n_nuc += detect_watershed(h, mpp)
                e = ch["eosin"]
                e = np.clip((e - np.percentile(e, 1)) /
                            max(np.percentile(e, 99) - np.percentile(e, 1), 1e-6), 0, 1)
                a = tiled_anisotropy(e, cfg)
                a = a[np.isfinite(a)]
                aniso_rows += [{"slide": p.name[:23], "anisotropy": float(v)} for v in a]
                n_tiles += 1
        if n_tiles == 0:
            logger.warning("[%d/%d] %s: no usable tiles (all blank or off-edge)",
                           si, len(slides), p.name[:34])
            continue
        area_mm2 = n_tiles * (TILE * mpp / 1000.0) ** 2
        rows.append({"slide": p.name[:23], "mpp_um_per_px": mpp, "n_tiles": n_tiles,
                     "n_nuclei": n_nuc,
                     "nuclear_density_per_mm2": n_nuc / area_mm2 if area_mm2 else np.nan})
        logger.info("[%d/%d] %s  mpp %.4f  %d tiles  %d nuclei  %.0f/mm2",
                    si, len(slides), p.name[:23], mpp, n_tiles, n_nuc,
                    rows[-1]["nuclear_density_per_mm2"])

    if not rows:
        raise SystemExit("no slide yielded measurements")
    df = pd.DataFrame(rows); df.to_csv(OUT / "stage3_cell_detection.csv", index=False)
    ad = pd.DataFrame(aniso_rows); ad.to_csv(OUT / "stage7_anisotropy.csv", index=False)

    logger.info("")
    logger.info("STAGE 3  nuclear density across %d slides: median %.0f per mm2 "
                "(IQR %.0f to %.0f)", len(df), df.nuclear_density_per_mm2.median(),
                df.nuclear_density_per_mm2.quantile(.25),
                df.nuclear_density_per_mm2.quantile(.75))
    logger.info("         NOT validated against human annotation; none exists for "
                "these slides, so the published 90 percent bar is not tested")
    logger.info("STAGE 7  fibre anisotropy, %d windows over %d slides: median %.4f "
                "(IQR %.4f to %.4f)", len(ad), ad.slide.nunique(),
                ad.anisotropy.median(), ad.anisotropy.quantile(.25),
                ad.anisotropy.quantile(.75))
    logger.info("         reported as a distribution, never per patient: the cutting "
                "angle is unknown on a single section and cannot be corrected")

    summary = {
        "cohort": "TCGA-BRCA diagnostic slides, BCSS-annotated cases",
        "n_slides": int(len(df)),
        "mpp_median": float(df.mpp_um_per_px.median()),
        "nuclear_density_median_per_mm2": float(df.nuclear_density_per_mm2.median()),
        "anisotropy_median": float(ad.anisotropy.median()),
        "anisotropy_n_windows": int(len(ad)),
        "stages_run": [3, 7],
        "stages_not_run": {"1,2,5,6": "TCGA has no consecutive sections",
                           "4": "needs GPU training; see notebooks/"},
        "cell_detection_validated": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", OUT / "summary.json")


if __name__ == "__main__":
    main()

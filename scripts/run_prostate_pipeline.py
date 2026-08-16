#!/usr/bin/env python
"""Arm 1: the full CODA pipeline on the 260-section prostate series.

    python scripts/run_prostate_pipeline.py [--downsample 16] [--limit N]

Stages 1, 2, 5, 6 and 7. Stage 4 is not implemented anywhere in this project
because it needs annotated training tiles and a GPU; stage 3 runs as a
method-to-method comparison only, since no human annotation exists.

WHAT IS DIFFERENT FROM THE LIVER ARM, AND WHY IT MATTERS

The landmarks are pairwise. 259 rows describe 260 sections: each row gives four
points on section n and the same four points located again on section n+1. A
landmark does not persist through the stack, so accumulated error cannot be the
residual about a line fitted down z. It is the cumulative resultant of the mean
pairwise displacement vectors, which is what detects a stack bending steadily in
one direction while every individual pair still looks well aligned.

The two observers are NOT repeated measurements of the same points. Each chose
their own anatomical features, a median 1286 um apart even at nearest match, so
their disagreement is not an annotation floor and is not reported as one. Only
observer 1 is used, and that is stated.

The block is 1.3 mm deep against 235 um for the liver. That is the reason this
series was chosen: the sectioning-angle comparison in stage 7 failed on the
liver because orthogonal planes through a 235 um slab measure slab geometry
rather than tissue, proven there by a shuffle control. The same control is run
here and the result is reported only if it passes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coda_my.loaders.kartasalo_prostate import (  # noqa: E402
    NATIVE_MPP_UM, SECTION_THICKNESS_UM, accumulated_tre_pairwise,
    load_pairwise_fiducials, pairwise_tre, section_paths,
)
from coda_my.qc import axial_vs_lateral_correlation, z_skip_validation  # noqa: E402
from coda_my.registration import RegistrationConfig, apply_rigid  # noqa: E402
from coda_my.registration_fix import (  # noqa: E402
    SearchConfig, register_stack_two_scale,
)

DATA = ROOT / "data/raw/kartasalo_prostate/extracted/Data_to_IDA"
OUT = ROOT / "results/prostate"
BASE_DS = 16
COARSE_EXTRA = 11
CODA_PANCREAS_OVERCOUNT = 12.3
logger = logging.getLogger("prostate")


def blockmean(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a.astype(np.float32)
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    return a[:h2, :w2].astype(np.float32).reshape(h2 // f, f, w2 // f, f).mean((1, 3))


def load_stack(image_dir: Path, downsample: int, limit: int | None,
               cache: Path) -> np.ndarray:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    if cache.exists():
        s = np.load(cache)
        logger.info("loaded cached stack %s", s.shape)
        return s
    paths = section_paths(image_dir)
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no sections under {image_dir}")
    with Image.open(paths[0]) as im:
        first = np.asarray(im.convert("L").reduce(downsample))
    stack = np.zeros((len(paths), *first.shape), dtype=np.uint8)
    stack[0] = first
    for i, p in enumerate(paths[1:], start=1):
        with Image.open(p) as im:
            a = np.asarray(im.convert("L").reduce(downsample))
        h, w = min(a.shape[0], first.shape[0]), min(a.shape[1], first.shape[1])
        stack[i, :h, :w] = a[:h, :w]
        if (i + 1) % 25 == 0:
            logger.info("loaded %d/%d sections", i + 1, len(paths))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, stack)
    return stack


def segment_lumina(volume: np.ndarray, mpp: float,
                   min_um: float = 40.0, max_um: float = 1200.0) -> np.ndarray:
    """Enclosed bright spaces inside tissue: glandular lumina in prostate."""
    out = np.zeros(volume.shape, dtype=bool)
    min_px = np.pi * (min_um / 2 / mpp) ** 2
    max_px = np.pi * (max_um / 2 / mpp) ** 2
    for i in range(len(volume)):
        img = volume[i]
        if img.max() <= 0:
            continue
        sat, tis = np.percentile(img, 85), np.percentile(img, 20)
        if sat - tis < 1e-6:
            continue
        band = (img >= tis + 0.55 * (sat - tis)) & (img < sat - 0.02 * (sat - tis))
        band = ndimage.binary_opening(band, np.ones((3, 3)))
        lab, n = ndimage.label(band)
        if n == 0:
            continue
        sizes = ndimage.sum(band, lab, range(1, n + 1))
        keep = [j + 1 for j, s in enumerate(sizes) if min_px <= s <= max_px]
        if keep:
            out[i] = np.isin(lab, keep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--downsample", type=int, default=BASE_DS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs/prostate.log"),
                                  logging.StreamHandler(sys.stdout)])

    fine_mpp = NATIVE_MPP_UM * args.downsample
    coarse_mpp = fine_mpp * COARSE_EXTRA
    logger.info("=" * 72)
    logger.info("PROSTATE, rigid at %.1f um/px, fine at %.2f um/px, sections %.1f um",
                coarse_mpp, fine_mpp, SECTION_THICKNESS_UM)

    fid = load_pairwise_fiducials(DATA / "fiducialcoordinates_prostate_observer1.txt")
    logger.info("landmarks: observer 1 only (the two observers marked different "
                "features, so their disagreement is not a noise floor)")

    # ---------------------------------------------- baseline before anything
    base = pairwise_tre(fid)
    base_atre = accumulated_tre_pairwise(base)
    logger.info("BASELINE, no transform: TRE mean %.1f um, median %.1f; "
                "ATRE mean %.1f um", base.tre_mean_um.mean(),
                base.tre_median_um.median(), base_atre.atre_um.mean())

    stack = load_stack(DATA / "prostate", args.downsample, args.limit,
                       ROOT / f"data/interim/prostate_ds{args.downsample}.npy")
    n = len(stack)
    logger.info("stack %s, %.2f GB", stack.shape, stack.nbytes / 1e9)

    # -------------------------------------------------- stage 1: registration
    coarse = np.stack([blockmean(stack[i], COARSE_EXTRA) for i in range(n)])
    t0 = time.time()
    registered, params, fields = register_stack_two_scale(
        coarse, stack, coarse_mpp, fine_mpp, RegistrationConfig(),
        SearchConfig(max_abs_deg=45.0), elastic=False)
    logger.info("STAGE 1 registration done in %.1f min", (time.time() - t0) / 60)
    np.save(OUT / "registered.npy", registered)

    corr = pd.DataFrame([{"section": i + 1, "correlation": p.get("correlation", np.nan),
                          "angle_deg": p.get("angle", np.nan),
                          "dy_px": p.get("dy_fine", np.nan),
                          "dx_px": p.get("dx_fine", np.nan),
                          "reference_offset": p.get("reference_offset", np.nan)}
                         for i, p in enumerate(params)])
    cfg = RegistrationConfig()
    corr["flagged"] = corr.correlation < cfg.min_correlation
    corr.to_csv(OUT / "stage1_correlation.csv", index=False)
    logger.info("STAGE 2 correlation median %.4f, flagged %d/%d",
                corr.correlation.median(), int(corr.flagged.sum()), n)

    # ------------------------------------------------------ stage 2: TRE/ATRE
    tf = [(p.get("angle", 0.0), p.get("dy_fine", 0.0), p.get("dx_fine", 0.0), fields[i])
          for i, p in enumerate(params)]
    reg_tre = pairwise_tre(fid, transforms=tf, shape=stack.shape[1:],
                           mpp=fine_mpp, downsample=args.downsample)
    reg_atre = accumulated_tre_pairwise(reg_tre)
    reg_tre.to_csv(OUT / "stage2_tre.csv", index=False)
    reg_atre.to_csv(OUT / "stage2_atre.csv", index=False)
    logger.info("STAGE 2 TRE mean %.1f um (baseline %.1f), ATRE mean %.1f um "
                "(baseline %.1f)", reg_tre.tre_mean_um.mean(), base.tre_mean_um.mean(),
                reg_atre.atre_um.mean(), base_atre.atre_um.mean())
    beat = reg_tre.tre_mean_um.mean() < base.tre_mean_um.mean()
    logger.info("STAGE 2 VERDICT: %s the do-nothing baseline",
                "BEATS" if beat else "FAILS TO BEAT")

    # ------------------------------------------- stages 5 and 6: volume, count
    from coda_my.registration import apply_elastic
    vol = np.zeros_like(stack)
    for i in range(n):
        p = params[i]
        img = apply_rigid(stack[i].astype(np.float32), p.get("angle", 0.0),
                          p.get("dy_fine", 0.0), p.get("dx_fine", 0.0))
        if fields[i] is not None:
            img = apply_elastic(img, *fields[i])
        vol[i] = np.clip(img, 0, 255)
    np.save(OUT / "volume.npy", vol)
    logger.info("STAGE 5 volume %s built", vol.shape)

    mask = segment_lumina(vol, fine_mpp)
    per_section = [int(ndimage.label(mask[i])[1]) for i in range(n)]
    lab3, n3 = ndimage.label(mask)
    ratio = sum(per_section) / max(n3, 1)
    logger.info("STAGE 6 2D total %d, 3D objects %d, overcounting %.2f-fold "
                "(CODA pancreas %.1f)", sum(per_section), n3, ratio,
                CODA_PANCREAS_OVERCOUNT)

    sizes = ndimage.sum(mask, lab3, range(1, n3 + 1)) if n3 else np.array([])
    present = np.zeros(n3 + 1, dtype=int)
    for i in range(n):
        for u in np.unique(lab3[i]):
            if u:
                present[u] += 1
    vox = fine_mpp * fine_mpp * SECTION_THICKNESS_UM
    rows = []
    for lo in (0, 1e4, 1e5, 1e6, 1e7):
        keep = np.where(sizes * vox >= lo)[0] + 1
        if not len(keep):
            continue
        rows.append({"min_object_volume_um3": lo, "n_3d_objects": len(keep),
                     "sum_2d_detections": int(present[keep].sum()),
                     "overcounting_ratio": round(present[keep].sum() / len(keep), 2),
                     "median_sections_spanned": float(np.median(present[keep]))})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "stage6_overcount_sensitivity.csv", index=False)
    logger.info("STAGE 6 sensitivity to object size:\n%s", sens.to_string(index=False))

    # ------------------------------------------------------------- stage 2/8
    ax = axial_vs_lateral_correlation(vol, mpp=fine_mpp,
                                      section_um=SECTION_THICKNESS_UM)
    ax.to_csv(OUT / "stage2_axial_lateral.csv", index=False)
    zs = z_skip_validation(vol, section_um=SECTION_THICKNESS_UM)
    zs.to_csv(OUT / "stage2_zskip.csv", index=False)
    logger.info("z-skip:\n%s", zs.to_string(index=False))

    summary = {
        "dataset": "Kartasalo mouse prostate, Etsin c76335fa, CC BY 4.0",
        "n_sections": n, "mpp_um": fine_mpp, "coarse_mpp_um": coarse_mpp,
        "section_thickness_um": SECTION_THICKNESS_UM,
        "block_depth_um": n * SECTION_THICKNESS_UM,
        "landmarks": "observer 1, pairwise; observers are not repeated measures",
        "baseline_tre_um": float(base.tre_mean_um.mean()),
        "baseline_atre_um": float(base_atre.atre_um.mean()),
        "registered_tre_um": float(reg_tre.tre_mean_um.mean()),
        "registered_atre_um": float(reg_atre.atre_um.mean()),
        "beats_baseline": bool(beat),
        "correlation_median": float(corr.correlation.median()),
        "n_flagged": int(corr.flagged.sum()),
        "n_3d_objects": int(n3), "total_2d_count": int(sum(per_section)),
        "overcounting_ratio": float(ratio),
        "coda_pancreas_reference": CODA_PANCREAS_OVERCOUNT,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", OUT / "summary.json")


if __name__ == "__main__":
    main()

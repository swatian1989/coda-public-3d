#!/usr/bin/env python
"""Stage 4 inference: apply the trained segmentation to the TCGA slides, on CPU.

    python scripts/run_tcga_segmentation.py [--max-slides 10]

Training needs a GPU and happens in notebooks/stage4_segmentation_colab.ipynb.
Inference does not: the model is about 90 MB, slides are tiled lazily from the
pyramid, and one tile at a time fits comfortably in memory. This is the whole
reason the two are separated.

WHAT THIS ADDS THAT THE EARLIER STAGES COULD NOT

Stage 3 counts nuclei but cannot say what tissue they sit in. With a tissue map,
the same slides yield composition (how much of the section is tumour, stroma,
inflammatory infiltrate, necrosis) and nuclear density WITHIN each compartment,
which is the measurement the source publication builds its biology on. A nuclear
density averaged over a whole slide mixes dense tumour with sparse fat and is
close to meaningless on its own.

THE MODEL IS NOT ASSUMED TO BE GOOD

The notebook reports per-class precision and recall against the 90 percent
acceptance gate the published protocol specifies. Any class that failed that
gate is carried through to here and named in the output, so a composition
fraction computed from an unreliable class is visibly flagged rather than
silently reported alongside reliable ones.
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

DATA = ROOT / "data/raw/tcga_brca"
MODELS = ROOT / "data/models"
OUT = ROOT / "results/tcga"
TILE = 512
logger = logging.getLogger("seg")


def load_model(path: Path):
    import torch
    import segmentation_models_pytorch as smp
    ck = torch.load(path, map_location="cpu")
    model = smp.DeepLabV3Plus(ck.get("encoder", "resnet34"), encoder_weights=None,
                              classes=ck.get("classes", 6))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    names = {int(k): v for k, v in ck.get("names", {}).items()}
    return model, names, ck


def slide_mpp(path: Path) -> float | None:
    import tifffile
    try:
        with tifffile.TiffFile(str(path)) as tf:
            if not len(tf.pages):
                return None
            desc = tf.pages[0].description or ""
    except Exception:
        return None
    for p in desc.replace("|", "\n").split("\n"):
        if "MPP" in p.upper():
            try:
                return float(p.split("=")[1].strip())
            except Exception:
                pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-slides", type=int, default=10)
    ap.add_argument("--tiles-per-slide", type=int, default=24)
    ap.add_argument("--model", default=str(MODELS / "stage4_deeplab.pt"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs/tcga_seg.log"),
                                  logging.StreamHandler(sys.stdout)])
    mp = Path(args.model)
    if not mp.exists():
        raise SystemExit(
            f"no model at {mp}\n"
            "Run notebooks/stage4_segmentation_colab.ipynb on a Colab T4, then put\n"
            "the downloaded stage4_deeplab.pt into data/models/. Training needs a\n"
            "GPU; this inference step does not.")

    import torch
    import tifffile
    import zarr

    model, names, ck = load_model(mp)
    logger.info("model %s, %d classes: %s", ck.get("arch"), ck.get("classes"),
                ", ".join(names.values()))

    # carry the acceptance result forward rather than assuming every class is usable
    failed = []
    pcm = OUT / "stage4_per_class_metrics.csv"
    if not pcm.exists():
        pcm = MODELS / "stage4_per_class_metrics.csv"
    if pcm.exists():
        m = pd.read_csv(pcm)
        failed = m.loc[~m.passes_90_gate.astype(bool), "class"].tolist()
        logger.info("classes failing the 90 percent gate in training: %s",
                    failed if failed else "none")
    else:
        logger.warning("no per-class metrics found; composition is reported without "
                       "knowing which classes met the acceptance gate")

    MEAN = np.array([0.485, 0.456, 0.406]); STD = np.array([0.229, 0.224, 0.225])
    slides = sorted(DATA.glob("*.svs"))[:args.max_slides]
    rng = np.random.default_rng(0)
    rows = []

    for si, p in enumerate(slides, 1):
        mpp = slide_mpp(p)
        if mpp is None:
            logger.warning("[%d/%d] %s unreadable or no MPP, skipped",
                           si, len(slides), p.name[:34])
            continue
        with tifffile.TiffFile(str(p)) as tf:
            s = tf.series[0]
            lv = s.levels if hasattr(s, "levels") else [s]
            low = np.asarray(lv[min(2, len(lv) - 1)].asarray())
            if low.ndim == 3 and low.shape[0] in (3, 4):
                low = np.moveaxis(low, 0, -1)
            low = low[..., :3]
            z = zarr.open(lv[0].aszarr(), mode="r")
            if not hasattr(z, "shape"):
                z = z["0"]                      # keys are unsorted; select by name
            scale = z.shape[0] / low.shape[0]

            g = low.mean(axis=2)
            tissue = ndimage.binary_opening(g < np.percentile(g, 75), np.ones((3, 3)))
            ys, xs = np.nonzero(tissue)
            if not len(ys):
                logger.warning("[%d/%d] %s no tissue", si, len(slides), p.name[:34])
                continue
            idx = rng.choice(len(ys), size=min(args.tiles_per_slide, len(ys)),
                             replace=False)

            counts = np.zeros(ck.get("classes", 6), dtype=np.int64)
            n_tiles = 0
            for i in idx:
                y, x = int(ys[i] * scale), int(xs[i] * scale)
                if y + TILE > z.shape[0] or x + TILE > z.shape[1]:
                    continue
                tile = np.asarray(z[y:y + TILE, x:x + TILE])[..., :3]
                if tile.mean() > 235:
                    continue
                t = ((tile / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
                with torch.no_grad():
                    pred = model(torch.from_numpy(t)).argmax(1).numpy()[0]
                counts += np.bincount(pred.ravel(), minlength=len(counts))
                n_tiles += 1
        if n_tiles == 0:
            logger.warning("[%d/%d] %s no usable tiles", si, len(slides), p.name[:34])
            continue

        tot = counts[1:].sum()
        row = {"slide": p.name[:23], "n_tiles": n_tiles, "mpp": mpp}
        for c, nm in names.items():
            if c == 0:
                continue
            row[f"frac_{nm}"] = round(float(counts[c] / max(tot, 1)), 4)
        rows.append(row)
        logger.info("[%d/%d] %s  %d tiles  " + "  ".join(
            f"{nm} {counts[c]/max(tot,1):.2f}" for c, nm in names.items() if c),
            si, len(slides), p.name[:23], n_tiles)

    if not rows:
        raise SystemExit("no slide produced a composition")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stage4_composition.csv", index=False)

    logger.info("")
    logger.info("STAGE 4  tissue composition across %d slides (median fraction)", len(df))
    for c in [c for c in df.columns if c.startswith("frac_")]:
        flag = "  <-- FAILED the 90 percent gate in training" \
            if c.replace("frac_", "") in failed else ""
        logger.info("   %-16s %.3f%s", c.replace("frac_", ""), df[c].median(), flag)
    if failed:
        logger.info("Fractions for the flagged classes are reported for completeness "
                    "and should not be treated as reliable; the published protocol "
                    "adds annotations and retrains until every class passes.")

    (OUT / "stage4_summary.json").write_text(json.dumps({
        "n_slides": int(len(df)), "tiles_per_slide": args.tiles_per_slide,
        "classes": names, "classes_failing_gate": failed,
        "median_fractions": {c.replace("frac_", ""): float(df[c].median())
                             for c in df.columns if c.startswith("frac_")},
    }, indent=2))
    logger.info("wrote %s", OUT / "stage4_composition.csv")


if __name__ == "__main__":
    main()

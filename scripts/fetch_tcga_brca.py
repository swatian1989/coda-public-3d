#!/usr/bin/env python
"""Arm 2: fetch TCGA-BRCA diagnostic slides from the GDC.

    python scripts/fetch_tcga_brca.py [--n 10] [--max-gb 12]

A deliberately small cohort. The prostate series occupies about 90 GB of the
available disk, and a 40-slide TCGA cohort at roughly 1 GB per slide would not
fit alongside it. Ten slides is enough to characterise nuclear density and
fibre anisotropy distributions across patients, and the cohort size is reported
with every result rather than being implied to be larger. Raise --n when disk
allows; the selection is seeded, so a larger n is a superset of the same
cohort.

DIAGNOSTIC SLIDES ONLY, AND WHY

TCGA ships two kinds of slide. Diagnostic slides (DX) are formalin-fixed and
paraffin-embedded. Tissue slides (TS, BS, MS) are frozen sections, cut from the
top, bottom or middle of a frozen block. Freezing produces ice-crystal artefact
that changes nuclear morphology and stromal texture, which is exactly what a
morphometric pipeline measures, so mixing the two would confound every result
with preparation method. Only DX slides are fetched.

WHAT THIS ARM CAN AND CANNOT DO, DECIDED BEFORE DOWNLOADING

Queried against the GDC API: TCGA-BRCA has 3,112 slide files and the maximum for
any single patient is seven. The section_location field returns TOP and BOTTOM,
labels that exist because slides are deliberately taken from opposite ends of a
block to sample different regions. There are no consecutive sections anywhere in
TCGA, for any cancer type, so stages 1, 2, 5 and 6 cannot run here and are not
attempted. Stage 3, stage 7 and stereology can.

SIZE

Whole slide images are large, commonly 300 MB to 1.5 GB each. The default caps
both the file count and the total download so a run cannot silently consume the
disk. Slides are streamed to disk and never held in memory; tiles are read
lazily at analysis time.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/tcga_brca"
GDC = "https://api.gdc.cancer.gov"
logger = logging.getLogger("tcga")


def query_slides(n: int, seed: int = 0) -> list[dict]:
    """Diagnostic SVS slides for TCGA-BRCA, sampled without size bias.

    Taking the n smallest files would be convenient and wrong. File size tracks
    the area of tissue on the slide, so the smallest slides are the ones with
    least tissue, and a cohort selected that way is systematically biased toward
    small specimens before a single measurement is made. The full list is
    retrieved and a deterministic pseudo-random sample is drawn instead, so the
    selection is reproducible from the seed and independent of size.
    """
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id",
                                     "value": ["TCGA-BRCA"]}},
            {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
            {"op": "in", "content": {"field": "experimental_strategy",
                                     "value": ["Diagnostic Slide"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.submitter_id",
        "format": "JSON", "size": "5000",
    }
    url = f"{GDC}/files?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=180) as fh:
        data = json.load(fh)
    hits = data["data"]["hits"]
    total = data["data"]["pagination"]["total"]

    # Restrict to slides that carry BCSS tissue annotations. Selecting the
    # cohort from the annotated 151 means the segmentation stage is validated on
    # the same slides it is applied to, rather than trained on one set of
    # patients and evaluated on another with no ground truth at all.
    bcss = ROOT / "data/reference/bcss_annotated_slides.csv"
    if bcss.exists():
        import csv
        with open(bcss) as fh:
            cases = {r["case"] for r in csv.DictReader(fh)}
        before = len(hits)
        hits = [h for h in hits
                if h.get("cases") and h["cases"][0]["submitter_id"] in cases]
        logger.info("restricted to BCSS-annotated cases: %d of %d slides remain",
                    len(hits), before)

    import random
    random.Random(seed).shuffle(hits)
    sizes = sorted(h["file_size"] for h in hits)
    logger.info("GDC reports %d diagnostic slides, %.0f MB median; sampling %d "
                "at random (seed %d), not by size",
                total, sizes[len(sizes) // 2] / 1e6, n, seed)
    # one slide per patient, so the cohort is patients rather than slides
    seen, keep = set(), []
    for h in hits:
        pid = h["cases"][0]["submitter_id"] if h.get("cases") else h["file_id"]
        if pid in seen:
            continue
        seen.add(pid)
        keep.append(h)
        if len(keep) >= n:
            break
    return keep


def download(fid: str, name: str, dest: Path, expect_bytes: int | None = None) -> bool:
    """Fetch one slide, refusing to publish a truncated file.

    Writing to a .part file and renaming on completion is not by itself enough.
    When a connection drops mid-transfer, read() returns empty, which is
    indistinguishable from a clean end of stream, so the loop exits normally and
    the rename fires on a partial file. That is how a 444 MB slide arrived as a
    52 MB file that opened with zero pages and crashed the analysis several
    steps later, far from the cause.

    The byte count is therefore checked against the size the API declared before
    the file is allowed to take its final name. A short read is a failure.
    """
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(f"{GDC}/data/{fid}",
                                 headers={"User-Agent": "coda-tcga"})
    try:
        got = 0
        with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
        if expect_bytes and got != expect_bytes:
            logger.error("  TRUNCATED %s: got %.0f MB of %.0f MB, discarding",
                         name[:40], got / 1e6, expect_bytes / 1e6)
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dest)
        return True
    except Exception as exc:
        logger.error("  failed %s: %s", name, str(exc)[:80])
        tmp.unlink(missing_ok=True)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="patients (one slide each)")
    ap.add_argument("--max-gb", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="selection seed; the cohort is reproducible from it")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs/tcga_fetch.log"),
                                  logging.StreamHandler(sys.stdout)])

    import urllib.parse  # noqa: F401  (used in query_slides)
    slides = query_slides(args.n, args.seed)
    total = sum(s["file_size"] for s in slides) / 1e9
    logger.info("%d patients, %.1f GB total, %.0f MB median",
                len(slides), total, 1e-6 * sorted(s["file_size"] for s in slides)[len(slides)//2])

    manifest, got, gb = [], 0, 0.0
    for i, s in enumerate(slides, 1):
        size_gb = s["file_size"] / 1e9
        if gb + size_gb > args.max_gb:
            logger.warning("stopping at %.1f GB to stay under the --max-gb cap; "
                           "%d slides not fetched", gb, len(slides) - i + 1)
            break
        dest = OUT / s["file_name"]
        if dest.exists() and dest.stat().st_size != s["file_size"]:
            logger.warning("[%d/%d] %s is %.0f MB, expected %.0f MB; refetching",
                           i, len(slides), s["file_name"][:34],
                           dest.stat().st_size / 1e6, s["file_size"] / 1e6)
            dest.unlink()
        if dest.exists():
            logger.info("[%d/%d] have %s", i, len(slides), s["file_name"][:44])
        else:
            logger.info("[%d/%d] %s (%.0f MB)", i, len(slides),
                        s["file_name"][:44], size_gb * 1000)
            if not download(s["file_id"], s["file_name"], dest, s["file_size"]):
                continue
        gb += size_gb
        got += 1
        manifest.append({"file_id": s["file_id"], "file_name": s["file_name"],
                         "case": s["cases"][0]["submitter_id"] if s.get("cases") else "",
                         "size_bytes": s["file_size"]})

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("done: %d slides, %.1f GB, manifest written", got, gb)
    logger.info("NOTE stages 1, 2, 5 and 6 cannot run on these slides. TCGA has no "
                "consecutive sections; the maximum for any patient is seven slides "
                "from different blocks.")


if __name__ == "__main__":
    main()

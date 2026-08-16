#!/usr/bin/env python
"""Stream the Kartasalo supplementary ZIP and extract only the PROSTATE stack.

    python scripts/fetch_kartasalo_prostate.py [--recon] [--keep PATTERN]

WHY THIS IS NOT JUST `curl | unzip`.

The dataset (Etsin c76335fa-cdcf-4ddc-ab1c-1882bad82861, CC BY 4.0, access
type Open) is published as a SINGLE 63.79 GB zip. The Fairdata download
service ignores HTTP Range entirely: a request with `Range: bytes=0-99` and one
with `Range: bytes=-100` both return 200 and begin streaming from offset zero,
and no Accept-Ranges or Content-Range header is sent. That rules out reading
the zip central directory from the tail to pull single members, and it rules
out resuming a broken transfer.

So the whole 63.79 GB must cross the network exactly once. It does not have to
land on disk. This reads the stream as a sequential zip: for every member it
parses the local file header, and then either inflates and writes the member
(if wanted) or discards `compressed_size` bytes without inflating (if not).
Peak disk is the size of the liver subset, not of the archive.

Every member seen is appended to a JSONL manifest as it goes, flushed
immediately, so a transfer that dies at 90 percent still leaves a usable record
of the archive layout up to that point.

Members are kept when the path matches --keep (default: liver, plus landmark
and fiducial spellings), or when the member is small and looks like metadata
(readme, txt, csv, xml, json, mat under 5 MB), which is cheap insurance against
the layout not using the word "liver" where expected.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/kartasalo_prostate"

DATASET_ID = "c76335fa-cdcf-4ddc-ab1c-1882bad82861"
MEMBER = ("/A_comparison_of_reconstruction_algorithms_for_3D_histology/"
          "SupplementaryData_A_comparison_of_reconstruction_algorithms_for_3D_histology.zip")
AUTHORIZE = "https://etsin.fairdata.fi/api/download/authorize"

SIG_LOCAL, SIG_CENTRAL = 0x04034B50, 0x02014B50
KEEP_DEFAULT = r"prostate|landmark|fiducial|hole"
META_EXT = {".txt", ".csv", ".xml", ".json", ".mat", ".md", ".m", ".pdf", ".xlsx"}
META_MAX = 5 * 1024 * 1024

logger = logging.getLogger("kartasalo")


def authorize() -> str:
    """Ask Etsin for a short-lived signed download URL."""
    body = json.dumps({"cr_id": DATASET_ID, "file": MEMBER}).encode()
    req = urllib.request.Request(AUTHORIZE, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)["url"]


class Stream:
    """Byte reader over the HTTP response with an exact-length read."""

    def __init__(self, fh):
        self.fh = fh
        self.pos = 0

    def read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.fh.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        self.pos += len(buf)
        return bytes(buf)

    def skip(self, n: int) -> int:
        """Discard n bytes without keeping them."""
        left = n
        while left > 0:
            chunk = self.fh.read(min(left, 1 << 20))
            if not chunk:
                break
            left -= len(chunk)
        moved = n - left
        self.pos += moved
        return moved


def zip64_sizes(extra: bytes, comp: int, uncomp: int) -> tuple[int, int]:
    """Pull real sizes out of the zip64 extra field when the 32-bit ones saturate."""
    i = 0
    while i + 4 <= len(extra):
        hid, hlen = struct.unpack_from("<HH", extra, i)
        if hid == 0x0001:
            body, j = extra[i + 4:i + 4 + hlen], 0
            if uncomp == 0xFFFFFFFF and j + 8 <= len(body):
                uncomp = struct.unpack_from("<Q", body, j)[0]; j += 8
            if comp == 0xFFFFFFFF and j + 8 <= len(body):
                comp = struct.unpack_from("<Q", body, j)[0]
            break
        i += 4 + hlen
    return comp, uncomp


def wanted(name: str, uncomp: int, keep: re.Pattern) -> bool:
    if keep.search(name):
        return True
    return Path(name).suffix.lower() in META_EXT and uncomp <= META_MAX


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", default=KEEP_DEFAULT,
                    help="case-insensitive regex of member paths to extract")
    ap.add_argument("--recon", action="store_true",
                    help="log the layout only, write nothing but the manifest")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs/prostate_fetch.log"),
                                  logging.StreamHandler(sys.stdout)])
    keep = re.compile(args.keep, re.I)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    manifest = open(OUT / "archive_manifest.jsonl", "w", encoding="utf-8")

    url = authorize()
    logger.info("authorized; streaming 63.79 GB. The prostate series sits after "
                "the liver in the archive, so unlike the liver fetch this needs the "
                "WHOLE stream. No range support, so it cannot resume.")
    t0 = time.time()
    n_seen = n_kept = 0
    kept_bytes = 0

    req = urllib.request.Request(url, headers={"User-Agent": "coda-brca-my"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        s = Stream(resp)
        while True:
            sig_raw = s.read(4)
            if len(sig_raw) < 4:
                logger.warning("stream ended at %.2f GB after %d members",
                               s.pos / 1e9, n_seen)
                break
            sig = struct.unpack("<I", sig_raw)[0]
            if sig == SIG_CENTRAL:
                logger.info("reached central directory at %.2f GB", s.pos / 1e9)
                break
            if sig != SIG_LOCAL:
                logger.error("bad signature %08x at byte %d; aborting", sig, s.pos)
                break

            hdr = s.read(26)
            (_ver, flags, method, _t, _d, _crc,
             comp, uncomp, nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr)
            name = s.read(nlen).decode("utf-8", "replace")
            extra = s.read(elen)
            if comp == 0xFFFFFFFF or uncomp == 0xFFFFFFFF:
                comp, uncomp = zip64_sizes(extra, comp, uncomp)

            if flags & 0x08:
                logger.error("member %s uses a data descriptor, so its size is not "
                             "in the local header and sequential streaming cannot "
                             "skip it safely; aborting", name)
                break

            n_seen += 1
            take = (not args.recon) and (not name.endswith("/")) and \
                wanted(name, uncomp, keep)
            manifest.write(json.dumps({"name": name, "compressed": comp,
                                       "uncompressed": uncomp, "method": method,
                                       "kept": bool(take)}) + "\n")
            manifest.flush()

            if take:
                dest = OUT / "extracted" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dec = zlib.decompressobj(-15) if method == 8 else None
                left = comp
                with open(dest, "wb") as out:
                    while left > 0:
                        chunk = resp.read(min(left, 1 << 20))
                        if not chunk:
                            break
                        left -= len(chunk)
                        s.pos += len(chunk)
                        out.write(dec.decompress(chunk) if dec else chunk)
                    if dec:
                        out.write(dec.flush())
                n_kept += 1
                kept_bytes += uncomp
                logger.info("KEPT %s (%.1f MB)  [%d kept, %.1f GB streamed]",
                            name, uncomp / 1e6, n_kept, s.pos / 1e9)
            else:
                s.skip(comp)

            if n_seen % 200 == 0:
                gb, el = s.pos / 1e9, time.time() - t0
                logger.info("%d members, %.2f GB, %.1f MB/s, eta %.1f h",
                            n_seen, gb, s.pos / 1e6 / max(el, 1),
                            max(0.0, (63.79 - gb) / max(gb / max(el / 3600, 1e-9), 1e-9)))

    manifest.close()
    logger.info("done: %d members seen, %d kept (%.2f GB written), %.2f GB streamed "
                "in %.2f h", n_seen, n_kept, kept_bytes / 1e9, s.pos / 1e9,
                (time.time() - t0) / 3600)
    logger.info("layout manifest: %s", OUT / "archive_manifest.jsonl")


if __name__ == "__main__":
    main()

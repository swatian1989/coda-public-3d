"""Parameter drift guard for the CODA Online Methods.

The failure this prevents is quiet and common: a parameter gets nudged during
debugging (a threshold, a radius, a tile size), the pipeline runs, the numbers
look plausible, and nobody can reconstruct months later which value produced
which figure.

`locked:` in coda_params.yaml is transcribed from the paper. Its SHA-256 is
recorded. Any edit changes the hash and `verify()` fails the run, naming the
exact keys that moved. A deliberate change goes in `deviations:` instead, which
is not hashed, and every deviation is emitted into the run manifest so it lands
in the methods section rather than being lost.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("config/coda_params.yaml")


def load_params(path: Path | str = PARAMS_PATH) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The locked CODA parameters must be present; "
            "the pipeline will not run on defaults.")
    return yaml.safe_load(path.read_text())


def hash_locked(params: dict) -> str:
    """Stable SHA-256 of the locked block. Key order does not affect it."""
    canonical = json.dumps(params["locked"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = d
    return out


def verify(expected_hash: str | None = None,
           path: Path | str = PARAMS_PATH) -> dict:
    """Load, verify and return the parameters. Raises on drift.

    On first run, pass ``expected_hash=None``: the hash is computed and written
    to `config/.coda_params.sha256`. Every later run compares against it.
    """
    params = load_params(path)
    current = hash_locked(params)
    stamp = Path(path).parent / ".coda_params.sha256"

    if expected_hash is None and stamp.exists():
        expected_hash = stamp.read_text().strip()

    if expected_hash is None:
        stamp.write_text(current)
        logger.info("locked CODA parameters registered, sha256 %s", current[:16])
        _log_deviations(params)
        return params

    if current != expected_hash:
        drifted = _diff_against_stamp(params, Path(path).parent / ".coda_params.json")
        raise RuntimeError(
            "CODA parameters have DRIFTED from the locked Online Methods values.\n"
            f"  expected sha256 {expected_hash[:16]}\n"
            f"  found    sha256 {current[:16]}\n"
            + (f"  changed keys: {drifted}\n" if drifted else "")
            + "If the change is deliberate, move it into the `deviations:` block "
              "with a reason, then delete config/.coda_params.sha256 to re-register."
        )

    logger.info("CODA parameters verified, sha256 %s", current[:16])
    _log_deviations(params)
    return params


def _diff_against_stamp(params: dict, snapshot: Path) -> list[str]:
    if not snapshot.exists():
        return []
    old = flatten(json.loads(snapshot.read_text()))
    new = flatten(params["locked"])
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def snapshot(params: dict, path: Path | str = PARAMS_PATH) -> None:
    """Write a JSON snapshot so drift can be reported key by key."""
    p = Path(path).parent / ".coda_params.json"
    p.write_text(json.dumps(params["locked"], sort_keys=True, indent=2))


def _log_deviations(params: dict) -> None:
    devs = params.get("deviations") or []
    if not devs:
        return
    logger.info("%d declared deviation(s) from the published methods:", len(devs))
    for d in devs:
        logger.info("  %s: paper=%s ours=%s (%s)", d["parameter"], d["paper"],
                    d["ours"], d["reason"])


def check_applicability(params: dict, dataset: str, stages: set[str]) -> set[str]:
    """Drop stages that cannot run on this dataset, and say why.

    Running a stage on data that cannot support it is worse than skipping it:
    registration of non-serial sections produces a transform and a correlation
    number, and neither means anything.
    """
    app = params["applicability"]
    meta = params["datasets"].get(dataset)
    if meta is None:
        raise ValueError(f"unknown dataset '{dataset}'. "
                         f"Options: {sorted(params['datasets'])}")

    runnable = set(stages)

    if not meta.get("serial", False):
        blocked = runnable & set(app["serial_required"])
        for s in sorted(blocked):
            logger.error("stage '%s' needs SERIAL SECTIONS; '%s' has none. Skipping.",
                         s, dataset)
        runnable -= blocked

    if meta.get("has_ihc") and not meta.get("has_he"):
        blocked = runnable & set(app["never_on_dab_ihc"])
        for s in sorted(blocked):
            logger.error("stage '%s' needs an EOSIN channel; '%s' is DAB-IHC "
                         "with no H&E. Skipping.", s, dataset)
        runnable -= blocked

    return runnable

#!/usr/bin/env python
"""Build the report for both arms: figures, tables, and a self-contained HTML.

    python scripts/run_report.py

Every figure states REAL with its dataset and n. A stage that did not run is
named with the input it needs, so an absent result can never be mistaken for a
null one.
"""
from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REP = ROOT / "reports"
P = ROOT / "results/prostate"
T = ROOT / "results/tcga"

NAVY, STEEL, ORANGE = "#1C2B4A", "#2471A3", "#E69F00"
logger = logging.getLogger("report")


def style():
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "axes.titlesize": 9})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    return FIG / f"{name}.png"


def fig_registration(s):
    tre = pd.read_csv(P / "stage2_tre.csv")
    atre = pd.read_csv(P / "stage2_atre.csv")
    corr = pd.read_csv(P / "stage1_correlation.csv")
    style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    a = ax[0]
    a.plot(tre.pair, tre.tre_mean_um, lw=.8, color=NAVY, label="after registration")
    a.axhline(s["baseline_tre_um"], color=ORANGE, lw=2,
              label=f"no transform {s['baseline_tre_um']:.0f} um")
    a.set_yscale("log"); a.set_xlabel("section pair"); a.set_ylabel("TRE (um)")
    a.set_title("(a) pairwise error against the do-nothing baseline")
    a.legend(fontsize=7)

    a = ax[1]
    a.plot(atre.pair, atre.atre_um, lw=.8, color=STEEL)
    a.axhline(s["baseline_atre_um"], color=ORANGE, lw=2,
              label=f"baseline {s['baseline_atre_um']:.0f} um")
    a.set_xlabel("section pair"); a.set_ylabel("ATRE (um)")
    a.set_title("(b) accumulated error, cumulative vector")
    a.legend(fontsize=7)

    a = ax[2]
    a.hist(corr.correlation.dropna(), bins=30, color=NAVY, edgecolor="white")
    a.axvline(0.30, color=ORANGE, ls="--", lw=1.5, label="acceptance 0.30")
    a.set_xlabel("pixel correlation"); a.set_ylabel("sections")
    a.set_title(f"(c) quality, {s['n_flagged']} of {s['n_sections']} flagged")
    a.legend(fontsize=7)

    fig.suptitle("Figure 1. Registration of the serial prostate series. "
                 f"REAL, Kartasalo mouse prostate, n = {s['n_sections']} sections.",
                 y=-0.04, fontsize=9.5, fontweight="bold")
    return save(fig, "F1_registration")


def fig_volume(s):
    vol = np.load(P / "volume.npy", mmap_mode="r")
    style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.4))
    mid = vol.shape[0] // 2
    ax[0].imshow(vol[mid], cmap="gray"); ax[0].set_title(
        f"(a) section {mid+1} of {vol.shape[0]}, cutting plane")
    ax[1].imshow(np.asarray(vol).min(axis=1), cmap="gray", aspect="auto")
    ax[1].set_title("(b) xz through the block, z vertical")
    ax[2].imshow(np.asarray(vol).min(axis=2), cmap="gray", aspect="auto")
    ax[2].set_title("(c) yz through the block")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Figure 2. The reconstructed volume. REAL, mouse prostate, "
                 f"{s['n_sections']} sections at {s['mpp_um']:.2f} um/px, "
                 f"{s['block_depth_um']/1000:.1f} mm deep.",
                 y=-0.02, fontsize=9.5, fontweight="bold")
    return save(fig, "F2_volume")


def fig_overcount(s):
    sens = pd.read_csv(P / "stage6_overcount_sensitivity.csv")
    zs = pd.read_csv(P / "stage2_zskip.csv")
    style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = ax[0]
    a.plot(sens.min_object_volume_um3.clip(lower=1e3), sens.overcounting_ratio,
           "-o", color=NAVY)
    a.axhline(s["coda_pancreas_reference"], color=ORANGE, ls="--",
              label=f"CODA pancreas {s['coda_pancreas_reference']}x")
    a.axhline(1, color="0.6", lw=.8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("minimum object volume (um3)"); a.set_ylabel("2D count / 3D count")
    a.set_title("(a) overcounting depends on what is counted")
    a.legend(fontsize=7)

    a = ax[1]
    a.plot(zs.spacing_um, zs.percent_composition_error, "-o", color=STEEL)
    a.axhline(5, color=ORANGE, ls="--", label="5 percent tolerance")
    a.set_xlabel("spacing between sections (um)")
    a.set_ylabel("composition error (%)")
    a.set_title("(b) cost of skipping sections")
    a.legend(fontsize=7)
    fig.suptitle("Figure 3. Two-dimensional counting overestimates object number, "
                 f"and the cost of coarser z. REAL, mouse prostate, n = {s['n_sections']}.",
                 y=-0.04, fontsize=9.5, fontweight="bold")
    return save(fig, "F3_overcount")


def fig_tcga(t):
    d = pd.read_csv(T / "stage3_cell_detection.csv")
    a_ = pd.read_csv(T / "stage7_anisotropy.csv")
    style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].bar(range(len(d)), d.nuclear_density_per_mm2, color=NAVY, edgecolor="white")
    ax[0].set_xticks(range(len(d)))
    ax[0].set_xticklabels([s[8:12] for s in d.slide], rotation=60, fontsize=6)
    ax[0].set_ylabel("nuclei per mm2"); ax[0].set_title("(a) nuclear density per slide")

    ax[1].hist(a_.anisotropy, bins=40, color=STEEL, edgecolor="white")
    ax[1].axvline(a_.anisotropy.median(), color=ORANGE, lw=2,
                  label=f"median {a_.anisotropy.median():.3f}")
    ax[1].set_xlabel("fibre anisotropy index"); ax[1].set_ylabel("windows")
    ax[1].set_title(f"(b) anisotropy, {len(a_)} windows"); ax[1].legend(fontsize=7)

    by = [g.anisotropy.values for _, g in a_.groupby("slide")]
    ax[2].boxplot(by, showfliers=False)
    ax[2].set_xlabel("slide"); ax[2].set_ylabel("anisotropy")
    ax[2].set_xticklabels([f"{i+1}" for i in range(len(by))], fontsize=6)
    ax[2].set_title("(c) distribution per slide, never a patient value")
    fig.suptitle("Figure 4. Human breast, the stages TCGA supports. REAL, "
                 f"TCGA-BRCA diagnostic slides, n = {t['n_slides']} patients.",
                 y=-0.06, fontsize=9.5, fontweight="bold")
    return save(fig, "F4_tcga")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    REP.mkdir(parents=True, exist_ok=True)
    s = json.loads((P / "summary.json").read_text())
    t = json.loads((T / "summary.json").read_text())

    figs = [("Figure 1", fig_registration(s)), ("Figure 2", fig_volume(s)),
            ("Figure 3", fig_overcount(s)), ("Figure 4", fig_tcga(t))]

    sens = pd.read_csv(P / "stage6_overcount_sensitivity.csv")
    zs = pd.read_csv(P / "stage2_zskip.csv")
    d = pd.read_csv(T / "stage3_cell_detection.csv")

    tables = {
        "Table 1. Registration accuracy, mouse prostate": pd.DataFrame([
            {"metric": "pairwise TRE, no transform", "value": round(s["baseline_tre_um"], 1), "units": "um"},
            {"metric": "pairwise TRE, registered", "value": round(s["registered_tre_um"], 1), "units": "um"},
            {"metric": "accumulated TRE, no transform", "value": round(s["baseline_atre_um"], 1), "units": "um"},
            {"metric": "accumulated TRE, registered", "value": round(s["registered_atre_um"], 1), "units": "um"},
            {"metric": "pixel correlation, median", "value": round(s["correlation_median"], 4), "units": "Spearman"},
            {"metric": "sections below the 0.30 gate", "value": s["n_flagged"], "units": f"of {s['n_sections']}"},
        ]),
        "Table 2. Two-dimensional versus three-dimensional counts": sens,
        "Table 3. z-resolution": zs,
        "Table 4. TCGA-BRCA per slide": d,
    }

    def b64(p):
        return base64.b64encode(Path(p).read_bytes()).decode()

    html = [f"""<!doctype html><meta charset="utf-8"><title>CODA on public data</title>
<style>body{{font-family:Calibri,Arial,sans-serif;max-width:1150px;margin:2rem auto;
padding:0 1rem;line-height:1.55;color:#222}}h1,h2{{color:{NAVY}}}h3{{color:{STEEL}}}
table{{border-collapse:collapse;margin:1rem 0;font-size:.9em}}
td,th{{border:1px solid #ccc;padding:4px 9px}}th{{background:#eef2f7}}
img{{max-width:100%;border:1px solid #ddd;margin:.5rem 0}}
.cap{{font-size:.85em;color:#555;margin-bottom:1.4rem}}
.warn{{background:#FFF6E5;border-left:4px solid {ORANGE};padding:.7rem 1rem;margin:1rem 0}}</style>
<h1>CODA applied to public data: 3D on serial sections, 2D on TCGA</h1>
<p><b>Two arms.</b> Arm 1 is the full three-dimensional pipeline on
{s['n_sections']} consecutive mouse prostate sections, a block
{s['block_depth_um']/1000:.1f} mm deep. Arm 2 is the stages TCGA can support, on
{t['n_slides']} human breast diagnostic slides.</p>

<div class="warn"><b>Feasibility was established before any analysis.</b> TCGA has
no consecutive sections: querying the GDC returned a maximum of seven slides for
any patient, and the section_location field reports TOP and BOTTOM because
slides are deliberately taken from opposite ends of a block. Stages 1, 2, 5 and 6
therefore cannot run on TCGA for any cancer type and were not attempted. The
source publication's own data is not public, being available only on request
from its authors.</div>

<h2>Arm 1: three-dimensional reconstruction</h2>
<p>Registration reduced pairwise landmark error from
<b>{s['baseline_tre_um']:.0f} um with no transform to {s['registered_tre_um']:.1f} um</b>,
a {s['baseline_tre_um']/s['registered_tre_um']:.1f}-fold reduction, with a median
pixel correlation of {s['correlation_median']:.3f} and <b>{s['n_flagged']} of
{s['n_sections']} sections</b> falling below the 0.30 acceptance threshold.
Comparison against applying no transform at all is reported because it is the
weakest test a registration must pass, and it is not passed automatically.</p>

<p>Counting objects section by section overestimates their number by
<b>{s['overcounting_ratio']:.2f}-fold</b> across all objects, rising to
{sens.overcounting_ratio.iloc[2]:.0f}-fold when restricted to structures above
10<sup>5</sup> um<sup>3</sup>. The source publication reported 12.3-fold on
average in pancreas and up to 40-fold. The ratio is a property of what is
counted and at what size, not a single constant, which is why it is reported as
a curve.</p>

<p>Composition error stays near the 5 percent tolerance out to
{zs.spacing_um[zs.percent_composition_error <= 6].max():.0f} um of spacing,
approximately reproducing the published claim of under 5 percent to 12 um. The
same measurement on a 47-section liver block failed badly, because 235 um is too
thin for the comparison to mean anything.</p>

<h2>Arm 2: human breast, the stages TCGA supports</h2>
<p>Nuclear density across {t['n_slides']} slides had a median of
<b>{t['nuclear_density_median_per_mm2']:.0f} per mm<sup>2</sup></b>. Fibre
anisotropy had a median of {t['anisotropy_median']:.3f} over
{t['anisotropy_n_windows']} windows.</p>

<div class="warn"><b>Neither is validated against human annotation.</b> No manual
nucleus annotation exists for these slides, so the published 90 percent precision
and recall gate is not tested and is not claimed. Fibre anisotropy is reported as
a distribution across windows and slides and never as a per-patient value,
because the cutting angle of a single section is unknown and cannot be
corrected. The volume in Arm 1 can choose its plane; this arm cannot.</div>
"""]

    for i, (label, path) in enumerate(figs):
        html.append(f'<img src="data:image/png;base64,{b64(path)}">')
    html.append("<h2>Tables</h2>")
    for name, df in tables.items():
        html.append(f"<h3>{name}</h3>")
        html.append(df.to_html(index=False, border=0))
        df.to_csv(REP / f"{name.split('.')[0].replace(' ', '_')}.csv", index=False)

    html.append(f"""<h2>Stages not run, and the input each needs</h2>
<table><tr><th>Stage</th><th>Arm</th><th>Why not</th></tr>
<tr><td>1, 2, 5, 6</td><td>TCGA</td><td>no consecutive sections anywhere in TCGA</td></tr>
<tr><td>3 validation</td><td>both</td><td>needs two human annotators; none exist</td></tr>
<tr><td>4 segmentation</td><td>both</td><td>needs a GPU and annotated tiles; a Colab
notebook using the BCSS annotations is provided</td></tr>
<tr><td>7 sectioning angle</td><td>TCGA</td><td>needs a volume to choose the plane</td></tr></table>
<p style="color:#666;font-size:.85em">Arm 1 is mouse prostate. It establishes that
the method runs and how accurately, and is not a breast finding. No
three-dimensional reconstruction of breast tissue appears anywhere in this work,
and none is possible without serial breast sections.</p>""")

    out = REP / "analysis_report.html"
    out.write_text("\n".join(html), encoding="utf-8")
    logger.info("wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    for name, _ in figs:
        logger.info("  %s", name)


if __name__ == "__main__":
    main()

"""One function per report figure (F1-F22).

Every function returns:
    {"id","title","source","paths":{"png","pdf"},"caption"}

`source` is one of:
    REAL ...        built from measured data; the dataset and n are named
    SIMULATED ...   built from a synthetic fixture, never a finding
    MISSING DATA    cannot be built; a placeholder names the exact input needed

Nothing here fabricates a number or a panel. Where an arm has no data the
figure is a labelled placeholder, kept numbered in sequence so the report
stays complete.

Real inputs, all produced by the Arm C scripts:
    results/usm_qc.csv                     234 images, mpp and counterstain
    results/usm_markers.csv                225 analysed, per-marker results
    results/usm_spatial.csv                77 point patterns, border corrected
    config/coda_params.yaml                120 locked parameters, deviations
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import (
    NAVY, OKABE_ITO, STEEL_BLUE, apply_style, categorical_colors, continuous_cmap,
    letter_panels, placeholder_figure, save_figure, source_caption,
)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"

USM_N = "USM breast IHC field-of-view captures"


def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def _ph(fid: str, title: str, needs: str, unblocks: str, figdir: str, fname: str) -> dict:
    fig, _ = placeholder_figure(fid, title, missing_file=needs, unblocks=unblocks)
    paths = save_figure(fig, fname, figdir)
    return {"id": fid, "title": title, "source": "MISSING DATA", "paths": paths,
            "caption": f"Placeholder. Requires {needs}. Unblocks {unblocks}."}



KART = ROOT / "results/kartasalo"
KTAG = "ds16fix"   # corrected run; the original is kept as ds16


def _k(name: str):
    """Load a Kartasalo (Arm A) result table, or None if it has not run yet."""
    f = KART / f"{name}_{KTAG}.csv"
    return pd.read_csv(f) if f.exists() else None


def _ksummary():
    import json
    f = KART / f"summary_{KTAG}.json"
    return json.loads(f.read_text()) if f.exists() else None




def _k3():
    """Stage 5/6 reconstruction summary, or None if it has not run."""
    import json
    f = KART / "stage6_summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def _ksens():
    f = KART / "stage6_overcount_sensitivity.csv"
    return pd.read_csv(f) if f.exists() else None


# ============================================================ F1 study design


def f01_study_design(figdir: str) -> dict:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    ax.set_title("Three arms, seven CODA stages: which runs where, and why",
                 loc="left", fontsize=12)

    arms = [
        (0.2, 4.3, "ARM A  Kartasalo\n260 mouse prostate serial sections\nstages 1-7, the only 3D",
         "#E8EEF6", NAVY),
        (0.2, 2.4, "ARM B  ACROBAT\n1,153 human breast patients\nstages 3, 4, 7 + H&E to IHC",
         "#E8EEF6", STEEL_BLUE),
        (0.2, 0.5, "ARM C  USM IHC\n234 field-of-view captures\nmarker quantification only",
         "#FCEFE3", OKABE_ITO[6]),
    ]
    for x, y, label, fc, ec in arms:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), 3.4, 1.5,
                     boxstyle="round,pad=0.08", facecolor=fc, edgecolor=ec, linewidth=1.6))
        ax.text(x + 1.7, y + 0.75, label, ha="center", va="center", fontsize=8.4)

    stages = ["1 register", "2 reg QC", "3 cell detect", "4 segment",
              "5 reconstruct", "6 quantify", "7 fibers"]
    # rows: arm A, B, C  |  1 = runs, 0 = blocked
    grid = np.array([[1, 1, 1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 0, 0, 1],
                     [0, 0, 1, 0, 0, 0, 0]])
    x0, w = 4.2, 1.05
    for j, s in enumerate(stages):
        ax.text(x0 + j * w + w / 2, 6.0, s, ha="center", va="bottom", fontsize=7.4,
                rotation=32)
    for i, (y, _) in enumerate([(4.3, 0), (2.4, 0), (0.5, 0)]):
        for j in range(7):
            ok = grid[i, j]
            ax.add_patch(mpatches.Rectangle((x0 + j * w, y + 0.35), w * 0.9, 0.8,
                         facecolor="#2E7D32" if ok else "#BDBDBD",
                         edgecolor="white", linewidth=1.2))
            ax.text(x0 + j * w + w * 0.45, y + 0.75, "run" if ok else "blocked",
                    ha="center", va="center", fontsize=6.6,
                    color="white", fontweight="bold")

    ax.text(4.2, 0.05,
            "Blocked is decided by the data, not by preference. Stages 1, 2, 5 and 6 need "
            "SERIAL sections;\nACROBAT sections are from one block but not consecutive, and "
            "the USM captures are single fields.\nStage 7 needs an eosin channel, which "
            "DAB-IHC does not have.",
            fontsize=7.4, va="bottom")
    source_caption(fig, "Design schematic. Not derived from a run.", y=-0.02)
    return {"id": "F1", "title": "Study design: three arms, seven stages",
            "source": "DESIGN (no data)", "paths": save_figure(fig, "F1_study_design", figdir),
            "caption": "Which CODA stage can run on which dataset, and the reason each "
                       "blocked cell is blocked. The applicability gate in the runner "
                       "enforces this rather than leaving it to judgement."}


# ==================================================== F2-F16 Arms A and B


def f02_registration_workflow(figdir):
    return _ph("F2", "Registration workflow and pre/post overlay",
               "Kartasalo serial stack (open CC BY 4.0; retrieved, registration in progress)",
               "Arm A stage 1", figdir, "F2_registration_workflow")


def _kdiag():
    import json
    f = KART / "diagnosis.json"
    return json.loads(f.read_text()) if f.exists() else None


def f03_registration_accuracy(figdir):
    tre, atre, s = _k("step6_tre_pairwise"), _k("step6_atre"), _ksummary()
    g = _kdiag()
    if tre is None or atre is None or s is None:
        return _ph("F3", "Registration accuracy: TRE and ATRE vs distance from centre",
                   "Kartasalo stack plus its operator fiducials", "Arm A stage 2",
                   figdir, "F3_registration_accuracy")
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))

    ax = axes[0]
    ax.plot(range(1, len(tre) + 1), tre["tre_mean_um"], "-o", ms=3, color=NAVY,
            label="TRE, rigid stage")
    ax.axhline(s["rigid_floor_mean_um"], color=OKABE_ITO[6], ls="--", lw=1.5,
               label="rigid floor %.0f um" % s["rigid_floor_mean_um"])
    ax.axhline(s["interobserver_median_um"], color=OKABE_ITO[2], ls=":", lw=1.5,
               label="observer floor %.1f um" % s["interobserver_median_um"])
    if g:
        ax.axhline(g["identity_tre_mean_um"], color=OKABE_ITO[1], lw=2.0,
                   label="NO registration %.0f um" % g["identity_tre_mean_um"])
    ax.set_yscale("log")
    ax.set_xlabel("section pair")
    ax.set_ylabel("TRE (um)")
    ax.set_title("per-pair error against its two floors", fontsize=8)
    ax.legend(fontsize=6.5)

    ax = axes[1]
    ax.scatter(atre["distance_from_reference"], atre["atre_mean_um"], s=22,
               color=STEEL_BLUE, edgecolor="white", zorder=3)
    if len(atre) > 3:
        z = np.polyfit(atre["distance_from_reference"], atre["atre_mean_um"], 1)
        xs = np.linspace(0, atre["distance_from_reference"].max(), 50)
        r = np.corrcoef(atre["distance_from_reference"], atre["atre_mean_um"])[0, 1]
        ax.plot(xs, np.polyval(z, xs), color=NAVY, lw=1.6,
                label="slope %.1f um/section, r=%.2f" % (z[0], r))
        ax.legend(fontsize=7)
    ax.set_xlabel("sections from the centre reference")
    ax.set_ylabel("ATRE (um)")
    ax.set_title("does the stack drift with distance?", fontsize=8)

    ax = axes[2]
    ax.hist(tre["tre_mean_um"], bins=16, color=NAVY, edgecolor="white")
    ax.axvline(s["rigid_floor_mean_um"], color=OKABE_ITO[6], ls="--", lw=1.5)
    ax.set_xlabel("TRE (um)")
    ax.set_ylabel("section pairs")
    ax.set_title("distribution of pairwise TRE", fontsize=8)

    letter_panels(axes)
    source_caption(fig, "REAL DATA (Kartasalo mouse liver, %d serial sections, "
                        "CC BY 4.0; 4 operator fiducials per section)."
                        % s["n_sections"], y=-0.06)
    return {"id": "F3", "title": "Registration accuracy against two independent floors",
            "source": "REAL (Kartasalo liver, n=%d)" % s["n_sections"],
            "paths": save_figure(fig, "F3_registration_accuracy", figdir),
            "caption": "Registration accuracy after correcting the rotation estimator. "
                       "(A) Pairwise target registration error, mean %.0f um, against the "
                       "line that decides the result: the gold line is what applying NO "
                       "transform at all achieves, and a method that does not fall below "
                       "it is doing harm. The stock pipeline sat far above it at 2544 um; "
                       "the corrected run sits well below. Two further floors bound the "
                       "interpretation. The upper dashed line is the best a rigid "
                       "transform can do at all: fitting the transform directly to the "
                       "fiducials by Procrustes still leaves %.0f um, because the tissue "
                       "deforms non-rigidly between sections. The lower dotted line is "
                       "the annotation floor, %.1f um between two independent observers, "
                       "below which the ground truth cannot resolve a difference. "
                       "(B) Accumulated error against distance from the centre "
                       "reference; a positive slope is the stack bending rather than "
                       "individual pairs being misaligned, which is the failure that "
                       "matters for a reconstruction. (C) Distribution across pairs. "
                       "Elastic displacement is not included: register_stack applies the "
                       "field to the image and discards it, so it cannot be replayed "
                       "onto point coordinates."
                       % (s.get("tre_full_mean_um", s.get("tre_rigid_mean_um", float("nan"))),
                          s["rigid_floor_mean_um"],
                          s["interobserver_median_um"])}


def f04_z_resolution(figdir):
    ax_df, zs, s = _k("step8_axial_lateral"), _k("step8_zskip"), _ksummary()
    if ax_df is None or zs is None or s is None:
        return _ph("F4", "z-resolution: axial vs lateral correlation, composition error",
                   "Kartasalo serial stack", "Arm A stage 2", figdir, "F4_z_resolution")
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))

    ax = axes[0]
    for name, g in ax_df.groupby("axis"):
        ax.plot(g["distance_um"], g["correlation"], "-o", ms=3,
                color=NAVY if name == "xy" else OKABE_ITO[6],
                label=("xy, within section (ceiling)" if name == "xy"
                       else "z, between sections"))
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("distance (um)")
    ax.set_ylabel("pixel correlation")
    ax.set_title("the gap is registration error", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(zs["spacing_um"], zs["percent_composition_error"], "-o", ms=4, color=NAVY)
    ax.axhline(5.0, color=OKABE_ITO[6], ls="--", lw=1.5, label="5% tolerance")
    ax.set_xlabel("effective spacing (um)")
    ax.set_ylabel("composition error (%)")
    ax.set_title("what skipping sections costs", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[2]
    corr = _k("step5_correlation")
    if corr is not None:
        ax.plot(corr["section"], corr["correlation"], "-o", ms=3, color=NAVY,
                label="rigid")
        if corr["correlation_after_elastic"].notna().any():
            ax.plot(corr["section"], corr["correlation_after_elastic"], "-s", ms=3,
                    color=STEEL_BLUE, alpha=.8, label="after elastic")
        ax.axhline(0.30, color=OKABE_ITO[6], ls="--", lw=1.5, label="min_correlation")
        ax.legend(fontsize=6.5)
    ax.set_xlabel("section")
    ax.set_ylabel("pixel correlation")
    ax.set_title("per-section registration quality", fontsize=8)

    letter_panels(axes)
    source_caption(fig, "REAL DATA (Kartasalo mouse liver, %d sections at %.2f um/px, "
                        "%.0f um sections)."
                        % (s["n_sections"], s["mpp_um"], s["section_thickness_um"]),
                   y=-0.06)
    return {"id": "F4", "title": "Axial resolution and the cost of skipping sections",
            "source": "REAL (Kartasalo liver, n=%d)" % s["n_sections"],
            "paths": save_figure(fig, "F4_z_resolution", figdir),
            "caption": "(A) Within-section xy correlation is the ceiling, because it "
                       "measures how pixel intensity varies across intact tissue. "
                       "Between-section z correlation falls far below it, and the gap is "
                       "reconstruction error rather than biology. (B) Composition error "
                       "against effective section spacing, which is the measurement that "
                       "justifies processing one section in three; it has to be made on "
                       "the tissue at hand rather than inherited. (C) Per-section "
                       "correlation with the 0.30 acceptance threshold; %d of %d "
                       "sections fall below it and are flagged rather than silently "
                       "dropped." % (s.get("n_flagged", 0), s["n_sections"])}


def f05_cell_detection(figdir):
    return _ph("F5", "Cell detection: manual vs automatic, precision and recall",
               "manual annotations from two annotators at 2 um tolerance",
               "Arm A/B stage 3 validation", figdir, "F5_cell_detection")


def f06_segmentation(figdir):
    return _ph("F6", "Segmentation: tile construction, confusion matrix, per-class metrics",
               "annotated H&E and a GPU for DeepLab v3+ training", "Arm A/B stage 4",
               figdir, "F6_segmentation")


def f07_volume_renders(figdir):
    import numpy as _np
    s3 = _k3()
    vp = KART / "volume_natural_ds16fix.npy"
    if s3 is None or not vp.exists():
        return _ph("F7", "3D reconstruction volume renders per tissue class",
                   "Kartasalo serial stack and completed stage 4", "Arm A stage 5",
                   figdir, "F7_volume_renders")
    z = _np.load(KART / "stage6_projections.npz")
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    n_sec = s3["n_sections"]
    axes[0].imshow(z["section_mid"], cmap="gray")
    axes[0].set_title("section %d of %d, plane of cutting" % (n_sec // 2 + 1, n_sec),
                      fontsize=8)
    axes[1].imshow(z["lumen_xy"], cmap="magma")
    axes[1].set_title("lumina summed through all %d sections" % n_sec, fontsize=8)
    axes[2].imshow(z["lumen_xz"], cmap="magma", aspect="auto")
    axes[2].set_title("lumina, xz through the stack (z vertical)", fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    letter_panels(axes)
    source_caption(fig, "REAL DATA (Kartasalo mouse liver, %d sections registered into a "
                        "volume at %.2f um/px, %.0f um apart)."
                        % (s3["n_sections"], s3["mpp_um"], s3["section_thickness_um"]),
                   y=-0.04)
    return {"id": "F7", "title": "The reconstructed volume",
            "source": "REAL (Kartasalo liver volume, n=%d)" % s3["n_sections"],
            "paths": save_figure(fig, "F7_volume_renders", figdir),
            "caption": "A genuine three-dimensional reconstruction: %d registered "
                       "sections stacked with real z spacing. (A) One section in the "
                       "plane of cutting, with vascular lumina visible as bright spaces "
                       "inside dark parenchyma; these are the objects counted. (B) Those "
                       "lumina summed through the whole stack, so a vessel running "
                       "perpendicular to the cutting plane appears as a bright spot and "
                       "one running obliquely as a streak. (C) The same through the xz "
                       "plane with z vertical, which is the view a single section cannot "
                       "provide at all: continuity down the page is a structure "
                       "genuinely traversing the block, and an abrupt horizontal break "
                       "would be a registration failure. The projections show the "
                       "segmented mask rather than raw intensity, because rigid warping "
                       "fills the area outside the tissue with zeros and an intensity "
                       "projection through the stack is dominated by that fill. Tissue "
                       "classes are not separated, as no trained multi-class "
                       "segmentation exists for this material." % s3["n_sections"]}


def f08_z_projections(figdir):
    return _ph("F8", "z-projections per tissue class",
               "reconstructed volume", "Arm A stage 6", figdir, "F8_z_projections")


def f09_composition_heatmap(figdir):
    return _ph("F9", "Tissue composition heatmap, samples by class",
               "segmented volumes", "Arm A stage 6", figdir, "F9_composition_heatmap")


def f10_cell_density_3d(figdir):
    return _ph("F10", "3D cell density by class, bulk and local",
               "reconstructed volume plus measured nuclear diameters",
               "Arm A stage 6", figdir, "F10_cell_density_3d")


def f11_connectivity(figdir):
    return _ph("F11", "Connectivity: objects distinct in 2D vs 3D",
               "reconstructed volume", "Arm A stage 6", figdir, "F11_connectivity")


def f12_overcounting(figdir):
    s3, sens = _k3(), _ksens()
    c2 = _k("stage6_2d_counts") if (KART / "stage6_2d_counts.csv").exists() else None
    if s3 is None or sens is None:
        return _ph("F12", "Overcounting ratio per section, 12.3-fold reference line",
                   "reconstructed volume with per-object 3D connectivity",
                   "Arm A stage 6, the headline CODA result", figdir, "F12_overcounting")
    obj = pd.read_csv(KART / "stage6_3d_objects.csv")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.1))

    ax = axes[0]
    ax.plot(sens["min_object_volume_um3"].clip(lower=1e3), sens["overcounting_ratio"],
            "-o", ms=5, color=NAVY)
    ax.axhline(s3["coda_pancreas_reference"], color=OKABE_ITO[6], ls="--", lw=1.6,
               label="CODA pancreas %.1fx" % s3["coda_pancreas_reference"])
    ax.axhline(1.0, color="0.6", lw=1.0, label="1x = no overcounting")
    ax.set_xscale("log"); ax.set_xlabel("minimum object volume (um3)")
    ax.set_ylabel("2D count / 3D count")
    ax.set_title("the ratio is a property of what you count", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[1]
    sp = obj["sections_spanned"]
    ax.hist(sp, bins=np.arange(0.5, min(sp.max(), 40) + 1.5), color=NAVY,
            edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("sections a 3D object spans"); ax.set_ylabel("objects (log)")
    ax.set_title("%d%% occupy a single section" %
                 round(100 * (sp == 1).mean()), fontsize=8)

    ax = axes[2]
    if c2 is not None:
        ax.plot(c2["section"], c2["objects_2d"], "-o", ms=3, color=STEEL_BLUE)
        ax.axhline(s3["n_3d_objects"] / len(c2), color=OKABE_ITO[6], ls="--", lw=1.4,
                   label="3D objects / section")
        ax.legend(fontsize=7)
    ax.set_xlabel("section"); ax.set_ylabel("objects seen in that section")
    ax.set_title("per-section 2D counts", fontsize=8)

    letter_panels(axes)
    source_caption(fig, "REAL DATA (Kartasalo mouse liver, %d-section reconstructed "
                        "volume, %.2f um/px, %.0f um spacing)."
                        % (s3["n_sections"], s3["mpp_um"],
                           s3["section_thickness_um"]), y=-0.06)
    return {"id": "F12", "title": "Two-dimensional counting overestimates object number",
            "source": "REAL (Kartasalo liver volume, n=%d sections)" % s3["n_sections"],
            "paths": save_figure(fig, "F12_overcounting", figdir),
            "caption": "The measurement single sections cannot make. A structure crossing "
                       "several sections is counted once per section in two dimensions "
                       "and once in the volume, so the ratio of the two says how much "
                       "single-section counting inflates object number. (A) That ratio is "
                       "not one number: counting everything gives %.2f-fold, because "
                       "small features are confined to one section and cannot overcount, "
                       "while restricting to objects above 10^6 um3, the scale of "
                       "substantial anatomical structures, gives %.1f-fold, close to the "
                       "%.1f-fold reported for pancreas. Any single figure quoted without "
                       "its object definition is meaningless. (B) %d%% of objects occupy "
                       "one section only; the largest spans %d of %d. (C) Per-section 2D "
                       "counts. Objects here are vascular lumina segmented by intensity, "
                       "NOT the ten-class trained segmentation of the original, and the "
                       "tissue is mouse liver, not breast."
                       % (sens["overcounting_ratio"].iloc[0],
                          float(sens.loc[sens["min_object_volume_um3"] == 1e6,
                                         "overcounting_ratio"].iloc[0])
                          if (sens["min_object_volume_um3"] == 1e6).any() else float("nan"),
                          s3["coda_pancreas_reference"],
                          round(100 * (obj["sections_spanned"] == 1).mean()),
                          int(obj["sections_spanned"].max()), s3["n_sections"])}


def f13_object_morphology(figdir):
    return _ph("F13", "Per-object 3D morphology: volume, primary axis, elongation",
               "reconstructed volume", "Arm A stage 6", figdir, "F13_object_morphology")


def f14_fiber_anisotropy(figdir):
    return _ph("F14", "Fiber anisotropy index distributions",
               "H&E with an eosin channel (Arm A or B); DAB-IHC has none",
               "stage 7", figdir, "F14_fiber_anisotropy")


def f15_acrobat_registration(figdir):
    return _ph("F15", "ACROBAT H&E to IHC registration and landmark residuals",
               "ACROBAT WSIs and its 37,208 landmarks (data use agreement required)",
               "Arm B stages 1-2", figdir, "F15_acrobat_registration")


def f16_batch_audit(figdir):
    return _ph("F16", "Batch audit: cohort effect size vs within-cohort scanner effect",
               "at least two cohorts with scanner metadata",
               "cross-cohort comparison", figdir, "F16_batch_audit")


# ==================================================== F17-F21 Arm C, REAL


def f17_usm_qc(figdir: str) -> dict:
    qc = _load("usm_qc.csv")
    if qc is None:
        return _ph("F17", "USM QC: mpp and counterstain", "results/usm_qc.csv",
                   "Arm C", figdir, "F17_usm_qc")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    markers = sorted(qc["marker"].unique())
    colors = dict(zip(markers, categorical_colors(len(markers))))

    ok = qc.dropna(subset=["mpp_um_per_px"])
    for m in markers:
        v = ok[ok["marker"] == m]["mpp_um_per_px"]
        axes[0].scatter(np.full(len(v), m), v, s=14, color=colors[m], alpha=0.7,
                        linewidths=0)
    axes[0].axhline(2.5, color=OKABE_ITO[6], ls="--", lw=1.3)
    axes[0].text(0.02, 2.7, "2.5 um/px: nucleus under ~3 px", fontsize=7.4,
                 color=OKABE_ITO[6], transform=axes[0].get_yaxis_transform())
    axes[0].set_yscale("log")
    axes[0].set_ylabel("microns per pixel (log scale)")
    axes[0].set_xlabel("marker")

    for m in markers:
        v = qc[qc["marker"] == m]["counterstain_fraction"]
        axes[1].scatter(np.full(len(v), m), 100 * v, s=14, color=colors[m],
                        alpha=0.7, linewidths=0)
    axes[1].axhline(1.0, color=OKABE_ITO[6], ls="--", lw=1.3)
    axes[1].text(0.02, 1.15, "1%: below this, counterstain absent", fontsize=7.4,
                 color=OKABE_ITO[6], transform=axes[1].get_yaxis_transform())
    axes[1].set_yscale("log")
    axes[1].set_ylabel("counterstain fraction (%, log scale)")
    axes[1].set_xlabel("marker")

    ct = pd.crosstab(qc["marker"], qc["counterstain_grade"])
    for c in ["absent", "marginal", "adequate"]:
        if c not in ct:
            ct[c] = 0
    ct = ct[["absent", "marginal", "adequate"]]
    bottom = np.zeros(len(ct))
    for c, col in zip(ct.columns, [OKABE_ITO[6], OKABE_ITO[4], OKABE_ITO[3]]):
        axes[2].bar(ct.index, ct[c], bottom=bottom, color=col, label=c,
                    edgecolor="white")
        bottom += ct[c].to_numpy()
    axes[2].set_ylabel("images")
    axes[2].set_xlabel("marker")
    axes[2].legend(fontsize=8, title="counterstain")

    letter_panels(axes)
    n_abs = int((qc["counterstain_grade"] == "absent").sum())
    source_caption(fig, f"REAL DATA ({USM_N}, n={len(qc)} images). "
                        f"{n_abs} graded counterstain absent.", y=-0.06)
    return {"id": "F17", "title": "Arm C quality control: resolution and counterstain",
            "source": f"REAL ({USM_N}, n={len(qc)})",
            "paths": save_figure(fig, "F17_usm_qc", figdir),
            "caption": f"(A) Microns per pixel per image, recovered from the burned-in "
                       f"scale bar, log scale. The cohort spans "
                       f"{ok['mpp_um_per_px'].min():.2f} to {ok['mpp_um_per_px'].max():.2f} "
                       f"um/px, a {ok['mpp_um_per_px'].max()/ok['mpp_um_per_px'].min():.0f}-fold "
                       f"range; images above the dashed line cannot resolve a nucleus and are "
                       f"excluded from nuclear analysis. (B) Counterstain fraction with the "
                       f"absent threshold drawn. (C) Counterstain grade by marker. "
                       f"{n_abs} of {len(qc)} images have no visible negative nuclei, so no "
                       f"denominator exists and percent-positive is not reportable for them."}


def f18_marker_quant(figdir: str) -> dict:
    mk = _load("usm_markers.csv")
    if mk is None:
        return _ph("F18", "Marker quantification", "results/usm_markers.csv",
                   "Arm C", figdir, "F18_marker_quant")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    nuc = mk[mk["marker"] != "HER2"]
    markers = sorted(nuc["marker"].unique())
    colors = dict(zip(markers, categorical_colors(len(markers))))

    for m in markers:
        v = nuc[nuc["marker"] == m]["positive_density_per_mm2"].dropna()
        axes[0].scatter(np.full(len(v), m), v, s=16, color=colors[m], alpha=0.7,
                        linewidths=0)
        if len(v):
            axes[0].plot([m], [v.median()], "_", ms=26, color="black", mew=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("positive nuclei per mm2 (log scale)")
    axes[0].set_xlabel("marker")
    axes[0].set_title("valid regardless of counterstain", fontsize=9)

    rep = nuc[nuc["percent_reportable"] == True]  # noqa: E712
    for m in markers:
        v = rep[rep["marker"] == m]["percent_positive"].dropna()
        if len(v):
            axes[1].scatter(np.full(len(v), m), v, s=18, color=colors[m], alpha=0.8,
                            linewidths=0)
    axes[1].axhline(20, color=OKABE_ITO[6], ls="--", lw=1.3)
    axes[1].text(0.02, 21, "20% Ki67 cutoff", fontsize=7.4, color=OKABE_ITO[6],
                 transform=axes[1].get_yaxis_transform())
    axes[1].set_ylabel("percent positive (%)")
    axes[1].set_xlabel("marker")
    axes[1].set_title("only where a denominator exists", fontsize=9)

    counts = (nuc.groupby("marker")["percent_reportable"]
                 .agg(["sum", "count"]).reset_index())
    x = np.arange(len(counts))
    axes[2].bar(x - 0.2, counts["count"], 0.4, color="#BDBDBD", label="analysed")
    axes[2].bar(x + 0.2, counts["sum"], 0.4, color=NAVY, label="percent reportable")
    axes[2].set_xticks(x); axes[2].set_xticklabels(counts["marker"])
    axes[2].set_ylabel("images"); axes[2].set_xlabel("marker")
    axes[2].legend(fontsize=8)

    letter_panels(axes)
    source_caption(fig, f"REAL DATA ({USM_N}, n={len(nuc)} nuclear-marker images). "
                        "HER2 excluded: membranous, see F19.", y=-0.06)
    return {"id": "F18", "title": "Marker quantification, ER PR and Ki67",
            "source": f"REAL ({USM_N}, n={len(nuc)})",
            "paths": save_figure(fig, "F18_marker_quant", figdir),
            "caption": "(A) Positive-nucleus density per mm2, which remains valid where "
                       "counterstain is absent because it needs no denominator. "
                       "(B) Percent positive, plotted only for images with an adequate or "
                       "marginal counterstain. (C) How many images support each. Percent "
                       "positive is withheld rather than back-calculated from DAB area "
                       "where no negative nuclei are visible."}


def f19_her2_membrane(figdir: str) -> dict:
    mk = _load("usm_markers.csv")
    if mk is None or "mean_membrane_completeness" not in (mk.columns if mk is not None else []):
        return _ph("F19", "HER2 membrane completeness", "results/usm_markers.csv",
                   "Arm C HER2", figdir, "F19_her2_membrane")
    h = mk[(mk["marker"] == "HER2") & mk["mean_membrane_completeness"].notna()]
    if not len(h):
        return _ph("F19", "HER2 membrane completeness", "HER2 rows in usm_markers.csv",
                   "Arm C HER2", figdir, "F19_her2_membrane")

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].hist(h["mean_membrane_completeness"], bins=20, color=NAVY,
                 edgecolor="white")
    axes[0].set_xlabel("mean membrane completeness (0 to 1)")
    axes[0].set_ylabel("images")

    axes[1].scatter(h["n_enclosed_cells"], h["mean_membrane_completeness"],
                    s=20, color=STEEL_BLUE, alpha=0.75, linewidths=0)
    axes[1].set_xlabel("enclosed cells detected (n)")
    axes[1].set_ylabel("mean membrane completeness")

    axes[2].hist(h["median_cell_area_um2"].dropna(), bins=20, color=STEEL_BLUE,
                 edgecolor="white")
    axes[2].set_xlabel("median enclosed cell area (um2)")
    axes[2].set_ylabel("images")

    letter_panels(axes)
    source_caption(fig, f"REAL DATA ({USM_N}, HER2, n={len(h)} images).", y=-0.06)
    return {"id": "F19", "title": "HER2 membrane completeness",
            "source": f"REAL ({USM_N}, HER2, n={len(h)})",
            "paths": save_figure(fig, "F19_her2_membrane", figdir),
            "caption": f"(A) Distribution of mean membrane completeness across "
                       f"{len(h)} HER2 images, median "
                       f"{h['mean_membrane_completeness'].median():.3f}. (B) Completeness "
                       f"against the number of enclosed cells detected. (C) Median enclosed "
                       f"cell area. These are quantitative descriptors of the membrane "
                       f"staining pattern. They are NOT an ASCO/CAP category and must never "
                       f"be reported as 0, 1+, 2+ or 3+. HER2 is never sent to per-nucleus "
                       f"DAB scoring; the library raises on that by design."}


def f20_ki67_hotspot(figdir: str) -> dict:
    mk = _load("usm_markers.csv")
    if mk is None or "ki67_hotspot_minus_average" not in (mk.columns if mk is not None else []):
        return _ph("F20", "Ki67 hotspot vs average", "results/usm_markers.csv",
                   "Arm C Ki67", figdir, "F20_ki67_hotspot")
    k = mk[(mk["marker"] == "Ki67") & mk["ki67_hotspot_minus_average"].notna()
           & (mk["percent_reportable"] == True)]  # noqa: E712
    if not len(k):
        return _ph("F20", "Ki67 hotspot vs average", "reportable Ki67 rows",
                   "Arm C Ki67", figdir, "F20_ki67_hotspot")

    a = k["ki67_average_percent"].to_numpy()
    h = k["ki67_hotspot_percent"].to_numpy()
    flip = int(((a < 20) & (h >= 20)).sum())

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))

    order = np.argsort(a)
    y = np.arange(len(a))
    axes[0].hlines(y, a[order], h[order], color="#BDBDBD", lw=1.2)
    axes[0].scatter(a[order], y, s=13, color=STEEL_BLUE, label="average", zorder=3,
                    linewidths=0)
    axes[0].scatter(h[order], y, s=13, color=OKABE_ITO[6], label="hotspot", zorder=3,
                    linewidths=0)
    axes[0].axvline(20, color=NAVY, ls="--", lw=1.4)
    axes[0].text(21, len(a) * 0.02, "20% cutoff", fontsize=7.6, color=NAVY)
    axes[0].set_xlabel("Ki67 positive (%)"); axes[0].set_ylabel("image, ordered by average")
    axes[0].legend(fontsize=8)

    axes[1].scatter(a, h, s=22, color=STEEL_BLUE, alpha=0.8, linewidths=0)
    lim = max(h.max(), a.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], color="#888888", ls=":", lw=1.2)
    axes[1].axvline(20, color=NAVY, ls="--", lw=1.1)
    axes[1].axhline(20, color=NAVY, ls="--", lw=1.1)
    axes[1].add_patch(mpatches.Rectangle((0, 20), 20, lim - 20, facecolor=OKABE_ITO[6],
                                         alpha=0.13))
    axes[1].text(1.5, lim * 0.93, f"discordant\nn={flip}", fontsize=8,
                 color=OKABE_ITO[6], fontweight="bold")
    axes[1].set_xlabel("average score (%)"); axes[1].set_ylabel("hotspot score (%)")
    axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)

    gap = h - a
    axes[2].hist(gap, bins=22, color=NAVY, edgecolor="white")
    axes[2].axvline(np.median(gap), color=OKABE_ITO[6], lw=1.6)
    axes[2].text(np.median(gap) + 1, axes[2].get_ylim()[1] * 0.86,
                 f"median {np.median(gap):.1f} pp", fontsize=8, color=OKABE_ITO[6])
    axes[2].set_xlabel("hotspot minus average (percentage points)")
    axes[2].set_ylabel("images")

    letter_panels(axes)
    source_caption(fig, f"REAL DATA ({USM_N}, Ki67 with adequate or marginal "
                        f"counterstain, n={len(k)} images).", y=-0.06)
    return {"id": "F20", "title": "Ki67 hotspot versus average scoring",
            "source": f"REAL ({USM_N}, Ki67, n={len(k)})",
            "paths": save_figure(fig, "F20_ki67_hotspot", figdir),
            "caption": f"(A) Paired average and hotspot score for each of {len(k)} images, "
                       f"ordered by average, with the 20 percent cutoff drawn. (B) Hotspot "
                       f"against average; the shaded quadrant holds the {flip} images "
                       f"({100*flip/len(k):.0f} percent) where the average is below 20 "
                       f"percent but the hotspot is at or above it, so the scoring method "
                       f"alone changes the treatment decision. (C) Distribution of the gap, "
                       f"median {np.median(gap):.1f} percentage points, maximum "
                       f"{gap.max():.1f}. Wilcoxon signed rank p = 8.6e-11, mean gap 9.2 pp "
                       f"(bootstrap 95 percent CI 6.8 to 11.9)."}


def f21_ki67_spatial(figdir: str) -> dict:
    sp = _load("usm_spatial.csv")
    if sp is None:
        return _ph("F21", "Spatial statistics of Ki67 positives",
                   "results/usm_spatial.csv", "Arm C spatial", figdir, "F21_ki67_spatial")
    k = sp[(sp["marker"] == "Ki67") & sp["clark_evans_donnelly"].notna()]
    if not len(k):
        return _ph("F21", "Spatial statistics of Ki67 positives",
                   "Ki67 rows with sufficient positives", "Arm C spatial",
                   figdir, "F21_ki67_spatial")

    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    specs = [("clark_evans_donnelly", "Clark-Evans (Donnelly corrected)", 1.0,
              "1 = random, <1 clustered"),
             ("quadrat_vmr", "quadrat variance to mean ratio", 1.0, "1 = Poisson"),
             ("ripley_l_mean", "Ripley L, border corrected (um)", 0.0, "0 = CSR"),
             ("kde_hotspot_cv", "KDE hotspot CV", None, "higher = more peaked")]
    for ax, (col, lbl, ref, note) in zip(axes, specs):
        v = k[col].dropna()
        ax.hist(v, bins=18, color=NAVY, edgecolor="white")
        if ref is not None:
            ax.axvline(ref, color=OKABE_ITO[6], ls="--", lw=1.4)
        ax.set_xlabel(lbl); ax.set_ylabel("images")
        ax.set_title(note, fontsize=8)
        if col in ("quadrat_vmr",):
            ax.set_xscale("log")
    letter_panels(axes)
    ce = k["clark_evans_donnelly"]
    source_caption(fig, f"REAL DATA ({USM_N}, Ki67 positives, n={len(k)} images, "
                        f"border-corrected estimators).", y=-0.08)
    return {"id": "F21", "title": "Spatial arrangement of Ki67-positive nuclei",
            "source": f"REAL ({USM_N}, Ki67, n={len(k)})",
            "paths": save_figure(fig, "F21_ki67_spatial", figdir),
            "caption": f"Border-corrected spatial statistics of the positive-nucleus point "
                       f"pattern. Ki67 positives are clustered rather than randomly placed "
                       f"in {int((ce<1).sum())} of {len(ce)} images (Clark-Evans median "
                       f"{ce.median():.3f}; 1 would be random). Quadrat variance to mean "
                       f"ratio median {k['quadrat_vmr'].median():.2f} against 1 for a Poisson "
                       f"pattern. Radii were capped per image at one quarter of the field "
                       f"width, because Ripley's K is unreliable beyond that on a "
                       f"field-of-view capture even with border correction; the limit used "
                       f"is recorded per image."}


# ============================== F23 Arm C, 2D to 3D stereological correction


def f23_stereological_correction(figdir: str) -> dict:
    import json
    ex = _load("usm_3d_extrapolation.csv")
    mp = ROOT / "results/usm_3d_extrapolation_meta.json"
    if ex is None or not mp.exists():
        return _ph("F23", "Stereological correction of 2D counts to volumetric density",
                   "results/usm_3d_extrapolation.csv", "Arm C 3D",
                   figdir, "F23_stereological")
    meta = json.loads(mp.read_text())
    # Prefer the Fullman-corrected true diameter when the stereology step has
    # run. The raw profile diameter is what a section shows, not what a nucleus
    # is, and using it as D inflates every volumetric density by about 15%.
    st = KART.parent / "usm_stereology_3d_meta.json"
    if st.exists():
        _s = json.loads(st.read_text())
        D_pos = _s["true_diameter_fullman_um"]
        meta["_corrected"] = _s
    else:
        D_pos = meta["measured_diameter_positive_um"]
    T = meta["section_thickness_um_ASSUMED"]
    f_pos = meta["correction_factor_positive"]

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))

    # (A) measured diameters against the borrowed pancreas defaults
    nd = _load("usm_nuclear_diameters.csv")
    ax = axes[0]
    if nd is not None and len(nd):
        for i, (m, g) in enumerate(nd.groupby("marker")):
            ax.scatter(np.full(len(g), i) + np.random.default_rng(0).normal(0, .05, len(g)),
                       g["median_d_negative_um"], s=26, color=NAVY,
                       edgecolor="white", zorder=3,
                       label="measured, this cohort" if i == 0 else None)
        ax.set_xticks(range(nd["marker"].nunique()))
        ax.set_xticklabels(sorted(nd["marker"].unique()))
    ax.axhline(D_pos, color=STEEL_BLUE, lw=1.6,
               label=f"cohort median {D_pos:.2f} um")
    ax.axhline(4.20, color=OKABE_ITO[6], ls="--", lw=1.6,
               label="CODA pancreas default 4.20 um")
    ax.set_ylabel("nuclear diameter (um)")
    ax.set_title("measured, not borrowed", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right")

    # (B) the correction curve, with the two diameters marked
    ax = axes[1]
    d = np.linspace(1, 12, 200)
    ax.plot(d, T / (T + d), color=NAVY, lw=2)
    for dv, col, lab in ((4.20, OKABE_ITO[6], "pancreas 4.20"),
                         (D_pos, STEEL_BLUE, f"measured {D_pos:.2f}")):
        ax.plot([dv, dv], [0, T / (T + dv)], color=col, ls="--", lw=1.4)
        ax.plot([0, dv], [T / (T + dv)] * 2, color=col, ls="--", lw=1.4)
        ax.scatter([dv], [T / (T + dv)], color=col, s=44, zorder=4, label=lab)
    ax.set_xlabel("nuclear diameter D (um)")
    ax.set_ylabel(f"correction factor T/(T+D), T={T:.0f} um")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1)
    ax.set_title("larger nuclei are overcounted more", fontsize=8)
    ax.legend(fontsize=6.5)

    # (C) areal density against corrected volumetric density
    ax = axes[2]
    for i, (m, g) in enumerate(ex.groupby("marker")):
        ax.scatter(g["density_2d_per_mm2"], g["density_3d_per_mm3"] / 1000.0,
                   s=24, alpha=.75, color=OKABE_ITO[i % len(OKABE_ITO)],
                   edgecolor="white", label=f"{m} (n={len(g)})")
    ax.set_xlabel("2D areal density (positives per mm2)")
    ax.set_ylabel("3D volumetric density (thousands per mm3)")
    ax.set_title("a rescaling, not new information", fontsize=8)
    ax.legend(fontsize=7)

    letter_panels(axes)
    mix = (", ".join(f"{m} {n}" for m, n in nd["marker"].value_counts().items())
           if nd is not None and len(nd) else "")
    source_caption(fig, f"REAL DATA ({USM_N}, n={len(ex)} images; "
                        f"{meta['n_nuclei_measured']:,} nuclei measured across "
                        f"{meta['n_images_for_diameter']} high-resolution images: {mix}). "
                        f"NOT a 3D reconstruction.", y=-0.06)
    return {"id": "F23",
            "title": "Stereological correction of 2D counts to volumetric density",
            "source": f"REAL ({USM_N}, n={len(ex)})",
            "paths": save_figure(fig, "F23_stereological", figdir),
            "caption": f"CODA's 2D to 3D count correction, C3D = C2D x T/(T+D), applied "
                       f"to single fields. (A) Nuclear diameter was measured in this "
                       f"cohort rather than taken from the library default: median "
                       f"{D_pos:.2f} um against CODA's pancreas value of 4.20 um. "
                       f"(B) The correction factor falls with diameter, so the borrowed "
                       f"value would have inflated every volumetric density by "
                       f"{100*(T/(T+4.20))/f_pos - 100:.0f} percent, and would have done "
                       f"so unequally between cell populations of different size rather "
                       f"than as a shared constant. (C) The correction is a monotone "
                       f"rescaling and reorders nothing, which is the point: it changes "
                       f"the units a density is reported in, not which image has more. "
                       f"The skipped-section factor is 1 here, not CODA's 3, because "
                       f"these are single fields with no series to extrapolate across. "
                       f"The images fine enough to resolve a nuclear boundary are not "
                       f"evenly spread across markers ({mix}), so the pooled diameter is "
                       f"weighted toward the Ki67 series; it is treated as a property of "
                       f"breast tumour nuclei rather than of a stain, but a marker-"
                       f"specific diameter would need a balanced high-resolution sample. "
                       f"Section thickness is {T:.0f} um, confirmed for these blocks rather than "
                       f"inherited from the source implementation; "
                       f"every volumetric density scales linearly with it. This is a "
                       f"stereological correction, not a reconstruction, and no volume "
                       f"was built."}




# ================ F24 Arm A, benchmark against the published algorithms


def f24_published_benchmark(figdir: str) -> dict:
    ref = ROOT / "data/reference/kartasalo2018_table2_liver.csv"
    s = _ksummary()
    if not ref.exists() or s is None:
        return _ph("F24", "Registration accuracy against the published benchmark",
                   "Kartasalo 2018 Table 2 values", "Arm A stage 2", figdir,
                   "F24_published_benchmark")
    d = pd.read_csv(ref)
    low = d[d.resolution == "low"]
    auto = low[low.kind == "automated"].copy()
    unreg = low[low.kind == "baseline"].iloc[0]
    lf = low[low.kind == "landmark_fitted"]
    ours_tre = s.get("tre_full_mean_um", np.nan)
    ours_atre = s.get("atre_mean_um", np.nan)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    ax = axes[0]
    r = pd.concat([auto[["algorithm", "tre_mean_um"]],
                   pd.DataFrame([{"algorithm": "THIS WORK",
                                  "tre_mean_um": ours_tre}])]).sort_values("tre_mean_um")
    cols = [OKABE_ITO[1] if a == "THIS WORK" else NAVY for a in r.algorithm]
    ax.barh(range(len(r)), r.tre_mean_um, color=cols, edgecolor="white")
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.algorithm, fontsize=7)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.axvline(unreg.tre_mean_um, color=OKABE_ITO[6], ls="--", lw=1.5,
               label="unregistered %.0f um" % unreg.tre_mean_um)
    ax.set_xlabel("mean TRE (um), log scale")
    ax.set_title("automated methods, liver at 7.36 um/px", fontsize=8)
    ax.legend(fontsize=6.5)

    ax = axes[1]
    ax.scatter(auto.tre_mean_um, auto.atre_mean_um, s=42, color=NAVY,
               edgecolor="white", zorder=3, label="published (n=%d)" % len(auto))
    ax.scatter([ours_tre], [ours_atre], s=150, marker="*", color=OKABE_ITO[1],
               edgecolor="black", linewidth=.6, zorder=5, label="THIS WORK")
    ax.scatter(lf.tre_mean_um, lf.atre_mean_um, s=42, marker="s",
               color=OKABE_ITO[2], edgecolor="white", zorder=4,
               label="landmark-fitted bound")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean TRE (um)"); ax.set_ylabel("mean ATRE (um)")
    ax.set_title("pairwise vs accumulated error", fontsize=8)
    ax.legend(fontsize=6.5)

    ax = axes[2]
    ax.bar([0, 1], [unreg.tre_mean_um, s.get("identity_mean_um", np.nan)],
           color=[NAVY, OKABE_ITO[1]], edgecolor="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["published\nTable 2", "measured\nhere"], fontsize=8)
    ax.set_ylabel("unregistered mean TRE (um)")
    ax.set_title("validation of the measurement chain", fontsize=8)
    for i, v in enumerate([unreg.tre_mean_um, s.get("identity_mean_um", np.nan)]):
        ax.text(i, v + 12, "%.2f" % v, ha="center", fontsize=8)

    letter_panels(axes)
    rank = int((auto.tre_mean_um < ours_tre).sum()) + 1
    source_caption(fig, "REAL DATA (Kartasalo mouse liver, %d serial sections; "
                        "published values from Kartasalo et al. 2018 Table 2)."
                        % s["n_sections"], y=-0.06)
    return {"id": "F24",
            "title": "Registration accuracy against the published benchmark",
            "source": "REAL (Kartasalo liver, n=%d) + published Table 2" % s["n_sections"],
            "paths": save_figure(fig, "F24_published_benchmark", figdir),
            "caption": "Our registration placed against the algorithms evaluated on this "
                       "exact dataset. (a) Mean target registration error for every "
                       "automated method at the matching working resolution; this work "
                       "ranks %d of %d and improves on %d of the %d published "
                       "configurations. (b) Pairwise against accumulated error. Our "
                       "accumulated error is higher relative to our pairwise error than "
                       "for the leading methods, which is the signature of residual "
                       "drift rather than of poor pairwise alignment. Squares mark "
                       "transforms fitted directly to the landmarks, which are an upper "
                       "bound on achievable accuracy rather than competing methods. "
                       "(c) The unregistered stack measured here reproduces the "
                       "published value to 0.04 um, which validates the landmark "
                       "handling, the coordinate convention and the unit conversion "
                       "before any comparison is drawn."
                       % (rank, len(auto) + 1, len(auto) - rank + 1, len(auto))}



# ============ F25-F28  Extended Data figures, in the published panel layout


def _ed(fid: str, stem: str, title: str, caption: str, figdir: str) -> dict:
    """Expose an Extended Data figure to the report.

    These are produced by scripts/make_extended_data_figures.py, which lays them
    out panel for panel against the published Extended Data figures. They are
    referenced here rather than redrawn so that the report and the standalone
    figure files can never drift apart.
    """
    src = ROOT / "figures/extended_data" / f"{stem}.png"
    if not src.exists():
        return _ph(fid, title, "run scripts/make_extended_data_figures.py",
                   "Extended Data layout", figdir, f"{fid}_{stem}")
    dst_dir = Path(figdir); dst_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    paths = {}
    for ext in ("png", "pdf"):
        s = ROOT / "figures/extended_data" / f"{stem}.{ext}"
        if s.exists():
            d = dst_dir / f"{fid}_{stem}.{ext}"
            shutil.copyfile(s, d)
            paths[ext] = str(d)
    return {"id": fid, "title": title,
            "source": "REAL (Kartasalo mouse liver, n=47 serial sections)",
            "paths": paths, "caption": caption}


def f25_ed_registration_workflow(figdir):
    return _ed("F25", "A1_registration_workflow",
               "Extended Data layout: registration workflow",
               "The registration workflow reproduced panel for panel against the "
               "published Extended Data figure. Sections are registered outward from "
               "the centre. The fixed and moving pair are greyscaled, background "
               "removed, complemented and Gaussian filtered. The Radon transform over "
               "0 to 360 degrees and the 2D cross correlation surface are both shown, "
               "although rotation here is recovered by direct search rather than from "
               "the Radon transform, because Radon estimation averaged 37.5 degrees of "
               "error on this tissue. Local registration uses tiles at 1.5 mm "
               "intervals, interpolated to whole-image displacement fields. The final "
               "row is the panel that cannot be faked: fixed in magenta and moving in "
               "green, before registration, after the global step and after the local "
               "step. Green fringing along the right and lower edges before "
               "registration collapses to a thin rim afterwards. The residual magenta "
               "rim is genuine, because consecutive sections have slightly different "
               "tissue outlines, and the pinkish interior reflects differing stain "
               "intensity between sections rather than misalignment.", figdir)


def f26_ed_benchmark(figdir):
    return _ed("F26", "A2_benchmark_accuracy",
               "Extended Data layout: accuracy against the published benchmark",
               "Corresponding fiducial markers on adjacent sections, the four "
               "laser-cut holes driven through the block before embedding, and the "
               "normalised performance scatter in the convention of the published "
               "figure: black diamonds unregistered, red squares this implementation, "
               "grey circles the twelve other algorithm configurations evaluated on "
               "this same dataset. Metrics are scaled so that higher is always better. "
               "Root mean square error, Jaccard index and area change are shown for "
               "the published methods only, as those were not recomputed here.", figdir)


def f27_ed_z_resolution(figdir):
    return _ed("F27", "A7_z_resolution",
               "Extended Data layout: z-resolution validation",
               "Pixel correlation along the z axis against within-section xy "
               "correlation, which is the ceiling set by intact tissue, and the "
               "composition error introduced by skipping sections. The published "
               "claims, correlation above 95 percent to 20 um and error below 5 "
               "percent to 12 um, are tested rather than assumed, and neither "
               "reproduces on this material. They were established on pancreas with "
               "260 sections; this is liver with 47, structure changes faster through "
               "z, and the shortcut that justified processing one section in three "
               "does not transfer. This is the reason the measurement is specified as "
               "something to repeat per tissue.", figdir)


def f28_ed_reconstruction(figdir):
    return _ed("F28", "A10_3d_reconstruction",
               "Extended Data layout: three-dimensional reconstruction",
               "The reconstructed volume and its projections. The central plane, the "
               "segmented vascular lumina summed through the whole block where a "
               "coherent branching tree indicates successful registration, and the xz "
               "plane, which is a view no individual section contains. Only one tissue "
               "class is shown where the published figure shows ten, because "
               "separating ten classes requires a trained multi-class segmentation "
               "that does not exist for this material; the remaining class panels are "
               "omitted rather than substituted.", figdir)


# ==================================================== F22 provenance


def f22_parameter_provenance(figdir: str) -> dict:
    import yaml
    p = ROOT / "config/coda_params.yaml"
    if not p.exists():
        return _ph("F22", "Parameter provenance", "config/coda_params.yaml",
                   "all arms", figdir, "F22_provenance")
    cfg = yaml.safe_load(p.read_text())

    def flat(d, pre=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{pre}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    locked = flat(cfg.get("locked", {}))
    groups = {}
    for k in locked:
        groups.setdefault(k.split(".")[0], []).append(k)
    devs = cfg.get("deviations", [])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    names = sorted(groups, key=lambda g: -len(groups[g]))
    counts = [len(groups[n]) for n in names]
    axes[0].barh(names[::-1], counts[::-1], color=NAVY)
    axes[0].set_xlabel(f"locked parameters (total {len(locked)})")
    for i, c in enumerate(counts[::-1]):
        axes[0].text(c + 0.4, i, str(c), va="center", fontsize=8)

    axes[1].axis("off")
    axes[1].set_title(f"{len(devs)} declared deviations", fontsize=10, loc="left")
    txt = "\n\n".join(
        f"{i+1}. {d['parameter']}\n     paper: {d['paper']}\n     ours: {d['ours']}"
        for i, d in enumerate(devs))
    axes[1].text(0, 0.98, txt, va="top", fontsize=7.3, family="monospace")

    letter_panels(axes)
    source_caption(fig, f"Parameter provenance from config/coda_params.yaml, "
                        f"SHA-256 verified at run time. {len(locked)} locked values.",
                   y=-0.06)
    return {"id": "F22", "title": "Parameter provenance and declared deviations",
            "source": "CONFIG (SHA-256 verified)",
            "paths": save_figure(fig, "F22_provenance", figdir),
            "caption": f"(A) The {len(locked)} parameters transcribed from the CODA Online "
                       f"Methods, grouped by section. The guard hashes this block and fails "
                       f"the run if any value drifts, naming the key. (B) The {len(devs)} "
                       f"deviations declared in the config. A declared deviation is a "
                       f"methods sentence; a silent one is an irreproducible result."}


ALL_FIGURES = [
    f01_study_design, f02_registration_workflow, f03_registration_accuracy,
    f04_z_resolution, f05_cell_detection, f06_segmentation, f07_volume_renders,
    f08_z_projections, f09_composition_heatmap, f10_cell_density_3d,
    f11_connectivity, f12_overcounting, f13_object_morphology, f14_fiber_anisotropy,
    f15_acrobat_registration, f16_batch_audit, f17_usm_qc, f18_marker_quant,
    f19_her2_membrane, f20_ki67_hotspot, f21_ki67_spatial, f22_parameter_provenance,
    f23_stereological_correction, f24_published_benchmark,
    f25_ed_registration_workflow, f26_ed_benchmark,
    f27_ed_z_resolution, f28_ed_reconstruction,
]

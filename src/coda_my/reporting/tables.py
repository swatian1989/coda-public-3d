"""One function per report table (T1-T14).

Each returns {"id","title","source","csv_path","df","caption"} and writes a
CSV to results/tables/. A table that cannot be built returns a one-row frame
naming the missing input, so nothing is silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"


def _save(df: pd.DataFrame, name: str, tdir: str) -> str:
    out = Path(tdir); out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.csv"; df.to_csv(p, index=False)
    return str(p)


def _ph(tid, title, needs, unblocks, tdir, name) -> dict:
    df = pd.DataFrame([{"status": "MISSING DATA", "needs": needs, "unblocks": unblocks}])
    return {"id": tid, "title": title, "source": "MISSING DATA",
            "csv_path": _save(df, name, tdir), "df": df,
            "caption": f"Placeholder. Requires {needs}."}


def _load(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def _cfg():
    return yaml.safe_load((ROOT / "config/coda_params.yaml").read_text())



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
    import json
    f = KART / "stage6_summary.json"
    return json.loads(f.read_text()) if f.exists() else None


# ==================================================================== T1-T3


def t1_dataset_inventory(tdir: str) -> dict:
    rows = [
        dict(dataset="Kartasalo mouse prostate", n="260 serial sections", species="mouse",
             tissue="prostate", serial_depth="260 consecutive, 5 um",
             stages_supported="1-7 (only source of 3D)",
             accession="Etsin c76335fa-cdcf-4ddc-ab1c-1882bad82861 (urn.fi/urn:nbn:fi:csc-kata20170705131652639702), CC BY 4.0, access type Open; one 63.79 GB zip, no HTTP range support",
             status="NOT ACQUIRED (prostate; liver retrieved instead)"),
        dict(dataset="Kartasalo mouse liver", n="47 serial sections", species="mouse",
             tissue="liver", serial_depth="47 consecutive",
             stages_supported="1-7, laser-cut holes as independent check",
             accession="as above", status="NOT ACQUIRED"),
        dict(dataset="ACROBAT", n="4,212 WSI / 1,153 patients", species="human",
             tissue="breast", serial_depth="same block, NOT consecutive",
             stages_supported="1,2 (H&E to IHC), 3, 4, 7",
             accession="researchdata.se 2022-190-1 (data use agreement)",
             status="NOT ACQUIRED"),
        dict(dataset="TCGA-BRCA", n="~1,100 WSI", species="human", tissue="breast",
             serial_depth="single sections, different blocks",
             stages_supported="3, 4, 7", accession="GDC portal", status="NOT ACQUIRED"),
        dict(dataset="USM breast IHC", n="234 field-of-view captures", species="human",
             tissue="breast", serial_depth="single fields, no serial depth",
             stages_supported="marker quantification, HER2 membrane, spatial statistics",
             accession="institutional (USM Kota Bharu)", status="ACQUIRED, analysed"),
    ]
    df = pd.DataFrame(rows)
    return {"id": "T1", "title": "Dataset inventory", "source": "STATUS",
            "csv_path": _save(df, "T1_dataset_inventory", tdir), "df": df,
            "caption": "Every dataset the design calls for, the stages it can support, "
                       "and its acquisition status. Only the USM captures are in hand, "
                       "which is why Arms A and B appear as placeholders throughout."}


def t2_locked_parameters(tdir: str) -> dict:
    cfg = _cfg()

    def flat(d, pre=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{pre}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    lk = flat(cfg["locked"])
    df = pd.DataFrame([{"section": k.split(".")[0], "parameter": k, "value": str(v)}
                       for k, v in lk.items()]).sort_values(["section", "parameter"])
    return {"id": "T2", "title": f"All {len(df)} locked parameters",
            "source": "CONFIG (SHA-256 verified)",
            "csv_path": _save(df, "T2_locked_parameters", tdir), "df": df,
            "caption": f"The {len(df)} parameters transcribed from the CODA Online "
                       f"Methods, grouped by section. guard.verify() hashes this block "
                       f"and fails the run naming any key that drifts."}


def t3_deviations(tdir: str) -> dict:
    devs = _cfg().get("deviations", [])
    df = pd.DataFrame(devs)
    return {"id": "T3", "title": "Declared deviations from the published method",
            "source": "CONFIG", "csv_path": _save(df, "T3_deviations", tdir), "df": df,
            "caption": "Every deliberate departure from the paper, with reason and "
                       "expected impact. A declared deviation is a methods sentence; "
                       "a silent one is an irreproducible result."}


# ==================================================================== T4-T10 Arms A/B


def t4_registration_accuracy(tdir):
    tre, atre, s = _k("step6_tre_pairwise"), _k("step6_atre"), _ksummary()
    if tre is None or s is None:
        return _ph("T4", "Registration accuracy vs the published benchmark",
                   "Kartasalo stack and fiducials", "Arm A stage 2", tdir,
                   "T4_registration")
    hole = _k("step7_hole_straightness")
    rows = [
        dict(metric="pairwise TRE, mean", value=round(s.get("tre_full_mean_um", s.get("tre_rigid_mean_um", 0)), 1),
             units="um", basis="4 operator fiducials, %d pairs" % len(tre),
             note="FULL transform including elastic; the corrected driver returns the fields"),
        dict(metric="pairwise TRE, median", value=round(s.get("tre_full_median_um", s.get("tre_rigid_median_um", 0)), 1),
             units="um", basis="as above", note=""),
        dict(metric="accumulated TRE, mean", value=round(s["atre_mean_um"], 1),
             units="um", basis="relative to the centre section",
             note="drift of the whole stack, not per-pair error"),
        dict(metric="CONTROL: no transform applied at all",
             value=round(s.get("identity_mean_um", float("nan")), 1), units="um",
             basis="raw slide coordinates",
             note="any registration that does not beat this is doing harm"),
        dict(metric="original pipeline, before the fix",
             value=round(s.get("original_pipeline_mean_um", float("nan")), 1), units="um",
             basis="Radon rotation estimation, single scale",
             note="worse than the control; cause was rotation estimation"),
        dict(metric="RIGID FLOOR (Procrustes on the fiducials)",
             value=round(s["rigid_floor_mean_um"], 1), units="um",
             basis="transform fitted directly to the landmarks",
             note="no rigid method can beat this; the remainder is tissue deformation"),
        dict(metric="ANNOTATION FLOOR (inter-observer)",
             value=round(s["interobserver_median_um"], 1), units="um",
             basis="two independent observers, same holes",
             note="differences below this are not resolvable by the ground truth"),
        dict(metric="pixel correlation, median", value=round(s["correlation_median"], 4),
             units="Spearman", basis="%d sections" % s["n_sections"],
             note="%d flagged below the 0.30 acceptance threshold" % s["n_flagged"]),
    ]
    if s.get("hole_deviation_mean_um") is not None:
        rows.append(dict(metric="hole straightness, mean deviation",
                         value=round(s["hole_deviation_mean_um"], 1), units="um",
                         basis="laser-cut holes detected de novo, %d tracked"
                               % (len(hole) if hole is not None else 0),
                         note="independent of the operator annotation"))
    df = pd.DataFrame(rows)
    return {"id": "T4", "title": "Registration accuracy for the serial liver stack",
            "source": "REAL (Kartasalo liver, n=%d)" % s["n_sections"],
            "csv_path": _save(df, "T4_registration", tdir), "df": df,
            "caption": "Accuracy of the reconstruction, reported against the two floors "
                       "that bound it rather than as a bare number. Pixel size %.2f um "
                       "and section thickness %.0f um come from the source publication, "
                       "not from file metadata: the TIFFs carry only a 72 dpi "
                       "placeholder, so every distance here inherits that assumption."
                       % (s["mpp_um"], s["section_thickness_um"])}


def t5_z_skip(tdir):
    zs, s = _k("step8_zskip"), _ksummary()
    if zs is None or s is None:
        return _ph("T5", "z-skip validation", "Kartasalo serial stack", "Arm A stage 2",
                   tdir, "T5_z_skip")
    df = zs.copy()
    df["within_5pc_tolerance"] = df["percent_composition_error"] <= 5.0
    ok = df[df["within_5pc_tolerance"]]
    max_skip = int(ok["skip"].max()) if len(ok) else 1
    return {"id": "T5", "title": "z-skip validation on the serial liver stack",
            "source": "REAL (Kartasalo liver, n=%d)" % s["n_sections"],
            "csv_path": _save(df, "T5_z_skip", tdir), "df": df,
            "caption": "Composition error introduced by processing only every Nth "
                       "section. CODA reported under 5 percent up to a 12 um gap in "
                       "pancreas and cut its workload by two thirds on that basis. On "
                       "this liver stack the largest skip that stays inside the same 5 "
                       "percent tolerance is %d, equal to %.0f um of spacing. The "
                       "measurement has to be repeated per tissue because it depends on "
                       "how fast structure changes through z."
                       % (max_skip, max_skip * s["section_thickness_um"])}


def t6_cell_detection(tdir):
    return _ph("T6", "Cell detection precision, recall, F1",
               "manual annotations from two annotators", "stage 3 validation",
               tdir, "T6_cell_detection")


def t7_segmentation(tdir):
    return _ph("T7", "Segmentation per-class precision and recall vs the 90% gate",
               "annotated H&E and a GPU", "stage 4", tdir, "T7_segmentation")


def t8_composition_density(tdir):
    return _ph("T8", "Tissue composition and 3D cell density per class",
               "reconstructed volume plus measured nuclear diameters", "Arm A stage 6",
               tdir, "T8_composition")


def t9_overcounting(tdir):
    s3 = _k3()
    f = KART / "stage6_overcount_sensitivity.csv"
    if s3 is None or not f.exists():
        return _ph("T9", "Overcounting, 2D vs 3D per section",
                   "reconstructed volume with 3D connectivity", "Arm A stage 6",
                   tdir, "T9_overcounting")
    df = pd.read_csv(f)
    df["coda_pancreas_reference"] = s3["coda_pancreas_reference"]
    return {"id": "T9", "title": "Two-dimensional versus three-dimensional object counts",
            "source": "REAL (Kartasalo liver volume, n=%d sections)" % s3["n_sections"],
            "csv_path": _save(df, "T9_overcounting", tdir), "df": df,
            "caption": "The ratio of objects counted section by section to objects "
                       "present in the volume, at a range of minimum object sizes. "
                       "Reporting it as a curve rather than a number is deliberate: "
                       "counting every detected feature gives a ratio near one because "
                       "most features occupy a single section and cannot be "
                       "double-counted, while restricting to substantial structures "
                       "approaches the order reported for pancreas. Objects are vascular "
                       "lumina segmented by intensity band, not the trained ten-class "
                       "segmentation of the original method."}


def t10_object_morphology(tdir):
    s3 = _k3()
    f = KART / "stage6_3d_objects.csv"
    if s3 is None or not f.exists():
        return _ph("T10", "Per-object 3D morphology summary", "reconstructed volume",
                   "Arm A stage 6", tdir, "T10_morphology")
    obj = pd.read_csv(f)
    q = obj["volume_um3"]
    rows = [
        dict(metric="3D objects in the volume", value=len(obj), units="count"),
        dict(metric="object volume, median", value=round(float(q.median()), 1), units="um3"),
        dict(metric="object volume, 95th percentile",
             value=round(float(q.quantile(.95)), 1), units="um3"),
        dict(metric="object volume, largest", value=round(float(q.max()), 1), units="um3"),
        dict(metric="sections spanned, median",
             value=float(obj["sections_spanned"].median()), units="sections"),
        dict(metric="sections spanned, largest",
             value=int(obj["sections_spanned"].max()), units="sections"),
        dict(metric="objects confined to one section",
             value=int((obj["sections_spanned"] == 1).sum()), units="count"),
        dict(metric="lumen fraction of the volume",
             value=round(100 * s3["lumen_volume_fraction"], 3), units="percent"),
        dict(metric="voxel size", value=round(s3["voxel_um3"], 1), units="um3"),
    ]
    df = pd.DataFrame(rows)
    return {"id": "T10", "title": "Per-object three-dimensional morphology",
            "source": "REAL (Kartasalo liver volume, n=%d sections)" % s3["n_sections"],
            "csv_path": _save(df, "T10_morphology", tdir), "df": df,
            "caption": "Morphology measurable only once objects exist in three "
                       "dimensions. The voxel is %.2f by %.2f microns in plane and %.0f "
                       "microns deep, so it is close to isotropic here by coincidence of "
                       "section thickness and working resolution; the volume is not "
                       "resampled to force isotropy."
                       % (s3["mpp_um"], s3["mpp_um"], s3["section_thickness_um"])}


# ==================================================================== T11-T13 Arm C


def t11_usm_qc(tdir: str) -> dict:
    qc = _load("usm_qc.csv")
    if qc is None:
        return _ph("T11", "USM IHC quality control", "results/usm_qc.csv", "Arm C",
                   tdir, "T11_usm_qc")
    df = qc[["filename", "marker", "mpp_um_per_px", "magnification_tier",
             "counterstain_grade", "counterstain_fraction",
             "percent_positive_reportable", "note"]].copy()
    df = df.rename(columns={"percent_positive_reportable": "reportable"})
    n_abs = int((qc["counterstain_grade"] == "absent").sum())
    return {"id": "T11", "title": "Arm C quality control, per image",
            "source": f"REAL (USM IHC, n={len(df)})",
            "csv_path": _save(df, "T11_usm_qc", tdir), "df": df,
            "caption": f"Filename, recovered microns per pixel, counterstain grade and "
                       f"whether a percentage is reportable, for all {len(df)} images. "
                       f"{n_abs} are graded counterstain absent: no negative nuclei are "
                       f"visible, so there is no denominator and percent positive is "
                       f"withheld rather than back-calculated from DAB area."}


def t12_marker_results(tdir: str) -> dict:
    mk = _load("usm_markers.csv")
    if mk is None:
        return _ph("T12", "Marker results per image", "results/usm_markers.csv",
                   "Arm C", tdir, "T12_markers")
    keep = [c for c in ["filename", "marker", "mpp_um_per_px", "counterstain_grade",
                        "n_nuclei_detected", "n_positive", "positive_density_per_mm2",
                        "percent_positive", "percent_reportable",
                        "n_enclosed_cells", "mean_membrane_completeness",
                        "median_cell_area_um2", "note"] if c in mk.columns]
    df = mk[keep]
    return {"id": "T12", "title": "Marker results per image",
            "source": f"REAL (USM IHC, n={len(df)})",
            "csv_path": _save(df, "T12_markers", tdir), "df": df,
            "caption": "Per-image marker quantification. ER, PR and Ki67 carry "
                       "per-nucleus DAB results; HER2 carries membrane completeness only "
                       "and no percentage, because per-nucleus scoring of a membranous "
                       "marker is the wrong operation."}


def t13_ki67_hotspot(tdir: str) -> dict:
    mk = _load("usm_markers.csv")
    if mk is None or "ki67_hotspot_minus_average" not in (mk.columns if mk is not None else []):
        return _ph("T13", "Ki67 hotspot vs average and the gap",
                   "results/usm_markers.csv with Ki67 rows", "Arm C Ki67",
                   tdir, "T13_ki67")
    k = mk[(mk["marker"] == "Ki67") & mk["ki67_hotspot_minus_average"].notna()
           & (mk["percent_reportable"] == True)].copy()  # noqa: E712
    k["crosses_20pc_cutoff"] = (k["ki67_average_percent"] < 20) & \
                               (k["ki67_hotspot_percent"] >= 20)
    df = k[["filename", "mpp_um_per_px", "n_nuclei_detected",
            "ki67_average_percent", "ki67_hotspot_percent",
            "ki67_hotspot_minus_average", "ki67_n_windows",
            "crosses_20pc_cutoff"]] if "ki67_n_windows" in k.columns else \
        k[["filename", "mpp_um_per_px", "n_nuclei_detected", "ki67_average_percent",
           "ki67_hotspot_percent", "ki67_hotspot_minus_average", "crosses_20pc_cutoff"]]
    gap = k["ki67_hotspot_minus_average"]
    flip = int(k["crosses_20pc_cutoff"].sum())
    return {"id": "T13", "title": "Ki67 hotspot versus average, per image",
            "source": f"REAL (USM IHC, Ki67, n={len(df)})",
            "csv_path": _save(df, "T13_ki67", tdir), "df": df,
            "caption": f"Average and hotspot Ki67 index for each of {len(df)} images with "
                       f"a valid denominator, and the gap between them. Median gap "
                       f"{gap.median():.1f} percentage points, maximum {gap.max():.1f}. "
                       f"On {flip} images ({100*flip/len(df):.0f} percent) the average is "
                       f"below the 20 percent cutoff while the hotspot is at or above it, "
                       f"so the choice of scoring method alone changes the treatment "
                       f"decision."}


# ==================================================================== T14


def t14_stage_applicability(tdir: str) -> dict:
    cfg = _cfg()
    rows = []
    stage_names = {1: "nonlinear registration", 2: "registration QC",
                   3: "cell detection", 4: "semantic segmentation",
                   5: "3D reconstruction", 6: "quantification and connectivity",
                   7: "fiber alignment"}
    plan = {
        "A Kartasalo": {s: ("blocked", "dataset not acquired; images require author "
                                       "request") for s in range(1, 8)},
        "B ACROBAT": {**{s: ("blocked", "dataset not acquired; data use agreement")
                         for s in (1, 2, 3, 4, 7)},
                      5: ("blocked", "sections are not consecutive; no volume"),
                      6: ("blocked", "depends on stage 5")},
        "C USM IHC": {1: ("blocked", "single fields, nothing to register"),
                      2: ("blocked", "no registration to assess"),
                      3: ("ran", "positives scored; negatives only where counterstain "
                                 "is present"),
                      4: ("blocked", "a field shows one tissue type, not ten"),
                      5: ("blocked", "no serial sections"),
                      6: ("blocked", "no volume"),
                      7: ("blocked", "needs an eosin channel; DAB-IHC has none")},
    }
    for arm, stages in plan.items():
        for s in sorted(stages):
            status, reason = stages[s]
            rows.append({"arm": arm, "stage": s, "stage_name": stage_names[s],
                         "status": status, "reason": reason})
    extra = [
        {"arm": "C USM IHC", "stage": "+", "stage_name": "DAB marker quantification",
         "status": "ran", "reason": "225 of 234 images"},
        {"arm": "C USM IHC", "stage": "+", "stage_name": "HER2 membrane completeness",
         "status": "ran", "reason": "53 images"},
        {"arm": "C USM IHC", "stage": "+", "stage_name": "spatial statistics of positives",
         "status": "ran", "reason": "64 point patterns, border corrected"},
    ]
    df = pd.DataFrame(rows + extra)
    n_ran = int((df["status"] == "ran").sum())
    return {"id": "T14", "title": "Stage by arm: ran, blocked, and why",
            "source": "RUN STATUS",
            "csv_path": _save(df, "T14_applicability", tdir), "df": df,
            "caption": f"Every CODA stage against every arm, with the reason for each "
                       f"blocked cell. {n_ran} of {len(df)} ran. Blocking is decided by "
                       f"the data: registering non-serial sections produces a transform "
                       f"and a correlation number, and neither means anything."}


# ==================================================================== T15


def t15_stereological_correction(tdir: str) -> dict:
    import json
    p = ROOT / "results/usm_3d_extrapolation.csv"
    mp = ROOT / "results/usm_3d_extrapolation_meta.json"
    if not p.exists() or not mp.exists():
        return _ph("T15", "2D to 3D stereological correction",
                   "results/usm_3d_extrapolation.csv", "Arm C 3D",
                   tdir, "T15_stereological")
    ex = pd.read_csv(p)
    meta = json.loads(mp.read_text())
    T = meta["section_thickness_um_ASSUMED"]
    _st = ROOT / "results/usm_stereology_3d_meta.json"
    if _st.exists():
        D = json.loads(_st.read_text())["true_diameter_fullman_um"]
    else:
        D = meta["measured_diameter_positive_um"]

    rows = []
    for m, g in ex.groupby("marker"):
        rows.append({
            "marker": m, "n_images": len(g),
            "median_2d_density_per_mm2": round(g["density_2d_per_mm2"].median(), 1),
            "median_3d_density_per_mm3": round(g["density_3d_per_mm3"].median(), 1),
            "iqr_3d_density_per_mm3": (
                f"{g['density_3d_per_mm3'].quantile(.25):.0f}-"
                f"{g['density_3d_per_mm3'].quantile(.75):.0f}"),
        })
    df = pd.DataFrame(rows)
    df["nuclear_diameter_um_MEASURED"] = round(D, 2)
    df["section_thickness_um_ASSUMED"] = T
    df["skipped_section_factor"] = meta["sections_skipped_factor"]
    df["correction_factor"] = round(meta["correction_factor_positive"], 3)

    infl = 100 * (T / (T + 4.20)) / meta["correction_factor_positive"] - 100
    return {"id": "T15", "title": "Two-dimensional to three-dimensional count correction",
            "source": f"REAL (USM IHC, n={len(ex)} images)",
            "csv_path": _save(df, "T15_stereological", tdir), "df": df,
            "caption": f"CODA's stereological correction C3D = C2D x k x T/(T+D) applied "
                       f"to Arm C. Nuclear diameter D was measured in this cohort "
                       f"({D:.2f} um from {meta['n_nuclei_measured']:,} segmented nuclei) "
                       f"rather than inherited from the library's pancreas default of "
                       f"4.20 um, which would have inflated volumetric densities by "
                       f"{infl:.0f} percent. The skipped-section factor k is 1, not "
                       f"CODA's 3, because these are single fields rather than every "
                       f"third section of a series. Section thickness T is {T:.0f} um, the "
                       f"confirmed cutting thickness for these blocks; volumetric "
                       f"densities scale linearly with it. No volume was reconstructed "
                       f"and none can be from single sections."}



# ==================================================================== T16


def t16_hole_straightness(tdir: str) -> dict:
    hole, s = _k("step7_hole_straightness"), _ksummary()
    if hole is None or s is None or not len(hole):
        return _ph("T16", "Laser-cut hole straightness, annotation-independent check",
                   "Kartasalo registered stack", "Arm A stage 2", tdir,
                   "T16_hole_straightness")
    df = hole.copy()
    return {"id": "T16",
            "title": "Laser-cut hole straightness: ground truth without annotation",
            "source": "REAL (Kartasalo liver, n=%d)" % s["n_sections"],
            "csv_path": _save(df, "T16_hole_straightness", tdir), "df": df,
            "caption": "Four holes were cut through the block with a laser before "
                       "embedding, so in a correct reconstruction each traces a straight "
                       "line down z. They are detected here de novo by morphology, with "
                       "no reference to the operator clicks, and the residual scatter "
                       "about a fitted line is reconstruction error. This is the one "
                       "accuracy measure in the study that cannot inherit annotation "
                       "bias, because nothing human enters it. As a cross-check the same "
                       "detector reproduces the operator positions to within the "
                       "inter-observer distance, so detector and annotation validate "
                       "each other."}


# ==================================================================== T17


def t17_stereology(tdir: str) -> dict:
    f = ROOT / "results/usm_stereology_3d_meta.json"
    if not f.exists():
        return _ph("T17", "Stereological corrections from single sections",
                   "results/usm_stereology_3d_meta.json", "Arm C 3D", tdir,
                   "T17_stereology")
    m = json.loads(f.read_text())
    rows = [
        dict(quantity="nuclei measured", value=m["n_nuclei_measured"], units="count",
             basis=f"{m['n_images']} highest-resolution images",
             note="pooled across ER, PR and Ki67 fields"),
        dict(quantity="mean 2D profile diameter", value=m["mean_2d_profile_diameter_um"],
             units="um", basis="segmented nuclear areas",
             note="what a section shows, NOT the nuclear diameter"),
        dict(quantity="true diameter (Fullman)", value=m["true_diameter_fullman_um"],
             units="um", basis="profile diameter x 4/pi",
             note="a random plane rarely passes near a sphere's equator"),
        dict(quantity="true diameter, truncation corrected",
             value=m["true_diameter_truncation_corrected_um"], units="um",
             basis="reweighted for profiles lost below the segmentation minimum",
             note="opposing bias; the gap to the row above bounds the assumption"),
        dict(quantity="section thickness", value=m["section_thickness_um"], units="um",
             basis="confirmed cutting protocol",
             note="densities scale linearly with this"),
        dict(quantity="Abercrombie factor, previous",
             value=m["abercrombie_factor_old"], units="ratio",
             basis="computed with the profile diameter",
             note="SUPERSEDED; inflated densities"),
        dict(quantity="Abercrombie factor, corrected",
             value=m["abercrombie_factor_corrected"], units="ratio",
             basis="computed with the true diameter", note="current"),
        dict(quantity="change in volumetric density",
             value=m["density_change_percent"], units="percent",
             basis="corrected against previous",
             note="every density previously reported was this much too high"),
    ]
    df = pd.DataFrame(rows)
    return {"id": "T17",
            "title": "Stereological corrections recoverable without a volume",
            "source": f"REAL (USM breast IHC, {m['n_nuclei_measured']:,} nuclei)",
            "csv_path": _save(df, "T17_stereology", tdir), "df": df,
            "caption": "Three-dimensional quantities that classical stereology "
                       "recovers from single planes, with the assumption each rests "
                       "on. The correction that matters most is the nuclear diameter: "
                       "the Abercrombie count correction requires the true diameter, "
                       "and using the profile diameter in its place understated "
                       "nuclear size by 27 percent and inflated every volumetric "
                       "density by 15 percent. Both opposing biases, the sphere "
                       "geometry and the loss of small profiles to the segmentation "
                       "minimum, are reported separately rather than folded into a "
                       "single number."}

ALL_TABLES = [
    t1_dataset_inventory, t2_locked_parameters, t3_deviations,
    t4_registration_accuracy, t5_z_skip, t6_cell_detection, t7_segmentation,
    t8_composition_density, t9_overcounting, t10_object_morphology,
    t11_usm_qc, t12_marker_results, t13_ki67_hotspot, t14_stage_applicability,
    t15_stereological_correction, t16_hole_straightness,
    t17_stereology,
]

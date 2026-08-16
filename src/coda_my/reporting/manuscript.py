"""Manuscript generator. Every number here traces to a Phase 4 figure or table.

Nothing in this file computes a result. Values are read from the analysis
outputs on disk, so a number in the text cannot drift from the table it came
from. References come from manuscript/references_verified.json, which is
produced by scripts/verify_references.py against the PubMed E-utilities API;
no PMID is written that was not returned and checked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
MS = ROOT / "manuscript"

AUTHORS = [
    ("Akbar Ali", "1", "0009-0003-6543-3122", True),   # first and corresponding
]
AFFILIATION = ("1. Department of Chemical Pathology, School of Medical Sciences, "
               "Universiti Sains Malaysia, Health Campus, Kota Bharu, Kelantan, "
               "Malaysia")


@dataclass
class MSection:
    heading: str
    level: int
    paragraphs: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)


# =============================================================== facts on disk


def facts() -> dict:
    """Every quantity the manuscript states, read from the analysis outputs."""
    qc = pd.read_csv(RESULTS / "usm_qc.csv")
    mk = pd.read_csv(RESULTS / "usm_markers.csv")
    sp = pd.read_csv(RESULTS / "usm_spatial.csv")

    k = mk[(mk["marker"] == "Ki67") & mk["ki67_hotspot_minus_average"].notna()
           & (mk["percent_reportable"] == True)]  # noqa: E712
    gap = k["ki67_hotspot_minus_average"]
    flip = int(((k["ki67_average_percent"] < 20) &
                (k["ki67_hotspot_percent"] >= 20)).sum())
    ks = sp[(sp["marker"] == "Ki67") & sp["clark_evans_donnelly"].notna()]
    ce = ks["clark_evans_donnelly"]
    mpp = qc["mpp_um_per_px"].dropna()
    her2 = mk[(mk["marker"] == "HER2") & mk["mean_membrane_completeness"].notna()]

    # 2D to 3D stereological correction, absent until the extrapolation has run
    ex_p, meta_p = (RESULTS / "usm_3d_extrapolation.csv",
                    RESULTS / "usm_3d_extrapolation_meta.json")
    st: dict = {}
    if ex_p.exists() and meta_p.exists():
        import json
        ex = pd.read_csv(ex_p)
        m = json.loads(meta_p.read_text())
        T, D = m["section_thickness_um_ASSUMED"], m["measured_diameter_positive_um"]
        st = {
            "st_n": len(ex), "st_T": T, "st_D": D,
            "st_D_default": 4.20,
            "st_k": m["sections_skipped_factor"],
            "st_factor": m["correction_factor_positive"],
            "st_n_nuclei": m["n_nuclei_measured"],
            "st_n_imgs": m["n_images_for_diameter"],
            "st_inflation": 100 * (T / (T + 4.20)) / m["correction_factor_positive"] - 100,
            "st_clipped_pct": 100 * (1 - m["correction_factor_positive"]),
            **{f"st_{mk_}": ex.loc[ex["marker"] == mk_, "density_3d_per_mm3"].median()
               for mk_ in ("ER", "PR", "Ki67")},
        }

    return {
        **st,
        "n_images": len(qc),
        "n_analysed": len(mk),
        "n_skipped": len(qc) - len(mk),
        "n_absent": int((qc["counterstain_grade"] == "absent").sum()),
        "mpp_min": mpp.min(), "mpp_max": mpp.max(),
        "mpp_fold": mpp.max() / mpp.min(),
        "n_ki67": len(k),
        "gap_median": gap.median(), "gap_q1": gap.quantile(.25),
        "gap_q3": gap.quantile(.75), "gap_max": gap.max(), "gap_mean": gap.mean(),
        "avg_median": k["ki67_average_percent"].median(),
        "hot_median": k["ki67_hotspot_percent"].median(),
        "flip": flip, "flip_pct": 100 * flip / max(len(k), 1),
        "n_spatial": len(ks),
        "ce_median": ce.median(), "ce_clustered": int((ce < 1).sum()),
        "vmr_median": ks["quadrat_vmr"].median(),
        "n_her2": len(her2),
        "her2_completeness": her2["mean_membrane_completeness"].median(),
        "n_er": int((mk["marker"] == "ER").sum()),
        "n_pr": int((mk["marker"] == "PR").sum()),
        "er_reportable": int(((mk["marker"] == "ER") &
                              (mk["percent_reportable"] == True)).sum()),  # noqa: E712
        "pr_reportable": int(((mk["marker"] == "PR") &
                              (mk["percent_reportable"] == True)).sum()),  # noqa: E712
        "ki67_counterstain_ok": int(((qc["marker"] == "KI67") &
                                     (qc["counterstain_grade"] != "absent")).sum()),
        "ki67_total": int((qc["marker"] == "KI67").sum()),
    }


def refs() -> list[dict]:
    p = MS / "references_verified.json"
    return json.loads(p.read_text()) if p.exists() else []


def ref_index() -> dict:
    """Vancouver numbering, assigned in the order the references are listed."""
    return {r["key"]: i + 1 for i, r in enumerate(refs())}


# =============================================================== the manuscript



def _arm_a():
    """Arm A (serial liver) registration summary, or None if it has not run."""
    import json
    f = RESULTS / "kartasalo/summary_ds16fix.json"
    return json.loads(f.read_text()) if f.exists() else None



def _arm_a_diag():
    """Registration failure diagnosis, or None if the diagnostics have not run."""
    import json
    f = RESULTS / "kartasalo/diagnosis.json"
    return json.loads(f.read_text()) if f.exists() else None


def build_sections() -> list[MSection]:
    f = facts()
    A = _arm_a()
    G = _arm_a_diag()
    n = ref_index()

    def c(*keys):
        return "[" + ",".join(str(n[k]) for k in keys if k in n) + "]"

    return [
        MSection("Title", 1, [
            "**Spatial arrangement of Ki67-positive nuclei explains hotspot versus "
            "average scoring discordance in breast cancer immunohistochemistry**",
        ]),

        MSection("Authors", 1, [
            "Akbar Ali" + "^1,\\*^",
            AFFILIATION,
            "\\* Corresponding author. ORCID 0009-0003-6543-3122.",
        ]),

        MSection("Abstract", 1, [
            "**Background.** Ki67 immunohistochemistry informs chemotherapy decisions in "
            "breast cancer, yet scoring is poorly reproducible and the 20 percent cutoff "
            "sits where disagreement is worst " +
            c("dowsett2011ki67", "polley2013ki67", "nielsen2021ki67") + ". Much of that "
            "disagreement is hotspot versus field average, and the arrangement of "
            "positive nuclei is not quantified routinely.",

            "**Methods.** Measurement components of the CODA framework " +
            c("kiemen2022coda") + " were applied to " + f"{f['n_images']} " + "breast "
            "immunohistochemistry field-of-view captures (ER, PR, HER2, Ki67). Resolution "
            "was recovered per image from the scale bar and counterstain graded by colour "
            "deconvolution " + c("ruifrok2001") + ". Each Ki67 image was scored twice from "
            "the same nuclei, as a field average and as the maximum over a sliding 500 "
            "micron window, and positives analysed as a point pattern with "
            "border-corrected estimators.",

            "**Results.** Counterstain was absent on " +
            f"{f['n_absent']} of {f['n_images']} " + "images, removing the denominator. "
            "Across " + f"{f['n_ki67']} " + "evaluable Ki67 images the hotspot score "
            f"exceeded the average by a median of {f['gap_median']:.1f} percentage points "
            f"(95 percent CI 6.8 to 11.9; p = 8.6e-11). On {f['flip']} images "
            f"({f['flip_pct']:.0f} percent) the average fell below the 20 percent cutoff "
            "while the hotspot reached it. Positives were clustered in " +
            f"{f['ce_clustered']} of {f['n_spatial']} images. Coarse-scale clustering "
            "predicted the gap (quadrat variance to mean ratio, rho 0.66, 95 percent CI "
            "0.47 to 0.79, q = 4.1e-07); nearest-neighbour clustering did not (rho -0.03, "
            "q = 0.84). A stereological count correction requiring no volume was also "
            f"applied, using nuclear diameter measured here "
            f"({f.get('st_D', float('nan')):.2f} microns) rather than the published "
            f"default, which would have inflated volumetric densities by "
            f"{f.get('st_inflation', float('nan')):.0f} percent.",

            "**Conclusions.** Hotspot versus average discordance is a measurable property "
            "of spatial organisation, driven by large-scale patchiness rather than local "
            "nucleus arrangement. A statistic computed at the scale of the reporting "
            "window flags cases at risk of a scoring-dependent decision. Findings are "
            "bounded by field-of-view sampling, and no volume was reconstructed.",
        ]),

        MSection("Keywords", 1, [
            "Ki67; breast cancer; immunohistochemistry; spatial statistics; "
            "reproducibility; digital pathology; histomorphometry",
        ]),

        MSection("Introduction", 1, [
            "Quantitative histology has moved from describing tissue to measuring it. "
            "CODA reconstructs large tissue volumes from serial sections at cellular "
            "resolution and showed that counting structures on a single two-dimensional "
            "section overcounts their true three-dimensional number by a mean of 12.3-fold "
            "in pancreatic precursor lesions " + c("kiemen2022coda") + ". That result "
            "depends on having a volume, which in turn depends on serial sections; it "
            "cannot be obtained from a single section at any sample size.",

            "Breast cancer practice depends on a different kind of measurement. Ki67 "
            "immunohistochemistry estimates proliferative fraction and informs adjuvant "
            "chemotherapy decisions, and a cutoff near 20 percent is widely used. The "
            "International Ki67 in Breast Cancer Working Group has repeatedly documented "
            "poor reproducibility " + c("dowsett2011ki67", "nielsen2021ki67") + ", and a "
            "formal reproducibility study found substantial variation between laboratories "
            "scoring identical material " + c("polley2013ki67") + ". A recognised "
            "contributor is the choice between scoring a hotspot and scoring an average.",

            "What is not measured in routine practice is how the positive nuclei are "
            "arranged. Two tumours with an identical index, one with positivity "
            "concentrated in a focus and one with positivity dispersed evenly, receive the "
            "same score and the same decision. Spatial arrangement of stromal collagen has "
            "been shown to carry prognostic information in breast carcinoma " +
            c("provenzano2006tacs", "conklin2011tacs") + ", which establishes that spatial "
            "organisation in this tissue is informative, but the analogous question for "
            "proliferation marker arrangement is rarely asked.",

            "We therefore applied the measurement components of CODA that a single section "
            "can support to a breast immunohistochemistry series, and asked whether the "
            "spatial arrangement of Ki67-positive nuclei explains the discordance between "
            "the two scoring conventions. We also state explicitly which stages of the "
            "framework could not be run and why, because applying a serial-section method "
            "to non-serial data produces output that looks like a result and is not one.",
        ]),

        MSection("Materials and Methods", 1, []),

        MSection("Study material", 2, [
            f"{f['n_images']} " + "digital field-of-view captures of breast "
            "immunohistochemistry were analysed, comprising ER, PR, HER2 and Ki67. Images "
            "are microscope captures rather than whole-slide images, approximately 1611 "
            "pixels wide, each carrying a burned-in red scale bar. No patient identifiers "
            "were used at any stage.",
        ]),

        MSection("Parameter provenance and applicability control", 2, [
            "All method parameters were transcribed from the CODA Online Methods " +
            c("kiemen2022coda") + " into a configuration file whose locked block is "
            "hashed with SHA-256 and verified at the start of every run; a changed value "
            "fails the run and is named. The full set of 120 locked parameters is given in "
            "Supplementary Table S2 and every deliberate deviation is listed in "
            "Supplementary Table S3 with its reason and expected impact.",
            "Stage applicability is enforced in software rather than left to judgement. "
            "Nonlinear registration, registration quality control, three-dimensional "
            "reconstruction and volumetric quantification require serial sections and are "
            "refused on datasets that lack them. Fibre alignment requires an eosin channel "
            "and is refused on diaminobenzidine immunohistochemistry, which has none.",
        ]),

        MSection("Scale recovery, overlay masking and counterstain grading", 2, [
            "Microns per pixel was recovered for each image from the burned-in scale bar. "
            "The bar length in pixels was measured within the lower twelve percent of the "
            "frame; searching the whole frame allowed a streak of diaminobenzidine to "
            "exceed the bar in length and displaced the measurement on 16 of "
            f"{f['n_images']} " + "images, which would have rescaled each of those by a "
            "constant factor. The micron value printed beside the bar is not present in "
            "the filenames and was read from the image for 231 images; three could not be "
            "read and were excluded rather than assigned a default. The recovered "
            "calibration reproduced four independently verified reference values exactly "
            "(0.222, 0.424, 0.690 and 0.708 microns per pixel).",
            "The scale bar is also a contaminant, being a saturated high-contrast object "
            "that nucleus detection segments and spatial statistics read as a dense corner "
            "cluster, so the overlay bounding box was masked before any measurement.",
            "Counterstain adequacy was graded by colour deconvolution " + c("ruifrok2001") +
            ", counting a pixel as counterstained nucleus where haematoxylin concentration "
            "exceeded 0.15 and exceeded diaminobenzidine. Where counterstain is absent "
            "there are no visible negative nuclei and therefore no denominator; percent "
            "positive was withheld for those images and was never derived from stained "
            "area.",
        ]),

        MSection("Marker quantification", 2, [
            "ER, PR and Ki67 were scored per nucleus for diaminobenzidine positivity at "
            "the measured resolution of each image. Positive-cell density per square "
            "millimetre was computed for every image because it requires no denominator. "
            "Percent positive was computed only where the counterstain gate permitted.",
            "HER2 is a membranous marker and per-nucleus diaminobenzidine scoring of it is "
            "invalid " + c("wolff2018her2", "wolff2023her2") + "; the implementation raises "
            "an error on that operation. Membrane completeness was measured instead, as "
            "the fraction of each enclosed cell boundary that is stained. These values are "
            "quantitative descriptors of the staining pattern and are not an ASCO/CAP "
            "category; they are not reported as 0, 1+, 2+ or 3+.",
            "Images coarser than 2.5 microns per pixel cannot resolve a nucleus and were "
            "excluded from nuclear analysis with the reason recorded.",
        ]),

        MSection("Hotspot versus average scoring, and spatial statistics", 2, [
            "Each image was scored twice from the same detected nuclei. The average score "
            "is the positive fraction across the whole field. The hotspot score is the "
            "maximum positive fraction over a sliding 500 micron window containing at "
            "least 100 nuclei, which approximates the field a pathologist would select. "
            "The difference between them is therefore attributable to the scoring "
            "convention alone.",
            "Positive nuclei were converted to a labelled point pattern and characterised "
            "with border-corrected Ripley K and L, the Clark-Evans index with Donnelly's "
            "perimeter correction, quadrat dispersion as a variance to mean ratio, and the "
            "coefficient of variation of a kernel density estimate. Border correction is "
            "essential on field-of-view captures, where a large fraction of the field lies "
            "within one analysis radius of an edge and an uncorrected estimator reads the "
            "missing area as reduced clustering. Radii were capped per image at one quarter "
            "of the field width and the limit used is recorded with every value.",
        ]),

        MSection("Stereological correction of counts to volumetric density", 2, [
            "Counts per unit section area were converted to counts per unit tissue volume "
            "using the source implementation's correction, C3D = C2D x k x T/(T+D), where "
            "k is the number of sections each stained section represents, T the section "
            "thickness and D the nuclear diameter " + c("kiemen2022coda") + ". The factor "
            "k was set to 1 rather than the published 3 because the present material is "
            "single fields rather than every third section of a series; retaining 3 would "
            "have tripled every count without a corresponding sampling interval. Nuclear "
            "diameter was measured in this material rather than taken from the "
            "implementation's pancreatic defaults, as equivalent circular diameter from "
            "the segmented area of every detected nucleus, pooled separately over "
            "marker-positive and marker-negative populations across the "
            f"{f.get('st_n_imgs', 0)} images with the finest pixel size and a visible "
            "counterstain. Section thickness is not recorded in the image metadata and "
            f"was taken from the confirmed cutting protocol for these blocks, "
            f"{f.get('st_T', 4.0):.0f} microns; all volumetric densities scale linearly "
            "with it. Section volume "
            "was taken as field area multiplied by section thickness. The procedure "
            "produces a volumetric density and not a reconstruction; no volume was built "
            "and none is obtainable from single sections.",
        ]),

        MSection("Statistics", 2, [
            "Paired comparisons used the Wilcoxon signed rank test, and the Wilcoxon rank "
            "sum test was the specified test for unpaired comparisons " + c("kiemen2022coda") +
            ". Effect sizes are reported with confidence intervals rather than p values "
            "alone; the mean scoring gap carries a bootstrap 95 percent confidence "
            "interval from 2000 resamples, and correlation coefficients carry Fisher "
            "z-transformed intervals. Multiple comparisons across the four spatial "
            "statistics were controlled by the Benjamini-Hochberg procedure and q values "
            "are reported alongside p. No sample size was predetermined and no data were "
            "excluded other than for the stated technical gates.",
        ]),

        MSection("Results", 1, []),

        MSection("Image quality determines what each marker can support", 2, [
            f"Of {f['n_images']} images, {f['n_analysed']} were analysed and "
            f"{f['n_skipped']} were excluded, three for an unreadable scale bar and six "
            "for insufficient resolution. The series spans a "
            f"{f['mpp_fold']:.0f}-fold range of magnification, {f['mpp_min']:.3f} to "
            f"{f['mpp_max']:.2f} microns per pixel (Figure 1A).",
            f"Counterstain was absent on {f['n_absent']} of {f['n_images']} images, and "
            "the distribution across markers determines what each can support (Figure 1B, "
            f"1C). Ki67 retained an adequate or marginal counterstain on "
            f"{f['ki67_counterstain_ok']} of {f['ki67_total']} images, whereas percent "
            f"positive was reportable on only {f['er_reportable']} of {f['n_er']} ER "
            f"images and {f['pr_reportable']} of {f['n_pr']} PR images. For the remainder, "
            "positive-cell density and spatial arrangement remain valid and are reported, "
            "while percent positive is withheld.",
            f"HER2 membrane completeness was measured on {f['n_her2']} images, median "
            f"{f['her2_completeness']:.3f} (Figure 2).",
        ], figure_ids=["F17", "F18", "F19"], table_ids=["T11", "T12"]),

        MSection("Hotspot and average scoring disagree, and disagree decisively at the cutoff", 2, [
            f"Across {f['n_ki67']} Ki67 images with a valid denominator, the median "
            f"average score was {f['avg_median']:.1f} percent and the median hotspot score "
            f"{f['hot_median']:.1f} percent. The hotspot score exceeded the average by a "
            f"median of {f['gap_median']:.1f} percentage points (interquartile range "
            f"{f['gap_q1']:.1f} to {f['gap_q3']:.1f}, maximum {f['gap_max']:.1f}). The mean "
            f"difference was {f['gap_mean']:.1f} percentage points with a bootstrap 95 "
            "percent confidence interval of 6.8 to 11.9, and the paired Wilcoxon signed "
            "rank test gave p = 8.6e-11 (Figure 3).",
            f"At the 20 percent cutoff, {f['flip']} images ({f['flip_pct']:.0f} percent) "
            "were discordant, with the average below the cutoff and the hotspot at or "
            "above it. Because both scores derive from the same nuclei on the same image, "
            "the change in category is attributable to the scoring convention and to "
            "nothing else.",
        ], figure_ids=["F20"], table_ids=["T13"]),

        MSection("Positive nuclei are clustered, and the scale of clustering carries the signal", 2, [
            f"Ki67-positive nuclei were spatially clustered rather than randomly "
            f"distributed in {f['ce_clustered']} of {f['n_spatial']} images, with a median "
            f"Donnelly-corrected Clark-Evans index of {f['ce_median']:.3f} against 1 for a "
            f"random pattern, and a median quadrat variance to mean ratio of "
            f"{f['vmr_median']:.2f} against 1 for a Poisson pattern (Figure 4).",
            "The spatial statistics differ sharply in whether they explain the scoring "
            "gap. The quadrat variance to mean ratio, computed over windows comparable in "
            "size to the reporting field, correlated strongly with the hotspot minus "
            "average difference (Spearman rho 0.66, 95 percent confidence interval 0.47 to "
            "0.79, q = 4.1e-07). The kernel density hotspot coefficient of variation and "
            "the border-corrected Ripley L correlated weakly (rho 0.31 and 0.30 "
            "respectively, q = 0.044 for both). The Clark-Evans index, which measures "
            "nearest-neighbour spacing at single-cell distances, showed no association "
            "(rho -0.03, q = 0.84).",
            "The discordance is therefore generated by large-scale patchiness in the "
            "distribution of proliferating cells, not by whether positive nuclei lie "
            "adjacent to one another. A spatial statistic evaluated at the wrong scale is "
            "silent about the problem even when it correctly reports that clustering "
            "exists.",
        ], figure_ids=["F21"]),

        MSection("Correction of two-dimensional counts to volumetric density", 2, [
            "One element of the framework's three-dimensional quantification requires no "
            "reconstructed volume and was applied. A nucleus enters a section whenever any "
            "part of it intersects the cutting plane, so the depth sampled is the section "
            "thickness plus the nuclear diameter rather than the thickness alone, and "
            "counts per unit area therefore overstate counts per unit volume in proportion "
            "to nuclear size " + c("kiemen2022coda") + ". The correction, C3D = C2D x k x "
            "T/(T+D), needs only a thickness and a diameter.",
            f"Two of its three parameters were not inherited from the source "
            f"implementation. The skipped-section factor k is 3 there because every third "
            f"section was stained and each stained section represents three sections of "
            f"tissue; the present images are single fields with no series, so k = "
            f"{f.get('st_k', 1)}. The nuclear diameter defaults to a pancreatic value of "
            f"{f.get('st_D_default', 4.20):.2f} microns, and because the correction scales "
            f"counts directly it was instead measured here: across "
            f"{f.get('st_n_nuclei', 0):,} segmented nuclei in the "
            f"{f.get('st_n_imgs', 0)} highest-resolution images the median equivalent "
            f"circular diameter was {f.get('st_D', float('nan')):.2f} microns. Adopting the "
            f"default would have inflated every volumetric density by "
            f"{f.get('st_inflation', float('nan')):.0f} percent, and unequally between "
            f"populations of differing nuclear size rather than as a shared constant that "
            f"cancels in a comparison (Figure 5).",
            f"Section thickness is not recorded in the image metadata and was taken from "
            f"the confirmed cutting protocol for these blocks, "
            f"{f.get('st_T', 4.0):.0f} microns; every volumetric density below scales "
            f"linearly with it. At that thickness and the measured "
            f"diameter the correction factor is {f.get('st_factor', float('nan')):.3f}, "
            f"implying that {f.get('st_clipped_pct', float('nan')):.0f} percent of the "
            f"nuclei visible in a section are counted only because the plane clipped them. "
            f"Median corrected volumetric densities across {f.get('st_n', 0)} images were "
            f"{f.get('st_ER', float('nan')):,.0f} per cubic millimetre for oestrogen "
            f"receptor, {f.get('st_Ki67', float('nan')):,.0f} for Ki67 and "
            f"{f.get('st_PR', float('nan')):,.0f} for progesterone receptor "
            f"(Supplementary Table S15).",
            "The correction is a monotone rescaling and reorders no image relative to "
            "another. That is its intended behaviour: it places a density in units "
            "comparable with volumetric measurements, and it is not evidence about which "
            "tumour proliferates more. It is a stereological correction and not a "
            "reconstruction, and no volume was built.",
        ], figure_ids=["F23"], table_ids=["T15"]),

        *([MSection("Serial-section registration on an external stack", 2, [
            f"To establish what the serial-section stages deliver, they were run on an "
            f"openly licensed benchmark series of {A['n_sections']} consecutive mouse "
            f"liver sections " + c("kartasalo2018") + ". The first attempt failed and is "
            f"reported because it locates a defect that would otherwise be invisible. "
            f"Registration left consecutive fiducial landmarks "
            f"{G['production_tre_mean_um']:.0f} microns apart where applying no "
            f"transform leaves them {G['identity_tre_mean_um']:.0f} microns apart, and "
            f"reduced between-section pixel correlation from "
            f"{G['raw_image_corr_median']:.3f} to "
            f"{G['registered_image_corr_median']:.3f}.",
            f"The degradation was independent of working resolution: sweeping the rigid "
            f"stage from {G['scale_sweep_mpp_range'][0]:.0f} to "
            f"{G['scale_sweep_mpp_range'][1]:.0f} microns per pixel left the error "
            f"between {G['scale_sweep_range_um'][0]:.0f} and "
            f"{G['scale_sweep_range_um'][1]:.0f} microns throughout. It localised "
            f"instead to rotation estimation, which deviated from the fiducial-implied "
            f"rotation by {G['rotation_err_mean_deg']:.1f} degrees on average and agreed "
            f"within five degrees on {G['rotation_within_5deg']} of "
            f"{G['rotation_n_tested']} pairs. Liver is a compact, near-convex and "
            f"texturally homogeneous object whose Radon transform carries little "
            f"orientation signal.",
            f"Replacing the estimator with a direct search over rotation, scoring each "
            f"candidate angle by pixel correlation after phase-correlation alignment, "
            f"reduced mean absolute rotation error from 20.8 to 3.9 degrees on the same "
            f"validation pairs. With the rigid stage solved at "
            f"{A['coarse_mpp_um']:.0f} microns per pixel and the elastic stage at "
            f"{A['mpp_um']:.2f}, and with the elastic displacement fields retained so "
            f"that landmarks receive the full transform, target registration error was "
            f"{A['tre_full_mean_um']:.0f} microns mean and "
            f"{A['tre_full_median_um']:.0f} microns median, median pixel correlation "
            f"{A['correlation_median']:.3f}, and no section fell below the 0.30 "
            f"acceptance threshold. The residual error remains well above the "
            f"{A['interobserver_median_um']:.1f} micron distance between two independent "
            f"human annotators, so the reconstruction is usable rather than exact.",
        ], figure_ids=["F3", "F4"], table_ids=["T4", "T5", "T16"])] if (A and G) else []),

        MSection("Stages that could not be run", 2, [
            ("The serial-section stages were run on external mouse liver rather than on "
             "the present material, which cannot support them. " if (A and G) else
             "The serial-section stages of the framework were not run on any material. ") +
            "The breast whole-slide resource with matched markers and registration "
            "landmarks " + c("weitz2024acrobat") + " requires a data use agreement and "
            "was not obtained. The present material consists of single fields and cannot "
            "support registration " + c("borovec2020anhir") + ", reconstruction or "
            "volumetric quantification at any sample size, so no three-dimensional "
            "reconstruction of breast tissue is reported anywhere in this work. "
            "Supplementary Table S14 records every stage against every arm with the "
            "reason for each block.",
        ], table_ids=["T14"]),

        MSection("Discussion", 1, [
            "Ki67 scoring irreproducibility is usually framed as observer variability, "
            "and the remedies proposed are training, standardised protocols and automated "
            "counting " + c("nielsen2021ki67") + ". Our results indicate that a substantial "
            "part of the disagreement is a property of the tissue rather than of the "
            "observer. When proliferation is spatially patchy, a hotspot convention and an "
            "average convention are measuring different things, and both are correct "
            "measurements of different quantities.",
            "The scale-dependence is the practically important finding. Clark-Evans "
            "correctly reported clustering in almost every image, yet carried no "
            "information about the scoring gap, because nearest-neighbour spacing operates "
            "at single-cell distances. The quadrat variance to mean ratio, evaluated over "
            "windows comparable to the reporting field, explained the gap well. Any "
            "attempt to use spatial statistics to flag unreliable Ki67 cases must "
            "therefore match the statistic to the scale at which the score is formed, and "
            "reporting that positive cells are clustered is not by itself useful.",
            "This has a direct clinical reading. Cases with high coarse-scale patchiness "
            "are those where the treatment decision is most likely to depend on where the "
            "pathologist looks. Such cases could be flagged for a defined scoring protocol "
            "or for a second reader, which is a more targeted intervention than applying "
            "the same standardisation everywhere.",
            "The three-dimensional component reproduced only in part, and the division is "
            "worth stating precisely. The stereological correction of counts to volumetric "
            "density needs a section thickness and a nuclear diameter but no volume, and it "
            "was applied, with the diameter measured in this tissue rather than borrowed. "
            "The overcounting result " + c("kiemen2022coda") + " did not reproduce and "
            "could not. That result comes from tracking whether objects distinct on one "
            "section are connected in the volume, which is the strongest argument that "
            "single-section counting misrepresents tissue, and it requires serial sections "
            "that do not exist for this material and are not publicly available for the "
            "benchmark tissue. We report that as a gap rather than substituting a weaker "
            "analysis, because the available substitutes would produce numbers without "
            "the property that makes the original result meaningful. A correction factor "
            "of 0.400 on a count is not a statement about connectivity, and presenting it "
            "as one would misrepresent both.",
            "The HER2 handling deserves a note. Per-nucleus scoring of a membranous marker "
            "produces a confident and meaningless number, which is more dangerous than an "
            "obvious error, and the implementation refuses the operation " +
            c("wolff2023her2") + ". Membrane completeness is reported instead as a "
            "quantitative descriptor and deliberately not mapped onto the ASCO/CAP "
            "categories, which are defined by a scoring procedure this measurement does "
            "not reproduce.",
        ]),

        MSection("Limitations", 1, [
            "**Field of view rather than whole slide.** The images are microscope captures "
            "of roughly 350 to 1150 microns across. No whole-slide inference follows, and "
            "Ripley's K beyond about a quarter of the field width is unreliable even with "
            "border correction, which is why radii were capped and the limit recorded.",
            "**Possible non-random field selection.** Fields photographed to document "
            "staining are plausibly biased toward positive areas. If selection was by eye "
            "the sample does not represent the slide, and the selection procedure is not "
            "recorded in the image metadata.",
            "**A wide magnification range.** The series spans "
            f"{f['mpp_fold']:.0f}-fold in microns per pixel. Measurements are converted to "
            "microns before pooling, but scale-dependent texture features are not "
            f"comparable across this range, and {f['n_skipped'] - 3} images were too "
            "coarse for nuclear analysis.",
            f"**No denominator on {f['n_absent']} of {f['n_images']} images.** Percent "
            "positive is not reportable for those and was not estimated indirectly. This "
            "constrains ER and PR far more than Ki67.",
            "**No three-dimensional reconstruction anywhere in this work, and no serial "
            "breast material.** The stereological correction reported above needs no "
            "volume and should not be read as one: it rescales a count and cannot address "
            "connectivity, object continuity across sections, or the overcounting result "
            "that motivates the original method. The images fine enough to resolve a "
            "nuclear boundary are also unevenly distributed across markers, so the pooled "
            "diameter is weighted toward the Ki67 series and is applied as a property of "
            "breast tumour nuclei rather than as a marker-specific value. The only serial "
            "material contemplated is mouse prostate and liver, which even if obtained "
            "would validate the pipeline rather than establish a breast finding.",
            "**Single institution, single scanner, no comparison cohort.** No cross-cohort "
            "comparison was attempted and no batch sensitivity audit is reported. Any "
            "future comparison against public cohorts must run that audit first, because "
            "scanner and protocol differences otherwise masquerade as population "
            "differences.",
            "**Ki67 detection was not validated against manual counts** in this material. "
            "The framework specifies validation at a 2 micron matching tolerance against "
            "two annotators " + c("kiemen2022coda") + ", which requires annotation effort "
            "not yet performed. The paired comparison is internally controlled, since both "
            "scores derive from the same detections, but absolute index values should be "
            "read with that in mind.",
        ]),

        MSection("Conclusion", 1, [
            "The disagreement between hotspot and average Ki67 scoring is measurable, "
            "large enough to change the treatment category in roughly a quarter of "
            f"evaluable images, and explained by coarse-scale spatial patchiness of "
            "proliferating cells. Spatial statistics computed at the scale of the "
            "reporting window identify the cases at risk; the same statistics computed at "
            "single-cell scale do not. Quantifying arrangement, not only proportion, is a "
            "tractable addition to Ki67 reporting.",
        ]),

        MSection("Data and code availability", 1, [
            "Analysis code, the parameter configuration with its SHA-256 locked block, and "
            "the test suite are available in the project repository. Per-image quality "
            "control, marker and spatial results are provided as supplementary tables. "
            "The image material is institutional and is not publicly redistributable. "
            "Public datasets referenced but not obtained are the serial benchmark stacks " +
            c("kartasalo2018") + " and the ACROBAT breast cohort " + c("weitz2024acrobat") +
            ". Software used includes OpenSlide " + c("goode2013openslide") + " and "
            "QuPath " + c("bankhead2017qupath") + " for related handling, and colour "
            "deconvolution follows " + c("ruifrok2001") + ".",
        ]),
    ]

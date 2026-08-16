"""Report prose. Every paragraph of the CODA report lives here.

Kept separate from the renderers so that editing what the report SAYS never
risks breaking how it is rendered. report.py imports build_sections from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Section:
    heading: str
    level: int
    paragraphs: list[str] = field(default_factory=list)
    figure_ids: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)


def _n(tab, tid, default=0):
    try:
        return len(tab[tid]["df"])
    except Exception:
        return default



def _arm_a():
    """Arm A registration summary, or None if that stage has not run."""
    import json
    from pathlib import Path as _P
    f = (_P(__file__).resolve().parents[3]
         / "results/kartasalo/summary_ds16fix.json")
    return json.loads(f.read_text()) if f.exists() else None



def _arm_a_diag():
    """Registration failure diagnosis, or None if the diagnostics have not run."""
    import json
    from pathlib import Path as _P
    f = (_P(__file__).resolve().parents[3] / "results/kartasalo/diagnosis.json")
    return json.loads(f.read_text()) if f.exists() else None



def _arm_a_3d():
    """Stage 5/6 reconstruction summary, or None if it has not run."""
    import json
    from pathlib import Path as _P
    f = (_P(__file__).resolve().parents[3] / "results/kartasalo/stage6_summary.json")
    return json.loads(f.read_text()) if f.exists() else None


def build_sections(fig: dict, tab: dict, stats: dict) -> list[Section]:
    n_missing_fig = sum(1 for f in fig.values() if f["source"] == "MISSING DATA")
    n_missing_tab = sum(1 for t in tab.values() if t["source"] == "MISSING DATA")
    n_real_fig = sum(1 for f in fig.values() if f["source"].startswith("REAL"))
    A = _arm_a()
    G = _arm_a_diag()
    V = _arm_a_3d()

    return [
        Section("Summary", 1, [
            "This report reproduces components of CODA (Kiemen et al., Nature Methods "
            "2022) on a Malaysian breast immunohistochemistry series, and states plainly "
            "which parts of the method could not be run and why.",
            "The design has three arms. Arm A is the only place the full seven-stage "
            "pipeline including three-dimensional reconstruction can run, because it is "
            "the only dataset with true serial sections. Arm B supplies human breast "
            "tissue with the same marker panel and 37,208 registration landmarks. Arm C "
            "is the institutional image series, which supports marker quantification and "
            "spatial analysis and nothing else.",
            f"**Arm A is now acquired; Arm B is not.** The benchmark repository ships the "
            f"evaluation software, not the sections, but the images themselves are "
            f"openly published (CC BY 4.0, access type Open) as a single 63.79 GB "
            f"archive, and the 47-section liver stack has been retrieved from it. "
            f"Arm B still requires a data use agreement. Consequently {n_real_fig} of 22 figures and "
            f"{22 - n_missing_fig - n_real_fig} others are built from real or "
            f"configuration data, while {n_missing_fig} figures and {n_missing_tab} "
            f"tables are labelled placeholders naming the exact input each needs. "
            f"Nothing is fabricated to fill a gap.",
            "**The headline result is from Arm C.** Across 62 Ki67 images with a valid "
            "denominator, scoring the same field by hotspot rather than by average "
            "raises the index by a median of 5.8 percentage points (mean 9.2, bootstrap "
            "95 percent confidence interval 6.8 to 11.9; Wilcoxon signed rank "
            "p = 8.6e-11, maximum gap 52.7 points). On 14 images, 23 percent of the "
            "series, the average lies below the 20 percent cutoff while the hotspot lies "
            "at or above it, so the choice of scoring method alone changes the treatment "
            "decision.",
            "**Spatial arrangement explains part of that discordance, and the scale "
            "matters.** Ki67-positive nuclei are spatially clustered rather than randomly "
            "placed in 51 of 53 images. Coarse-scale clustering, measured as the quadrat "
            "variance to mean ratio over windows comparable to the reporting field, "
            "correlates strongly with the hotspot-minus-average gap (Spearman rho 0.66, "
            "95 percent CI 0.47 to 0.79, q = 4.1e-07). Nearest-neighbour clustering "
            "measured by the Clark-Evans index does not (rho -0.03, q = 0.84). A "
            "statistic computed at the wrong spatial scale carries no information about "
            "the reproducibility problem.",
            "**Three caveats govern everything below.** These are field-of-view captures "
            "rather than whole slides, so no whole-slide inference follows and spatial "
            "statistics are bounded by the field. Counterstain is absent on 150 of 234 "
            "images, which removes the denominator and makes percent-positive "
            "unreportable for those; density and spatial pattern remain valid. No part "
            "of this work performs three-dimensional reconstruction on breast tissue, "
            "and none of it can, because no serial breast sections exist here.",
        ], figure_ids=["F1"], table_ids=["T1", "T14"]),

        Section("Methods", 1, [
            "Parameters are transcribed from the CODA Online Methods into a "
            "configuration file whose locked block is hashed with SHA-256 and verified "
            "at the start of every run; a drifted value fails the run and is named. "
            f"T2 lists all {_n(tab,'T2',120)} locked parameters grouped by Online "
            f"Methods section. T3 lists every declared deviation with its reason and "
            f"expected impact, and each is referenced inline where it applies.",
            "The runner also enforces applicability. Stages 1, 2, 5 and 6 require serial "
            "sections and are refused on any dataset that lacks them, rather than "
            "producing a transform and a correlation coefficient that would look like "
            "results. Registering sections that are not consecutive yields numbers "
            "without meaning, and the gate exists to prevent exactly that.",
        ], figure_ids=["F22"], table_ids=["T2", "T3"]),

        Section("Scale calibration and quality control (Arm C)", 2, [
            "The images are microscope field-of-view captures with a burned-in red scale "
            "bar, and the bar is both the calibration and a contaminant. It gives "
            "microns per pixel, without which every distance is in pixels; it is also a "
            "saturated high-contrast object that a nucleus detector segments and a "
            "spatial statistic reads as a dense corner cluster. The overlay region is "
            "masked before any measurement.",
            "The micron value printed beside the bar is not in the filenames and cannot "
            "be read reliably by the detector, so it was recovered by isolating the "
            "saturated red text and reading it, for 231 of 234 images. Three could not "
            "be read and carry a null value with a stated reason rather than a default. "
            "The recovered calibration reproduces all four independently verified "
            "reference values exactly: 0.222, 0.424, 0.690 and 0.708 microns per pixel.",
            "**One correction was necessary and is worth recording.** The bar detector "
            "takes the longest run of red pixels, and on heavily stained images a streak "
            "of brown diaminobenzidine outran the bar itself, placing the detected bar "
            "in the middle of the tissue. Restricting the search to the bottom twelve "
            "percent of the frame corrected 16 of 234 images. Left uncorrected, each of "
            "those would have been rescaled by a constant factor that survives every "
            "subsequent step and reaches the figures intact.",
            "**The cohort spans a 38-fold range of magnification**, 0.197 to 7.50 microns "
            "per pixel, not the 3-fold range that four sample images had suggested. At "
            "the coarse end a nucleus occupies about one pixel. Images were therefore "
            "tiered, and the 7 coarser than 2.5 microns per pixel are excluded from "
            "nuclear analysis with the reason recorded. Scale-dependent texture "
            "measurements are not comparable across this range at all.",
            "Counterstain was graded by deconvolution rather than an RGB heuristic. It "
            "is absent on 150 of 234 images, and the distribution is uneven in a way "
            "that decides what each marker can support: Ki67 retains a usable "
            "counterstain on 70 of 76 images, whereas ER, PR and HER2 do not on the "
            "large majority. Where counterstain is absent there are no visible negative "
            "nuclei, so no denominator exists. Percent positive is withheld for those "
            "images and is never back-calculated from stained area.",
        ], figure_ids=["F17"], table_ids=["T11"]),

        Section("Marker quantification (Arm C)", 2, [
            "225 of 234 images were analysed; 9 were skipped, 3 for an unreadable scale "
            "bar and 6 for insufficient resolution, each with the reason recorded.",
            "ER, PR and Ki67 were scored per nucleus for diaminobenzidine positivity. "
            "Positive-cell density per square millimetre is reported for every image "
            "because it requires no denominator. Percent positive is reported only where "
            "the counterstain gate permits, which is 9 of 62 ER images and 2 of 39 PR "
            "images.",
            "HER2 was never sent to per-nucleus scoring. It is a membranous marker, and "
            "per-nucleus diaminobenzidine scoring of it produces a confident meaningless "
            "number, which is worse than an error; the library raises on the attempt by "
            "design. Membrane completeness was measured instead across 53 images, median "
            "0.998. These are quantitative descriptors of the staining pattern and are "
            "not an ASCO/CAP category. They must never be reported as 0, 1+, 2+ or 3+.",
        ], figure_ids=["F18", "F19"], table_ids=["T12"]),

        Section("Ki67 hotspot versus average, and the spatial arrangement (Arm C)", 2, [
            "Ki67 scoring is irreproducible in practice because observers disagree on "
            "whether to score a hotspot or an average, and the 20 percent cutoff that "
            "drives chemotherapy decisions sits where that disagreement is worst. Both "
            "scores were computed from the same nuclei on the same image, so the "
            "difference is attributable to the scoring convention alone and to nothing "
            "else.",
            "The hotspot score exceeds the average by a median of 5.8 percentage points "
            "(interquartile range 0.9 to 15.2, maximum 52.7). The mean difference is 9.2 "
            "points with a bootstrap 95 percent confidence interval of 6.8 to 11.9, and "
            "the paired Wilcoxon signed rank test gives p = 8.6e-11. Of 62 images, 31 "
            "fall below the cutoff on both conventions, 17 fall at or above it on both, "
            "and **14 (23 percent) are discordant**, with the average below 20 percent "
            "and the hotspot at or above it.",
            "Positive nuclei are not randomly arranged. Clark-Evans, corrected for edge "
            "effects by Donnelly's perimeter term, has a median of 0.686 and is below 1, "
            "indicating clustering, in 51 of 53 images. The quadrat variance to mean "
            "ratio has a median of 6.63 against 1 for a Poisson pattern. Border "
            "correction matters more than usual here: on a field-of-view capture a large "
            "fraction of the field lies within one analysis radius of an edge, and an "
            "uncorrected estimator reads the missing area as reduced clustering. Radii "
            "were capped per image at one quarter of the field width and the limit is "
            "recorded alongside every value.",
            "**The scale at which clustering is measured determines whether it carries "
            "information.** Quadrat variance to mean ratio, computed over windows "
            "comparable to the reporting field, correlates strongly with the "
            "hotspot-minus-average gap (Spearman rho 0.66, 95 percent CI 0.47 to 0.79, "
            "Benjamini-Hochberg q = 4.1e-07). The kernel density hotspot coefficient of "
            "variation and the border-corrected Ripley L correlate weakly (rho 0.31 and "
            "0.30, q = 0.044 for both). Clark-Evans, which measures nearest-neighbour "
            "spacing at single-cell distances, does not correlate at all (rho -0.03, "
            "q = 0.84). The discordance is produced by large-scale patchiness, not by "
            "whether positive nuclei touch one another, and a statistic computed at the "
            "wrong scale is silent about it.",
            "This is the question a percentage cannot answer. A clustered 18 percent and "
            "a dispersed 18 percent receive the same score and the same treatment "
            "decision, and these statistics separate them.",
        ], figure_ids=["F20", "F21"], table_ids=["T13"]),

        Section("The three-dimensional component that a single section can support", 2, [
            "A three-dimensional reconstruction was not built and cannot be. Building "
            "one requires a hundred or more consecutive sections through a single "
            "block, and Arm C is single fields from many patients, which is breadth "
            "rather than depth. The applicability gate refuses stages 1, 2, 5 and 6 on "
            "this dataset for that reason and the refusal is recorded in T14.",
            "**One part of CODA's three-dimensional quantification does not need a "
            "volume, and it was run.** A nucleus appears in a section whenever any part "
            "of it intersects the cutting plane, so the depth actually sampled is the "
            "section thickness plus the nuclear diameter rather than the thickness "
            "alone. Counting nuclei per unit area therefore overestimates the number "
            "per unit volume, and overestimates it most for the largest nuclei. The "
            "correction is C3D = C2D x k x T/(T+D), and it requires only a thickness "
            "and a diameter.",
            "**Two of its three parameters were deliberately not inherited from the "
            "paper.** The skipped-section factor k is 3 in CODA because every third "
            "section was stained and each stained section stands for three sections of "
            "tissue. There is no series here, so k is 1; leaving it at 3 would have "
            "tripled every count for no reason. The nuclear diameter D defaults in the "
            "library to CODA's pancreas measurements, and the protocol requires "
            "measuring it in the tissue at hand because the correction scales counts "
            "directly. Measured across 29,440 segmented nuclei in the 25 "
            "highest-resolution images, the median equivalent circular diameter in this "
            "cohort is 6.00 um against the pancreas default of 4.20 um. Using the "
            "borrowed value would have inflated every volumetric density by 22 percent, "
            "and would have done so unequally between populations of different nuclear "
            "size rather than as a shared constant that cancels in a comparison.",
            "The third parameter is not recorded in the images and comes from the "
            "cutting protocol. Section thickness is 4 um, confirmed for these blocks "
            "rather than inherited from the source implementation, and every volumetric "
            "density scales linearly with it. At 4 um and a 6.00 um diameter the "
            "correction factor is "
            "0.400, meaning three fifths of the nuclei visible in a section are "
            "counted only because the plane clipped them.",
            "Median corrected volumetric densities are 41,120 per mm3 for oestrogen "
            "receptor, 36,583 for Ki67 and 19,666 for progesterone receptor. The "
            "correction is a monotone rescaling and reorders no image against another, "
            "which is exactly what it should do: it changes the units a density is "
            "reported in so that it is comparable with volumetric measurements, and it "
            "is not new information about which tumour proliferates more. It is "
            "reported here as a stereological correction, never as a reconstruction.",
        ], figure_ids=["F23"], table_ids=["T15", "T17"]),

        *( [Section("Arm A: serial-section registration, its failure, and the fix", 2, [
            f"The benchmark serial dataset was retrieved and all {A['n_sections']} "
            f"sections of the mouse liver series were registered. This is not breast "
            f"tissue and no biological claim follows from it; it is the only material "
            f"here on which the serial-section stages can run at all. The first attempt "
            f"failed outright, the cause was traced, and the corrected run succeeded. "
            f"All three steps are reported, because the failure is what makes the "
            f"correction credible.",
            f"**The first run made the stack worse than leaving it alone.** Consecutive "
            f"fiducials sit {G['identity_tre_mean_um']:.0f} microns apart with no "
            f"transform applied. After the stock pipeline they were "
            f"{G['production_tre_mean_um']:.0f} microns apart, "
            f"{G['ratio_worse_than_identity']:.1f} times worse than doing nothing, with "
            f"median pixel correlation {G['registered_image_corr_median']:.3f} where the "
            f"unregistered stack scored {G['raw_image_corr_median']:.3f}. Images and "
            f"landmarks agreed the transform was harmful, so this was not an artefact of "
            f"how landmarks were mapped.",
            f"**The first explanation was wrong and was discarded.** The configuration "
            f"declares a global registration resolution of 80 microns per pixel that "
            f"nothing in the code reads, so the rigid stage had been running about "
            f"eleven times finer than specified. Sweeping it from "
            f"{G['scale_sweep_mpp_range'][0]:.0f} to {G['scale_sweep_mpp_range'][1]:.0f} "
            f"microns per pixel moved the error only between "
            f"{G['scale_sweep_range_um'][0]:.0f} and {G['scale_sweep_range_um'][1]:.0f} "
            f"microns, every setting still worse than doing nothing. Scale was not the "
            f"cause.",
            f"**The cause was rotation estimation.** Against the rotation implied by the "
            f"fiducials, the Radon-based estimator averaged "
            f"{G['rotation_err_mean_deg']:.1f} degrees of error and agreed within five "
            f"degrees on {G['rotation_within_5deg']} of {G['rotation_n_tested']} pairs. "
            f"Liver is a compact, near-convex, texturally homogeneous object, so its "
            f"Radon transform carries little orientation signal to lock onto. Tens of "
            f"degrees of rotation error on a specimen nine millimetres across displaces "
            f"tissue by millimetres, which is the magnitude that was observed.",
            f"**The replacement searches the objective directly, and was validated "
            f"before it was trusted.** For each candidate angle the moving section is "
            f"rotated, translation is recovered by phase correlation, and the pair is "
            f"scored with the same pixel correlation the pipeline uses to judge quality; "
            f"the best-scoring angle wins. Measured against the same fiducial ground "
            f"truth, mean absolute rotation error falls from 20.8 to 3.9 degrees and "
            f"agreement within five degrees rises from 2 of 20 pairs to 15. The search "
            f"is bounded to plus or minus 45 degrees, which is a stated prior about hand "
            f"mounted serial sections rather than a tuned parameter, and it clips no "
            f"true value: the largest fiducial-implied rotation anywhere in this series "
            f"is 34.1 degrees. registration.py was not modified; the replacement lives "
            f"in a separate module.",
            f"**The corrected run succeeds on every measure.** Median pixel correlation "
            f"rose from {G['registered_image_corr_median']:.3f} to "
            f"{A['correlation_median']:.3f}, and the number of sections failing the 0.30 "
            f"acceptance threshold fell from {29} to {A['n_flagged']}. Target "
            f"registration error is now {A['tre_full_mean_um']:.0f} microns mean and "
            f"{A['tre_full_median_um']:.0f} microns median, against "
            f"{A['identity_mean_um']:.0f} for applying no transform. Accumulated error "
            f"fell from {2473:.0f} to {A['atre_mean_um']:.0f} microns, and hole "
            f"straightness from 1331 to {A['hole_deviation_mean_um']:.0f} microns. "
            f"Runtime fell from 199.8 minutes to {A['runtime_min']:.1f}, because solving "
            f"the rigid stage on a coarse copy is both more robust and far cheaper than "
            f"solving it at the elastic stage's resolution.",
            f"**Two details of that comparison matter.** The error is now BELOW the "
            f"{A['rigid_floor_mean_um']:.0f} micron rigid floor, which is not a "
            f"contradiction: that floor is the best a purely rigid transform can do, and "
            f"the corrected driver returns the elastic displacement fields so the "
            f"landmarks receive the same non-rigid transform the images do. The stock "
            f"pipeline discards those fields, so its landmark error could only ever "
            f"describe the rigid part. And the remaining error is still an order of "
            f"magnitude above the {A['interobserver_median_um']:.1f} micron distance "
            f"between the two human observers, so there is headroom left rather than a "
            f"solved problem.",
        ], figure_ids=["F3", "F4", "F24"], table_ids=["T4", "T5", "T16"])] if (A and G) else [] ),

        *( [Section("Arm A: a three-dimensional volume, and what it shows that a section cannot", 2, [
            f"With registration working, the {V['n_sections']} sections were stacked into "
            f"a volume at {V['mpp_um']:.2f} microns per pixel and "
            f"{V['section_thickness_um']:.0f} micron spacing. This is a genuine "
            f"reconstruction and not the stereological correction reported for Arm C: "
            f"objects exist in it, can be traced through z, and can be measured.",
            f"**The measurement a single section cannot make.** A structure crossing "
            f"several sections is counted once in each of them by two-dimensional "
            f"counting, and once in the volume. The ratio of the two is how much "
            f"single-section counting inflates object number, and it is the central claim "
            f"of the original method, which reported a {V['coda_pancreas_reference']:.1f}-fold "
            f"overestimate in pancreas. Reproduced here on real serial sections, the "
            f"answer is that there is no single ratio. Counting every detected feature "
            f"gives {V['overcounting_ratio']:.2f}-fold, because "
            f"{round(100*V['objects_in_one_section_only']/max(V['n_3d_objects'],1))} "
            f"percent of features occupy one section only and cannot be double counted. "
            f"Restricting to objects above a million cubic microns, the scale of "
            f"substantial anatomical structures, gives 10.1-fold, the same order as the "
            f"published pancreas figure. The largest structure spans "
            f"{V['max_sections_spanned']} of {V['n_sections']} sections.",
            "**So the honest form of the result is a curve, not a number.** Overcounting "
            "is real and large for structures big enough to cross sections, and absent "
            "for features that are not. Quoting a single fold-change without stating what "
            "was counted and above what size conveys almost nothing, and the same figure "
            "can be made to vary by more than an order of magnitude by moving that "
            "threshold alone.",
            "**What this reproduction is not.** The objects are vascular lumina separated "
            "by an intensity band, not the ten tissue classes of a trained segmentation, "
            "because no annotated training data or trained model exists for this "
            "material and fabricating one would be worse than doing without. The tissue "
            "is mouse liver. Neither the ratio nor the morphology is a breast result, and "
            "no breast serial sections exist in this study to make one.",
        ], figure_ids=["F7", "F12"], table_ids=["T9", "T10"])] if V else [] ),

        *( [Section("Figures in the published Extended Data layout", 2, [
            "The four figures below reproduce specific published Extended Data panel "
            "sets on this data, panel for panel, so that a reader familiar with the "
            "original can compare like with like rather than taking a summary on "
            "trust. They are generated by a standalone script and referenced here, so "
            "the report and the figure files cannot drift apart.",
            "**Four further panel sets are deliberately absent.** Cell detection "
            "precision and recall require two human annotators, and none exist for "
            "this material; the two automatic detectors run here disagree threefold, "
            "so presenting them as a manual-versus-automatic comparison would "
            "misdescribe what was measured. The segmentation training design and its "
            "confusion matrices require annotated tiles and a trained model. The "
            "labelled tissue class panel requires multi-class segmentation. Each is "
            "named with its missing input rather than being filled with something "
            "else, because a reader must never be able to mistake an absent result "
            "for a null one.",
        ], figure_ids=["F25", "F26", "F27", "F28"])] if (A and G) else [] ),

        Section("Arms A and B: not run", 2, [
            "Arm A would provide the registration accuracy benchmark and the "
            "two-dimensional versus three-dimensional overcounting result, which is the "
            "headline finding of the original paper. An earlier version of this report "
            "stated that the images required a request to the authors; that was wrong. "
            "The benchmark repository contains the evaluation framework only, but the "
            "sections are openly published under CC BY 4.0 with access type Open, as a "
            "single 63.79 GB archive whose download service ignores HTTP range requests, "
            "so the whole archive must be streamed to reach any part of it. The "
            "47-section liver stack has been retrieved and registration is under way; "
            "the figures and tables that depend on it are placeholders only until those "
            "results land.",
            "Arm B would validate haematoxylin and eosin to immunohistochemistry "
            "registration against 37,208 landmarks on breast tissue with this exact "
            "marker panel, and is the only place fibre alignment could run, because "
            "diaminobenzidine immunohistochemistry has no eosin channel. It requires a "
            "data use agreement. F15, F16, T6 and T7 are placeholders.",
            "Neither is a methodological obstacle. Both are access steps that a human "
            "must complete, and the pipeline that consumes them is written and tested.",
        ], figure_ids=((["F2", "F5", "F6", "F8", "F9", "F10", "F11",
                         "F13", "F14", "F15", "F16"] if V else
                        ["F2", "F5", "F6", "F7", "F8", "F9", "F10", "F11",
                         "F12", "F13", "F14", "F15", "F16"]) if (A and G) else
                       ["F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11",
                        "F12", "F13", "F14", "F15", "F16"]),
           table_ids=((["T6", "T7", "T8"] if V else
                       ["T6", "T7", "T8", "T9", "T10"]) if (A and G) else
                      ["T4", "T5", "T6", "T7", "T8", "T9", "T10", "T16"])),

        Section("Limitations", 1, [
            "**Field of view, not whole slide.** Arm C images are microscope captures "
            "roughly 350 to 1150 microns across. No whole-slide inference follows from "
            "them, and Ripley's K beyond about a quarter of the field width is "
            "unreliable even with border correction.",
            "**Possible non-random field selection.** Photographs taken to document "
            "positive staining are biased toward positive areas. If the fields were "
            "chosen by eye then the sample does not represent the slide. How the fields "
            "were selected is not recorded in the image metadata and should be stated "
            "explicitly by whoever captured them.",
            "**A 38-fold magnification range.** Every measurement is converted to microns "
            "before pooling, but scale-dependent texture features are not comparable "
            "across this range, and 7 images are too coarse for nuclear analysis at all.",
            "**No denominator on 150 of 234 images.** Percent positive is not reportable "
            "for those and has not been estimated by any indirect route.",
            "**No three-dimensional analysis anywhere in this work, and none on breast.** "
            "Arm A is the only source of serial sections and it is mouse prostate and "
            "liver. Even complete, it would validate the pipeline rather than establish "
            "a breast finding.",
            "**Sectioning angle is uncorrected.** Fibre alignment on a single section "
            "depends on the angle the structure was cut at, and the original work could "
            "correct for this only because it had the volume. Stage 7 did not run here, "
            "so the issue does not affect the present results, but it constrains any "
            "future single-section fibre measurement.",
            "**One institution, one scanner, no comparison cohort.** No cross-cohort "
            "comparison is attempted, so no batch audit is reported. Any future "
            "comparison against public cohorts must run the batch sensitivity audit "
            "first, because scanner and protocol differences will otherwise masquerade "
            "as population differences.",
        ]),

        Section("Reproducibility", 1, [
            f"Configuration `{stats['config_path']}`, SHA-256 of the merged locked block "
            f"`{stats['config_hash_sha256']}`, seed {stats['project_seed']}. "
            f"Git SHA **{stats['git_sha']}**. Python {stats['python_version']} on "
            f"{stats['platform']}.",
            "**Software environment.** Every third-party library used, with the version "
            "installed and the role it plays, so the environment can be rebuilt from "
            "this report without reading the source.",
            "SOFTWARE_TABLE_PLACEHOLDER",
            "Runtime on the machine used, CPU only: quality control over 234 images "
            "about 4 minutes; marker quantification about 25 minutes; spatial statistics "
            "about 8 minutes; report generation about 3 minutes. Stage 4 segmentation "
            "would require a GPU and did not run.",
            "Exact commands to regenerate every artefact in this report:",
            "```\n"
            "python -m pytest tests/ -q\n"
            "python scripts/run_usm_qc.py\n"
            "python scripts/run_usm_markers.py\n"
            "python scripts/run_usm_spatial.py\n"
            "python scripts/run_report.py\n"
            "```",
        ]),
    ]

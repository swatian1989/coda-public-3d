"""Contract tests. Synthetic data only, no downloads, no real slides."""

import numpy as np
import pandas as pd
import pytest
from scipy import ndimage

from coda_my.cohort_compare import (_cliffs_delta, _eta_squared,
                                    audit_batch_sensitivity, compare_features,
                                    match_cohorts)
from coda_my.deconv import deconvolve, estimate_stain_vectors, rgb_to_od
from coda_my.fibers import FiberConfig, anisotropy_index, tiled_anisotropy
from coda_my.ihc import IHCConfig, hotspot_vs_average, score_marker, to_point_pattern


# ----------------------------------------------------------------- deconv


def test_od_monotonic():
    """Darker pixels must give higher optical density."""
    assert rgb_to_od(np.array([[[50, 50, 50]]])).mean() > \
           rgb_to_od(np.array([[[200, 200, 200]]])).mean()


def test_white_is_zero_od():
    assert np.allclose(rgb_to_od(np.full((4, 4, 3), 255, np.uint8)), 0.0, atol=1e-9)


def test_deconvolution_separates_stains():
    img = np.full((60, 60, 3), 240, np.uint8)
    img[10:30, 10:30] = [110, 80, 160]     # hematoxylin-like
    ch = deconvolve(img)
    assert ch["hematoxylin"][10:30, 10:30].mean() > ch["hematoxylin"][40:, 40:].mean()


def test_stain_vectors_are_unit_norm():
    rng = np.random.default_rng(0)
    img = rng.integers(80, 220, (200, 200, 3), dtype=np.uint8)
    for v in estimate_stain_vectors(img).values():
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)


# ----------------------------------------------------------------- fibers


def _aligned(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n))
    a[::8, :] = 1.0
    return ndimage.gaussian_filter(a, 1.5) + rng.normal(0, 0.02, (n, n))


def _isotropic(n=300, seed=1):
    rng = np.random.default_rng(seed)
    i = ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 2.0)
    return (i - i.min()) / (i.max() - i.min())


def test_aligned_beats_isotropic():
    cfg = FiberConfig(min_eosin=0.0)
    assert anisotropy_index(_aligned(), cfg) > 0.8
    assert anisotropy_index(_isotropic(), cfg) < 0.3


def test_anisotropy_is_rotation_invariant():
    """A rotated fiber bundle is still a fiber bundle. If this fails, the index
    is measuring image axes rather than tissue."""
    cfg = FiberConfig(min_eosin=0.0)
    a = _aligned()
    r = ndimage.rotate(a, 37, reshape=False, order=1)
    assert abs(anisotropy_index(a, cfg) - anisotropy_index(r, cfg)) < 0.15


def test_anisotropy_bounded():
    cfg = FiberConfig(min_eosin=0.0)
    for img in (_aligned(), _isotropic()):
        assert 0.0 <= anisotropy_index(img, cfg) <= 1.0


def test_tiled_returns_distribution():
    """We report distributions, not single numbers, because of the sectioning
    angle problem."""
    t = tiled_anisotropy(_aligned(), FiberConfig(min_eosin=0.0, window_um=50, mpp=0.5))
    assert len(t) > 1


def test_empty_region_returns_nan_not_zero():
    """Zero is a meaningful anisotropy value. Blank tissue must give NaN."""
    assert np.isnan(anisotropy_index(np.zeros((100, 100)), FiberConfig()))


# -------------------------------------------------------------------- IHC


def test_her2_rejected_for_nuclear_scoring():
    """HER2 is membranous. Nuclear DAB scoring would produce a confident,
    meaningless number, which is worse than an error."""
    img = np.full((50, 50, 3), 200, np.uint8)
    with pytest.raises(ValueError, match="membranous"):
        score_marker(img, "HER2")


def test_unknown_marker_rejected():
    img = np.full((50, 50, 3), 200, np.uint8)
    with pytest.raises(ValueError, match="marker must be"):
        score_marker(img, "CD8")


def test_point_pattern_schema():
    nuc = pd.DataFrame({"x_um": [1.0, 2.0], "y_um": [3.0, 4.0],
                        "positive": [True, False]})
    pp = to_point_pattern(nuc)
    assert list(pp.columns) == ["x_um", "y_um", "habitat"]
    assert set(pp["habitat"]) <= {0, 1}


def test_hotspot_exceeds_average():
    """Construct a slide with one dense positive focus. The hotspot score must
    exceed the average, which is the whole Ki67 reproducibility problem."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 3000, 4000)
    y = rng.uniform(0, 3000, 4000)
    pos = (x < 600) & (y < 600)            # positives confined to one corner
    nuc = pd.DataFrame({"x_um": x, "y_um": y, "positive": pos})
    r = hotspot_vs_average(nuc, window_um=500)
    assert r["hotspot_percent"] > r["average_percent"]
    assert r["hotspot_minus_average"] > 0


def test_uniform_marker_has_small_hotspot_gap():
    rng = np.random.default_rng(1)
    n = 4000
    nuc = pd.DataFrame({"x_um": rng.uniform(0, 3000, n),
                        "y_um": rng.uniform(0, 3000, n),
                        "positive": rng.random(n) < 0.2})
    r = hotspot_vs_average(nuc, window_um=500)
    assert r["hotspot_minus_average"] < 15


# --------------------------------------------------------- cohort compare


def _cohorts(seed=0):
    rng = np.random.default_rng(seed)
    md = pd.DataFrame({
        "cohort": ["USM_Malaysia"] * 80 + ["TCGA"] * 120,
        "scanner": ["Aperio"] * 80 + ["Hamamatsu"] * 60 + ["Aperio"] * 60,
        "age": np.r_[rng.normal(48, 9, 80), rng.normal(58, 11, 120)],
        "stage": rng.choice(["I", "II", "III"], 200),
        "grade": rng.choice([1, 2, 3], 200),
        "subtype": rng.choice(["LumA", "LumB", "TNBC", "HER2"], 200),
    })
    feats = pd.DataFrame({
        "real_signal": np.r_[rng.normal(0.5, 0.1, 80), rng.normal(0.42, 0.1, 120)],
        "null_feature": rng.normal(1.6, 0.2, 200),
        "scanner_artifact": np.r_[rng.normal(1, 0.1, 80), rng.normal(2, 0.1, 60),
                                  rng.normal(1, 0.1, 60)],
    })
    return feats, md


def test_batch_audit_catches_scanner_confounding():
    """THE critical test. A feature driven by scanner must be flagged even
    though it separates cohorts significantly."""
    feats, md = _cohorts()
    a = audit_batch_sensitivity(feats, md).set_index("feature")
    assert not a.loc["scanner_artifact", "trustworthy"]
    assert a.loc["real_signal", "trustworthy"]


def test_matching_balances_arms():
    feats, md = _cohorts()
    m = match_cohorts(md)
    counts = m.groupby("cohort")["matched"].sum()
    assert counts["USM_Malaysia"] > 0 and counts["TCGA"] > 0


def test_comparison_requires_matching_first():
    feats, md = _cohorts()
    with pytest.raises(ValueError, match="match_cohorts"):
        compare_features(feats, md, matched_only=True)


def test_null_feature_not_significant():
    feats, md = _cohorts()
    r = compare_features(feats, match_cohorts(md)).set_index("feature")
    assert r.loc["null_feature", "q"] > 0.05


def test_effect_size_reported_not_just_p():
    feats, md = _cohorts()
    r = compare_features(feats, match_cohorts(md))
    assert "cliffs_delta" in r.columns and "effect_magnitude" in r.columns


def test_cliffs_delta_sign_and_bounds():
    a, b = np.arange(100.0), np.arange(100.0) + 50
    assert -1.0 <= _cliffs_delta(a, b) <= 0.0
    assert _cliffs_delta(b, a) > 0


def test_eta_squared_zero_for_identical_groups():
    g = np.random.default_rng(0).normal(size=50)
    assert _eta_squared([g.copy(), g.copy()]) < 1e-9


# ------------------------------------------------------- CODA registration


def _structured_tissue(n=160, seed=0):
    """Anisotropic synthetic tissue. Blobby isotropic textures make rotation
    recovery meaningless, so the fixture must have orientation."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    base = np.zeros((n, n))
    for _ in range(9):
        y, x = rng.integers(20, n - 20, 2)
        r = rng.integers(6, 16)
        yy, xx = np.ogrid[:n, :n]
        base[((yy - y) ** 2 / r ** 2 + (xx - x) ** 2 / (r * 2.2) ** 2) < 1] = 1.0
    base += 0.35 * np.sin(np.linspace(0, 9, n))[None, :]
    base = ndi.gaussian_filter(base, 2)
    return (base - base.min()) / (base.max() - base.min())


def test_rotation_recovered_within_quantisation():
    """Radon-based angle estimation must be accurate to the angular step."""
    from coda_my.registration import (RegistrationConfig, estimate_rotation,
                                      preprocess)
    cfg = RegistrationConfig()
    base = _structured_tissue()
    errs = []
    for angle in (-11.0, -4.0, 3.0, 9.0):
        moved = ndimage.rotate(base, angle, reshape=False, order=1)
        est = estimate_rotation(preprocess(1 - base, cfg),
                                preprocess(1 - moved, cfg), cfg)
        errs.append(abs(-est - angle))
    assert np.median(errs) < 2.0, f"median rotation error {np.median(errs):.1f} deg"


def test_translation_recovered():
    from coda_my.registration import estimate_translation
    base = _structured_tissue()
    moved = ndimage.shift(base, (7, -5), order=1)
    dy, dx = estimate_translation(base, moved)
    assert abs(dy + 7) < 1.5 and abs(dx - 5) < 1.5


def test_defective_section_flagged_not_silently_accepted():
    """A torn section must be detectable, otherwise it corrupts the stack."""
    from coda_my.registration import RegistrationConfig, global_register
    cfg = RegistrationConfig()
    good = _structured_tissue()
    noise = np.random.default_rng(3).random(good.shape)
    _, params = global_register(1 - good, 1 - noise, cfg)
    assert params["correlation"] < cfg.min_correlation


def test_stack_registers_to_centre_not_neighbour():
    """Chaining neighbour-to-neighbour accumulates error; CODA registers all
    sections to the centre section."""
    from coda_my.registration import RegistrationConfig, register_stack
    base = _structured_tissue()
    rng = np.random.default_rng(1)
    imgs = [1 - ndimage.shift(ndimage.rotate(base, rng.uniform(-8, 8),
                                             reshape=False, order=1),
                              rng.uniform(-5, 5, 2), order=1) for _ in range(5)]
    reg, params = register_stack(imgs, RegistrationConfig(), elastic=False)
    assert len(reg) == 5
    assert params[2]["correlation"] == 1.0        # centre is its own reference


# ----------------------------------------------------- 3D reconstruction


def _branched_volume():
    """One connected 3D object that looks like several separate ones in 2D.
    This is the CODA overcounting phenomenon, constructed deliberately."""
    vol = np.zeros((30, 120, 120), np.int16)
    vol[:, 10:110, 10:110] = 1
    for z in range(30):
        vol[z, 55:65, 55:65] = 2
        if z > 10:
            off = (z - 10) * 2
            vol[z, 55:65, 55 - off:65 - off] = 2
            vol[z, 55:65, 55 + off:65 + off] = 2
    return vol


def test_connectivity_finds_one_object():
    from coda_my.reconstruct import label_connected_objects
    _, n = label_connected_objects(_branched_volume(), 2)
    assert n == 1, f"branched structure is one object, found {n}"


def test_2d_overcounts_3d():
    """THE CODA result. 2D section counting must exceed the 3D truth."""
    from coda_my.reconstruct import overcounting_ratio
    r = overcounting_ratio(_branched_volume(), 2)
    assert r["ratio"].mean() > 1.0
    assert r["ratio"].max() >= 3.0


def test_composition_excludes_background():
    """Denominator must be tissue voxels; otherwise composition depends on how
    much empty slide was scanned."""
    from coda_my.reconstruct import tissue_composition
    df = tissue_composition(_branched_volume(), {1: "stroma", 2: "lesion"})
    assert abs(df["fraction"].sum() - 1.0) < 1e-6


def test_cell_count_correction_reduces_count():
    """Effective thickness is section + nuclear diameter, so the correction
    must pull the naive count DOWN, and more for larger nuclei."""
    from coda_my.reconstruct import ReconstructionConfig, extrapolate_3d_cell_count
    cfg = ReconstructionConfig()
    out = extrapolate_3d_cell_count({"pdac": 10_000, "ecm": 10_000}, cfg)
    naive = 10_000 * cfg.sections_skipped
    assert out["pdac"] < naive and out["ecm"] < naive
    assert out["pdac"] < out["ecm"]          # 6.7 um nuclei vs 2.5 um


def test_unknown_nuclear_diameter_warns_not_guesses():
    from coda_my.reconstruct import extrapolate_3d_cell_count
    out = extrapolate_3d_cell_count({"breast_tumour": 1000})
    assert out["breast_tumour"] == 3000.0    # uncorrected, flagged in the log


def test_mismatched_section_shapes_rejected():
    from coda_my.reconstruct import stack_to_volume
    with pytest.raises(ValueError, match="differing shapes"):
        stack_to_volume([np.zeros((10, 10), np.int16), np.zeros((12, 10), np.int16)])


def test_isotropic_resampling_changes_z_scale():
    """2 x 2 x 12 um voxels must become 12 x 12 x 12 um."""
    from coda_my.reconstruct import ReconstructionConfig, stack_to_volume
    cfg = ReconstructionConfig()
    sections = [np.ones((120, 120), np.int16) for _ in range(12)]
    vol = stack_to_volume(sections, cfg)
    assert vol.shape[1] == pytest.approx(120 * cfg.xy_mpp / cfg.isotropic_voxel_um, abs=1)


# ------------------------------------------------------------------- QC


def test_tre_zero_for_identical_landmarks():
    from coda_my.qc import target_registration_error
    lm = np.random.default_rng(0).uniform(0, 500, (50, 2))
    assert target_registration_error(lm, lm)["tre_mean_um"] == 0.0


def test_cell_detection_one_to_one_matching():
    """One manual point must not absorb several detections, or precision is
    inflated by duplicate detections."""
    from coda_my.qc import cell_detection_metrics
    manual = np.array([[10.0, 10.0]])
    detected = np.array([[10.0, 10.0], [10.5, 10.5], [11.0, 11.0]])
    m = cell_detection_metrics(detected, manual, tolerance_um=2.0, mpp=0.5)
    assert m["tp"] == 1 and m["fp"] == 2


def test_z_skip_validation_runs():
    from coda_my.qc import z_skip_validation
    vol = np.zeros((24, 60, 60), np.int16)
    vol[:, 5:55, 5:55] = 1
    vol[:, 20:40, 20:40] = 2
    df = z_skip_validation(vol)
    assert len(df) == 5 and (df["percent_composition_error"] < 5).all()


# ------------------------------------------------- CODA segmentation stage


def _crops(seed=0):
    rng = np.random.default_rng(seed)
    return {1: [rng.integers(80, 200, (180, 180, 3), dtype=np.uint8) for _ in range(9)],
            2: [rng.integers(80, 200, (90, 90, 3), dtype=np.uint8) for _ in range(9)],
            3: [rng.integers(80, 200, (60, 60, 3), dtype=np.uint8) for _ in range(9)]}


def test_tile_geometry_matches_paper():
    """9000 / 500 must give 324 small tiles per big tile."""
    from coda_my.segmentation import SegmentationConfig
    assert SegmentationConfig().tiles_per_big == 324


def test_training_tile_classes_are_balanced():
    """The overlay must equalise PIXELS per class, not just place boxes at
    random. Random placement reproduces the tissue's natural imbalance and rare
    classes are then learned poorly however many tiles you build."""
    from coda_my.segmentation import SegmentationConfig, build_training_tile
    cfg = SegmentationConfig(big_tile_px=1500, small_tile_px=500)
    _, mask = build_training_tile(_crops(), cfg, rng=np.random.default_rng(0))
    _, counts = np.unique(mask[mask > 0], return_counts=True)
    assert counts.min() / counts.max() > 0.7, "class pixels are not balanced"


def test_training_tile_reaches_fill_target():
    from coda_my.segmentation import SegmentationConfig, build_training_tile
    cfg = SegmentationConfig(big_tile_px=1500, small_tile_px=500)
    _, mask = build_training_tile(_crops(), cfg, rng=np.random.default_rng(0))
    assert (mask > 0).mean() >= 0.9 * cfg.fill_fraction


def test_dataset_tile_counts_scale_correctly():
    from coda_my.segmentation import SegmentationConfig, build_dataset
    cfg = SegmentationConfig(big_tile_px=1000, small_tile_px=500,
                             n_train_big_tiles=2, n_val_big_tiles=1)
    d = build_dataset(_crops(), cfg)
    assert len(d["train_images"]) == 2 * cfg.tiles_per_big
    assert len(d["val_images"]) == 1 * cfg.tiles_per_big


def test_acceptance_is_a_gate_not_a_target():
    """A class below 90% precision OR recall must fail the whole model."""
    from coda_my.segmentation import check_acceptance
    ok, failing = check_acceptance({"a": 0.95, "b": 0.82}, {"a": 0.93, "b": 0.91})
    assert not ok and failing == ["b"]
    ok2, _ = check_acceptance({"a": 0.95, "b": 0.91}, {"a": 0.93, "b": 0.92})
    assert ok2


def test_acceptance_checks_recall_too():
    from coda_my.segmentation import check_acceptance
    ok, failing = check_acceptance({"a": 0.99}, {"a": 0.60})
    assert not ok and failing == ["a"]


# ------------------------------------------------- scale bar & IHC FOV gates


def _fov(with_bar=True, with_counterstain=False, n=400, seed=0):
    """Synthetic IHC field of view with a burned-in red scale bar."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n, 3), 235, np.uint8)
    for _ in range(60):                                    # brown DAB nuclei
        y, x = rng.integers(20, n - 20, 2)
        yy, xx = np.ogrid[:n, :n]
        m = (yy - y) ** 2 + (xx - x) ** 2 < 36
        img[m] = [110, 70, 30]
    if with_counterstain:
        for _ in range(120):                               # blue negative nuclei
            y, x = rng.integers(20, n - 20, 2)
            yy, xx = np.ogrid[:n, :n]
            m = (yy - y) ** 2 + (xx - x) ** 2 < 30
            img[m] = [120, 120, 190]
    if with_bar:
        img[n - 20:n - 16, 10:110] = [220, 30, 30]         # 100 px bar
    return img


def test_scale_bar_length_recovered():
    from coda_my.scalebar import detect_scale_bar
    bar = detect_scale_bar(_fov(), label_um=50.0)
    assert abs(bar.length_px - 100) <= 3
    assert abs(bar.mpp - 0.5) < 0.02


def test_no_bar_raises_rather_than_guessing():
    """Silently defaulting mpp would mis-scale every downstream measurement by
    a constant factor, which survives all the way to a figure."""
    from coda_my.scalebar import detect_scale_bar
    with pytest.raises(ValueError, match="no red scale bar"):
        detect_scale_bar(_fov(with_bar=False))


def test_overlay_mask_covers_bar_and_is_small():
    from coda_my.scalebar import mask_overlay_region
    m = mask_overlay_region(_fov())
    assert m[385, 50]              # bar covered
    assert m.mean() < 0.15         # but not swallowing the tissue


def test_counterstain_grading_distinguishes_cases():
    from coda_my.scalebar import has_counterstain
    absent, f0 = has_counterstain(_fov(with_counterstain=False))
    present, f1 = has_counterstain(_fov(with_counterstain=True))
    assert absent == "absent"
    assert f1 > f0
    assert present in ("marginal", "adequate")


def test_mpp_without_label_is_none_not_guessed():
    from coda_my.scalebar import detect_scale_bar
    bar = detect_scale_bar(_fov(), label_um=None)
    assert bar.mpp is None


def test_parse_label_from_filename():
    from coda_my.scalebar import parse_label_from_filename
    assert parse_label_from_filename("cap_80um.png") == 80.0
    assert parse_label_from_filename("cap.png") is None


# ------------------------------------------------------------------ HER2


def _membrane_fov(n=400):
    """Chicken-wire membrane pattern: complete circumferential staining."""
    img = np.full((n, n, 3), 235, np.uint8)
    step = 40
    for y in range(20, n - 20, step):
        img[y:y + 4, 20:n - 20] = [110, 70, 30]
    for x in range(20, n - 20, step):
        img[20:n - 20, x:x + 4] = [110, 70, 30]
    img[n - 20:n - 16, 10:110] = [220, 30, 30]
    return img


def test_her2_detects_enclosed_cells():
    from coda_my.her2 import HER2Config, membrane_completeness
    r = membrane_completeness(_membrane_fov(), HER2Config(mpp=0.5))
    assert r["n_enclosed_cells"] > 10
    assert r["mean_completeness"] > 0.8


def test_her2_module_is_separate_from_nuclear_scoring():
    """HER2 must never route through per-nucleus DAB scoring."""
    from coda_my.ihc import score_marker
    with pytest.raises(ValueError, match="membranous"):
        score_marker(_membrane_fov(), "HER2")


# ------------------------------------------------------- parameter guard


def test_all_locked_sections_present():
    """Every Online Methods section must be represented."""
    from coda_my.guard import load_params
    locked = load_params()["locked"]
    for section in ("acquisition", "registration", "registration_qc",
                    "cell_detection", "segmentation", "reconstruction",
                    "quantification", "connectivity", "fiber_alignment",
                    "z_projection", "statistics"):
        assert section in locked, f"missing locked section: {section}"


def test_key_paper_values_are_correct():
    """Spot-check values transcribed from the Online Methods. If any of these
    is wrong, everything downstream is wrong in a way that still looks fine."""
    from coda_my.guard import load_params
    L = load_params()["locked"]
    assert L["acquisition"]["section_thickness_um"] == 4
    assert L["acquisition"]["he_stain_interval"] == 3
    assert L["registration"]["registration_mpp"] == 80.0
    assert L["registration"]["elastic_tile_interval_um"] == 1500
    assert L["registration"]["field_smoothing_sigma_px"] == 2
    assert L["registration"]["n_reference_candidates"] == 3
    assert L["cell_detection"]["kmeans_clusters"] == 100
    assert L["cell_detection"]["match_radius_um"] == 2.0
    assert L["segmentation"]["big_tile_px"] == 9000
    assert L["segmentation"]["small_tile_px"] == 500
    assert L["segmentation"]["tiles_per_big_tile"] == 324
    assert L["segmentation"]["n_train_tiles"] == 6480
    assert L["segmentation"]["n_val_tiles"] == 1620
    assert L["segmentation"]["fill_fraction"] == 0.65
    assert L["segmentation"]["validation_patience"] == 5
    assert L["reconstruction"]["isotropic_voxel_um"] == [12, 12, 12]
    assert L["fiber_alignment"]["window_area_um2"] == 2500
    assert L["statistics"]["test"] == "wilcoxon_rank_sum"


def test_tile_arithmetic_is_self_consistent():
    from coda_my.guard import load_params
    S = load_params()["locked"]["segmentation"]
    assert (S["big_tile_px"] // S["small_tile_px"]) ** 2 == S["tiles_per_big_tile"]
    assert S["n_train_big_tiles"] * S["tiles_per_big_tile"] == S["n_train_tiles"]
    assert S["n_val_big_tiles"] * S["tiles_per_big_tile"] == S["n_val_tiles"]


def test_voxel_volume_matches_isotropic_size():
    from coda_my.guard import load_params
    R = load_params()["locked"]["reconstruction"]
    side = R["isotropic_voxel_um"][0]
    assert abs(R["voxel_volume_mm3"] - (side ** 3) * 1e-9) < 1e-12


def test_hash_is_order_independent():
    from coda_my.guard import hash_locked
    a = {"locked": {"x": 1, "y": {"b": 2, "a": 3}}}
    b = {"locked": {"y": {"a": 3, "b": 2}, "x": 1}}
    assert hash_locked(a) == hash_locked(b)


def test_hash_changes_when_a_value_changes():
    from coda_my.guard import hash_locked
    a = {"locked": {"registration": {"elastic_tile_interval_um": 1500}}}
    b = {"locked": {"registration": {"elastic_tile_interval_um": 1200}}}
    assert hash_locked(a) != hash_locked(b)


def test_serial_stages_blocked_on_single_section_data():
    """The important gate. Registering non-serial sections yields a transform
    and a correlation, and neither means anything."""
    from coda_my.guard import check_applicability, load_params
    p = load_params()
    runnable = check_applicability(
        p, "tcga", {"registration", "reconstruction", "connectivity",
                    "cell_detection", "segmentation"})
    assert runnable == {"cell_detection", "segmentation"}


def test_fiber_alignment_blocked_on_dab_ihc():
    """Fiber alignment reads the eosin channel. DAB-IHC has none."""
    from coda_my.guard import check_applicability, load_params
    p = load_params()
    runnable = check_applicability(p, "usm_ihc_fov",
                                   {"fiber_alignment", "cell_detection"})
    assert runnable == {"cell_detection"}


def test_all_stages_run_on_serial_data():
    from coda_my.guard import check_applicability, load_params
    p = load_params()
    stages = {"registration", "reconstruction", "connectivity",
              "cell_detection", "segmentation", "fiber_alignment"}
    assert check_applicability(p, "kartasalo_prostate", stages) == stages


def test_unknown_dataset_rejected():
    from coda_my.guard import check_applicability, load_params
    with pytest.raises(ValueError, match="unknown dataset"):
        check_applicability(load_params(), "made_up", {"cell_detection"})


def test_deviations_are_fully_documented():
    """Each deviation needs all four fields, or it is not a methods sentence."""
    from coda_my.guard import load_params
    for d in load_params().get("deviations", []):
        for field in ("parameter", "paper", "ours", "reason"):
            assert field in d and d[field], f"deviation missing '{field}': {d}"

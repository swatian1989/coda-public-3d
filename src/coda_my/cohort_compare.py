"""Malaysia vs TCGA cohort comparison, with batch effect taken seriously.

The single biggest threat to this study is that scanner, stain protocol,
fixation time and lab differ between USM and TCGA, and every one of those
produces morphological differences that look exactly like population
differences. Deep models can predict TCGA tissue source site from H&E alone;
the site signal is that strong. If you scan in Kota Bharu and compare to TCGA,
you WILL find differences, and almost none of them will be biological.

Three defences, all implemented here:

1. A third cohort. If Malaysia differs from TCGA AND from the third cohort in
   the same direction, that is a signal. If Malaysia is simply the odd one out
   on every feature, that is your scanner.
2. Clinical matching. Malaysian breast cancer presents younger, at later stage,
   with a higher TNBC fraction. Without matching you measure stage, not
   ancestry.
3. A batch-sensitivity audit. Features that separate cohorts strongly but also
   separate scanners within a cohort are flagged as untrustworthy.

Ethnicity note. Malaysia is Malay, Chinese and Indian. "Malaysian" is not an
ancestry group. Analyses that pool them will be flagged by reviewers who work
in this area, and pooling can hide or fabricate effects. Record and stratify.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

MATCH_COVARIATES = ("age", "stage", "grade", "subtype")
ETHNICITY_LEVELS = ("Malay", "Chinese", "Indian", "Other")


def audit_batch_sensitivity(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    cohort_col: str = "cohort",
    scanner_col: str = "scanner",
) -> pd.DataFrame:
    """Flag features whose cohort difference is confounded by scanner.

    For every feature, computes the effect size for cohort and, separately, for
    scanner WITHIN a single cohort. A feature that separates scanners within one
    cohort cannot be trusted to separate populations across cohorts.

    Returns a table with cohort_eta2, within_cohort_scanner_eta2, and a
    ``trustworthy`` flag. Report this table. Do not quietly drop the failures.
    """
    rows = []
    for col in features.columns:
        joined = features[[col]].join(metadata, how="inner").dropna(
            subset=[col, cohort_col])
        if joined[cohort_col].nunique() < 2:
            continue

        groups = [g[col].to_numpy() for _, g in joined.groupby(cohort_col)]
        cohort_eta2 = _eta_squared(groups)

        scanner_eta2 = np.nan
        if scanner_col in joined.columns:
            biggest = joined[cohort_col].value_counts().idxmax()
            sub = joined[joined[cohort_col] == biggest].dropna(subset=[scanner_col])
            if sub[scanner_col].nunique() >= 2:
                scanner_eta2 = _eta_squared(
                    [g[col].to_numpy() for _, g in sub.groupby(scanner_col)])

        rows.append({
            "feature": col,
            "cohort_eta2": cohort_eta2,
            "within_cohort_scanner_eta2": scanner_eta2,
            "trustworthy": bool(
                np.isnan(scanner_eta2) or scanner_eta2 < 0.5 * cohort_eta2),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        n_bad = int((~out["trustworthy"]).sum())
        logger.info("%d of %d features are scanner-confounded", n_bad, len(out))
        if n_bad > 0.3 * len(out):
            logger.warning(
                "More than 30%% of features are scanner-confounded. Stain "
                "normalisation is not sufficient here. Consider rescanning a "
                "subset of TCGA slides on your own scanner, or restricting the "
                "comparison to features that survive this audit.")
    return out


def _eta_squared(groups: list[np.ndarray]) -> float:
    """Effect size for a one-way comparison. Proportion of variance explained."""
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return np.nan
    all_v = np.concatenate(groups)
    grand = all_v.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((all_v - grand) ** 2).sum()
    return float(ss_between / ss_total) if ss_total > 0 else np.nan


def match_cohorts(
    metadata: pd.DataFrame,
    cohort_col: str = "cohort",
    reference: str = "USM_Malaysia",
    covariates: tuple[str, ...] = MATCH_COVARIATES,
    caliper: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Propensity-match comparison cohorts to the reference on clinical variables.

    Without this, any difference you find is at least partly stage and age.
    Malaysian breast cancer presents younger and later-stage than the TCGA
    cohort, and both change tissue morphology independently of ancestry.

    Returns metadata with a ``matched`` boolean column added.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    md = metadata.copy()
    md["matched"] = False

    usable = [c for c in covariates if c in md.columns]
    if not usable:
        raise ValueError(f"none of {covariates} present in metadata")

    design = pd.get_dummies(md[usable], drop_first=True, dummy_na=True)
    ok = design.notna().all(axis=1)
    x = StandardScaler().fit_transform(design[ok].astype(float))
    is_ref = (md.loc[ok, cohort_col] == reference).to_numpy()
    if is_ref.sum() < 10 or (~is_ref).sum() < 10:
        raise ValueError("need at least 10 cases per arm to match")

    ps = LogisticRegression(max_iter=1000).fit(x, is_ref).predict_proba(x)[:, 1]
    idx = md.index[ok]
    ref_ps = ps[is_ref]
    pool_ps, pool_idx = ps[~is_ref], idx[~is_ref]

    sd = float(np.std(ps))
    rng = np.random.default_rng(seed)
    available = np.ones(len(pool_ps), dtype=bool)
    md.loc[idx[is_ref], "matched"] = True

    for p in rng.permutation(ref_ps):
        if not available.any():
            break
        d = np.abs(pool_ps - p)
        d[~available] = np.inf
        j = int(np.argmin(d))
        if d[j] <= caliper * sd:
            md.loc[pool_idx[j], "matched"] = True
            available[j] = False

    logger.info("matched %d reference to %d comparison cases",
                int(is_ref.sum()),
                int(md["matched"].sum() - is_ref.sum()))
    return md


def compare_features(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    cohort_col: str = "cohort",
    matched_only: bool = True,
    stratify_by: str | None = None,
) -> pd.DataFrame:
    """Test each feature across cohorts, with FDR control and effect sizes.

    Reports Cliff's delta as the effect size rather than a p value alone. With
    hundreds of slides, trivial differences reach significance; the effect size
    is what tells you whether the difference matters.
    """
    md = metadata
    if matched_only:
        if "matched" not in md.columns:
            raise ValueError("run match_cohorts() first, or pass matched_only=False")
        md = md[md["matched"]]

    strata = [(None, md)] if stratify_by is None else list(md.groupby(stratify_by))

    rows = []
    for stratum, sub in strata:
        joined = features.join(sub[[cohort_col]], how="inner")
        cohorts = sorted(joined[cohort_col].dropna().unique())
        if len(cohorts) < 2:
            continue
        for col in features.columns:
            groups = {c: joined.loc[joined[cohort_col] == c, col].dropna().to_numpy()
                      for c in cohorts}
            if any(len(g) < 5 for g in groups.values()):
                continue
            ref = groups[cohorts[0]]
            for c in cohorts[1:]:
                other = groups[c]
                u, p = stats.mannwhitneyu(ref, other, alternative="two-sided")
                rows.append({
                    "stratum": stratum, "feature": col,
                    "cohort_a": cohorts[0], "cohort_b": c,
                    "n_a": len(ref), "n_b": len(other),
                    "median_a": float(np.median(ref)),
                    "median_b": float(np.median(other)),
                    "cliffs_delta": _cliffs_delta(ref, other),
                    "p": float(p),
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q"] = multipletests(out["p"], method="fdr_bh")[1]
        out["effect_magnitude"] = pd.cut(
            out["cliffs_delta"].abs(), [0, 0.147, 0.33, 0.474, 1.0],
            labels=["negligible", "small", "medium", "large"], include_lowest=True)
    return out.sort_values("p").reset_index(drop=True) if not out.empty else out


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Nonparametric effect size in [-1, 1]. Sign follows a minus b."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return np.nan
    greater = sum((a[:, None] > b[None, :]).sum() for _ in [0])
    less = (a[:, None] < b[None, :]).sum()
    return float((greater - less) / (n_a * n_b))

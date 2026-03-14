"""
Posture–Gait–Cognitive Load Analysis (PD) — Δ-first pipeline + H7 robustness + H8 exploratory
-----------------------------------------------------------------------------------------

- Fails fast if required columns are missing or IDs are duplicated.
- Writes: derived dataset, results CSVs, plots (matplotlib), and run_report.json.

How to run:
  python pd_posture_delta_pipeline.py --csv "Final_data_frame_filtered_enrollment_pdg_merged_fes_merged_P-N_duration.csv"

Outputs:
  ./paired_delta_outputs/
    derived_dataset_with_deltas.csv
    metadata.json
    column_rename_map.json
    run_report.json
    results_tables/*.csv
    figures/*.png
"""

from __future__ import annotations

import os
import re
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# HARD dependency for H8 (fail fast if not installed)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# -----------------------------
# CLI + Configuration
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PD posture Δ-first analysis pipeline (H1–H8).")
    p.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV file."
    )
    p.add_argument(
        "--out",
        type=str,
        default="./paired_delta_outputs",
        help="Output directory."
    )
    return p.parse_args()


@dataclass
class Config:
    csv_path: str
    out_dir: str

    # Expected core columns in YOUR file
    id_col: str = "study_id"
    group_col: str = "group"       # TD, PIGD, PD_UNCLASSIFIED
    sex_col: str = "pohlavi"       # 'M'/'F'
    age_col: str = "age_0"

    duration_col: str = "duration_pd"
    ledd_col: str = "levodopa_equivalent"

    # Cognition and gait cols (NOTE: time_meanP-meanN will be sanitized to time_meanP_meanN)
    cog_cols: Tuple[str, ...] = ("moca", "tmt_a", "tmt_b", "tmt_a_chyby", "tmt_b_chyby", "jolo")
    gait_cols: Tuple[str, ...] = ("w_ex_time_meanN1N2", "w_ex_time_meanP1P2", "time_meanP_meanN")

    # Primary posture columns required for Δ endpoints
    posture_cols_required_for_primary_deltas: Tuple[str, ...] = (
        "pre_rest_tcc", "pre_cog_tcc",
        "pre_stand_ucc", "post_stand_ucc",
        "pre_rest_pisa", "post_rest_pisa",
    )

    # Auto-detect long-format posture columns
    posture_pattern: str = r"^(pre|post)_(rest|stand|cog)_(tcc|ucc|pisa|back)$"

    # Multiple testing thresholds
    alpha_primary_fdr: float = 0.05   # FDR for 3 primary Δ endpoints
    alpha_secondary_fdr: float = 0.05 # FDR for secondary families

    # Robust SE for OLS
    ols_cov_type: str = "HC3"

    # Plots
    dpi: int = 200


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def require_cols(df: pd.DataFrame, cols: List[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required {label} columns: {missing}")

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def sanitize_columns_for_patsy(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Make column names safe for statsmodels/patsy formulas:
    - Replace '-' with '_'
    - Replace spaces with '_'
    - Replace '/' with '_'
    Returns (df_renamed, mapping_old_to_new).
    """
    old_cols = df.columns.tolist()
    new_cols = []
    for c in old_cols:
        c2 = c.strip()
        c2 = c2.replace(" ", "_").replace("-", "_").replace("/", "_")
        new_cols.append(c2)

    mapping = {o: n for o, n in zip(old_cols, new_cols) if o != n}
    if mapping:
        df = df.rename(columns=mapping).copy()
    return df, mapping

def bh_fdr(pvals: pd.Series, alpha: float) -> pd.DataFrame:
    """Benjamini–Hochberg: returns p, q, reject_fdr in a dataframe."""
    p = pvals.copy()
    mask = p.notna()
    q = pd.Series(np.nan, index=p.index, dtype=float)
    reject = pd.Series(False, index=p.index, dtype=bool)
    if mask.sum() > 0:
        rej, qvals, _, _ = multipletests(p[mask].values, alpha=alpha, method="fdr_bh")
        q.loc[mask] = qvals
        reject.loc[mask] = rej
    return pd.DataFrame({"p": p, "q": q, "reject_fdr": reject})

def describe_series(x: pd.Series) -> Dict:
    x = x.dropna()
    if x.shape[0] == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan}
    q1 = float(np.nanpercentile(x, 25))
    q3 = float(np.nanpercentile(x, 75))
    return {
        "n": int(x.shape[0]),
        "mean": float(np.nanmean(x)),
        "sd": float(np.nanstd(x, ddof=1)) if x.shape[0] > 1 else np.nan,
        "median": float(np.nanmedian(x)),
        "iqr": float(q3 - q1),
        "min": float(np.nanmin(x)),
        "max": float(np.nanmax(x)),
    }

def cohen_dz_paired(delta: pd.Series) -> float:
    """Cohen's dz for paired design: mean(delta)/sd(delta)."""
    d = delta.dropna().values
    if d.shape[0] < 2:
        return np.nan
    sd = float(np.std(d, ddof=1))
    if sd == 0:
        return np.nan
    return float(np.mean(d) / sd)

def rank_biserial_from_wilcoxon(delta: pd.Series) -> float:
    """
    Rank-biserial correlation for Wilcoxon signed-rank:
      r_rb = (W_pos - W_neg) / (W_pos + W_neg)
    Computed explicitly from data (no synthetic assumptions).
    """
    d = delta.dropna().values
    if d.shape[0] < 2:
        return np.nan
    d = d[d != 0]  # Wilcoxon convention
    if d.shape[0] < 2:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    W_pos = float(np.sum(ranks[d > 0]))
    W_neg = float(np.sum(ranks[d < 0]))
    denom = W_pos + W_neg
    if denom == 0:
        return np.nan
    return float((W_pos - W_neg) / denom)


# -----------------------------
# Modeling helpers
# -----------------------------

def fit_ols(formula: str, data: pd.DataFrame, cov_type: str):
    """OLS with robust SE (HC3 by default); drops missing rows automatically."""
    return smf.ols(formula, data=data, missing="drop").fit(cov_type=cov_type)

def extract_term(res, term: str) -> Dict:
    """Extract beta/SE/p/CI for a term from a fitted statsmodels result."""
    if term not in res.params.index:
        return {"term": term, "beta": np.nan, "se": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    beta = float(res.params[term])
    se = float(res.bse[term])
    p = float(res.pvalues[term])
    ci = res.conf_int().loc[term].tolist()
    return {"term": term, "beta": beta, "se": se, "p": p, "ci_low": float(ci[0]), "ci_high": float(ci[1])}

def omnibus_wald_for_prefix(res, term_prefix: str) -> Dict:
    """
    Omnibus Wald test on all dummy-coded terms that start with a prefix, e.g.:
      term_prefix="C(group)"  -> selects terms like "C(group)[T.PIGD]" etc.

    Returns a dict with stat, df_num, p, etc.
    """
    terms = [t for t in res.params.index if t.startswith(term_prefix + "[T.")]
    if len(terms) == 0:
        return {"factor": term_prefix, "n_terms": 0, "stat": np.nan, "df_num": np.nan, "df_denom": np.nan, "p": np.nan, "test": "NA"}

    names = list(res.params.index)
    R = np.zeros((len(terms), len(names)), dtype=float)
    for i, t in enumerate(terms):
        R[i, names.index(t)] = 1.0

    w = res.wald_test(R)
    stat = float(np.asarray(w.statistic).squeeze())
    df_num = float(getattr(w, "df_num", len(terms)))
    df_denom = float(getattr(w, "df_denom", np.nan))
    p = float(w.pvalue)

    return {"factor": term_prefix, "n_terms": int(len(terms)), "stat": stat, "df_num": df_num, "df_denom": df_denom, "p": p, "test": "Wald"}


# -----------------------------
# Plotting helpers (matplotlib only)
# -----------------------------

def plot_delta_distribution(delta: pd.Series, title: str, out_path: str, dpi: int) -> None:
    d = delta.dropna().values
    plt.figure()
    plt.hist(d, bins=20)
    plt.axvline(0, linewidth=1)
    if d.shape[0] > 0:
        plt.axvline(float(np.mean(d)), linewidth=1)
    plt.title(title)
    plt.xlabel("Δ value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def plot_delta_by_group(df: pd.DataFrame, delta_col: str, group_col: str, title: str, out_path: str, dpi: int) -> None:
    sub = df[[delta_col, group_col]].dropna()
    if sub.empty:
        return
    groups = list(pd.unique(sub[group_col]))
    data = [sub.loc[sub[group_col] == g, delta_col].values for g in groups]

    plt.figure()
    plt.boxplot(data, labels=groups, showfliers=True)
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.ylabel(delta_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()

def plot_observed_means_time_condition(long_df: pd.DataFrame, metric: str, out_path: str, dpi: int) -> None:
    """
    Descriptive plot: observed mean per time×condition for a given metric.
    (This is safe and useful even if MixedLM fails.)
    """
    sub = long_df[long_df["metric"] == metric].dropna(subset=["value", "time", "condition"])
    if sub.empty:
        return

    means = sub.groupby(["condition", "time"], as_index=False)["value"].mean()

    time_order = ["pre", "post"]
    cond_order = ["rest", "stand", "cog"]

    xs, ys, labels = [], [], []
    idx = 0
    for cond in cond_order:
        for t in time_order:
            v = means.loc[(means["condition"] == cond) & (means["time"] == t), "value"]
            if v.shape[0] == 0:
                continue
            xs.append(idx)
            ys.append(float(v.iloc[0]))
            labels.append(f"{cond}_{t}")
            idx += 1

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xticks(xs, labels, rotation=45, ha="right")
    plt.title(f"Observed means: {metric} by time×condition")
    plt.ylabel("Posture value")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    args = parse_args()
    cfg = Config(csv_path=args.csv, out_dir=args.out)

    # Output directories
    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "results_tables"))
    ensure_dir(os.path.join(cfg.out_dir, "figures"))

    # Load CSV
    df = pd.read_csv(cfg.csv_path)

    # Sanitize column names for Patsy/statsmodels formulas (Option A)
    df, col_map = sanitize_columns_for_patsy(df)
    save_json(col_map, os.path.join(cfg.out_dir, "column_rename_map.json"))

    # Fail-fast columns
    require_cols(df, [cfg.id_col, cfg.group_col, cfg.sex_col, cfg.age_col], "core")
    require_cols(df, list(cfg.posture_cols_required_for_primary_deltas), "primary posture Δ")

    # Fail-fast ID uniqueness
    if df[cfg.id_col].duplicated().any():
        dup_n = int(df[cfg.id_col].duplicated().sum())
        raise ValueError(f"Duplicate IDs detected in '{cfg.id_col}': {dup_n} duplicates. Resolve upstream before analysis.")

    # Numeric coercion for relevant columns
    numeric_cols = [cfg.age_col, cfg.duration_col, cfg.ledd_col]
    numeric_cols += list(cfg.cog_cols) + list(cfg.gait_cols) + list(cfg.posture_cols_required_for_primary_deltas)
    coerce_numeric(df, [c for c in numeric_cols if c in df.columns])

    # Sex recode (model-friendly)
    # Your file: pohlavi in {'M','F'} -> sex_bin: F=0, M=1
    df["sex_bin"] = df[cfg.sex_col].astype(str).str.strip().str.upper().map({"F": 0, "M": 1})
    df["sex_bin"] = pd.to_numeric(df["sex_bin"], errors="coerce")

    # Group categorical
    df[cfg.group_col] = df[cfg.group_col].astype("category")

    # Metadata snapshot
    meta = {
        "n_rows": int(df.shape[0]),
        "columns": df.columns.tolist(),
        "id_col": cfg.id_col,
        "group_col": cfg.group_col,
        "sex_col_raw": cfg.sex_col,
        "sex_col_model": "sex_bin",
        "age_col": cfg.age_col,
        "duration_col_present": cfg.duration_col in df.columns,
        "ledd_col_present": cfg.ledd_col in df.columns,
        "group_counts": df[cfg.group_col].value_counts(dropna=False).to_dict(),
        "sex_counts_raw": df[cfg.sex_col].value_counts(dropna=False).to_dict(),
    }
    save_json(meta, os.path.join(cfg.out_dir, "metadata.json"))

    # -----------------------------
    # Δ variables (primary)
    # -----------------------------
    df["delta_tcc_cog"] = df["pre_cog_tcc"] - df["pre_rest_tcc"]
    df["delta_ucc_walk_stand"] = df["post_stand_ucc"] - df["pre_stand_ucc"]
    df["delta_pisa_walk_rest"] = df["post_rest_pisa"] - df["pre_rest_pisa"]

    # Optional extra deltas if columns exist
    def add_delta_if_exists(new: str, a: str, b: str) -> None:
        if a in df.columns and b in df.columns:
            df[new] = pd.to_numeric(df[a], errors="coerce") - pd.to_numeric(df[b], errors="coerce")

    add_delta_if_exists("delta_ucc_walk_rest", "post_rest_ucc", "pre_rest_ucc")
    add_delta_if_exists("delta_tcc_walk_rest", "post_rest_tcc", "pre_rest_tcc")
    add_delta_if_exists("delta_back_walk_rest", "post_rest_back", "pre_rest_back")

    # Save derived dataset
    derived_path = os.path.join(cfg.out_dir, "derived_dataset_with_deltas.csv")
    df.to_csv(derived_path, index=False)

    # -----------------------------
    # QC: posture summary stats (useful to show "not z-normalized")
    # -----------------------------
    posture_pattern = re.compile(cfg.posture_pattern, re.IGNORECASE)
    posture_cols = sorted([c for c in df.columns if posture_pattern.match(c)])
    qc_rows = []
    for c in posture_cols:
        qc_rows.append({"column": c, **describe_series(df[c])})
    pd.DataFrame(qc_rows).to_csv(os.path.join(cfg.out_dir, "results_tables", "QC_posture_summary_stats.csv"), index=False)

    # -----------------------------
    # Primary Δ tests (3 endpoints) + FDR across these 3 only
    # -----------------------------
    primary_endpoints = ["delta_tcc_cog", "delta_ucc_walk_stand", "delta_pisa_walk_rest"]
    primary_rows = []

    for dcol in primary_endpoints:
        delta = df[dcol]
        desc = describe_series(delta)
        d_nonan = delta.dropna().values

        # Shapiro normality test (standard; used only to choose t vs Wilcoxon)
        shapiro_p = np.nan
        if 3 <= d_nonan.shape[0] <= 5000:
            try:
                shapiro_p = float(stats.shapiro(d_nonan).pvalue)
            except Exception:
                shapiro_p = np.nan

        use_wilcoxon = (not np.isnan(shapiro_p)) and (shapiro_p < 0.05)

        if use_wilcoxon:
            w = stats.wilcoxon(d_nonan, zero_method="wilcox", alternative="two-sided", mode="auto")
            p = float(w.pvalue)
            stat = float(w.statistic)
            effect = rank_biserial_from_wilcoxon(delta)
            test_name = "wilcoxon_signed_rank"
            effect_name = "rank_biserial"
        else:
            t = stats.ttest_1samp(d_nonan, popmean=0.0)
            p = float(t.pvalue)
            stat = float(t.statistic)
            effect = cohen_dz_paired(delta)
            test_name = "one_sample_t_on_delta"
            effect_name = "cohen_dz"

        primary_rows.append({
            "endpoint": dcol,
            "test": test_name,
            "n": desc["n"],
            "mean": desc["mean"],
            "sd": desc["sd"],
            "median": desc["median"],
            "iqr": desc["iqr"],
            "statistic": stat,
            "p": p,
            "normality_shapiro_p": shapiro_p,
            effect_name: effect,
        })

        plot_delta_distribution(
            delta,
            title=f"{dcol} distribution",
            out_path=os.path.join(cfg.out_dir, "figures", f"Fig_primary_{dcol}_dist.png"),
            dpi=cfg.dpi
        )

        plot_delta_by_group(
            df,
            delta_col=dcol,
            group_col=cfg.group_col,
            title=f"{dcol} by phenotype group",
            out_path=os.path.join(cfg.out_dir, "figures", f"Fig_primary_{dcol}_by_group.png"),
            dpi=cfg.dpi
        )

    primary_df = pd.DataFrame(primary_rows)
    primary_fdr = bh_fdr(primary_df["p"], alpha=cfg.alpha_primary_fdr)
    primary_df = pd.concat([primary_df, primary_fdr.reset_index(drop=True)], axis=1)
    primary_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "PRIMARY_delta_tests_with_FDR.csv"), index=False)

    # -----------------------------
    # H1: Baseline predicts vulnerability (Δ ~ baseline + age + sex)
    # FDR across the 3 baseline terms only.
    # -----------------------------
    h1_specs = [
        ("delta_tcc_cog", "pre_rest_tcc"),
        ("delta_ucc_walk_stand", "pre_stand_ucc"),
        ("delta_pisa_walk_rest", "pre_rest_pisa"),
    ]
    h1_models, h1_terms = [], []
    baseline_ps = []

    for outcome, baseline in h1_specs:
        formula = f"{outcome} ~ {baseline} + {cfg.age_col} + sex_bin"
        res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)

        model_label = f"H1:{outcome}_on_{baseline}"
        h1_models.append({"model": model_label, "formula": formula, "n": int(res.nobs), "r2": float(res.rsquared)})

        # baseline term (part of FDR family)
        b = extract_term(res, baseline)
        b.update({"model": model_label})
        h1_terms.append(b)
        baseline_ps.append(b["p"])

        # covariates (reported but not in baseline-term FDR family)
        for tname in [cfg.age_col, "sex_bin"]:
            tr = extract_term(res, tname)
            tr.update({"model": model_label})
            h1_terms.append(tr)

    h1_models_df = pd.DataFrame(h1_models)
    h1_terms_df = pd.DataFrame(h1_terms)

    # FDR across baseline terms only, in the order of h1_specs
    fdr_baseline = bh_fdr(pd.Series(baseline_ps), alpha=cfg.alpha_secondary_fdr).reset_index(drop=True)
    baseline_mask = h1_terms_df["term"].isin([b for _, b in h1_specs])
    if int(baseline_mask.sum()) == len(h1_specs):
        h1_terms_df.loc[baseline_mask, "q"] = fdr_baseline["q"].values
        h1_terms_df.loc[baseline_mask, "reject_fdr"] = fdr_baseline["reject_fdr"].values

    h1_models_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H1_models.csv"), index=False)
    h1_terms_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H1_terms.csv"), index=False)

    # -----------------------------
    # H2: Gait load explains Δ changes (Spearman + adjusted OLS)
    # FDR across all H2 tests executed.
    # -----------------------------
    h2_specs = [
        ("delta_ucc_walk_stand", "w_ex_time_meanP1P2"),
        ("delta_ucc_walk_stand", "time_meanP_meanN"),
        ("delta_pisa_walk_rest", "w_ex_time_meanP1P2"),
    ]
    h2_rows, h2_pvals = [], []

    for outcome, gait in h2_specs:
        if gait not in df.columns:
            continue

        # Spearman
        x = df[gait]
        y = df[outcome]
        mask = x.notna() & y.notna()
        if int(mask.sum()) >= 5:
            rho, p = stats.spearmanr(x[mask].values, y[mask].values)
            h2_rows.append({"test": "spearman", "outcome": outcome, "predictor": gait, "n": int(mask.sum()),
                            "rho": float(rho), "p": float(p)})
            h2_pvals.append(float(p))
        else:
            h2_rows.append({"test": "spearman", "outcome": outcome, "predictor": gait, "n": int(mask.sum()),
                            "rho": np.nan, "p": np.nan})
            h2_pvals.append(np.nan)

        # OLS adjusted
        formula = f"{outcome} ~ {gait} + {cfg.age_col} + sex_bin"
        res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)
        term = extract_term(res, gait)
        h2_rows.append({
            "test": "ols_HC3",
            "outcome": outcome,
            "predictor": gait,
            "n": int(res.nobs),
            "beta": term["beta"],
            "se": term["se"],
            "ci_low": term["ci_low"],
            "ci_high": term["ci_high"],
            "p": term["p"],
            "formula": formula
        })
        h2_pvals.append(term["p"])

    h2_df = pd.DataFrame(h2_rows)
    h2_fdr = bh_fdr(pd.Series(h2_pvals), alpha=cfg.alpha_secondary_fdr).reset_index(drop=True)
    h2_df["q"] = h2_fdr["q"].values
    h2_df["reject_fdr"] = h2_fdr["reject_fdr"].values
    h2_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H2_gait_links.csv"), index=False)

    # -----------------------------
    # H3: Cognition moderates ΔTCC_cog
    # Primary: MoCA. Secondary: other cognitive tests separately + FDR within secondary family.
    # -----------------------------
    h3_primary_rows = []
    h3_secondary_rows = []
    secondary_ps = []

    if "moca" in df.columns:
        formula = f"delta_tcc_cog ~ moca + {cfg.age_col} + sex_bin"
        res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)
        term = extract_term(res, "moca")
        h3_primary_rows.append({"model": "H3_primary_moca", "n": int(res.nobs), "formula": formula, **term})

    for pred in ["tmt_a", "tmt_b", "tmt_a_chyby", "tmt_b_chyby", "jolo"]:
        if pred not in df.columns:
            continue
        formula = f"delta_tcc_cog ~ {pred} + {cfg.age_col} + sex_bin"
        res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)
        term = extract_term(res, pred)
        h3_secondary_rows.append({"model": f"H3_secondary_{pred}", "n": int(res.nobs), "formula": formula, **term})
        secondary_ps.append(term["p"])

    h3_primary_df = pd.DataFrame(h3_primary_rows)
    h3_secondary_df = pd.DataFrame(h3_secondary_rows)
    if not h3_secondary_df.empty:
        fdr = bh_fdr(pd.Series(secondary_ps), alpha=cfg.alpha_secondary_fdr).reset_index(drop=True)
        h3_secondary_df["q"] = fdr["q"].values
        h3_secondary_df["reject_fdr"] = fdr["reject_fdr"].values

    h3_primary_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H3_primary_moca.csv"), index=False)
    h3_secondary_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H3_secondary_cognition.csv"), index=False)

    # -----------------------------
    # H4: Phenotype moderation on Δ (ANCOVA)
    # Δ ~ C(group) + age + sex
    # Save omnibus group test (per Δ) + group coefficients.
    # FDR across the 3 omnibus p-values.
    # -----------------------------
    h4_omni_rows, h4_coef_rows = [], []
    omni_ps = []

    for outcome in primary_endpoints:
        formula = f"{outcome} ~ C({cfg.group_col}) + {cfg.age_col} + sex_bin"
        res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)

        omni = omnibus_wald_for_prefix(res, f"C({cfg.group_col})")
        omni_row = {"outcome": outcome, "n": int(res.nobs), "formula": formula, **omni}
        h4_omni_rows.append(omni_row)
        omni_ps.append(omni_row["p"])

        for t in res.params.index:
            if t.startswith(f"C({cfg.group_col})[T."):
                h4_coef_rows.append({"outcome": outcome, "n": int(res.nobs), "formula": formula, **extract_term(res, t)})

    h4_omni_df = pd.DataFrame(h4_omni_rows)
    fdr = bh_fdr(pd.Series(omni_ps), alpha=cfg.alpha_secondary_fdr).reset_index(drop=True)
    h4_omni_df["q"] = fdr["q"].values
    h4_omni_df["reject_fdr"] = fdr["reject_fdr"].values

    pd.DataFrame(h4_coef_rows).to_csv(os.path.join(cfg.out_dir, "results_tables", "H4_group_coefficients_on_deltas.csv"), index=False)
    h4_omni_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H4_group_omnibus_on_deltas.csv"), index=False)

    # -----------------------------
    # H5: Sex × group interaction on ΔUCC
    # ΔUCC ~ C(group) * sex + age
    # Save omnibus interaction test + term estimates (no FDR by default; exploratory).
    # -----------------------------
    h5_models_rows, h5_terms_rows = [], []
    if "delta_ucc_walk_stand" in df.columns:
        formula = f"delta_ucc_walk_stand ~ C({cfg.group_col}) * sex_bin + {cfg.age_col}"
        res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)

        inter_terms = [t for t in res.params.index if t.startswith(f"C({cfg.group_col})[T.") and ":sex_bin" in t]

        omni_inter = {"factor": f"C({cfg.group_col}):sex_bin", "n_terms": 0, "stat": np.nan, "df_num": np.nan, "df_denom": np.nan, "p": np.nan, "test": "NA"}
        if len(inter_terms) > 0:
            names = list(res.params.index)
            R = np.zeros((len(inter_terms), len(names)), dtype=float)
            for i, t in enumerate(inter_terms):
                R[i, names.index(t)] = 1.0
            w = res.wald_test(R)
            omni_inter = {
                "factor": f"C({cfg.group_col}):sex_bin",
                "n_terms": int(len(inter_terms)),
                "stat": float(np.asarray(w.statistic).squeeze()),
                "df_num": float(getattr(w, "df_num", len(inter_terms))),
                "df_denom": float(getattr(w, "df_denom", np.nan)),
                "p": float(w.pvalue),
                "test": "Wald",
            }

        h5_models_rows.append({"model": "H5_sex_x_group_on_delta_ucc", "n": int(res.nobs), "formula": formula, **omni_inter})
        for t in inter_terms:
            h5_terms_rows.append({"model": "H5_sex_x_group_on_delta_ucc", **extract_term(res, t)})

    pd.DataFrame(h5_models_rows).to_csv(os.path.join(cfg.out_dir, "results_tables", "H5_models.csv"), index=False)
    pd.DataFrame(h5_terms_rows).to_csv(os.path.join(cfg.out_dir, "results_tables", "H5_terms.csv"), index=False)

    # -----------------------------
    # H6: LEDD association with ΔUCC and ΔPISA (association only)
    # Δ ~ LEDD (+ duration) + age + sex
    # FDR across outcomes tested.
    # -----------------------------
    h6_rows, h6_ps = [], []
    have_ledd = cfg.ledd_col in df.columns
    have_dur = cfg.duration_col in df.columns

    if have_ledd:
        for outcome in ["delta_ucc_walk_stand", "delta_pisa_walk_rest"]:
            covs = [cfg.ledd_col]
            if have_dur:
                covs.append(cfg.duration_col)
            covs += [cfg.age_col, "sex_bin"]
            formula = f"{outcome} ~ " + " + ".join(covs)
            res = fit_ols(formula, df, cov_type=cfg.ols_cov_type)
            term = extract_term(res, cfg.ledd_col)
            h6_rows.append({"outcome": outcome, "n": int(res.nobs), "formula": formula, **term})
            h6_ps.append(term["p"])

    h6_df = pd.DataFrame(h6_rows)
    if not h6_df.empty:
        fdr = bh_fdr(pd.Series(h6_ps), alpha=cfg.alpha_secondary_fdr).reset_index(drop=True)
        h6_df["q"] = fdr["q"].values
        h6_df["reject_fdr"] = fdr["reject_fdr"].values
    h6_df.to_csv(os.path.join(cfg.out_dir, "results_tables", "H6_LEDD_associations.csv"), index=False)

    # -----------------------------
    # H7: Mixed-effects robustness models (per metric)
    # -----------------------------
    h7_model_rows, h7_term_rows = [], []

    if len(posture_cols) > 0:
        long = df[[cfg.id_col, cfg.group_col, cfg.age_col, "sex_bin"] + posture_cols].copy()
        long = long.melt(
            id_vars=[cfg.id_col, cfg.group_col, cfg.age_col, "sex_bin"],
            value_vars=posture_cols,
            var_name="measure",
            value_name="value"
        )

        parts = long["measure"].str.extract(cfg.posture_pattern, flags=re.IGNORECASE, expand=True)
        long["time"] = parts[0].str.lower()
        long["condition"] = parts[1].str.lower()
        long["metric"] = parts[2].str.lower()
        long.dropna(subset=["value", "time", "condition", "metric"], inplace=True)

        for metric in sorted(long["metric"].unique()):
            plot_observed_means_time_condition(
                long_df=long,
                metric=metric,
                out_path=os.path.join(cfg.out_dir, "figures", f"Fig_H7_observed_means_{metric}.png"),
                dpi=cfg.dpi
            )

        for metric in sorted(long["metric"].unique()):
            sub = long[long["metric"] == metric].copy()
            sub[cfg.group_col] = sub[cfg.group_col].astype("category")
            sub["time"] = sub["time"].astype("category")
            sub["condition"] = sub["condition"].astype("category")

            formula_a = f"value ~ C(time) * C(condition) + C({cfg.group_col}) + {cfg.age_col} + sex_bin"
            formula_b = f"value ~ C(time) * C(condition) * C({cfg.group_col}) + {cfg.age_col} + sex_bin"

            for label, formula in [("A_timeXcond_plus_group", formula_a), ("B_timeXcondXgroup", formula_b)]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # Use statsmodels' guaranteed API
                        md = sm.MixedLM.from_formula(formula, groups=sub[cfg.id_col], data=sub, missing="drop")
                        mres = md.fit(reml=False, method="lbfgs")
                        converged = bool(getattr(mres, "converged", True))

                        h7_model_rows.append({
                            "metric": metric,
                            "model": label,
                            "formula": formula,
                            "n_obs": int(mres.nobs),
                            "n_groups": int(sub[cfg.id_col].nunique()),
                            "llf": float(mres.llf),
                            "aic": float(getattr(mres, "aic", np.nan)),
                            "bic": float(getattr(mres, "bic", np.nan)),
                            "converged": converged,
                            "status": "ok",
                        })

                        for t in mres.params.index:
                            if t == "Intercept":
                                continue
                            beta = float(mres.params[t])
                            se = float(mres.bse[t]) if hasattr(mres, "bse") and t in mres.bse.index else np.nan
                            p = float(mres.pvalues[t]) if hasattr(mres, "pvalues") and t in mres.pvalues.index else np.nan
                            ci = mres.conf_int().loc[t].tolist() if hasattr(mres, "conf_int") and t in mres.conf_int().index else [np.nan, np.nan]
                            h7_term_rows.append({
                                "metric": metric,
                                "model": label,
                                "term": t,
                                "beta": beta,
                                "se": se,
                                "p": p,
                                "ci_low": float(ci[0]),
                                "ci_high": float(ci[1]),
                            })

                    except Exception as e:
                        h7_model_rows.append({
                            "metric": metric,
                            "model": label,
                            "formula": formula,
                            "status": "failed",
                            "error": str(e),
                        })

    pd.DataFrame(h7_model_rows).to_csv(os.path.join(cfg.out_dir, "results_tables", "H7_mixedlm_models.csv"), index=False)
    pd.DataFrame(h7_term_rows).to_csv(os.path.join(cfg.out_dir, "results_tables", "H7_mixedlm_terms.csv"), index=False)

    # -----------------------------
    # H8: Responder phenotypes (exploratory clustering on Δ vectors)
    # (HARD dependency: sklearn must exist)
    # -----------------------------
    h8_meta = []
    delta_features = [
        "delta_tcc_cog",
        "delta_ucc_walk_stand",
        "delta_pisa_walk_rest",
        "delta_ucc_walk_rest",
        "delta_tcc_walk_rest",
        "delta_back_walk_rest",
    ]
    delta_features = [c for c in delta_features if c in df.columns]

    if len(delta_features) >= 3:
        X = df[delta_features].copy()
        mask = X.notna().all(axis=1)
        Xc = X.loc[mask].values
        ids = df.loc[mask, cfg.id_col].values

        if Xc.shape[0] >= 10:
            scaler = StandardScaler()
            Xz = scaler.fit_transform(Xc)

            best_k, best_sil, best_model = None, -np.inf, None
            for k in [2, 3, 4, 5]:
                if Xz.shape[0] < (k + 5):
                    continue
                km = KMeans(n_clusters=k, n_init=50, random_state=42)
                labels = km.fit_predict(Xz)
                sil = float(silhouette_score(Xz, labels))
                if sil > best_sil:
                    best_sil, best_k, best_model = sil, k, km

            if best_model is not None and best_k is not None:
                labels = best_model.predict(Xz)
                cluster_df = pd.DataFrame({cfg.id_col: ids, "cluster": labels})
                df = df.merge(cluster_df, on=cfg.id_col, how="left")

                sizes = df["cluster"].value_counts(dropna=True).sort_index()
                sizes.to_csv(os.path.join(cfg.out_dir, "results_tables", "H8_cluster_sizes.csv"))

                anchors = []
                for col in [
                    "mdsupdrs_iii", "mdsupdrs_pigd", "mdsupdrs_td",
                    "fes_score", "pdq_soucet",
                    "w_ex_time_meanP1P2", "time_meanP_meanN",
                    cfg.age_col, "sex_bin"
                ]:
                    if col in df.columns:
                        anchors.append(col)

                summary = df.groupby("cluster")[delta_features + anchors].agg(["count", "mean", "std", "median"])
                summary.to_csv(os.path.join(cfg.out_dir, "results_tables", "H8_cluster_summary.csv"))

                dist = df.groupby("cluster")[cfg.group_col].value_counts(dropna=False).rename("n").reset_index()
                dist.to_csv(os.path.join(cfg.out_dir, "results_tables", "H8_cluster_by_group.csv"), index=False)

                h8_meta.append({
                    "status": "ok",
                    "n_clustered": int(mask.sum()),
                    "features": delta_features,
                    "best_k": int(best_k),
                    "silhouette": float(best_sil),
                })
            else:
                h8_meta.append({"status": "skipped", "reason": "No suitable k found", "features": delta_features})
        else:
            h8_meta.append({"status": "skipped", "reason": "Too few complete rows for clustering", "features": delta_features})
    else:
        h8_meta.append({"status": "skipped", "reason": "Not enough delta features available", "features": delta_features})

    pd.DataFrame(h8_meta).to_csv(os.path.join(cfg.out_dir, "results_tables", "H8_cluster_meta.csv"), index=False)

    # -----------------------------
    # Run report
    # -----------------------------
    run_report = {
        "outputs_dir": os.path.abspath(cfg.out_dir),
        "input_csv": os.path.abspath(cfg.csv_path),
        "derived_dataset_path": os.path.abspath(derived_path),
        "n_subjects": int(df.shape[0]),
        "group_counts": df[cfg.group_col].value_counts(dropna=False).to_dict(),
        "primary_endpoints": primary_endpoints,
        "delta_features_for_H8": delta_features,
    }
    save_json(run_report, os.path.join(cfg.out_dir, "run_report.json"))

    print("DONE.")
    print("Input CSV:", os.path.abspath(cfg.csv_path))
    print("Outputs dir:", os.path.abspath(cfg.out_dir))
    print("Derived dataset:", os.path.abspath(derived_path))
    print("Primary results:", os.path.join(cfg.out_dir, "results_tables", "PRIMARY_delta_tests_with_FDR.csv"))


if __name__ == "__main__":
    main()

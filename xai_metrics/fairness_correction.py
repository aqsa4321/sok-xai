#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1: Fairness evaluation hygiene.

Post-processes mimic_xai_disparity_full_raw.csv (in ~/sok/) to add:
  1.1 BH (fdr_bh) correction over 44 valid tests
  1.2 Rank-biserial r as Mann-Whitney effect size (recomputes U)
  1.3 Bootstrap 95% CI for (mean(group1) - mean(group0))
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, bootstrap
from statsmodels.stats.multitest import multipletests

HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(HERE, "mimic_full.csv")
RAW_CSV    = os.path.join(HERE, "mimic_xai_disparity_full_raw.csv")
OUT_CSV    = os.path.join(HERE, "fairness_corrected.csv")
OUT_TXT    = os.path.join(HERE, "fairness_corrected.txt")

RANDOM_STATE = 42
METRICS = ["pred_gap_fidelity", "instability", "inconsistency", "sparsity"]


def label_gender(df):    return df["gender_F"].astype(int)
def label_insurance(df): return df["insurance_Medicare"].astype(int)
def label_age(df):       return (df["anchor_age"] >= 65).astype(int)
def label_race(df):
    white_cols = [c for c in df.columns if c.startswith("race_WHITE")]
    is_white = df[white_cols].sum(axis=1) > 0
    return (~is_white).astype(int)


ATTRIBUTES = [
    ("gender",    "M (0)",             "F (1)",             label_gender),
    ("insurance", "not-Medicare (0)",  "Medicare (1)",      label_insurance),
    ("race",      "WHITE (0)",         "not-WHITE (1)",     label_race),
    ("age",       "<65 (0)",           ">=65 (1)",          label_age),
]


def build_group_lookup(sample_indices):
    df = pd.read_csv(DATA_PATH)
    subset = df.loc[df.index.isin(sample_indices)].copy()
    print(f"  matched {len(subset)}/{len(sample_indices)} sample_idx values.")
    group_lookup = {}
    for attr_name, _, _, fn in ATTRIBUTES:
        labels = fn(subset)
        group_lookup[attr_name] = labels.to_dict()
        n1 = int(labels.sum()); n0 = len(labels) - n1
        print(f"  {attr_name}: group0={n0}, group1={n1}")
    return group_lookup


def analyze_one_test(vals0, vals1, is_trivial):
    n0, n1 = len(vals0), len(vals1)
    if is_trivial or n0 == 0 or n1 == 0:
        return {"n0": n0, "n1": n1,
                "mean0": float(np.mean(vals0)) if n0 else np.nan,
                "mean1": float(np.mean(vals1)) if n1 else np.nan,
                "u_stat": np.nan, "p_raw": np.nan, "rank_biserial_r": np.nan,
                "ci_low": np.nan, "ci_high": np.nan, "ci_contains_zero": np.nan}
    u_stat, p_raw = mannwhitneyu(vals0, vals1, alternative="two-sided")
    r = 1.0 - (2.0 * u_stat) / (n0 * n1)
    def diff_means(a, b): return np.mean(b) - np.mean(a)
    res = bootstrap((np.asarray(vals0), np.asarray(vals1)),
                    statistic=diff_means, n_resamples=10_000,
                    confidence_level=0.95, method="percentile",
                    random_state=RANDOM_STATE, vectorized=False, paired=False)
    return {"n0": n0, "n1": n1,
            "mean0": float(np.mean(vals0)), "mean1": float(np.mean(vals1)),
            "u_stat": float(u_stat), "p_raw": float(p_raw),
            "rank_biserial_r": float(r),
            "ci_low": float(res.confidence_interval.low),
            "ci_high": float(res.confidence_interval.high),
            "ci_contains_zero": bool(res.confidence_interval.low <= 0.0 <= res.confidence_interval.high)}


def main():
    print("Loading raw per-sample metrics...")
    raw = pd.read_csv(RAW_CSV)
    unique_indices = raw["sample_idx"].unique().tolist()
    print(f"  {len(raw)} rows, {raw['method'].nunique()} methods, "
          f"{len(unique_indices)} unique samples.")
    print("Rebuilding group labels from MIMIC demographics...")
    group_lookup = build_group_lookup(unique_indices)

    rows = []
    for method in sorted(raw["method"].unique()):
        method_df = raw[raw["method"] == method]
        for attr_name, g0_label, g1_label, _ in ATTRIBUTES:
            grouped = method_df.assign(group=method_df["sample_idx"].map(group_lookup[attr_name])).dropna(subset=["group"])
            grouped["group"] = grouped["group"].astype(int)
            for metric in METRICS:
                vals0 = grouped.loc[grouped["group"] == 0, metric].values
                vals1 = grouped.loc[grouped["group"] == 1, metric].values
                is_trivial = (method == "EBM" and metric == "inconsistency"
                              and np.all(vals0 == 0) and np.all(vals1 == 0))
                s = analyze_one_test(vals0, vals1, is_trivial)
                s.update({"method": method, "attribute": attr_name,
                          "group0": g0_label, "group1": g1_label,
                          "metric": metric, "trivial_excluded_from_bh": is_trivial})
                rows.append(s)
    out = pd.DataFrame(rows)

    valid_mask = ~out["trivial_excluded_from_bh"]
    reject, p_bh, _, _ = multipletests(out.loc[valid_mask, "p_raw"].values,
                                       alpha=0.05, method="fdr_bh")
    out["p_bh"] = np.nan
    out["significant_bh_0.05"] = False
    out.loc[valid_mask, "p_bh"] = p_bh
    out.loc[valid_mask, "significant_bh_0.05"] = reject

    col_order = ["method", "attribute", "metric", "group0", "group1",
                 "n0", "n1", "mean0", "mean1",
                 "u_stat", "p_raw", "p_bh", "significant_bh_0.05",
                 "rank_biserial_r", "ci_low", "ci_high", "ci_contains_zero",
                 "trivial_excluded_from_bh"]
    out = out[col_order]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    with open(OUT_TXT, "w") as f:
        f.write("Fairness evaluation with BH correction, effect size, and 95% bootstrap CI.\n")
        f.write(f"BH applied over {int(valid_mask.sum())} valid tests (4 EBM-inconsistency rows excluded).\n\n")
        header = (f"{'method':<6} {'attribute':<10} {'metric':<20} "
                  f"{'n0':>4} {'n1':>4} {'p_raw':>10} {'p_bh':>10} "
                  f"{'r':>7} {'CI_low':>10} {'CI_high':>10} {'sig':>5}")
        f.write(header + "\n" + "-" * len(header) + "\n")
        for _, r in out.iterrows():
            if r["trivial_excluded_from_bh"]:
                f.write(f"{r['method']:<6} {r['attribute']:<10} {r['metric']:<20} "
                        f"{int(r['n0']):>4} {int(r['n1']):>4} "
                        f"{'n/a':>10} {'n/a':>10} {'n/a':>7} "
                        f"{'n/a':>10} {'n/a':>10} {'n/a':>5}\n")
            else:
                sig = "*" if r["significant_bh_0.05"] else ""
                f.write(f"{r['method']:<6} {r['attribute']:<10} {r['metric']:<20} "
                        f"{int(r['n0']):>4} {int(r['n1']):>4} "
                        f"{r['p_raw']:>10.4f} {r['p_bh']:>10.4f} "
                        f"{r['rank_biserial_r']:>7.3f} "
                        f"{r['ci_low']:>10.4f} {r['ci_high']:>10.4f} {sig:>5}\n")
    print(f"Saved: {OUT_TXT}")

    n_sig_raw = int((out.loc[valid_mask, "p_raw"] < 0.05).sum())
    n_sig_bh  = int((out.loc[valid_mask, "p_bh"]  < 0.05).sum())
    n_ci_ne0  = int((~out.loc[valid_mask, "ci_contains_zero"].astype(bool)).sum())
    print(f"\nOf {int(valid_mask.sum())} valid tests:")
    print(f"  Significant at raw p < 0.05:       {n_sig_raw}")
    print(f"  Significant after BH  (q < 0.05):  {n_sig_bh}")
    print(f"  CI excludes zero:                  {n_ci_ne0}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Post-process mimic_xai_disparity_full_raw.csv to produce three disparity-analysis
tables (all/class0/class1), each with per-group mean ± std + Mann-Whitney U p-value.

Requires that mimic_xai_disparity_full.py has finished and written:
  - mimic_xai_disparity_full_raw.csv   (per-sample metric values)

Outputs:
  - mimic_xai_disparity_full_all_results.txt
  - mimic_xai_disparity_full_class0_results.txt
  - mimic_xai_disparity_full_class1_results.txt
  - mimic_xai_disparity_full_all_summary.csv
  - mimic_xai_disparity_full_class0_summary.csv
  - mimic_xai_disparity_full_class1_summary.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ---------- config ----------
DATA_PATH = "/home/aqsa/sok/mimic_full.csv"
RAW_CSV   = "/home/aqsa/sok/mimic_xai_disparity_full_raw.csv"
OUT_DIR   = "/home/aqsa/sok"
TARGET    = "mortality"

METRICS_LIST = [
    ("pred_gap_fidelity", "Prediction-Gap Fidelity (lower = better fidelity)"),
    ("instability",       "Instability (lower = more stable)"),
    ("inconsistency",     "Inconsistency (lower = more consistent)"),
    ("sparsity",          "Sparsity (count of features above threshold)"),
]
ATTRS = ["gender", "insurance", "race", "age"]
GROUP_LABELS = {
    "gender":    ("M (0)", "F (1)"),
    "insurance": ("not-Medicare (0)", "Medicare (1)"),
    "race":      ("WHITE (0)", "not-WHITE (1)"),
    "age":       ("<65 (0)", ">=65 (1)"),
}


def derive_groups(df_raw):
    g = {}
    if "gender_F" in df_raw.columns:
        g["gender"] = (df_raw["gender_F"] > 0.5).astype(int).values
    if "insurance_Medicare" in df_raw.columns:
        g["insurance"] = (df_raw["insurance_Medicare"] > 0.5).astype(int).values
    if "race_WHITE" in df_raw.columns:
        g["race"] = (df_raw["race_WHITE"] < 0.5).astype(int).values
    if "anchor_age" in df_raw.columns:
        g["age"] = (df_raw["anchor_age"] >= 65).astype(int).values
    return g


def compute_disparities(df_metrics, groups, true_labels, label_filter=None):
    """label_filter: None (use all), 0 (class 0 only), 1 (class 1 only)."""
    if label_filter is not None:
        keep = (true_labels[df_metrics["sample_idx"].values] == label_filter)
        df_metrics = df_metrics[keep].reset_index(drop=True)

    rows = []
    for method in df_metrics["method"].unique():
        sub = df_metrics[df_metrics["method"] == method]
        for attr in ATTRS:
            if attr not in groups:
                continue
            g_full = groups[attr]
            g_for_method = g_full[sub["sample_idx"].values]
            for metric, _ in METRICS_LIST:
                vals = sub[metric].values
                v0 = vals[g_for_method == 0]
                v1 = vals[g_for_method == 1]
                v0 = v0[~np.isnan(v0)]
                v1 = v1[~np.isnan(v1)]
                if len(v0) < 2 or len(v1) < 2:
                    p = np.nan
                elif np.allclose(v0, v0[0]) and np.allclose(v1, v1[0]) and v0[0] == v1[0]:
                    p = np.nan      # both groups constant + equal -> undefined
                else:
                    try:
                        _, p = mannwhitneyu(v0, v1, alternative="two-sided")
                    except Exception:
                        p = np.nan
                rows.append({
                    "method":    method,
                    "attribute": attr,
                    "group0":    GROUP_LABELS[attr][0],
                    "group1":    GROUP_LABELS[attr][1],
                    "metric":    metric,
                    "n0":        int(len(v0)),
                    "n1":        int(len(v1)),
                    "mean0":     float(np.mean(v0)) if len(v0) else np.nan,
                    "std0":      float(np.std(v0))  if len(v0) else np.nan,
                    "mean1":     float(np.mean(v1)) if len(v1) else np.nan,
                    "std1":      float(np.std(v1))  if len(v1) else np.nan,
                    "p_value":   float(p),
                    "significant_0.05": bool(p < 0.05) if not np.isnan(p) else False,
                })
    return pd.DataFrame(rows)


def write_paper_style_table(summary_df, fh, exp_name, n_samples):
    methods = sorted(summary_df["method"].unique().tolist())
    fh.write(f"\nExperiment: {exp_name}  (n = {n_samples} samples)\n")
    for metric, desc in METRICS_LIST:
        fh.write(f"\n{'=' * 78}\n  {desc}\n{'=' * 78}\n")
        header = f"  {'Attribute':<12}" + "".join(f"{m:>14}" for m in methods)
        fh.write(header + "\n  " + "-" * (len(header) - 2) + "\n")
        for attr in ATTRS:
            row = summary_df[(summary_df["metric"] == metric) &
                             (summary_df["attribute"] == attr)]
            if row.empty:
                continue
            cells = []
            for m in methods:
                cell = row[row["method"] == m]
                if cell.empty:
                    cells.append("    n/a   ")
                else:
                    p = cell["p_value"].iloc[0]
                    if np.isnan(p):
                        cells.append("    n/a   ")
                    else:
                        marker = "*" if cell["significant_0.05"].iloc[0] else " "
                        cells.append(f"   {p:.3f}{marker}  ")
            fh.write(f"  {attr:<12}" + "".join(f"{c:>14}" for c in cells) + "\n")

        fh.write("\n  Per-group  mean ± std  (n):\n")
        for attr in ATTRS:
            row = summary_df[(summary_df["metric"] == metric) &
                             (summary_df["attribute"] == attr)]
            if row.empty:
                continue
            fh.write(f"    {attr}:\n")
            for _, r in row.iterrows():
                fh.write(
                    f"      {r['method']:<6} "
                    f"{r['group0']}: {r['mean0']:.4f} ± {r['std0']:.4f} "
                    f"(n={r['n0']})  | "
                    f"{r['group1']}: {r['mean1']:.4f} ± {r['std1']:.4f} "
                    f"(n={r['n1']})\n"
                )
    fh.write("\n* = significant at p<0.05 (Mann-Whitney U, two-sided)\n")
    fh.write("n/a = test not computable (constant values or too few samples)\n")


def main():
    assert os.path.exists(RAW_CSV), f"{RAW_CSV} not found — has the disparity job finished?"
    print(f"Loading raw per-sample metrics from {RAW_CSV} ...")
    raw_df = pd.read_csv(RAW_CSV)
    print(f"  {len(raw_df)} rows ({raw_df['method'].nunique()} methods x "
          f"{raw_df['sample_idx'].nunique()} samples)")

    print(f"Loading source data for sensitive attributes from {DATA_PATH} ...")
    df_raw = pd.read_csv(DATA_PATH)
    groups = derive_groups(df_raw)
    true_labels = df_raw[TARGET].astype(int).values
    print(f"  groups derived: {list(groups.keys())}")

    n_total = raw_df["sample_idx"].nunique()
    sample_ids = raw_df["sample_idx"].unique()
    n_class0 = int((true_labels[sample_ids] == 0).sum())
    n_class1 = int((true_labels[sample_ids] == 1).sum())
    print(f"  sample composition: {n_class0} class-0 + {n_class1} class-1\n")

    experiments = [
        ("all",    None,    n_total),
        ("class0", 0,       n_class0),
        ("class1", 1,       n_class1),
    ]

    for exp_name, label_filter, n in experiments:
        print(f">>> Experiment '{exp_name}' (n={n})")
        summary = compute_disparities(raw_df, groups, true_labels, label_filter)
        csv_path = os.path.join(OUT_DIR,
                                f"mimic_xai_disparity_full_{exp_name}_summary.csv")
        txt_path = os.path.join(OUT_DIR,
                                f"mimic_xai_disparity_full_{exp_name}_results.txt")
        summary.to_csv(csv_path, index=False)
        with open(txt_path, "w") as fh:
            fh.write(f"MIMIC-IV XAI Disparity Analysis  (Dai et al. 2022)\n")
            fh.write(f"Black-box: XGBoost  |  Glassbox: EBM\n")
            fh.write(f"Test: Mann-Whitney U (two-sided)\n")
            fh.write(f"Reporting: mean ± std per group + p-value\n")
            write_paper_style_table(summary, fh, exp_name, n)
        print(f"  -> {csv_path}")
        print(f"  -> {txt_path}")

    print("\nAll three experiments done.")


if __name__ == "__main__":
    main()

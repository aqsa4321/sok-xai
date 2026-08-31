#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMIC-IV XAI Disparity Analysis.

Implements the four explanation-disparity metrics from Dai et al. (AIES'22):
  - Prediction-gap fidelity
  - Instability (stability)
  - Inconsistency (consistency)
  - Sparsity

For each (XAI method, sensitive attribute, metric) tuple, computes per-sample
metric values, splits by the binary attribute, and runs a Mann-Whitney U test
to check whether the distributions differ between groups.

Models:
  - XGBoost (black-box)    -> SHAP (KernelExplainer), LIME
  - EBM     (glassbox)     -> EBM's own local explanations

Sensitive attributes (all binarized):
  - gender:    F vs M
  - insurance: Medicare vs not-Medicare
  - race:      not-WHITE vs WHITE
  - age:       >=65 vs <65

Outputs to /home/aqsa/sok/:
  - mimic_xai_disparity_results.txt   (paper-style table)
  - mimic_xai_disparity_summary.csv   (group means + p-values, tidy long format)
  - mimic_xai_disparity_raw.csv       (per-sample metric values, for plotting)
"""

# ---------------------------------------------------------------------------
# Silence noisy third-party output
# ---------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('shap').setLevel(logging.ERROR)
logging.getLogger('lime').setLevel(logging.ERROR)

from functools import partialmethod
try:
    from tqdm import tqdm as _tqdm_base
    _tqdm_base.__init__ = partialmethod(_tqdm_base.__init__, disable=True)
except Exception:
    pass
try:
    from tqdm.auto import tqdm as _tqdm_auto
    _tqdm_auto.__init__ = partialmethod(_tqdm_auto.__init__, disable=True)
except Exception:
    pass

import json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier


# ===========================================================================
# Paths and config
# ===========================================================================

DATA_PATH    = "/home/aqsa/sok/mimic_full.csv"
MODELS_DIR   = "/home/aqsa/sok/models_mimic_full"
XGB_PATH     = os.path.join(MODELS_DIR, "XGBoost_run0.joblib")
EBM_PATH     = os.path.join(MODELS_DIR, "EBM_run0.joblib")

OUT_TXT      = "/home/aqsa/sok/mimic_xai_disparity_full_results.txt"
OUT_SUMMARY  = "/home/aqsa/sok/mimic_xai_disparity_full_summary.csv"
OUT_RAW      = "/home/aqsa/sok/mimic_xai_disparity_full_raw.csv"

TARGET       = "mortality"
RANDOM_STATE = 42

# Paper hyperparameters (Section 4.1.4), adapted to our scale.
FIDELITY_K     = 5      # top-k features to keep "important"
FIDELITY_M     = 20     # noisings for fidelity (paper uses 1000, slow)
FIDELITY_SIGMA = 0.1

STABILITY_M    = 5      # explanations on perturbed inputs
STABILITY_SIGMA = 0.1

CONSISTENCY_M  = 5      # repeated explanations of the same input

SPARSITY_T     = 0.01   # threshold for "significant" attribution


# ===========================================================================
# Data loading
# ===========================================================================

def load_data():
    df = pd.read_csv(DATA_PATH)

    feature_names = [c for c in df.columns if c != TARGET]
    X_raw = df[feature_names].copy()
    y     = df[TARGET].astype(int).values

    # We need an unscaled copy to derive sensitive-attribute group labels
    # AND a scaled copy that matches what the model was trained on.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw.values.astype(np.float32))
    X_train, X_test, _, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    return df, X_raw, X_scaled, y, feature_names, X_train


def derive_groups(df_raw):
    """Build binary group labels from the unscaled CSV (still has one-hot cols
    for categoricals + raw anchor_age in years)."""
    g = {}

    # gender: F vs M
    if "gender_F" in df_raw.columns and "gender_M" in df_raw.columns:
        g["gender"] = (df_raw["gender_F"] > 0.5).astype(int).values  # 1 = F
        g_labels = {"gender": ("M", "F")}
    else:
        g["gender"] = None

    # insurance: Medicare vs not-Medicare
    if "insurance_Medicare" in df_raw.columns:
        g["insurance"] = (df_raw["insurance_Medicare"] > 0.5).astype(int).values
    else:
        g["insurance"] = None

    # race: not-WHITE vs WHITE   (1 = not-WHITE, 0 = WHITE)
    if "race_WHITE" in df_raw.columns:
        g["race"] = (df_raw["race_WHITE"] < 0.5).astype(int).values
    else:
        g["race"] = None

    # age: >=65 vs <65
    if "anchor_age" in df_raw.columns:
        g["age"] = (df_raw["anchor_age"] >= 65).astype(int).values
    else:
        g["age"] = None

    labels = {
        "gender":    ("M (0)", "F (1)"),
        "insurance": ("not-Medicare (0)", "Medicare (1)"),
        "race":      ("WHITE (0)", "not-WHITE (1)"),
        "age":       ("<65 (0)", ">=65 (1)"),
    }
    return g, labels


# ===========================================================================
# Model wrappers
# ===========================================================================

class ModelBundle:
    """Holds all three explainers, plus prediction wrappers, plus the fitted
    StandardScaler so we can convert raw rows -> scaled rows on demand."""

    def __init__(self, X_scaled, X_train_scaled, feature_names):
        self.feature_names = feature_names
        self.X_scaled = X_scaled
        self.X_train_scaled = X_train_scaled

        print("Loading XGBoost...")
        self.xgb = joblib.load(XGB_PATH)

        print("Loading EBM...")
        self.ebm = joblib.load(EBM_PATH)

        bg_size = min(50, len(X_train_scaled))
        bg = X_train_scaled[:bg_size]

        print("Building SHAP KernelExplainer (on XGBoost)...")
        self.shap_explainer = shap.KernelExplainer(self.xgb.predict_proba, bg)

        print("Building LIME explainer (on XGBoost)...")
        self.lime_explainer = LimeTabularExplainer(
            bg,
            feature_names=feature_names,
            class_names=["Survived", "Died"],
            discretize_continuous=True,
            random_state=RANDOM_STATE,
        )

    # ------------------ prediction wrappers ----------------------
    def xgb_predict_proba(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        return self.xgb.predict_proba(X)

    def ebm_predict_proba(self, X):
        if X.ndim == 1: X = X.reshape(1, -1)
        df = pd.DataFrame(X, columns=self.feature_names)
        return self.ebm.predict_proba(df)

    # ------------------ explanation generators -------------------
    def explain_shap(self, x, seed=None):
        """Return class-1 attributions, length n_features. Stochastic (used for consistency)."""
        if seed is not None:
            np.random.seed(seed)
        x2 = x.reshape(1, -1)
        sv = self.shap_explainer.shap_values(x2, nsamples=500, silent=True)
        if isinstance(sv, list):
            return np.asarray(sv[1][0] if len(sv) > 1 else sv[0][0]).flatten()
        arr = np.asarray(sv)
        if arr.ndim == 3:   return arr[0, :, 1].flatten()
        if arr.ndim == 2:   return arr[0].flatten()
        return arr.flatten()

    def explain_lime(self, x, seed=None):
        """Return class-1-aligned attribution vector, length n_features."""
        if seed is not None:
            np.random.seed(seed)
        exp = self.lime_explainer.explain_instance(
            x.flatten(), self.xgb_predict_proba,
            num_features=len(self.feature_names), num_samples=200,
        )
        d = dict(exp.as_list())
        out = np.zeros(len(self.feature_names))
        for i, fn in enumerate(self.feature_names):
            for lf, w in d.items():
                if fn in lf:
                    out[i] = w
                    break
        return out

    def explain_ebm(self, x, seed=None):
        """EBM local explanation. Deterministic; `seed` ignored."""
        x2 = x.reshape(1, -1)
        x_df = pd.DataFrame(x2, columns=self.feature_names)
        local_exp = self.ebm.explain_local(x_df)
        try:
            d = local_exp.data(0)
            if isinstance(d, dict) and "scores" in d:
                arr = np.asarray(d["scores"])
                if len(arr) >= len(self.feature_names):
                    return arr[:len(self.feature_names)]
        except Exception:
            pass
        out = np.zeros(len(self.feature_names))
        for i in range(len(self.feature_names)):
            try:
                fe = local_exp.data(i)
                if isinstance(fe, dict) and "scores" in fe:
                    sd = fe["scores"]
                    out[i] = float(np.mean(sd)) if hasattr(sd, "__len__") else float(sd)
            except Exception:
                pass
        return out


# ===========================================================================
# Paper-style metrics (Dai et al. 2022)
# ===========================================================================

def _class1_proba(predict_proba_fn, x):
    out = np.asarray(predict_proba_fn(x.reshape(1, -1)))
    if out.ndim == 2 and out.shape[1] == 2:
        return float(out[0, 1])
    return float(out.flatten()[-1])


def m_pred_gap_fidelity(predict_proba_fn, x, expl,
                        k=FIDELITY_K, m=FIDELITY_M, sigma=FIDELITY_SIGMA):
    """Definition 3.2: noise the NON-top-k features, measure |h(x) - h(x~)|."""
    top_k = np.argsort(np.abs(expl))[::-1][:k]
    not_top = np.setdiff1d(np.arange(len(x)), top_k)
    h_x = _class1_proba(predict_proba_fn, x)
    gaps = []
    for _ in range(m):
        x_tilde = x.copy()
        x_tilde[not_top] = x_tilde[not_top] + np.random.normal(0, sigma, size=not_top.shape)
        h_xt = _class1_proba(predict_proba_fn, x_tilde)
        gaps.append(abs(h_x - h_xt))
    return float(np.mean(gaps))


def m_instability(explain_fn, x,
                  m=STABILITY_M, sigma=STABILITY_SIGMA):
    """Definition 3.3: noise ALL features, measure ||E(x) - E(x~)||_1 averaged."""
    e0 = np.asarray(explain_fn(x)).flatten()
    dists = []
    for _ in range(m):
        x_tilde = x + np.random.normal(0, sigma, size=x.shape)
        try:
            et = np.asarray(explain_fn(x_tilde)).flatten()
            dists.append(np.sum(np.abs(e0 - et)))
        except Exception:
            continue
    return float(np.mean(dists)) if dists else 0.0


def m_inconsistency(explain_fn, x, m=CONSISTENCY_M, deterministic=False):
    """Definition 3.4: ||E_0(x) - E_j(x)||_1 averaged across j=1..m."""
    if deterministic:
        return 0.0   # EBM is deterministic; 0 by definition
    e0 = np.asarray(explain_fn(x, seed=0)).flatten()
    dists = []
    for j in range(1, m + 1):
        try:
            ej = np.asarray(explain_fn(x, seed=j * 100 + 7)).flatten()
            dists.append(np.sum(np.abs(e0 - ej)))
        except Exception:
            continue
    return float(np.mean(dists)) if dists else 0.0


def m_sparsity(expl, t=SPARSITY_T):
    """Definition 3.5: count features with |attribution| > t."""
    return int(np.sum(np.abs(np.asarray(expl).flatten()) > t))


# ===========================================================================
# Per-sample evaluation loop
# ===========================================================================

def evaluate_sample(i, x_scaled, bundle, results):
    """Compute all 4 metrics for all 3 explainers on this single sample."""
    explainers = [
        ("SHAP", bundle.explain_shap, bundle.xgb_predict_proba, False),
        ("LIME", bundle.explain_lime, bundle.xgb_predict_proba, False),
        ("EBM",  bundle.explain_ebm,  bundle.ebm_predict_proba, True),
    ]
    for method_name, explain_fn, predict_fn, is_deterministic in explainers:
        try:
            expl = explain_fn(x_scaled, seed=0)
        except Exception as e:
            print(f"    [warn] sample {i} / {method_name}: explanation failed: {e}")
            continue

        try:
            fid = m_pred_gap_fidelity(predict_fn, x_scaled, expl)
        except Exception:
            fid = np.nan
        try:
            sta = m_instability(explain_fn, x_scaled)
        except Exception:
            sta = np.nan
        try:
            con = m_inconsistency(explain_fn, x_scaled,
                                  deterministic=is_deterministic)
        except Exception:
            con = np.nan
        try:
            spa = m_sparsity(expl)
        except Exception:
            spa = np.nan

        results.append({
            "sample_idx": i,
            "method":     method_name,
            "pred_gap_fidelity": fid,
            "instability":       sta,
            "inconsistency":     con,
            "sparsity":          spa,
        })


# ===========================================================================
# Disparity tests
# ===========================================================================

METRICS_LIST = [
    ("pred_gap_fidelity", "Prediction-Gap Fidelity (lower = better fidelity)"),
    ("instability",       "Instability (lower = more stable)"),
    ("inconsistency",     "Inconsistency (lower = more consistent)"),
    ("sparsity",          "Sparsity (count of features above threshold)"),
]

ATTRS = ["gender", "insurance", "race", "age"]


def compute_disparities(df_metrics, groups, group_labels):
    """For every (method, attribute, metric), run Mann-Whitney U on the
    two sample groups and assemble a tidy summary."""
    rows = []
    for method in df_metrics["method"].unique():
        sub = df_metrics[df_metrics["method"] == method]
        for attr in ATTRS:
            if groups[attr] is None:
                continue
            g_for_method = groups[attr][sub["sample_idx"].values]
            for metric, _desc in METRICS_LIST:
                vals = sub[metric].values
                vals0 = vals[g_for_method == 0]
                vals1 = vals[g_for_method == 1]
                # drop NaNs
                vals0 = vals0[~np.isnan(vals0)]
                vals1 = vals1[~np.isnan(vals1)]
                if len(vals0) < 2 or len(vals1) < 2:
                    p = np.nan
                else:
                    try:
                        _, p = mannwhitneyu(vals0, vals1, alternative="two-sided")
                    except Exception:
                        p = np.nan
                rows.append({
                    "method":   method,
                    "attribute": attr,
                    "group0":    group_labels[attr][0],
                    "group1":    group_labels[attr][1],
                    "metric":    metric,
                    "n0":        int(len(vals0)),
                    "n1":        int(len(vals1)),
                    "mean0":     float(np.mean(vals0)) if len(vals0) else np.nan,
                    "mean1":     float(np.mean(vals1)) if len(vals1) else np.nan,
                    "p_value":   float(p),
                    "significant_0.05": bool(p < 0.05) if not np.isnan(p) else False,
                })
    return pd.DataFrame(rows)


# ===========================================================================
# Pretty printing of paper-style table
# ===========================================================================

def write_paper_style_table(summary_df, fh):
    methods = sorted(summary_df["method"].unique().tolist())
    for metric, desc in METRICS_LIST:
        fh.write(f"\n{'=' * 78}\n")
        fh.write(f"  {desc}\n")
        fh.write(f"{'=' * 78}\n")
        header = f"  {'Attribute':<12}" + "".join(f"{m:>14}" for m in methods)
        fh.write(header + "\n")
        fh.write("  " + "-" * (len(header) - 2) + "\n")
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
                    sig = cell["significant_0.05"].iloc[0]
                    marker = "*" if sig else " "
                    cells.append(f"   {p:.3f}{marker}  ")
            fh.write(f"  {attr:<12}" + "".join(f"{c:>14}" for c in cells) + "\n")

        # Means per group, per method
        fh.write("\n  Per-group means:\n")
        for attr in ATTRS:
            row = summary_df[(summary_df["metric"] == metric) &
                             (summary_df["attribute"] == attr)]
            if row.empty:
                continue
            fh.write(f"    {attr}:\n")
            for _, r in row.iterrows():
                fh.write(f"      {r['method']:<6} "
                         f"{r['group0']}: mean={r['mean0']:.4f} (n={r['n0']})  | "
                         f"{r['group1']}: mean={r['mean1']:.4f} (n={r['n1']})\n")
    fh.write("\n* = significant at p<0.05 (Mann-Whitney U, two-sided)\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 78)
    print("MIMIC-IV XAI DISPARITY ANALYSIS")
    print("=" * 78)

    df_raw, X_raw, X_scaled, y, feature_names, X_train_scaled = load_data()
    print(f"\nLoaded {len(df_raw)} samples, {len(feature_names)} features")
    print(f"Class balance: 0={int(np.sum(y==0))}  1={int(np.sum(y==1))}")

    groups, group_labels = derive_groups(df_raw)
    print("\nGroup sizes:")
    for attr in ATTRS:
        if groups[attr] is None:
            print(f"  {attr}: column not found, skipping")
            continue
        g = groups[attr]
        l0, l1 = group_labels[attr]
        print(f"  {attr:<10} {l0}: {int(np.sum(g==0))}   {l1}: {int(np.sum(g==1))}")

    bundle = ModelBundle(X_scaled, X_train_scaled, feature_names)

    # ---- per-sample metric loop ----
    # ---- Match per-sample XAI: 300 class-0 + 100 class-1 from TEST set ----
    from sklearn.model_selection import train_test_split as _tts
    idx_all = np.arange(len(y))
    _, idx_test = _tts(idx_all, test_size=0.2,
                       random_state=RANDOM_STATE, stratify=y)
    y_test_only = pd.Series(y, index=idx_all).iloc[idx_test]
    c0_pool = y_test_only[y_test_only == 0].index.tolist()
    c1_pool = y_test_only[y_test_only == 1].index.tolist()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(c0_pool)
    rng.shuffle(c1_pool)
    sample_indices = c0_pool[:300] + c1_pool[:100]
    print(f"\nSampling {len(sample_indices)} from TEST set "
          f"(300 surv + 100 deaths) — same as per-sample XAI run")
    print("(SHAP is slowest; expect ~30-60s per sample.)\n")
    results = []
    for k, i in enumerate(sample_indices):
        if k % 10 == 0:
            print(f"  sample {k+1}/{len(sample_indices)} (row index {i})",
                  flush=True)
        evaluate_sample(i, X_scaled[i], bundle, results)

    raw_df = pd.DataFrame(results)
    raw_df.to_csv(OUT_RAW, index=False)
    print(f"\nWrote per-sample raw metrics: {OUT_RAW}")

    # ---- disparity tests ----
    print("\nComputing disparity tests (Mann-Whitney U)...")
    summary_df = compute_disparities(raw_df, groups, group_labels)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote summary: {OUT_SUMMARY}")

    # ---- pretty table ----
    with open(OUT_TXT, "w") as fh:
        fh.write("MIMIC-IV XAI Disparity Analysis (Dai et al. 2022 framework)\n")
        fh.write("Black-box: XGBoost (XGBoost_run0.joblib)\n")
        fh.write("Glassbox:  EBM     (EBM_run0.joblib)\n")
        fh.write(f"Samples:   {len(sample_indices)} from test set (300 surv + 100 deaths, same as per-sample XAI)\n")
        fh.write(f"Test:      Mann-Whitney U (two-sided)\n")
        write_paper_style_table(summary_df, fh)
    print(f"Wrote paper-style table: {OUT_TXT}")

    # Also print to stdout
    print("\n" + "=" * 78)
    with open(OUT_TXT) as fh:
        print(fh.read())


if __name__ == "__main__":
    main()

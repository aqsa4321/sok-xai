#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2: Calibration (Brier score) and Model Complexity (parameter counts)
for the 6 models used in the XAI evaluation:
  - LSTM on Breast Cancer
  - SVM on Parkinson's
  - XGBoost on MIMIC-IV (full)
  - EBM on all three (Breast, Parkinson, MIMIC-full)

Reads saved model files from ~/sok/models_*/ and their saved X_test/y_test
arrays -- no retraining required.

Outputs to ~/sok/:
  - calibration_complexity.csv
  - calibration_complexity.txt   (paper-style table + methodology notes)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss

# Suppress TF chatter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "calibration_complexity.csv")
OUT_TXT = os.path.join(HERE, "calibration_complexity.txt")


# ---------------------------------------------------------------------------
# Model registry: (name, dataset, models_dir, model_filename, kind)
#   kind: "keras" (LSTM .h5), "sklearn" (SVM/XGBoost .joblib), "ebm" (EBM .joblib)
# We use *_run0 for each -- the disparity/XAI analyses used run0 too.
# ---------------------------------------------------------------------------
MODELS = [
    ("LSTM",    "Breast Cancer",   "models_breast",      "LSTM_run0.h5",       "keras"),
    ("SVM",     "Parkinson's",     "models_parkinson",   "SVM_run0.joblib",    "sklearn"),
    ("XGBoost", "MIMIC-IV (full)", "models_mimic_full",  "XGBoost_run0.joblib","sklearn"),
    ("EBM",     "Breast Cancer",   "models_breast",      "EBM_run0.joblib",    "ebm"),
    ("EBM",     "Parkinson's",     "models_parkinson",   "EBM_run0.joblib",    "ebm"),
    ("EBM",     "MIMIC-IV (full)", "models_mimic_full",  "EBM_run0.joblib",    "ebm"),
]


# ---------------------------------------------------------------------------
# Complexity: total trainable parameter count
# ---------------------------------------------------------------------------
def count_keras_params(model):
    """Count trainable parameters in a Keras model.
    Uses Keras's built-in count_params(); equivalent to
    sum(np.prod(v.shape) for v in model.trainable_variables)."""
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))


def count_svm_params(model):
    """For an RBF-kernel SVM, a natural analog to 'trainable parameters' is
    the number of learned real-valued coefficients: the dual coefficients
    (one per support vector per class-pair) plus the intercept.

    For binary classification with n_SV support vectors:
        params = n_SV (dual_coef) + n_SV * n_features (support_vectors_) + 1 (intercept)
    We report `dual_coef + intercept` as the standard analog, and also
    `n_SV` and `n_SV * n_features` for transparency."""
    dual = np.asarray(model.dual_coef_).size
    intercept = np.asarray(model.intercept_).size
    n_sv = int(np.asarray(model.support_vectors_).shape[0])
    n_features = int(np.asarray(model.support_vectors_).shape[1])
    total = dual + intercept
    return {
        "n_support_vectors":       n_sv,
        "n_features":              n_features,
        "dual_coef":               dual,
        "intercept":               intercept,
        "trainable_params":        total,
    }


def count_xgboost_params(model):
    """XGBoost has no single 'parameter count' in the deep-learning sense.
    Standard analogs used in the literature:
      - total number of tree leaves across all boosting rounds
      - total number of tree nodes (splits + leaves)
    XGBoost's Booster.trees_to_dataframe() exposes each node, so we count
    both."""
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    df = booster.trees_to_dataframe()
    n_trees = int(df["Tree"].nunique())
    n_leaves = int((df["Feature"] == "Leaf").sum())
    n_splits = int((df["Feature"] != "Leaf").sum())
    n_nodes = n_leaves + n_splits
    return {
        "n_trees":            n_trees,
        "n_leaves":           n_leaves,
        "n_split_nodes":      n_splits,
        "n_total_nodes":      n_nodes,
        "trainable_params":   n_leaves,   # leaves carry the learned real-valued weights
    }


def count_ebm_params(model):
    """EBM is a Generalized Additive Model: for each feature (and each pairwise
    interaction), it learns a piecewise-constant shape function on binned inputs.
    A natural 'trainable parameter' count is the total number of bins across
    all shape functions plus the intercept.

    interpret.glassbox.ExplainableBoostingClassifier exposes:
      - term_features_        (list of feature indices per shape function)
      - term_scores_          (list of arrays; each array is bin-scores for that term)
      - intercept_

    Total params = sum(len(scores)) over all terms + 1 (intercept)."""
    total_bins = int(sum(np.asarray(s).size for s in model.term_scores_))
    n_terms = int(len(model.term_features_))
    n_mains = int(sum(1 for t in model.term_features_ if len(t) == 1))
    n_pairs = int(sum(1 for t in model.term_features_ if len(t) == 2))
    intercept = int(np.asarray(model.intercept_).size)
    return {
        "n_terms":            n_terms,
        "n_main_effects":     n_mains,
        "n_pairwise_terms":   n_pairs,
        "total_bin_scores":   total_bins,
        "intercept":          intercept,
        "trainable_params":   total_bins + intercept,
    }


# ---------------------------------------------------------------------------
# Load test data for a given models_dir
# ---------------------------------------------------------------------------
def load_test(models_dir):
    root = os.path.join(HERE, models_dir)
    X = np.load(os.path.join(root, "X_test.npy"))
    y = np.load(os.path.join(root, "y_test.npy"))
    return X, y


# ---------------------------------------------------------------------------
# Compute Brier + parameter count for one row
# ---------------------------------------------------------------------------
def analyze(name, dataset, models_dir, model_file, kind):
    print(f"\n>>> {name} on {dataset}  ({models_dir}/{model_file})")
    root = os.path.join(HERE, models_dir)
    X_test, y_test = load_test(models_dir)

    if kind == "keras":
        model = load_model(os.path.join(root, model_file), compile=False)
        # LSTM was trained on (samples, 1 timestep, n_features)
        X_in = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        y_prob = model.predict(X_in, verbose=0).ravel()
        complexity = {"trainable_params": count_keras_params(model),
                      "method": "keras trainable_variables (sum of shapes)"}
    elif kind == "sklearn":
        model = joblib.load(os.path.join(root, model_file))
        y_prob = model.predict_proba(X_test)[:, 1]
        if name == "SVM":
            info = count_svm_params(model)
            complexity = {**info, "method": "n_SV * (dual_coef entries) + intercept"}
        elif name == "XGBoost":
            info = count_xgboost_params(model)
            complexity = {**info,
                          "method": "n_leaves via booster.trees_to_dataframe()"}
        else:
            complexity = {"trainable_params": np.nan, "method": "n/a"}
    elif kind == "ebm":
        model = joblib.load(os.path.join(root, model_file))
        y_prob = model.predict_proba(X_test)[:, 1]
        info = count_ebm_params(model)
        complexity = {**info,
                      "method": "sum(len(term_scores_)) over all terms + intercept"}
    else:
        raise ValueError(f"Unknown kind: {kind}")

    brier = float(brier_score_loss(y_test, y_prob))
    print(f"    Brier score:      {brier:.4f}")
    print(f"    Trainable params: {complexity.get('trainable_params')}")
    return brier, complexity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rows = []
    for name, dataset, models_dir, model_file, kind in MODELS:
        brier, complexity = analyze(name, dataset, models_dir, model_file, kind)
        rows.append({
            "model":            name,
            "dataset":          dataset,
            "brier_score":      brier,
            "trainable_params": complexity.get("trainable_params"),
            "counting_method":  complexity.get("method"),
            **{k: v for k, v in complexity.items()
               if k not in ("trainable_params", "method")},
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")

    # Paper-style table
    with open(OUT_TXT, "w") as f:
        f.write("Task 2: Calibration (Brier score) and Model Complexity.\n")
        f.write("="*80 + "\n\n")

        f.write("Test-set Brier scores (lower is better; 0 = perfect calibration,\n")
        f.write("0.25 = random for balanced binary).\n\n")
        header = f"{'Model':<10} {'Dataset':<20} {'Brier':>10} {'Trainable params':>20}"
        f.write(header + "\n" + "-"*len(header) + "\n")
        for r in rows:
            f.write(f"{r['model']:<10} {r['dataset']:<20} "
                    f"{r['brier_score']:>10.4f} "
                    f"{str(r['trainable_params']):>20}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Methodology for parameter counts (Part B)\n")
        f.write("="*80 + "\n\n")
        f.write("Different model families use different notions of 'parameter'; we report\n")
        f.write("a natural analog for each family and note the counting method explicitly.\n\n")
        f.write("LSTM (Keras): count of scalars in model.trainable_variables\n")
        f.write("  -> sum(np.prod(v.shape) for v in model.trainable_variables)\n")
        f.write("  Equivalent to model.count_params() when all layers are trainable.\n\n")
        f.write("SVM (sklearn.svm.SVC, RBF kernel): dual-coef entries + intercept\n")
        f.write("  -> model.dual_coef_.size + model.intercept_.size\n")
        f.write("  We also report n_support_vectors and n_features for context.\n\n")
        f.write("XGBoost (xgboost.XGBClassifier): number of tree leaves\n")
        f.write("  -> (model.get_booster().trees_to_dataframe()['Feature']=='Leaf').sum()\n")
        f.write("  Each leaf holds one learned real-valued output. We also report\n")
        f.write("  n_trees, n_split_nodes, and n_total_nodes for context.\n\n")
        f.write("EBM (interpret.glassbox.ExplainableBoostingClassifier): total bin scores\n")
        f.write("  -> sum(np.asarray(s).size for s in model.term_scores_) + intercept\n")
        f.write("  Each shape function is a piecewise-constant table over binned input.\n")
        f.write("  We also report the number of main-effect terms and pairwise terms.\n\n")
        f.write("Cross-family caveat: an LSTM's 50,000 float weights are not directly\n")
        f.write("comparable to an XGBoost's 500 tree leaves or an EBM's 3,000 bin scores.\n")
        f.write("These numbers characterise the size of the hypothesis space within each\n")
        f.write("family; they should not be interpreted as an equivalent 'complexity' unit.\n")
    print(f"Saved: {OUT_TXT}")

    # Console summary
    print("\n" + "="*70)
    print(f"{'Model':<10} {'Dataset':<20} {'Brier':>10} {'Params':>15}")
    print("-"*70)
    for r in rows:
        print(f"{r['model']:<10} {r['dataset']:<20} "
              f"{r['brier_score']:>10.4f} "
              f"{str(r['trainable_params']):>15}")
    print("="*70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMIC-IV demo: in-hospital mortality prediction.
Same structure as Breast Cancer / Parkinson's pipeline.
  - 11 models x 5 runs each, mean +/- std reported
  - All trained models saved to /home/aqsa/sok/models_mimic/ for later XAI use
  - Scaler and split indices also saved for reproducible XAI
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classical ML
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

# DL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_PATH    = "/home/aqsa/sok/mimic_full.csv"
RESULTS_PATH = "/home/aqsa/sok/mimic_full_results.txt"
MODELS_DIR   = "/home/aqsa/sok/models_mimic_full"
TARGET       = "mortality"
N_RUNS       = 5

os.makedirs(MODELS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(save_artifacts=False):
    """
    Load mimic.csv, scale, and split. Same pattern as breast.csv pipeline,
    plus stratify=y because the positive class is small (9/88).
    If save_artifacts=True, save scaler + feature names + split indices once
    so XAI later can reproduce the exact same X_train/X_test the models saw.
    """
    data = pd.read_csv(DATA_PATH)
    feature_names = [c for c in data.columns if c != TARGET]
    X = data[feature_names].values.astype(np.float32)
    y = data[TARGET].values.astype(np.int64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Keep indices so we can recover which row is which after splitting
    idx = np.arange(len(X_scaled))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_scaled, y, idx, test_size=0.2, random_state=42, stratify=y
    )

    if save_artifacts:
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
        with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
            json.dump(feature_names, f, indent=2)
        np.save(os.path.join(MODELS_DIR, "idx_train.npy"), idx_train)
        np.save(os.path.join(MODELS_DIR, "idx_test.npy"),  idx_test)
        np.save(os.path.join(MODELS_DIR, "X_train.npy"),   X_train)
        np.save(os.path.join(MODELS_DIR, "X_test.npy"),    X_test)
        np.save(os.path.join(MODELS_DIR, "y_train.npy"),   y_train)
        np.save(os.path.join(MODELS_DIR, "y_test.npy"),    y_test)
        print(f"  [saved] scaler, feature_names, split indices, X/y arrays -> {MODELS_DIR}")

    return X_train, X_test, y_train, y_test


def compute_metrics(y_true, y_pred):
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average="weighted", zero_division=0),
        recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    )


# ---------------------------------------------------------------------------
# Deep learning models -- each saves to {MODELS_DIR}/{name}_run{run_idx}.h5
# ---------------------------------------------------------------------------
def _dl_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=0),
    ]


def gru_mine(run_idx=0):
    X_train, X_test, y_train, y_test = load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test  = X_test.reshape(X_test.shape[0],   1, X_test.shape[1])

    model = Sequential()
    model.add(GRU(64, activation="relu",
                  input_shape=(X_train.shape[1], X_train.shape[2]),
                  return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=_dl_callbacks(), verbose=0)

    save_path = os.path.join(MODELS_DIR, f"GRU_run{run_idx}.h5")
    model.save(save_path)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    return compute_metrics(y_test, y_pred)


def lstm_mine(run_idx=0):
    X_train, X_test, y_train, y_test = load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test  = X_test.reshape(X_test.shape[0],   1, X_test.shape[1])

    model = Sequential()
    model.add(LSTM(128, activation="tanh",
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=_dl_callbacks(), verbose=0)

    save_path = os.path.join(MODELS_DIR, f"LSTM_run{run_idx}.h5")
    model.save(save_path)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    return compute_metrics(y_test, y_pred)


def cnn_mine(run_idx=0):
    X_train, X_test, y_train, y_test = load_data()
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],   X_test.shape[1],  1)

    model = Sequential()
    model.add(Conv1D(128, kernel_size=3, activation="relu",
                     input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=100, batch_size=32,
              validation_split=0.2, callbacks=_dl_callbacks(), verbose=0)

    save_path = os.path.join(MODELS_DIR, f"CNN_run{run_idx}.h5")
    model.save(save_path)

    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    return compute_metrics(y_test, y_pred)


# ---------------------------------------------------------------------------
# Classical ML -- each saves to {MODELS_DIR}/{name}_run{run_idx}.joblib
# ---------------------------------------------------------------------------
def _ml_runner(make_model, name):
    def _run(run_idx=0):
        X_train, X_test, y_train, y_test = load_data()
        model = make_model()
        model.fit(X_train, y_train)
        save_path = os.path.join(MODELS_DIR, f"{name}_run{run_idx}.joblib")
        joblib.dump(model, save_path)
        y_pred = model.predict(X_test)
        return compute_metrics(y_test, y_pred)
    return _run


knn_mine      = _ml_runner(lambda: KNeighborsClassifier(n_neighbors=5),                              "KNN")
svm_mine      = _ml_runner(lambda: SVC(kernel="rbf", C=1.0, gamma="scale",
                                       probability=True, random_state=42),                            "SVM")
rf_mine       = _ml_runner(lambda: RandomForestClassifier(n_estimators=100, random_state=42),        "RF")
dt_mine       = _ml_runner(lambda: DecisionTreeClassifier(random_state=42),                          "DT")
adaboost_mine = _ml_runner(lambda: AdaBoostClassifier(n_estimators=100, random_state=42),            "AdaBoost")
gbm_mine      = _ml_runner(lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),    "GBM")
xgb_mine      = _ml_runner(lambda: XGBClassifier(n_estimators=100, use_label_encoder=False,
                                                 eval_metric="logloss", random_state=42,
                                                 verbosity=0),                                        "XGBoost")
lgbm_mine     = _ml_runner(lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),    "LightGBM")

ebm_mine = _ml_runner(lambda: ExplainableBoostingClassifier(random_state=42), "EBM")
# ---------------------------------------------------------------------------
# 5-run averaging
# ---------------------------------------------------------------------------
def run_model_multiple_times(model_function, num_runs=N_RUNS, name=""):
    print(f"\n>>> Running {name} {num_runs} times...")
    accs, precs, recs, f1s = [], [], [], []
    for i in range(num_runs):
        tf.random.set_seed(i * 42)
        np.random.seed(i * 42)
        a, p, r, f = model_function(run_idx=i)
        accs.append(a); precs.append(p); recs.append(r); f1s.append(f)
        print(f"  Run {i+1}/{num_runs}: Acc={a:.4f}  Prec={p:.4f}  Rec={r:.4f}  F1={f:.4f}")

    res = {
        "name": name,
        "accuracy_mean":  np.mean(accs),  "accuracy_std":  np.std(accs),
        "precision_mean": np.mean(precs), "precision_std": np.std(precs),
        "recall_mean":    np.mean(recs),  "recall_std":    np.std(recs),
        "f1_mean":        np.mean(f1s),   "f1_std":        np.std(f1s),
    }
    print(f"  ---> {name}: "
          f"Acc {res['accuracy_mean']:.4f}±{res['accuracy_std']:.4f} | "
          f"Prec {res['precision_mean']:.4f}±{res['precision_std']:.4f} | "
          f"Rec {res['recall_mean']:.4f}±{res['recall_std']:.4f} | "
          f"F1 {res['f1_mean']:.4f}±{res['f1_std']:.4f}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Save scaler / split / arrays once up front
    print("Saving preprocessing artifacts for later XAI use...")
    _ = load_data(save_artifacts=True)

    models = [
        ("GRU",      gru_mine),
        ("LSTM",     lstm_mine),
        ("CNN",      cnn_mine),
        ("KNN",      knn_mine),
        ("SVM",      svm_mine),
        ("RF",       rf_mine),
        ("DT",       dt_mine),
        ("AdaBoost", adaboost_mine),
        ("GBM",      gbm_mine),
        ("XGBoost",  xgb_mine),
        ("LightGBM", lgbm_mine),
        ("EBM", ebm_mine),
    ]

    all_results = []
    for name, fn in models:
        try:
            res = run_model_multiple_times(fn, num_runs=N_RUNS, name=name)
            all_results.append(res)
        except Exception as e:
            print(f"  !! {name} failed: {e}")
            all_results.append({
                "name": name,
                "accuracy_mean":  np.nan, "accuracy_std":  np.nan,
                "precision_mean": np.nan, "precision_std": np.nan,
                "recall_mean":    np.nan, "recall_std":    np.nan,
                "f1_mean":        np.nan, "f1_std":        np.nan,
            })

    # Summary table
    header = f"{'Model':<10} {'Accuracy':<22} {'Precision':<22} {'Recall':<22} {'F1-Score':<22}"
    sep = "=" * 100
    print("\n" + sep)
    print(header)
    print(sep)
    with open(RESULTS_PATH, "w") as f:
        f.write(header + "\n")
        f.write(sep + "\n")
        for r in all_results:
            line = (f"{r['name']:<10} "
                    f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}     "
                    f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}     "
                    f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}     "
                    f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}")
            print(line)
            f.write(line + "\n")

    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"All trained models saved to {MODELS_DIR}/")

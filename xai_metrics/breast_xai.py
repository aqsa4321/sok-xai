#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Breast Cancer XAI evaluation.
  - LSTM (black-box, loaded from models_breast/LSTM_run0.h5)
  - EBM  (glassbox, loaded from models_breast/EBM_run0.joblib)
  - Three XAI methods: SHAP (KernelExplainer on LSTM), LIME (on LSTM), EBM
  - Nine metrics per sample, mean +/- std reported, split by class

Output: /home/aqsa/sok/breast_xai_results.txt

echo "--- MIMIC ---"; tail -3 /home/aqsa/sok/mimic_xai_v2_output.txt
echo "--- BREAST ---"; tail -3 /home/aqsa/sok/breast_xai_v1_output.txt
echo "--- PARKINSON ---"; tail -3 /home/aqsa/sok/parkinson_xai_v1_output.txt

"""

# ---------------------------------------------------------------------------
# Silence noisy third-party output BEFORE importing libraries
# ---------------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('shap').setLevel(logging.ERROR)
logging.getLogger('lime').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

import math
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier

import tensorflow as tf
from tensorflow.keras.models import load_model


# ===========================================================================
# Metric functions -- identical to Parkinson XAI (copied verbatim)
# ===========================================================================

def compute_infidelity(model, x_sample, explanation_values, explanation_function,
                      n_samples=20, noise_std=0.1, epsilon_min=1e-8):
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]
    infidelity_scores = []
    for _ in range(n_samples):
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample - perturbation
        perturbed_pred = model(x_perturbed.reshape(1, -1))[0]
        if len(perturbed_pred.shape) > 0:
            perturbed_pred = perturbed_pred[1] if len(perturbed_pred) > 1 else perturbed_pred[0]
        explanation_approx = np.dot(perturbation, explanation_values)
        model_diff = original_pred - perturbed_pred
        infidelity_score = (explanation_approx - model_diff) ** 2
        infidelity_scores.append(infidelity_score)
    return float(np.mean(infidelity_scores))


def compute_sensitivity(model, x_sample, explanation_function,
                       n_samples=20, radius=0.1, epsilon_min=1e-8):
    x_sample = np.array(x_sample).flatten()
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    max_sensitivity = 0.0
    for _ in range(n_samples):
        perturbation = np.random.normal(0, 1, x_sample.shape)
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > epsilon_min:
            random_radius = radius * np.random.random()
            perturbation = (perturbation / perturbation_norm) * random_radius
        else:
            perturbation = np.zeros_like(x_sample)
        y = x_sample + perturbation
        try:
            perturbed_explanation = np.array(explanation_function(y)).flatten()
            sensitivity = np.linalg.norm(perturbed_explanation - original_explanation)
            max_sensitivity = max(max_sensitivity, sensitivity)
        except Exception:
            continue
    return float(max_sensitivity)


def compute_faithfulness(model, x_sample, explanation_function,
                         subset_size=5, n_subsets=100, baseline_type='zero',
                         training_mean=None):
    x_sample = np.array(x_sample).flatten()
    d = len(x_sample)
    subset_size = min(subset_size, d)

    def _class1_proba(x):
        out = np.array(model(x.reshape(1, -1)))
        if out.ndim == 2 and out.shape[1] == 2:
            return float(out[0, 1])
        return float(out.flatten()[-1])

    explanation = np.array(explanation_function(x_sample)).flatten()
    original_output = _class1_proba(x_sample)
    attribution_sums = []
    output_differences = []
    for _ in range(n_subsets):
        subset_indices = np.random.choice(d, size=subset_size, replace=False)
        x_baseline = x_sample.copy()
        if baseline_type == 'zero':
            x_baseline[subset_indices] = 0.0
        else:
            x_baseline[subset_indices] = training_mean[subset_indices]
        baseline_output = _class1_proba(x_baseline)
        attribution_sums.append(np.sum(explanation[subset_indices]))
        output_differences.append(original_output - baseline_output)
    if len(attribution_sums) < 2 or np.var(attribution_sums) == 0 or np.var(output_differences) == 0:
        return 0.0
    correlation = np.corrcoef(attribution_sums, output_differences)[0, 1]
    return 0.0 if np.isnan(correlation) else float(correlation)


def compute_complexity(explanation_values):
    abs_attributions = np.abs(np.array(explanation_values).flatten())
    total_attribution = np.sum(abs_attributions)
    if total_attribution == 0:
        return 0.0
    prob_distribution = abs_attributions / total_attribution
    prob_distribution = prob_distribution[prob_distribution > 0]
    return float(-np.sum(prob_distribution * np.log(prob_distribution)))


def compute_relative_input_stability(model, x_sample, explanation_function,
                                     n_samples=20, noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    x_sample = np.array(x_sample).flatten()
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])
    valid_perturbations = []
    attempts = 0
    while len(valid_perturbations) < n_samples and attempts < n_samples * 10:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation
        if np.argmax(model(x_perturbed.reshape(1, -1))[0]) == original_pred:
            valid_perturbations.append(x_perturbed)
        attempts += 1
    if not valid_perturbations:
        return 0.0
    max_instability = 0.0
    for x_perturbed in valid_perturbations:
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()
        explanation_diff = original_explanation - perturbed_explanation
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                     epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp
        input_diff = x_sample - x_perturbed
        safe_original_input = np.where(np.abs(x_sample) < epsilon_min,
                                       epsilon_min, x_sample)
        input_relative = input_diff / safe_original_input
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        input_norm = np.linalg.norm(input_relative, ord=p_norm)
        max_instability = max(max_instability, explanation_norm / max(input_norm, epsilon_min))
    if max_instability <= 0:
        return 0.0
    return math.log(float(max_instability))


def compute_relative_output_stability(model_predict_proba, x_sample, explanation_function,
                                      n_samples=20, noise_std=0.05, epsilon_min=1e-6, p_norm=2):
    x_sample = np.array(x_sample).flatten()
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_logits = model_predict_proba(x_sample.reshape(1, -1))[0]
    original_pred = np.argmax(original_logits)
    max_instability = 0.0
    valid_count = 0
    for _ in range(n_samples * 10):
        if valid_count >= n_samples:
            break
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation
        perturbed_logits = model_predict_proba(x_perturbed.reshape(1, -1))[0]
        if np.argmax(perturbed_logits) != original_pred:
            continue
        valid_count += 1
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()
        explanation_diff_norm = np.linalg.norm(original_explanation - perturbed_explanation, ord=p_norm)
        explanation_orig_norm = np.linalg.norm(original_explanation, ord=p_norm)
        logit_diff_norm = np.linalg.norm(original_logits - perturbed_logits, ord=p_norm)
        explanation_relative_change = explanation_diff_norm / max(explanation_orig_norm, epsilon_min)
        max_instability = max(max_instability, explanation_relative_change / max(logit_diff_norm, epsilon_min))
    if max_instability <= 0:
        return 0.0
    return math.log(float(max_instability))


def compute_relative_representation_stability(model, x_sample, explanation_function,
                                              representation_layer, n_samples=50,
                                              noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    x_sample = np.array(x_sample).flatten()
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_representation = np.array(representation_layer(x_sample)).flatten()
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])
    valid_perturbations = []
    attempts = 0
    while len(valid_perturbations) < n_samples and attempts < n_samples * 10:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation
        if np.argmax(model(x_perturbed.reshape(1, -1))[0]) == original_pred:
            perturbed_representation = np.array(representation_layer(x_perturbed)).flatten()
            valid_perturbations.append((x_perturbed, perturbed_representation))
        attempts += 1
    if not valid_perturbations:
        return 0.0
    max_instability = 0.0
    for x_perturbed, perturbed_representation in valid_perturbations:
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()
        explanation_diff = original_explanation - perturbed_explanation
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                     epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp
        representation_diff = original_representation - perturbed_representation
        safe_original_rep = np.where(np.abs(original_representation) < epsilon_min,
                                     epsilon_min, original_representation)
        representation_relative = representation_diff / safe_original_rep
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        representation_norm = np.linalg.norm(representation_relative, ord=p_norm)
        max_instability = max(max_instability, explanation_norm / max(representation_norm, epsilon_min))
    if max_instability <= 0:
        return 0.0
    return math.log(float(max_instability))


def compute_completeness(model, x_sample, explanation_values, epsilon_min=1e-8):
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()
    baseline = np.zeros_like(x_sample)
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]
    baseline_pred = model(baseline.reshape(1, -1))[0]
    if len(baseline_pred.shape) > 0:
        baseline_pred = baseline_pred[1] if len(baseline_pred) > 1 else baseline_pred[0]
    attribution_sum = np.sum(explanation_values)
    model_diff = original_pred - baseline_pred
    return float(attribution_sum - model_diff)


def compute_model_parameter_randomization(model_predict_func, x_sample, explanation_function,
                                          trained_model=None, model_type='auto',
                                          randomization_type='full', similarity_metric='spearman_abs',
                                          n_samples=5, random_seed=42):
    from scipy.stats import spearmanr, pearsonr
    np.random.seed(random_seed)

    def _compute_similarity(exp1, exp2, metric):
        exp1, exp2 = np.array(exp1).flatten(), np.array(exp2).flatten()
        if len(exp1) != len(exp2):
            min_len = min(len(exp1), len(exp2))
            exp1, exp2 = exp1[:min_len], exp2[:min_len]
        if len(exp1) == 0 or (np.allclose(exp1, 0) and np.allclose(exp2, 0)):
            return 1.0
        try:
            if metric == 'spearman_abs':
                corr, _ = spearmanr(np.abs(exp1), np.abs(exp2))
            elif metric == 'spearman':
                corr, _ = spearmanr(exp1, exp2)
            elif metric == 'pearson':
                corr, _ = pearsonr(exp1, exp2)
            elif metric == 'cosine':
                exp1_norm = exp1 / (np.linalg.norm(exp1) + 1e-8)
                exp2_norm = exp2 / (np.linalg.norm(exp2) + 1e-8)
                corr = np.dot(exp1_norm, exp2_norm)
            else:
                return 0.0
            return 0.0 if np.isnan(corr) else abs(corr)
        except Exception:
            return 0.0

    def _randomize_tensorflow_model(model):
        try:
            random_model = tf.keras.models.clone_model(model)
            random_model.build(model.input_shape)
            for layer in random_model.layers:
                if hasattr(layer, 'kernel') and layer.kernel is not None:
                    layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.01))
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias.assign(tf.random.normal(layer.bias.shape, stddev=0.01))
                if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None:
                    layer.recurrent_kernel.assign(tf.random.normal(layer.recurrent_kernel.shape, stddev=0.01))
            return random_model
        except Exception as e:
            print(f"  [warn] TF randomization failed: {e}")
            return None

    def _randomize_sklearn_model(model):
        try:
            from sklearn.base import clone
            random_model = clone(model)
            n_features = len(x_sample)
            dummy_X = np.random.normal(0, 0.1, (100, n_features))
            dummy_y = np.random.randint(0, 2, 100)
            random_model.fit(dummy_X, dummy_y)
            if hasattr(random_model, 'coef_'):
                random_model.coef_ = np.random.normal(0, 0.01, random_model.coef_.shape)
            if hasattr(random_model, 'intercept_'):
                random_model.intercept_ = np.random.normal(0, 0.01, random_model.intercept_.shape)
            return random_model
        except Exception:
            return None

    try:
        original_explanation = np.array(explanation_function(x_sample)).flatten()
        if trained_model is None:
            similarities = []
            for _ in range(n_samples):
                random_explanation = np.random.normal(0, np.std(original_explanation),
                                                      original_explanation.shape)
                similarities.append(_compute_similarity(original_explanation, random_explanation,
                                                        similarity_metric))
            return float(np.mean(similarities))
        similarities = []
        for _ in range(n_samples):
            if model_type == 'tensorflow':
                random_model = _randomize_tensorflow_model(trained_model)
            else:
                random_model = _randomize_sklearn_model(trained_model)
            if random_model is not None:
                try:
                    random_explanation = explanation_function(x_sample, model_override=random_model)
                except TypeError:
                    random_explanation = np.random.normal(0, np.std(original_explanation),
                                                          original_explanation.shape)
                random_explanation = np.array(random_explanation).flatten()
            else:
                random_explanation = np.random.normal(0, np.std(original_explanation),
                                                      original_explanation.shape)
            similarities.append(_compute_similarity(original_explanation, random_explanation,
                                                    similarity_metric))
        return float(np.mean(similarities)) if similarities else float('nan')
    except Exception as e:
        print(f"  [warn] MPR failed: {e}")
        return float('nan')


# ===========================================================================
# Breast Cancer XAI evaluator -- LSTM (black-box) + EBM (glassbox)
# ===========================================================================

class BreastXAIEvaluator:
    DATA_PATH    = "/home/aqsa/sok/breast.csv"
    MODELS_DIR   = "/home/aqsa/sok/models_breast"
    LSTM_PATH    = os.path.join(MODELS_DIR, "LSTM_run0.h5")
    EBM_PATH     = os.path.join(MODELS_DIR, "EBM_run0.joblib")
    RESULTS_PATH = "/home/aqsa/sok/breast_xai_results.txt"
    TARGET       = "diagnosis"

    def __init__(self, n_metric_samples=50, n_class0_samples=30, n_class1_samples=30,
                 random_state=42):
        self.n_metric_samples = n_metric_samples
        self.n_class0_samples = n_class0_samples
        self.n_class1_samples = n_class1_samples
        self.random_state = random_state

        self._load_data()
        self._load_models()
        self._select_samples()

    # ----------------------------- setup ---------------------------------
    def _load_data(self):
        print("Loading and preparing data...")
        df = pd.read_csv(self.DATA_PATH)
        drop_cols = [c for c in ['id', 'Unnamed: 32'] if c in df.columns]
        df = df.drop(columns=drop_cols)
        df[self.TARGET] = LabelEncoder().fit_transform(df[self.TARGET])
        X = df.drop(columns=[self.TARGET])
        y = df[self.TARGET].astype(int)

        self.feature_names = X.columns.tolist()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X_full = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        self.y_full = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2,
            random_state=self.random_state, stratify=self.y_full,
        )
        print(f"  Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Class counts (full): {dict(self.y_full.value_counts())}")
        print(f"  Train: {self.X_train.shape}, Test: {self.X_test.shape}")

    def _load_models(self):
        print("\nLoading LSTM and EBM...")
        # LSTM
        self.lstm_model = load_model(self.LSTM_PATH, compile=False)
        # quick perf sanity check
        X_test_3d = self.X_test.values.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])
        lstm_pred = (self.lstm_model.predict(X_test_3d, verbose=0) > 0.5).astype(int).flatten()
        print(f"  LSTM loaded | "
              f"acc={accuracy_score(self.y_test, lstm_pred):.4f} "
              f"prec={precision_score(self.y_test, lstm_pred, average='weighted', zero_division=0):.4f} "
              f"rec={recall_score(self.y_test, lstm_pred, average='weighted', zero_division=0):.4f} "
              f"f1={f1_score(self.y_test, lstm_pred, average='weighted', zero_division=0):.4f}")

        # EBM
        self.ebm_model = joblib.load(self.EBM_PATH)
        ebm_pred = self.ebm_model.predict(self.X_test)
        print(f"  EBM  loaded | "
              f"acc={accuracy_score(self.y_test, ebm_pred):.4f} "
              f"prec={precision_score(self.y_test, ebm_pred, average='weighted', zero_division=0):.4f} "
              f"rec={recall_score(self.y_test, ebm_pred, average='weighted', zero_division=0):.4f} "
              f"f1={f1_score(self.y_test, ebm_pred, average='weighted', zero_division=0):.4f}")

    def _select_samples(self):
        self.class0_indices = self.y_full[self.y_full == 0].index[:self.n_class0_samples].tolist()
        self.class1_indices = self.y_full[self.y_full == 1].index[:self.n_class1_samples].tolist()
        print(f"\n  Selected {len(self.class0_indices)} class-0 samples (benign)")
        print(f"  Selected {len(self.class1_indices)} class-1 samples (malignant)")

    # ---------------------- prediction wrappers --------------------------
    def lstm_predict_proba(self, X, model_override=None):
        """LSTM wrapper that returns (N, 2) probabilities like sklearn."""
        model = model_override if model_override is not None else self.lstm_model
        if hasattr(X, 'values'):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X3d = X.reshape(X.shape[0], 1, X.shape[1])
        p1 = model.predict(X3d, verbose=0).flatten()
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        return np.column_stack([1 - p1, p1])

    def ebm_predict_proba(self, X):
        if hasattr(X, 'values'):
            X_arr = X.values
        else:
            X_arr = X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_df = pd.DataFrame(X_arr, columns=self.feature_names)
        return self.ebm_model.predict_proba(X_df)

    def get_lstm_representation(self, x_sample):
        """Pre-sigmoid logit of the class-1 probability (LSTM has no decision_function)."""
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)
        proba = self.lstm_predict_proba(x_sample)[:, 1]
        return np.log(proba / (1 - proba))

    def get_ebm_representation(self, x_sample):
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)
        p = self.ebm_model.predict_proba(x_sample)[:, 1]
        eps = 1e-7
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    # ----------------------- public evaluation API ------------------------
    def evaluate_shap(self, class_type='both'):
        print("\nEvaluating SHAP (KernelExplainer on LSTM)...")
        return self._evaluate_method(self._evaluate_shap_single_sample, class_type, 'SHAP')

    def evaluate_lime(self, class_type='both'):
        print("\nEvaluating LIME (on LSTM)...")
        return self._evaluate_method(self._evaluate_lime_single_sample, class_type, 'LIME')

    def evaluate_ebm(self, class_type='both'):
        print("\nEvaluating EBM...")
        return self._evaluate_method(self._evaluate_ebm_single_sample, class_type, 'EBM')

    def _evaluate_method(self, single_sample_fn, class_type, method_label):
        if class_type == 'class0':
            groups = [('class0', self.class0_indices)]
        elif class_type == 'class1':
            groups = [('class1', self.class1_indices)]
        else:
            groups = [('class0', self.class0_indices), ('class1', self.class1_indices)]

        results = {}
        for class_name, indices in groups:
            print(f"\n  Processing {class_name.upper()} samples for {method_label}...")
            class_results = []
            for i, idx in enumerate(indices):
                print(f"    {class_name} sample {i+1}/{len(indices)} (row {idx})", flush=True)
                try:
                    class_results.append(single_sample_fn(idx))
                except Exception as e:
                    print(f"      [error] {e}")
            results[class_name] = self._compute_average_metrics(class_results) if class_results else None
        return results

    # ----------------------- per-sample evaluation ------------------------
    def _evaluate_shap_single_sample(self, idx):
        x_sample = self.X_full.loc[idx].values  # already scaled

        bg_size = min(50, len(self.X_train))
        background = self.X_train.values[:bg_size]
        shap_explainer = shap.KernelExplainer(self.lstm_predict_proba, background)

        def get_shap_explanation(x, model_override=None):
            x = x.reshape(1, -1) if x.ndim == 1 else x
            if model_override is not None:
                # MPR: rebuild a KernelExplainer pointing at the random model
                def predict_with_override(X_):
                    return self.lstm_predict_proba(X_, model_override=model_override)
                tmp = shap.KernelExplainer(predict_with_override, background)
                sv = tmp.shap_values(x, nsamples=100, silent=True)
            else:
                sv = shap_explainer.shap_values(x, nsamples=100, silent=True)
            if isinstance(sv, list):
                return sv[1][0] if len(sv) > 1 else sv[0][0]
            arr = np.asarray(sv)
            if arr.ndim == 3:
                return arr[0, :, 1]
            if arr.ndim == 2:
                return arr[0]
            return arr.flatten()

        shap_explanation = np.array(get_shap_explanation(x_sample)).flatten()
        if len(shap_explanation) != len(self.feature_names):
            raise ValueError(f"SHAP shape mismatch: {len(shap_explanation)} vs {len(self.feature_names)}")

        return self._compute_all_metrics(
            predict_fn=self.lstm_predict_proba,
            x_sample=x_sample,
            explanation=shap_explanation,
            explain_fn=get_shap_explanation,
            representation_fn=self.get_lstm_representation,
            trained_model=self.lstm_model,
            model_type='tensorflow',
        )

    def _evaluate_lime_single_sample(self, idx):
        x_sample = self.X_full.loc[idx].values

        bg_size = min(50, len(self.X_train))
        lime_explainer = LimeTabularExplainer(
            self.X_train.values[:bg_size],
            feature_names=self.feature_names,
            class_names=["Benign", "Malignant"],
            discretize_continuous=True,
            random_state=self.random_state,
        )

        def get_lime_explanation(x, model_override=None):
            x = x.reshape(1, -1)[0] if x.ndim == 1 else x.flatten()
            if model_override is not None:
                def predict_fn(X_):
                    return self.lstm_predict_proba(X_, model_override=model_override)
            else:
                predict_fn = self.lstm_predict_proba
            explanation = lime_explainer.explain_instance(
                x, predict_fn,
                num_features=len(self.feature_names), num_samples=200,
            )
            explanation_dict = dict(explanation.as_list())
            attributions = []
            for feature_name in self.feature_names:
                weight = 0.0
                for lime_feature, lime_weight in explanation_dict.items():
                    if feature_name in lime_feature:
                        weight = lime_weight
                        break
                attributions.append(weight)
            return np.array(attributions)

        lime_explanation = get_lime_explanation(x_sample)

        return self._compute_all_metrics(
            predict_fn=self.lstm_predict_proba,
            x_sample=x_sample,
            explanation=lime_explanation,
            explain_fn=get_lime_explanation,
            representation_fn=self.get_lstm_representation,
            trained_model=self.lstm_model,
            model_type='tensorflow',
        )

    def _evaluate_ebm_single_sample(self, idx):
        x_sample = self.X_full.loc[idx].values

        def get_ebm_explanation(x, model_override=None):
            x = x.reshape(1, -1) if x.ndim == 1 else x
            x_df = pd.DataFrame(x, columns=self.feature_names)
            model = model_override if model_override is not None else self.ebm_model
            local_exp = model.explain_local(x_df)
            try:
                overall_exp = local_exp.data(0)
                if isinstance(overall_exp, dict) and 'scores' in overall_exp:
                    scores = np.array(overall_exp['scores'])
                    if len(scores) > len(self.feature_names):
                        scores = scores[:len(self.feature_names)]
                    return scores
            except Exception:
                pass
            scores = []
            for i in range(len(self.feature_names)):
                try:
                    feat_exp = local_exp.data(i)
                    if isinstance(feat_exp, dict) and 'scores' in feat_exp:
                        sd = feat_exp['scores']
                        score = float(np.mean(sd)) if isinstance(sd, (list, np.ndarray)) and len(sd) > 0 else float(sd)
                    else:
                        score = 0.0
                    scores.append(score)
                except Exception:
                    scores.append(0.0)
            return np.array(scores)

        ebm_explanation = get_ebm_explanation(x_sample)

        return self._compute_all_metrics(
            predict_fn=self.ebm_predict_proba,
            x_sample=x_sample,
            explanation=ebm_explanation,
            explain_fn=get_ebm_explanation,
            representation_fn=self.get_ebm_representation,
            trained_model=self.ebm_model,
            model_type='sklearn',
        )

    # ----------------------- metric aggregation ---------------------------
    def _compute_all_metrics(self, predict_fn, x_sample, explanation, explain_fn,
                             representation_fn, trained_model, model_type):
        n = self.n_metric_samples
        return {
            'infidelity':                    compute_infidelity(predict_fn, x_sample, explanation, explain_fn, n_samples=n),
            'complexity':                    compute_complexity(explanation),
            'sensitivity':                   compute_sensitivity(predict_fn, x_sample, explain_fn, n_samples=n),
            'ris':                           compute_relative_input_stability(predict_fn, x_sample, explain_fn, n_samples=n),
            'ros':                           compute_relative_output_stability(predict_fn, x_sample, explain_fn, n_samples=n),
            'rrs':                           compute_relative_representation_stability(predict_fn, x_sample, explain_fn, representation_fn, n_samples=n),
            'faithfulness':                  compute_faithfulness(predict_fn, x_sample, explain_fn),
            'completeness':                  compute_completeness(predict_fn, x_sample, explanation),
            'model_parameter_randomization': compute_model_parameter_randomization(predict_fn, x_sample, explain_fn, trained_model=trained_model, model_type=model_type, randomization_type='full', n_samples=n),
        }

    def _compute_average_metrics(self, results_list):
        if not results_list:
            return None
        metric_values = {metric: [] for metric in results_list[0].keys()}
        for result in results_list:
            for metric, value in result.items():
                if not np.isnan(value) and not np.isinf(value):
                    metric_values[metric].append(value)
        stats = {}
        for metric, values in metric_values.items():
            if values:
                stats[metric] = {
                    'mean': float(np.mean(values)),
                    'std':  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'n':    len(values),
                }
            else:
                stats[metric] = {'mean': np.nan, 'std': np.nan, 'n': 0}
        return stats

    # ----------------------- printing & saving ----------------------------
    def print_and_save_results(self, results, method_name, fh):
        header = f"\n{method_name.upper()} RESULTS:\n" + "=" * 60
        print(header); fh.write(header + "\n")
        for class_name, class_results in results.items():
            if class_results is None:
                line = f"\n{class_name.upper()}: No results"
                print(line); fh.write(line + "\n")
                continue
            block = [f"\n{class_name.upper()}:", "-" * 40]
            for metric, s in class_results.items():
                label = metric.replace('_', ' ').title()
                if np.isnan(s['mean']):
                    block.append(f"  {label}: NaN (n=0)")
                else:
                    block.append(f"  {label}: {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")
            for line in block:
                print(line); fh.write(line + "\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    evaluator = BreastXAIEvaluator(
        n_metric_samples=50,
        n_class0_samples=30,
        n_class1_samples=30,
    )

    with open(BreastXAIEvaluator.RESULTS_PATH, "w") as fh:
        fh.write("Breast Cancer XAI Evaluation Results\n")
        fh.write("Black-box: LSTM | Glassbox: EBM\n")
        fh.write("=" * 60 + "\n")

        shap_results = evaluator.evaluate_shap(class_type='both')
        evaluator.print_and_save_results(shap_results, "SHAP", fh)

        lime_results = evaluator.evaluate_lime(class_type='both')
        evaluator.print_and_save_results(lime_results, "LIME", fh)

        ebm_results = evaluator.evaluate_ebm(class_type='both')
        evaluator.print_and_save_results(ebm_results, "EBM", fh)

    print(f"\nResults saved to {BreastXAIEvaluator.RESULTS_PATH}")


if __name__ == "__main__":
    main()

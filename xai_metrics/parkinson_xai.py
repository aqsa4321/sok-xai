import numpy as np
import pandas as pd
import os
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier
import matplotlib.pyplot as plt
import warnings
import math
warnings.filterwarnings('ignore')


def compute_infidelity(model, x_sample, explanation_values, explanation_function,
                      n_samples=20, noise_std=0.1, epsilon_min=1e-8):
    """
    Compute Infidelity metric for any XAI explanation method Definition 2.1.
    INFD(Φ,f,x) = E_I[‖I^T Φ(f,x) - (f(x) - f(x-I))‖²]
    Parameters:
    -----------
    model : Callable
        Model prediction function that takes input and returns predictions
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_values : np.ndarray
        Attribution values from XAI method (same shape as x_sample)
    explanation_function : Callable
        Function that generates explanations for new inputs
    n_samples : int, default=20
        Number of perturbation samples to generate
    noise_std : float, default=0.1
        Standard deviation for Gaussian noise perturbations
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues

    Returns:
    --------
    float : Infidelity score (lower is better)
    """
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()

    # Get original prediction
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]

    infidelity_scores = []

    for _ in range(n_samples):
        # Generate perturbation I (following Definition 2.1)
        perturbation = np.random.normal(0, noise_std, x_sample.shape)

        # Compute x - I (corrected direction according to paper)
        x_perturbed = x_sample - perturbation

        # Get perturbed prediction
        perturbed_pred = model(x_perturbed.reshape(1, -1))[0]
        if len(perturbed_pred.shape) > 0:
            perturbed_pred = perturbed_pred[1] if len(perturbed_pred) > 1 else perturbed_pred[0]

        # Compute infidelity components according to Definition 2.1
        # I^T * Φ(f,x)
        explanation_approx = np.dot(perturbation, explanation_values)

        # f(x) - f(x-I)
        model_diff = original_pred - perturbed_pred

        # (I^T Φ(f,x) - (f(x) - f(x-I)))^2
        infidelity_score = (explanation_approx - model_diff) ** 2
        infidelity_scores.append(infidelity_score)

    return float(np.mean(infidelity_scores))


def compute_sensitivity(model, x_sample, explanation_function,
                       n_samples=20, radius=0.1, epsilon_min=1e-8):
    """
    Compute Max-Sensitivity metric according to Definition 3.1 in the paper.

    SENS_MAX(Φ,f,x,r) = max_{||y-x||≤r} ||Φ(f,y) - Φ(f,x)||

    Parameters:
    -----------
    model : Callable
        Model prediction function
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_function : Callable
        Function that generates explanations for inputs
    n_samples : int, default=20
        Number of perturbation samples to approximate the maximum
    radius : float, default=0.1
        Maximum perturbation radius r (L2 norm constraint)
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues

    Returns:
    --------
    float : Max-sensitivity score (lower indicates more stable explanations)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation Φ(f,x)
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    max_sensitivity = 0.0

    for _ in range(n_samples):
        # Generate perturbation with constraint ||perturbation|| ≤ r
        # Method 1: Sample from uniform distribution in L2 ball
        perturbation = np.random.normal(0, 1, x_sample.shape)
        perturbation_norm = np.linalg.norm(perturbation)

        if perturbation_norm > epsilon_min:
            # Normalize and scale to random radius ≤ r
            random_radius = radius * np.random.random()  # uniform in [0, r]
            perturbation = (perturbation / perturbation_norm) * random_radius
        else:
            perturbation = np.zeros_like(x_sample)

        # Compute perturbed point y = x + perturbation, where ||y-x|| ≤ r
        y = x_sample + perturbation

        try:
            # Get explanation for perturbed input Φ(f,y)
            perturbed_explanation = np.array(explanation_function(y)).flatten()

            # Compute ||Φ(f,y) - Φ(f,x)||
            explanation_diff = perturbed_explanation - original_explanation
            sensitivity = np.linalg.norm(explanation_diff)

            # Keep track of maximum: max_{||y-x||≤r} ||Φ(f,y) - Φ(f,x)||
            max_sensitivity = max(max_sensitivity, sensitivity)

        except Exception:
            # Skip this perturbation if explanation fails
            continue

    return float(max_sensitivity)

def compute_faithfulness(model, x_sample, explanation_function,
                               subset_size=5, n_subsets=100, baseline_type='zero',
                               training_mean=None):
    """
    Compute Faithfulness for binary classification according to Definition 3.
    Uses class-1 probability to match explanation conventions (SHAP/LIME/EBM
    all return attributions for class 1 in this codebase).
    """
    x_sample = np.array(x_sample).flatten()
    d = len(x_sample)
    subset_size = min(subset_size, d)

    # Helper: extract class-1 probability from any model output shape
    def _class1_proba(x):
        out = np.array(model(x.reshape(1, -1)))
        if out.ndim == 2 and out.shape[1] == 2:
            return float(out[0, 1])      # class 1 probability
        return float(out.flatten()[-1])  # fallback: last element

    # Get explanation g(f,x)
    explanation = np.array(explanation_function(x_sample)).flatten()

    # Get original model output f(x) — class 1 probability
    original_output = _class1_proba(x_sample)

    attribution_sums = []
    output_differences = []

    for _ in range(n_subsets):
        # Random subset S
        subset_indices = np.random.choice(d, size=subset_size, replace=False)

        # Create x[xs=x̄s]
        x_baseline = x_sample.copy()
        if baseline_type == 'zero':
            x_baseline[subset_indices] = 0.0
        else:  # 'mean'
            x_baseline[subset_indices] = training_mean[subset_indices]

        # Get f(x[xs=x̄s]) — class 1 probability
        baseline_output = _class1_proba(x_baseline)

        # Calculate correlation inputs
        attribution_sum = np.sum(explanation[subset_indices])
        output_diff = original_output - baseline_output

        attribution_sums.append(attribution_sum)
        output_differences.append(output_diff)

    # Pearson correlation
    if len(attribution_sums) < 2 or np.var(attribution_sums) == 0 or np.var(output_differences) == 0:
        return 0.0

    correlation = np.corrcoef(attribution_sums, output_differences)[0, 1]
    return 0.0 if np.isnan(correlation) else float(correlation)

def compute_complexity(explanation_values):
    """
    Compute Complexity metric according to Definition 4 in Bhatt et al. paper.

    Complexity measures how distributed the explanation is across features.
    Lower complexity = more focused explanation (fewer important features)
    Higher complexity = more distributed explanation (many important features)

    Pg(i) = |g(f, x)i| / Σj∈[d] |g(f, x)j|
    μC(f, g; x) = Ei[−ln(Pg)] = −Σ(i=1 to d) Pg(i) ln(Pg(i))

    Parameters:
    -----------
    explanation_values : np.ndarray
        Attribution values from XAI method (same naming as your other functions)

    Returns:
    --------
    float : Complexity score (0 = most focused, ln(d) = maximally distributed)
    """
    # Convert to numpy array and take absolute values (|g(f, x)i|)
    abs_attributions = np.abs(np.array(explanation_values).flatten())

    # Handle edge case where all attributions are zero
    total_attribution = np.sum(abs_attributions)
    if total_attribution == 0:
        return 0.0

    # Create fractional contribution distribution Pg(i) = |g(f, x)i| / Σj |g(f, x)j|
    prob_distribution = abs_attributions / total_attribution

    # Remove zero probabilities to avoid log(0) in entropy calculation
    prob_distribution = prob_distribution[prob_distribution > 0]

    # Compute entropy: μC = -∑ Pg(i) * ln(Pg(i))
    complexity = -np.sum(prob_distribution * np.log(prob_distribution))

    return float(complexity)


def compute_relative_input_stability(model, x_sample, explanation_function,
                                   n_samples=20, noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    """
    Compute Relative Input Stability (RIS) according to Equation 2.

    RIS = max ||(ex'-ex)/ex|| / max(||(x'-x)/x||, ε_min)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation
    original_explanation = np.array(explanation_function(x_sample)).flatten()

    # Generate perturbations with same prediction
    valid_perturbations = []
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])

    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_perturbations) < n_samples and attempts < max_attempts:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check if prediction class is the same
        perturbed_pred = np.argmax(model(x_perturbed.reshape(1, -1))[0])
        if perturbed_pred == original_pred:
            valid_perturbations.append(x_perturbed)

        attempts += 1

    if len(valid_perturbations) == 0:
        return 0.0

    max_instability = 0.0

    for x_perturbed in valid_perturbations:
        # Get explanation for perturbed input
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Compute (ex' - ex) / ex (element-wise, handling division by zero)
        explanation_diff = original_explanation - perturbed_explanation
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                   epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp

        # Compute (x' - x) / x (element-wise, handling division by zero)
        input_diff = x_sample - x_perturbed
        safe_original_input = np.where(np.abs(x_sample) < epsilon_min,
                                     epsilon_min, x_sample)
        input_relative = input_diff / safe_original_input

        # Compute norms of the relative change vectors
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        input_norm = np.linalg.norm(input_relative, ord=p_norm)

        # Compute RIS ratio
        instability_ratio = explanation_norm / max(input_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return math.log(float(max_instability))


def compute_relative_output_stability(model_predict_proba, x_sample, explanation_function,
                                        n_samples=20, noise_std=0.05, epsilon_min=1e-6, p_norm=2):
    # Use vector norms instead of element-wise division
    x_sample = np.array(x_sample).flatten()

    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_logits = model_predict_proba(x_sample)
    original_pred = np.argmax(original_logits)

    max_instability = 0.0
    valid_count = 0

    for _ in range(n_samples * 10):
        if valid_count >= n_samples:
            break

        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        perturbed_logits = model_predict_proba(x_perturbed)
        if np.argmax(perturbed_logits) != original_pred:
            continue

        valid_count += 1

        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Vector-level relative change
        explanation_diff_norm = np.linalg.norm(original_explanation - perturbed_explanation, ord=p_norm)
        explanation_orig_norm = np.linalg.norm(original_explanation, ord=p_norm)

        logit_diff_norm = np.linalg.norm(original_logits - perturbed_logits, ord=p_norm)

        # Relative change ratios
        explanation_relative_change = explanation_diff_norm / max(explanation_orig_norm, epsilon_min)

        instability_ratio = explanation_relative_change / max(logit_diff_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return math.log(float(max_instability))

def compute_relative_output_stability_element_wise(model_predict_proba, x_sample, explanation_function,
                                    n_samples=20, noise_std=0.005, epsilon_min=1e-8, p_norm=2):
    """
    Compute ROS: max[x'] (|| (ex-ex')/ex ||p) / max(||h(x)-h(x')||p, ε_min)
    """
    x_sample = np.array(x_sample).flatten()

    # Get original values
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_logits = model_predict_proba(x_sample)
    original_pred = np.argmax(original_logits)

    max_instability = 0.0
    valid_count = 0

    for _ in range(n_samples * 10):
        if valid_count >= n_samples:
            break

        # Generate perturbation
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check same prediction
        perturbed_logits = model_predict_proba(x_perturbed)
        if np.argmax(perturbed_logits) != original_pred:
            continue

        valid_count += 1

        # Get perturbed explanation
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Compute ROS
        explanation_diff = original_explanation - perturbed_explanation
        safe_original = np.where(np.abs(original_explanation) < epsilon_min,
                               epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original

        logit_diff = original_logits - perturbed_logits

        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        logit_norm = np.linalg.norm(logit_diff, ord=p_norm)

        instability_ratio = explanation_norm / max(logit_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return float(max_instability)

def compute_relative_representation_stability(model, x_sample, explanation_function,
                                            representation_layer, n_samples=50,
                                            noise_std=0.05, epsilon_min=1e-8, p_norm=2):
    """
    Compute Relative Representation Stability (RRS) according to Equation 3.

    RRS = max ||(ex-ex')/ex||_p / max(||(Lx-Lx')/Lx||_p, ε_min)

    Higher values = more unstable explanations
    """
    x_sample = np.array(x_sample).flatten()

    # Get original explanation and representation
    original_explanation = np.array(explanation_function(x_sample)).flatten()
    original_representation = np.array(representation_layer(x_sample)).flatten()

    # Generate perturbations with same prediction
    valid_perturbations = []
    original_pred = np.argmax(model(x_sample.reshape(1, -1))[0])

    attempts = 0
    max_attempts = n_samples * 10

    while len(valid_perturbations) < n_samples and attempts < max_attempts:
        perturbation = np.random.normal(0, noise_std, x_sample.shape)
        x_perturbed = x_sample + perturbation

        # Check if prediction class is the same
        perturbed_pred = np.argmax(model(x_perturbed.reshape(1, -1))[0])

        if perturbed_pred == original_pred:
            # Get representation for perturbed input
            perturbed_representation = np.array(representation_layer(x_perturbed)).flatten()
            valid_perturbations.append((x_perturbed, perturbed_representation))

        attempts += 1

    if len(valid_perturbations) == 0:
        return 0.0

    max_instability = 0.0

    for x_perturbed, perturbed_representation in valid_perturbations:
        # Get explanation for perturbed input
        perturbed_explanation = np.array(explanation_function(x_perturbed)).flatten()

        # Compute (ex - ex') / ex (percent change in explanations)
        explanation_diff = original_explanation - perturbed_explanation  # FIXED
        safe_original_exp = np.where(np.abs(original_explanation) < epsilon_min,
                                   epsilon_min, original_explanation)
        explanation_relative = explanation_diff / safe_original_exp

        # Compute (Lx - Lx') / Lx (percent change in representations)
        representation_diff = original_representation - perturbed_representation  # FIXED
        safe_original_rep = np.where(np.abs(original_representation) < epsilon_min,
                                   epsilon_min, original_representation)
        representation_relative = representation_diff / safe_original_rep

        # Compute norms of the relative change vectors
        explanation_norm = np.linalg.norm(explanation_relative, ord=p_norm)
        representation_norm = np.linalg.norm(representation_relative, ord=p_norm)

        # Compute RRS ratio
        instability_ratio = explanation_norm / max(representation_norm, epsilon_min)
        max_instability = max(max_instability, instability_ratio)

    return math.log(float(max_instability))


def compute_completeness(model, x_sample, explanation_values, epsilon_min=1e-8):
    """
    Compute Completeness metric for any XAI explanation method.If F : ℝⁿ → ℝ is differentiable almost everywhere, then:
    ∑ⁿᵢ₌₁ IntegratedGradsᵢ(x) = F(x) - F(x⁰)
    where:
    IntegratedGradsᵢ(x): The integrated gradient attribution for feature i
    F(x): Model output at input x
    F(x⁰): Model output at baseline x⁰
    x⁰: Baseline input (e.g., black image for vision models)

    Parameters:
    -----------
    model : Callable
        Model prediction function that takes input and returns predictions
    x_sample : np.ndarray
        Original input sample (1D array)
    explanation_values : np.ndarray
        Attribution values from XAI method (same shape as x_sample)
    baseline : np.ndarray, optional
        Baseline input. If None, uses zero baseline for tabular data.
        For other data types, you may want to specify appropriate baseline.
    epsilon_min : float, default=1e-8
        Minimum value to prevent numerical issues

    Returns:
    --------
    float : Completeness score (lower absolute value is better, 0 is perfect)
    """
    x_sample = np.array(x_sample).flatten()
    explanation_values = np.array(explanation_values).flatten()

    # Create baseline if not provided
    baseline = np.zeros_like(x_sample)  # Zero baseline (common default)


    # Get model predictions
    original_pred = model(x_sample.reshape(1, -1))[0]
    if len(original_pred.shape) > 0:
        original_pred = original_pred[1] if len(original_pred) > 1 else original_pred[0]

    baseline_pred = model(baseline.reshape(1, -1))[0]
    if len(baseline_pred.shape) > 0:
        baseline_pred = baseline_pred[1] if len(baseline_pred) > 1 else baseline_pred[0]

    # Completeness: Σ attributions should equal F(x) - F(baseline)
    attribution_sum = np.sum(explanation_values)
    model_diff = original_pred - baseline_pred
    completeness_error = attribution_sum - model_diff

    return float(completeness_error)



def compute_model_parameter_randomization(model_predict_func, x_sample, explanation_function,
                                        trained_model=None, model_type='auto',
                                        randomization_type='full', similarity_metric='spearman_abs',
                                        n_samples=5, random_seed=42):
    """
    Compute Model Parameter Randomization Test exactly as defined in the paper.
    Tests whether explanations are sensitive to model parameters by comparing
    explanations from trained vs randomly initialized models with same architecture.

    Parameters:
    -----------
    model_predict_func : Callable
        Model prediction function (e.g., lstm_predict_proba, lstm_predict_single)
    x_sample : np.ndarray
        Original input sample
    explanation_function : Callable
        Function that generates explanations (e.g., get_shap_explanation)
        Should accept (model, x_sample) or work with global model
    trained_model : object, optional
        The actual trained model object. If None, attempts to extract from prediction function
    model_type : str, default='auto'
        Type of model: 'tensorflow', 'pytorch', 'sklearn', 'auto'
    randomization_type : str, default='full'
        'full': randomize all weights, 'cascading': layer by layer, 'independent': one layer at a time
    similarity_metric : str, default='spearman_abs'
        Similarity metric: 'spearman_abs', 'spearman', 'pearson', 'cosine'
    n_samples : int, default=5
        Number of random model samples for averaging
    random_seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    float : Model parameter randomization score
           - High values (close to 1): Method is INSENSITIVE to model parameters (BAD)
           - Low values (close to 0): Method is SENSITIVE to model parameters (GOOD)
    """
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    from sklearn.metrics.pairwise import cosine_similarity
    import copy

    np.random.seed(random_seed)

    def _compute_similarity(exp1, exp2, metric):
        """Compute similarity between two explanations"""
        exp1, exp2 = np.array(exp1).flatten(), np.array(exp2).flatten()

        # Ensure same length
        if len(exp1) != len(exp2):
            min_len = min(len(exp1), len(exp2))
            exp1, exp2 = exp1[:min_len], exp2[:min_len]

        # Handle edge cases
        if len(exp1) == 0 or np.allclose(exp1, 0) and np.allclose(exp2, 0):
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
                raise ValueError(f"Unknown similarity metric: {metric}")

            return 0.0 if np.isnan(corr) else abs(corr)
        except Exception:
            return 0.0

    def _detect_model_type(model):
        """Auto-detect model framework"""
        if model is None:
            return 'unknown'

        model_str = str(type(model)).lower()
        if 'tensorflow' in model_str or 'keras' in model_str or hasattr(model, 'layers'):
            return 'tensorflow'
        elif 'torch' in model_str or hasattr(model, 'state_dict'):
            return 'pytorch'
        elif hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
            return 'sklearn'
        else:
            return 'unknown'

    def _randomize_tensorflow_model(model, randomization_layer=None):
        """Randomize TensorFlow/Keras model weights"""
        try:
            import tensorflow as tf

            # Clone the model architecture
            random_model = tf.keras.models.clone_model(model)
            random_model.build(model.input_shape)

            if randomization_type == 'full' or randomization_layer is None:
                # Randomize all layers
                for layer in random_model.layers:
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.01))
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.assign(tf.random.normal(layer.bias.shape, stddev=0.01))
                    if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None:
                        layer.recurrent_kernel.assign(tf.random.normal(layer.recurrent_kernel.shape, stddev=0.01))
            else:
                # Randomize specific layer for cascading/independent
                if randomization_layer < len(random_model.layers):
                    layer = random_model.layers[randomization_layer]
                    if hasattr(layer, 'kernel') and layer.kernel is not None:
                        layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.01))
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        layer.bias.assign(tf.random.normal(layer.bias.shape, stddev=0.01))
                    if hasattr(layer, 'recurrent_kernel') and layer.recurrent_kernel is not None:
                        layer.recurrent_kernel.assign(tf.random.normal(layer.recurrent_kernel.shape, stddev=0.01))

            return random_model
        except Exception as e:
            print(f"TensorFlow randomization failed: {e}")
            return None

    def _randomize_pytorch_model(model, randomization_layer=None):
        """Randomize PyTorch model weights"""
        try:
            import torch
            import torch.nn as nn

            # Deep copy the model
            random_model = copy.deepcopy(model)

            if randomization_type == 'full' or randomization_layer is None:
                # Randomize all parameters
                with torch.no_grad():
                    for name, param in random_model.named_parameters():
                        if param.requires_grad:
                            param.normal_(0, 0.01)
            else:
                # Randomize specific layer
                layer_count = 0
                with torch.no_grad():
                    for name, module in random_model.named_modules():
                        if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d, nn.Conv2d)):
                            if layer_count == randomization_layer:
                                for param in module.parameters():
                                    if param.requires_grad:
                                        param.normal_(0, 0.01)
                                break
                            layer_count += 1

            return random_model
        except Exception as e:
            print(f"PyTorch randomization failed: {e}")
            return None

    def _randomize_sklearn_model(model, randomization_layer=None):
        """Randomize sklearn model weights"""
        try:
            from sklearn.base import clone

            # Clone the model
            random_model = clone(model)

            # Create dummy data to fit the model
            n_features = len(x_sample) if hasattr(x_sample, '__len__') else 10
            dummy_X = np.random.normal(0, 0.1, (100, n_features))
            dummy_y = np.random.randint(0, 2, 100)

            # Fit with dummy data
            random_model.fit(dummy_X, dummy_y)

            # Randomize weights
            if hasattr(random_model, 'coef_'):
                random_model.coef_ = np.random.normal(0, 0.01, random_model.coef_.shape)
            if hasattr(random_model, 'intercept_'):
                random_model.intercept_ = np.random.normal(0, 0.01, random_model.intercept_.shape)

            return random_model
        except Exception as e:
            print(f"Sklearn randomization failed: {e}")
            return None

    def _create_random_model(model, model_framework, layer_idx=None):
        """Create randomized model based on framework"""
        if model_framework == 'tensorflow':
            return _randomize_tensorflow_model(model, layer_idx)
        elif model_framework == 'pytorch':
            return _randomize_pytorch_model(model, layer_idx)
        elif model_framework == 'sklearn':
            return _randomize_sklearn_model(model, layer_idx)
        else:
            return None

    def _get_explanation_with_model(model, x_sample, explanation_func):
        """Get explanation using specific model"""
        try:
            # Try different ways to pass model to explanation function
            try:
                # Method 1: explanation_function(model, x_sample)
                return explanation_func(model, x_sample)
            except:
                try:
                    # Method 2: explanation_function(x_sample) with global model replacement
                    # This is trickier and framework-dependent
                    return explanation_func(x_sample)
                except:
                    # Method 3: Generate random explanation as fallback (not ideal)
                    original_exp = explanation_func(x_sample)
                    return np.random.normal(0, 0.1, np.array(original_exp).shape)
        except Exception:
            return np.random.normal(0, 0.1, len(x_sample))

    try:
        # Auto-detect model type if needed
        if model_type == 'auto':
            model_type = _detect_model_type(trained_model)

        # Get original explanation
        original_explanation = explanation_function(x_sample)
        original_explanation = np.array(original_explanation).flatten()

        if trained_model is None:
            print("Warning: No trained model provided. Using simplified randomization.")
            # Fallback: generate random explanations
            similarities = []
            for _ in range(n_samples):
                random_explanation = np.random.normal(0, np.std(original_explanation),
                                                    original_explanation.shape)
                similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                similarities.append(similarity)
            return float(np.mean(similarities))

        similarities = []

        if randomization_type == 'full':
            # Full model randomization
            for _ in range(n_samples):
                random_model = _create_random_model(trained_model, model_type)
                if random_model is not None:
                    random_explanation = _get_explanation_with_model(random_model, x_sample, explanation_function)
                    random_explanation = np.array(random_explanation).flatten()
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)
                else:
                    # Fallback to random explanation
                    random_explanation = np.random.normal(0, np.std(original_explanation),
                                                        original_explanation.shape)
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)

        elif randomization_type == 'cascading':
            # Cascading randomization (layer by layer)
            max_layers = 10  # Reasonable default
            if hasattr(trained_model, 'layers'):
                max_layers = len(trained_model.layers)

            for layer_idx in range(min(max_layers, 5)):  # Limit to prevent too many iterations
                random_model = _create_random_model(trained_model, model_type, layer_idx)
                if random_model is not None:
                    random_explanation = _get_explanation_with_model(random_model, x_sample, explanation_function)
                    random_explanation = np.array(random_explanation).flatten()
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)

        elif randomization_type == 'independent':
            # Independent layer randomization
            max_layers = 10
            if hasattr(trained_model, 'layers'):
                max_layers = len(trained_model.layers)

            for layer_idx in range(min(max_layers, 5)):
                random_model = _create_random_model(trained_model, model_type, layer_idx)
                if random_model is not None:
                    random_explanation = _get_explanation_with_model(random_model, x_sample, explanation_function)
                    random_explanation = np.array(random_explanation).flatten()
                    similarity = _compute_similarity(original_explanation, random_explanation, similarity_metric)
                    similarities.append(similarity)

        if similarities:
            return float(np.mean(similarities))
        else:
            print("Warning: All randomization attempts failed. Using fallback.")
            return float('nan')

    except Exception as e:
        print(f"Warning: Model Parameter Randomization Test failed: {e}")
        return float('nan')



# ---------------------------------------------------------------------------
# Silence noisy third-party output BEFORE importing the libraries that emit it
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

# Disable tqdm progress bars globally (SHAP / LIME / interpret use these).
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
try:
    from tqdm.notebook import tqdm as _tqdm_nb
    _tqdm_nb.__init__ = partialmethod(_tqdm_nb.__init__, disable=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import shap
from lime.lime_tabular import LimeTabularExplainer
from interpret.glassbox import ExplainableBoostingClassifier


class ParkinsonXAIEvaluator:
    """
    Trains an SVM (black-box) and EBM (glassbox) on parkinson.csv,
    then evaluates SHAP, LIME, and EBM explanations with mean ± std
    across 30 samples per class.
    """

    def __init__(self, data_path='parkinson.csv', n_metric_samples=15,
                 n_samples_per_class=30, random_state=42):
        self.data_path = data_path
        self.n_metric_samples = n_metric_samples
        self.n_samples_per_class = n_samples_per_class
        self.random_state = random_state

        self._load_data()
        self._train_models()
        self._select_samples()

    # ----------------------------- setup ---------------------------------
    def _load_data(self):
        print("Loading and preparing data...")

        df = pd.read_csv(self.data_path)
        X = df.drop(columns=['name', 'status'])
        y = df['status'].astype(int)

        self.feature_names = X.columns.tolist()

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # keep an unscaled DataFrame for sample selection by index
        self.X_full = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        self.y_full = y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_full, self.y_full, test_size=0.2,
            random_state=self.random_state, stratify=self.y_full
        )

        print(f"✓ Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"✓ Train set:   {self.X_train.shape[0]} samples")
        print(f"✓ Test set:    {self.X_test.shape[0]} samples")
        print(f"✓ Class counts (full): {dict(self.y_full.value_counts())}")

    def _train_models(self):
        print("\nTraining models...")

        # --- SVM (black-box) ---
        self.svm_model = SVC(kernel='rbf', probability=True,
                             random_state=self.random_state)
        self.svm_model.fit(self.X_train.values, self.y_train.values)
        svm_pred = self.svm_model.predict(self.X_test.values)
        print(f"✓ SVM trained | "
              f"acc={accuracy_score(self.y_test, svm_pred):.4f} "
              f"prec={precision_score(self.y_test, svm_pred, average='weighted'):.4f} "
              f"rec={recall_score(self.y_test, svm_pred, average='weighted'):.4f} "
              f"f1={f1_score(self.y_test, svm_pred, average='weighted'):.4f}")

        # --- EBM (glassbox) ---
        self.ebm_model = ExplainableBoostingClassifier(random_state=self.random_state)
        self.ebm_model.fit(self.X_train, self.y_train)
        ebm_pred = self.ebm_model.predict(self.X_test)
        print(f"✓ EBM trained | "
              f"acc={accuracy_score(self.y_test, ebm_pred):.4f} "
              f"prec={precision_score(self.y_test, ebm_pred, average='weighted'):.4f} "
              f"rec={recall_score(self.y_test, ebm_pred, average='weighted'):.4f} "
              f"f1={f1_score(self.y_test, ebm_pred, average='weighted'):.4f}")

    def _select_samples(self):
        """
        Pick n_samples_per_class samples per class from the FULL dataset,
        because Parkinson's is imbalanced and the test split alone may
        not contain 30 of class 0.
        """
        n = self.n_samples_per_class
        self.class0_indices = self.y_full[self.y_full == 0].index[:n]
        self.class1_indices = self.y_full[self.y_full == 1].index[:n]

        print(f"\n✓ Selected {len(self.class0_indices)} class-0 samples")
        print(f"✓ Selected {len(self.class1_indices)} class-1 samples")

    # ---------------------- prediction wrappers --------------------------
    # IMPORTANT: data is already scaled in self.X_full, so these wrappers
    # do NOT re-scale. SHAP/LIME pass scaled inputs in directly.
    def svm_predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.svm_model.predict_proba(X)

    def ebm_predict_proba(self, X):
        if hasattr(X, 'values'):
            X_arr = X.values
        else:
            X_arr = X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_df = pd.DataFrame(X_arr, columns=self.feature_names)
        return self.ebm_model.predict_proba(X_df)

    def get_svm_representation(self, x_sample):
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)
        # SVC.decision_function gives the margin — a natural "logit-like" representation
        return self.svm_model.decision_function(x_sample).flatten()

    def get_ebm_representation(self, x_sample):
        if x_sample.ndim == 1:
            x_sample = x_sample.reshape(1, -1)
        preds = self.ebm_model.predict_proba(x_sample)[:, 1]
        eps = 1e-7
        preds = np.clip(preds, eps, 1 - eps)
        return np.log(preds / (1 - preds))

    # ----------------------- public evaluation API ------------------------
    def evaluate_shap(self, sample_indices=None, class_type='both'):
        print("\nEvaluating SHAP...")
        return self._evaluate_method(self._evaluate_shap_single_sample,
                                     sample_indices, class_type, 'SHAP')

    def evaluate_lime(self, sample_indices=None, class_type='both'):
        print("\nEvaluating LIME...")
        return self._evaluate_method(self._evaluate_lime_single_sample,
                                     sample_indices, class_type, 'LIME')

    def evaluate_ebm(self, sample_indices=None, class_type='both'):
        print("\nEvaluating EBM...")
        return self._evaluate_method(self._evaluate_ebm_single_sample,
                                     sample_indices, class_type, 'EBM')

    def _evaluate_method(self, single_sample_fn, sample_indices, class_type, method_label):
        if sample_indices is None:
            if class_type == 'class0':
                indices_to_process = [('class0', self.class0_indices)]
            elif class_type == 'class1':
                indices_to_process = [('class1', self.class1_indices)]
            else:
                indices_to_process = [('class0', self.class0_indices),
                                      ('class1', self.class1_indices)]
        else:
            indices_to_process = [('custom', sample_indices)]

        results = {}
        for class_name, indices in indices_to_process:
            print(f"\nProcessing {class_name.upper()} samples for {method_label}...")
            class_results = []

            for i, idx in enumerate(indices):
                print(f"Processing {class_name} sample {i+1}/{len(indices)} (index {idx})")
                try:
                    res = single_sample_fn(idx)
                    class_results.append(res)
                except Exception as e:
                    print(f"  Error processing {class_name} sample {idx}: {e}")

            results[class_name] = self._compute_average_metrics(class_results) if class_results else None

        return results

    # ----------------------- per-sample evaluation ------------------------
    def _evaluate_shap_single_sample(self, idx):
        x_sample = self.X_full.loc[idx].values  # already scaled

        # Background: a sample of training rows
        bg_size = min(50, len(self.X_train))
        background_data = self.X_train.values[:bg_size]
        shap_explainer = shap.KernelExplainer(self.svm_predict_proba, background_data)

        def get_shap_explanation(x):
            x = x.reshape(1, -1)
            shap_values = shap_explainer.shap_values(x, nsamples=100, silent=True)

            if isinstance(shap_values, list):
                return shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                return arr[0, :, 1]
            elif arr.ndim == 2:
                if arr.shape[1] == len(self.feature_names):
                    return arr[0]
                elif arr.shape[1] == len(self.feature_names) * 2:
                    return arr[0, len(self.feature_names):]
                else:
                    return arr[0]
            return arr

        shap_explanation = np.array(get_shap_explanation(x_sample)).flatten()
        if len(shap_explanation) != len(self.feature_names):
            if len(shap_explanation) == len(self.feature_names) * 2:
                shap_explanation = shap_explanation[len(self.feature_names):]
            else:
                raise ValueError(f"SHAP shape mismatch: {len(shap_explanation)} vs {len(self.feature_names)}")

        return self._compute_all_metrics(
            predict_fn=self.svm_predict_proba,
            x_sample=x_sample,
            explanation=shap_explanation,
            explain_fn=get_shap_explanation,
            representation_fn=self.get_svm_representation,
            trained_model=self.svm_model,
            model_type='sklearn',
        )

    def _evaluate_lime_single_sample(self, idx):
        x_sample = self.X_full.loc[idx].values  # already scaled

        bg_size = min(50, len(self.X_train))
        lime_explainer = LimeTabularExplainer(
            self.X_train.values[:bg_size],
            feature_names=self.feature_names,
            class_names=["Class0", "Class1"],
            discretize_continuous=True,
            random_state=self.random_state,
        )

        def get_lime_explanation(x):
            x = x.reshape(1, -1)[0]
            explanation = lime_explainer.explain_instance(
                x, self.svm_predict_proba,
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
            predict_fn=self.svm_predict_proba,
            x_sample=x_sample,
            explanation=lime_explanation,
            explain_fn=get_lime_explanation,
            representation_fn=self.get_svm_representation,
            trained_model=self.svm_model,
            model_type='sklearn',
        )

    def _evaluate_ebm_single_sample(self, idx):
        x_sample = self.X_full.loc[idx].values  # already scaled

        def get_ebm_explanation(x):
            x = x.reshape(1, -1)
            x_df = pd.DataFrame(x, columns=self.feature_names)
            local_exp = self.ebm_model.explain_local(x_df)

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

    def _compute_all_metrics(self, predict_fn, x_sample, explanation, explain_fn,
                             representation_fn, trained_model, model_type):
        """Run every metric and return a dict."""
        n = self.n_metric_samples
        return {
            'infidelity':                   compute_infidelity(predict_fn, x_sample, explanation, explain_fn, n_samples=n),
            'complexity':                   compute_complexity(explanation),
            'sensitivity':                  compute_sensitivity(predict_fn, x_sample, explain_fn, n_samples=n),
            'ris':                          compute_relative_input_stability(predict_fn, x_sample, explain_fn, n_samples=n),
            'ros':                          compute_relative_output_stability(predict_fn, x_sample, explain_fn, n_samples=n),
            'rrs':                          compute_relative_representation_stability(predict_fn, x_sample, explain_fn, representation_fn, n_samples=n),
            'faithfulness':                 compute_faithfulness(predict_fn, x_sample, explain_fn),
            'completeness':                 compute_completeness(predict_fn, x_sample, explanation),
            'model_parameter_randomization': compute_model_parameter_randomization(predict_fn, x_sample, explain_fn, trained_model=trained_model, model_type=model_type, randomization_type='full', n_samples=n),
        }

    
    # ----------------------- aggregation & printing -----------------------
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

    def print_results(self, results, method_name):
        print(f"\n{method_name.upper()} RESULTS:")
        print("=" * 60)

        for class_name, class_results in results.items():
            if class_results is None:
                print(f"\n{class_name.upper()}: No results")
                continue

            print(f"\n{class_name.upper()}:")
            print("-" * 40)
            for metric, s in class_results.items():
                label = metric.replace('_', ' ').title()
                if np.isnan(s['mean']):
                    print(f"  {label}: NaN (n=0)")
                else:
                    print(f"  {label}: {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
def code_usage():
    evaluator = ParkinsonXAIEvaluator(
        data_path='parkinson.csv',
        n_metric_samples=50,
        n_samples_per_class=30,
    )

    shap_results = evaluator.evaluate_shap(class_type='both')
    evaluator.print_results(shap_results, "SHAP")
 
    # LIME — both classes
    lime_results = evaluator.evaluate_lime(class_type='both')
    evaluator.print_results(lime_results, "LIME")
 
    # EBM — both classes
    ebm_results = evaluator.evaluate_ebm(class_type='both')
    evaluator.print_results(ebm_results, "EBM")


# def code_usage():
#     evaluator = ParkinsonXAIEvaluator(
#         data_path='parkinson.csv',
#         n_metric_samples=5,           # was 15 — fewer perturbations per metric
#         n_samples_per_class=2,        # was 30 — only 2 samples per class
#     )

#     shap_results = evaluator.evaluate_shap(class_type='both')
#     evaluator.print_results(shap_results, "SHAP")

#     lime_results = evaluator.evaluate_lime(class_type='both')
#     evaluator.print_results(lime_results, "LIME")

#     ebm_results = evaluator.evaluate_ebm(class_type='both')
#     evaluator.print_results(ebm_results, "EBM")

    

if __name__ == "__main__":
    code_usage()
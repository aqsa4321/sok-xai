# ---------------------------------------------------------------------------
# Environment variables MUST be set BEFORE importing tensorflow / matplotlib.
# ---------------------------------------------------------------------------
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'        # force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # 0=all, 1=info, 2=warning, 3=error
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('shap').setLevel(logging.ERROR)
logging.getLogger('lime').setLevel(logging.ERROR)
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

# Disable tqdm progress bars globally (LIME and others use these).
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

# Use non-interactive matplotlib backend (no display on the server).
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Now safe to import the heavy stuff
# ---------------------------------------------------------------------------
import glob
import math
import copy
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import mark_boundaries
from typing import Callable

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from lime import lime_image

# PyTorch is referenced inside calculate_model_parameter_randomization for the
# pytorch branch but we never actually use it (model is TensorFlow). Make the
# import optional so the env doesn't need torch.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False
    
# import os
# import numpy as np
# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# import math
# from scipy.stats import pearsonr
# import numpy as np
# import cv2
# import math
# import numpy as np
# import torch
# import torch.nn.functional as F
# from typing import Callable
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   

# import numpy as np
# import cv2
# import math

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # 0=all, 1=info, 2=warning, 3=error
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
# os.environ['PYTHONWARNINGS'] = 'ignore'

# import warnings
# warnings.filterwarnings('ignore')

# import logging
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# logging.getLogger('shap').setLevel(logging.ERROR)
# logging.getLogger('lime').setLevel(logging.ERROR)
# try:
#     import absl.logging
#     absl.logging.set_verbosity(absl.logging.ERROR)
# except ImportError:
#     pass

# # Disable tqdm progress bars globally (LIME and others use these).
# from functools import partialmethod
# try:
#     from tqdm import tqdm as _tqdm_base
#     _tqdm_base.__init__ = partialmethod(_tqdm_base.__init__, disable=True)
# except Exception:
#     pass
# try:
#     from tqdm.auto import tqdm as _tqdm_auto
#     _tqdm_auto.__init__ = partialmethod(_tqdm_auto.__init__, disable=True)
# except Exception:
#     pass
# try:
#     from tqdm.notebook import tqdm as _tqdm_nb
#     _tqdm_nb.__init__ = partialmethod(_tqdm_nb.__init__, disable=True)
# except Exception:
#     pass

# # ---------------------------------------------------------------------------
# # Imports
# # ---------------------------------------------------------------------------
# import glob
# import numpy as np
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(0)

# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import cv2
# import pandas as pd
# from lime import lime_image

# import torch
# import torch.nn as nn
# import numpy as np
# from scipy.stats import spearmanr
# from skimage.metrics import structural_similarity as ssim
# import copy
# import tensorflow as tf


# import numpy as np
# import matplotlib.pyplot as plt



def calculate_infidelity(explanation, model, img_array, num_samples=5, perturbation_std=0.1):
    """
    Calculate infidelity metric for an explanation.
    INFD(Φ,f,x) = E_I[‖I^T Φ(f,x) - (f(x) - f(x-I))‖²]
    Args:
        explanation: Explanation vector/map (same shape as input image)
        model: Trained model
        img_array: Preprocessed input image array
        num_samples: Number of perturbation samples
        perturbation_std: Standard deviation for Gaussian perturbations

    Returns:
        infidelity_score: Float value representing infidelity
    """
    original_pred = model.predict(img_array, verbose=0)[0][0]
    infidelity_scores = []

    for _ in range(num_samples):
        # Generate random perturbation
        perturbation = np.random.normal(0, perturbation_std, img_array.shape)

        # Apply perturbation
        perturbed_img = img_array + perturbation
        perturbed_img = np.clip(perturbed_img, -2.5, 2.5)  # Clip to reasonable range for preprocessed images

        # Get model prediction on perturbed image
        perturbed_pred = model.predict(perturbed_img, verbose=0)[0][0]

        # Calculate output difference
        output_diff = original_pred - perturbed_pred

        # Flatten perturbation and explanation for dot product
        perturbation_flat = perturbation.flatten()

        # Handle different explanation shapes
        if len(explanation.shape) == 2:  # Saliency map (H, W)
            explanation_resized = cv2.resize(explanation, (img_array.shape[2], img_array.shape[1]))
            explanation_expanded = np.stack([explanation_resized] * img_array.shape[3], axis=-1)
        elif len(explanation.shape) == 3:  # (H, W, C)
            explanation_expanded = explanation
        else:  # (1, H, W, C)
            explanation_expanded = explanation[0]

        explanation_flat = explanation_expanded.flatten()

        # Ensure same length
        min_len = min(len(perturbation_flat), len(explanation_flat))
        perturbation_flat = perturbation_flat[:min_len]
        explanation_flat = explanation_flat[:min_len]

        # Calculate dot product
        dot_product = np.dot(perturbation_flat, explanation_flat)

        # Calculate infidelity for this sample
        infidelity = (dot_product - output_diff) ** 2
        infidelity_scores.append(infidelity)

    return np.mean(infidelity_scores)

def calculate_max_sensitivity(explanation_func, model, img_array, radius=0.1, num_samples=5,
                           preserve_prediction=True):
    """
    Calculate max-sensitivity metric for an explanation method.
    SENS_MAX(Φ,f,x,r) = max_{‖y-x‖≤r} ‖Φ(f,y) - Φ(f,x)‖
    Args:
        explanation_func: Function that takes (model, img_array) and returns explanation
        model: Trained model
        img_array: Preprocessed input image array
        radius: Perturbation radius
        num_samples: Number of samples to test
        preserve_prediction: Whether to only consider perturbations that don't change the predicted class

    Returns:
        max_sensitivity: Maximum change in explanation over all valid perturbations
    """
    # Get original explanation and model prediction
    original_explanation = explanation_func(model, img_array)
    original_output = model.predict(img_array, verbose=0)[0][0]
    original_prediction = int(original_output > 0.5)  # Binary classification: 0 or 1

    sensitivities = []
    valid_samples = 0
    attempted_samples = 0

    while valid_samples < num_samples and attempted_samples < num_samples * 5:
        attempted_samples += 1

        # Generate random perturbation within radius
        perturbation = np.random.normal(0, 1, img_array.shape)
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > 0:
            perturbation = (perturbation / perturbation_norm) * radius

        # Apply perturbation
        perturbed_img = img_array + perturbation
        perturbed_img = np.clip(perturbed_img, -2.5, 2.5)

        # Check if perturbation preserves the prediction class
        if preserve_prediction:
            perturbed_output = model.predict(perturbed_img, verbose=0)[0][0]
            perturbed_prediction = int(perturbed_output > 0.5)

            if original_prediction != perturbed_prediction:
                continue  # Skip this perturbation - changes the prediction class

        valid_samples += 1

        # Get explanation for perturbed image
        perturbed_explanation = explanation_func(model, perturbed_img)

        # Handle different explanation shapes and calculate difference
        if len(original_explanation.shape) != len(perturbed_explanation.shape):
            # Handle shape mismatch (e.g., different methods returning different shapes)
            if len(original_explanation.shape) == 2:
                original_flat = original_explanation.flatten()
            else:
                original_flat = original_explanation.flatten()

            if len(perturbed_explanation.shape) == 2:
                perturbed_flat = perturbed_explanation.flatten()
            else:
                perturbed_flat = perturbed_explanation.flatten()

            # Ensure same length
            min_len = min(len(original_flat), len(perturbed_flat))
            original_flat = original_flat[:min_len]
            perturbed_flat = perturbed_flat[:min_len]

            sensitivity = np.linalg.norm(original_flat - perturbed_flat)
        else:
            sensitivity = np.linalg.norm(original_explanation - perturbed_explanation)

        sensitivities.append(sensitivity)

    if len(sensitivities) == 0:
        return calculate_max_sensitivity(explanation_func, model, img_array,
                                       radius=radius/2, num_samples=num_samples,
                                       preserve_prediction=False)

    return max(sensitivities)



def calculate_relative_input_stability(explanation_func, model, img_array, num_samples=5, perturbation_std=0.05,
                  preserve_prediction=True, p_norm=2, epsilon_min=1e-8):
    """
    Calculate Relative Input Stability (RIS) metric for an explanation method.
    RIS(x, x', ex, ex') = max_{x'} ||（ex - ex')/ex||_p / max(||(x - x')/x||_p, ε_min)
    Args:
        explanation_func: Function that takes (model, img_array) and returns explanation
        model: Trained model
        img_array: Preprocessed input image array (shape: 1, H, W, C)
        num_samples: Number of perturbation samples (default: 50 as in paper)
        perturbation_std: Standard deviation for Gaussian perturbations
        preserve_prediction: Whether to only consider perturbations that preserve predicted class
        p_norm: Order of the norm to use (default: 2 for L2 norm)
        epsilon_min: Small constant to prevent division by zero

    Returns:
        ris_score: Float value representing RIS (higher = more unstable)
    """
    # Get original explanation and prediction
    original_explanation = explanation_func(model, img_array)
    original_output = model.predict(img_array, verbose=0)
    original_prediction = np.argmax(original_output, axis=1)[0]  # For multi-class classification

    ris_scores = []
    valid_samples = 0
    attempted_samples = 0
    max_attempts = num_samples * 10  # Allow more attempts to find valid perturbations

    while valid_samples < num_samples and attempted_samples < max_attempts:
        attempted_samples += 1

        # Generate Gaussian noise perturbation
        perturbation = np.random.normal(0, perturbation_std, img_array.shape)

        # Apply perturbation and clip to valid range
        perturbed_img = img_array + perturbation
        perturbed_img = np.clip(perturbed_img, -2.5, 2.5)

        # Check if prediction is preserved (if required)
        if preserve_prediction:
            perturbed_output = model.predict(perturbed_img, verbose=0)
            perturbed_prediction = np.argmax(perturbed_output, axis=1)[0]

            if original_prediction != perturbed_prediction:
                continue  # Skip this perturbation - changes the prediction class

        valid_samples += 1

        # Get explanation for perturbed image
        perturbed_explanation = explanation_func(model, perturbed_img)

        # Normalize explanations to flat arrays
        def normalize_explanation(explanation):
            if explanation is None:
                return np.array([])

            if isinstance(explanation, tuple):
                explanation = explanation[0]

            explanation = np.array(explanation)

            # Handle different shapes
            if len(explanation.shape) == 4:  # (1, H, W, C)
                explanation = explanation[0]
            elif len(explanation.shape) == 2:  # (H, W) - typical for GradCAM
                pass
            elif len(explanation.shape) == 3:  # (H, W, C)
                if explanation.shape[2] == 3:
                    explanation = np.mean(explanation, axis=2)

            return explanation.flatten()

        original_flat = normalize_explanation(original_explanation)
        perturbed_flat = normalize_explanation(perturbed_explanation)

        # Ensure same length
        min_len = min(len(original_flat), len(perturbed_flat))
        original_flat = original_flat[:min_len]
        perturbed_flat = perturbed_flat[:min_len]

        # Calculate percent change in explanations (numerator)
        # RIS formula: ||(e_x' - e_x) / e_x||_p / max(||(x' - x) / x||_p, epsilon_min)
        explanation_diff = perturbed_flat - original_flat

        # Handle division by zero in explanation differences
        original_safe = original_flat.copy()
        original_safe[np.abs(original_safe) < epsilon_min] = epsilon_min
        percent_change_explanation = explanation_diff / original_safe
        explanation_change_norm = np.linalg.norm(percent_change_explanation, ord=p_norm)

        # Calculate percent change in inputs (denominator)
        input_diff = perturbed_img - img_array
        img_flat = img_array.flatten()
        input_diff_flat = input_diff.flatten()

        # Handle division by zero in input differences
        img_safe = img_flat.copy()
        img_safe[np.abs(img_safe) < epsilon_min] = epsilon_min
        percent_change_input = input_diff_flat / img_safe
        input_change_norm = np.linalg.norm(percent_change_input, ord=p_norm)

        # Calculate RIS for this sample
        denominator = max(input_change_norm, epsilon_min)
        ris_sample = explanation_change_norm / denominator

        ris_scores.append(ris_sample)

        # Progress indicator
        # if valid_samples % 10 == 0:
        #     print(f"Processed {valid_samples}/{num_samples} valid samples...")

    # if len(ris_scores) == 0:
    #     print("Warning: No valid perturbations found!")
    #     return float('inf')  # Infinite instability if no valid samples

    # print(f"Successfully processed {valid_samples} valid samples out of {attempted_samples} attempts")
    # print(f"Perturbation success rate: {valid_samples/attempted_samples:.2%}")

    # Return maximum RIS (as per paper definition)
    ris_max = max(ris_scores)
    ris_mean = np.mean(ris_scores)

    # print(f"RIS - Max: {ris_max:.6f}, Mean: {ris_mean:.6f}, Std: {np.std(ris_scores):.6f}")

    return math.log(float(ris_max))


def calculate_relative_output_stability(explanation_func, model, img_array, num_samples=5, perturbation_std=0.05,
                  preserve_prediction=True, p_norm=2, epsilon_min=1e-8):
    """
    Calculate Relative Output Stability (ROS) metric for an explanation method.
    ROS(x, x', ex, ex') = max_{x'} ||(ex - ex')/ex||_p / max(||h(x) - h(x')||_p, ε_min)
    Args:
        explanation_func: Function that takes (model, img_array) and returns explanation
        model: Trained model
        img_array: Preprocessed input image array (shape: 1, H, W, C)
        num_samples: Number of perturbation samples (default: 50 as in paper)
        perturbation_std: Standard deviation for Gaussian perturbations
        preserve_prediction: Whether to only consider perturbations that preserve predicted class
        p_norm: Order of the norm to use (default: 2 for L2 norm)
        epsilon_min: Small constant to prevent division by zero

    Returns:
        ros_score: Float value representing ROS (higher = more unstable)
    """
    # Get original explanation and output logits
    original_explanation = explanation_func(model, img_array)
    original_output = model.predict(img_array, verbose=0)
    original_logits = original_output[0]  # Raw logits/probabilities
    original_prediction = np.argmax(original_logits)

    ros_scores = []
    valid_samples = 0
    attempted_samples = 0
    max_attempts = num_samples * 10

    # print(f"Calculating ROS with {num_samples} samples...")
    # print(f"Original prediction class: {original_prediction}")

    while valid_samples < num_samples and attempted_samples < max_attempts:
        attempted_samples += 1

        # Generate Gaussian noise perturbation
        perturbation = np.random.normal(0, perturbation_std, img_array.shape)

        # Apply perturbation and clip to valid range
        perturbed_img = img_array + perturbation
        perturbed_img = np.clip(perturbed_img, -2.5, 2.5)

        # Get perturbed output logits
        perturbed_output = model.predict(perturbed_img, verbose=0)
        perturbed_logits = perturbed_output[0]
        perturbed_prediction = np.argmax(perturbed_logits)

        # Check if prediction is preserved (if required)
        if preserve_prediction and original_prediction != perturbed_prediction:
            continue

        valid_samples += 1

        # Get explanation for perturbed image
        perturbed_explanation = explanation_func(model, perturbed_img)

        # Normalize explanations to flat arrays
        def normalize_explanation(explanation):
            if explanation is None:
                return np.array([])

            if isinstance(explanation, tuple):
                explanation = explanation[0]

            explanation = np.array(explanation)

            # Handle different shapes
            if len(explanation.shape) == 4:  # (1, H, W, C)
                explanation = explanation[0]
            elif len(explanation.shape) == 2:  # (H, W) - typical for GradCAM
                pass
            elif len(explanation.shape) == 3:  # (H, W, C)
                if explanation.shape[2] == 3:
                    explanation = np.mean(explanation, axis=2)

            return explanation.flatten()

        original_flat = normalize_explanation(original_explanation)
        perturbed_flat = normalize_explanation(perturbed_explanation)

        # Ensure same length
        min_len = min(len(original_flat), len(perturbed_flat))
        original_flat = original_flat[:min_len]
        perturbed_flat = perturbed_flat[:min_len]

        # Calculate percent change in explanations (numerator)
        # ROS formula: ||(e_x' - e_x) / e_x||_p / max(||h(x) - h(x')||_p, epsilon_min)
        explanation_diff = perturbed_flat - original_flat

        # Handle division by zero in explanation differences
        original_safe = original_flat.copy()
        original_safe[np.abs(original_safe) < epsilon_min] = epsilon_min
        percent_change_explanation = explanation_diff / original_safe
        explanation_change_norm = np.linalg.norm(percent_change_explanation, ord=p_norm)

        # Calculate change in output logits (denominator)
        logits_diff = perturbed_logits - original_logits
        logits_change_norm = np.linalg.norm(logits_diff, ord=p_norm)

        # Calculate ROS for this sample
        denominator = max(logits_change_norm, epsilon_min)
        ros_sample = explanation_change_norm / denominator

        ros_scores.append(ros_sample)

        # Progress indicator
        if valid_samples % 10 == 0:
            print(f"Processed {valid_samples}/{num_samples} valid samples...")

    # if len(ros_scores) == 0:
    #     print("Warning: No valid perturbations found!")
    #     return float('inf')

    # print(f"Successfully processed {valid_samples} valid samples out of {attempted_samples} attempts")
    # print(f"Perturbation success rate: {valid_samples/attempted_samples:.2%}")

    # Return maximum ROS (as per paper definition)
    ros_max = max(ros_scores)
    ros_mean = np.mean(ros_scores)

    # print(f"ROS - Max: {ros_max:.6f}, Mean: {ros_mean:.6f}, Std: {np.std(ros_scores):.6f}")

    return math.log(float(ros_max))


def calculate_rrs(explanation_func, model, img_array, layer_names=None,
                           num_samples=5, perturbation_std=0.05, preserve_prediction=True,
                           p_norm=2, epsilon_min=1e-8, normalize_representations=False):
    """
    Calculate Relative Representation Stability (RRS) metric - CORRECTED VERSION.
    RRS(x, x', ex, ex') = max_{x'} ||(ex - ex')/ex||_p / max(||(Lx - Lx')/Lx||_p, ε_min)
    The key fix: Handle the scale difference between representations and inputs/logits
    """

    # Setup layers (same as before)
    if layer_names is None:
        layer_names = []
        for layer in model.layers:
            layer_name = layer.name
            if any(pattern in layer_name for pattern in ['conv2_block3_out', 'conv3_block4_out',
                                                        'conv4_block6_out', 'conv5_block3_out']):
                layer_names.append(layer_name)

        if not layer_names:
            total_layers = len(model.layers)
            indices = [total_layers//4, total_layers//2, 3*total_layers//4, total_layers-2]
            layer_names = [model.layers[i].name for i in indices if i < total_layers]

    if isinstance(layer_names, str):
        layer_names = [layer_names]

    print(f"Using layers for RRS: {layer_names}")

    # Create intermediate model
    from tensorflow.keras.models import Model
    intermediate_outputs = []
    for layer_name in layer_names:
        try:
            intermediate_layer = model.get_layer(layer_name)
            intermediate_outputs.append(intermediate_layer.output)
        except ValueError:
            print(f"Warning: Layer '{layer_name}' not found, skipping...")

    if not intermediate_outputs:
        raise ValueError("No valid layers found for RRS computation")

    intermediate_model = Model(inputs=model.input, outputs=intermediate_outputs)

    # Get original explanation and representations
    original_explanation = explanation_func(model, img_array)
    original_representations = intermediate_model.predict(img_array, verbose=0)
    original_output = model.predict(img_array, verbose=0)
    original_prediction = np.argmax(original_output, axis=1)[0]

    # Handle single layer case
    if len(layer_names) == 1:
        original_representations = [original_representations]

    # Concatenate and optionally normalize representations
    original_repr_list = [repr.flatten() for repr in original_representations]
    original_repr_flat = np.concatenate(original_repr_list)

    if normalize_representations:
        # Normalize to unit norm to handle scale issues
        original_repr_norm = np.linalg.norm(original_repr_flat)
        if original_repr_norm > epsilon_min:
            original_repr_flat = original_repr_flat / original_repr_norm

    rrs_scores = []
    valid_samples = 0
    attempted_samples = 0
    max_attempts = num_samples * 10

    # print(f"Calculating RRS with {num_samples} samples...")
    # print(f"Original prediction class: {original_prediction}")
    # print(f"Representation shape: {original_repr_flat.shape}")
    # print(f"Normalize representations: {normalize_representations}")

    while valid_samples < num_samples and attempted_samples < max_attempts:
        attempted_samples += 1

        # Generate perturbation
        perturbation = np.random.normal(0, perturbation_std, img_array.shape)
        perturbed_img = img_array + perturbation
        perturbed_img = np.clip(perturbed_img, -2.5, 2.5)

        # Check prediction preservation
        if preserve_prediction:
            perturbed_output = model.predict(perturbed_img, verbose=0)
            perturbed_prediction = np.argmax(perturbed_output, axis=1)[0]
            if original_prediction != perturbed_prediction:
                continue

        valid_samples += 1

        # Get perturbed explanation and representations
        perturbed_explanation = explanation_func(model, perturbed_img)
        perturbed_representations = intermediate_model.predict(perturbed_img, verbose=0)

        # Handle single layer case
        if len(layer_names) == 1:
            perturbed_representations = [perturbed_representations]

        # Concatenate and optionally normalize representations
        perturbed_repr_list = [repr.flatten() for repr in perturbed_representations]
        perturbed_repr_flat = np.concatenate(perturbed_repr_list)

        if normalize_representations:
            # Normalize to unit norm
            perturbed_repr_norm = np.linalg.norm(perturbed_repr_flat)
            if perturbed_repr_norm > epsilon_min:
                perturbed_repr_flat = perturbed_repr_flat / perturbed_repr_norm

        # Normalize explanations
        def normalize_explanation(explanation):
            if explanation is None:
                return np.array([])
            if isinstance(explanation, tuple):
                explanation = explanation[0]
            explanation = np.array(explanation)
            if len(explanation.shape) == 4:
                explanation = explanation[0]
            elif len(explanation.shape) == 2:
                pass
            elif len(explanation.shape) == 3:
                if explanation.shape[2] == 3:
                    explanation = np.mean(explanation, axis=2)
            return explanation.flatten()

        original_flat = normalize_explanation(original_explanation)
        perturbed_flat = normalize_explanation(perturbed_explanation)

        # Ensure same length
        min_len = min(len(original_flat), len(perturbed_flat))
        original_flat = original_flat[:min_len]
        perturbed_flat = perturbed_flat[:min_len]

        # Calculate explanation change (numerator)
        explanation_diff = perturbed_flat - original_flat
        original_safe = original_flat.copy()
        original_safe[np.abs(original_safe) < epsilon_min] = epsilon_min
        percent_change_explanation = explanation_diff / original_safe
        explanation_change_norm = np.linalg.norm(percent_change_explanation, ord=p_norm)

        # Calculate representation change (denominator) - THREE OPTIONS:

        # OPTION 1: Absolute difference (recommended for normalized representations)
        if normalize_representations:
            repr_diff = original_repr_flat - perturbed_repr_flat
            repr_change_norm = np.linalg.norm(repr_diff, ord=p_norm)

        # OPTION 2: Relative difference (as per paper formula)
        else:
            repr_diff = original_repr_flat - perturbed_repr_flat
            original_repr_safe = original_repr_flat.copy()
            original_repr_safe[np.abs(original_repr_safe) < epsilon_min] = epsilon_min
            percent_change_repr = repr_diff / original_repr_safe
            repr_change_norm = np.linalg.norm(percent_change_repr, ord=p_norm)

        # Calculate RRS for this sample
        denominator = max(repr_change_norm, epsilon_min)
        rrs_sample = explanation_change_norm / denominator
        rrs_scores.append(rrs_sample)

    #     if valid_samples % 10 == 0:
    #         print(f"Processed {valid_samples}/{num_samples} valid samples...")

    # if len(rrs_scores) == 0:
    #     print("Warning: No valid perturbations found!")
    #     return float('inf')

    # print(f"Successfully processed {valid_samples} valid samples out of {attempted_samples} attempts")
    # print(f"Perturbation success rate: {valid_samples/attempted_samples:.2%}")

    rrs_max = max(rrs_scores)
    rrs_mean = np.mean(rrs_scores)

    # print(f"RRS - Max: {rrs_max:.6f}, Mean: {rrs_mean:.6f}, Std: {np.std(rrs_scores):.6f}")

    return math.log(float(rrs_max))



def calculate_complexity(explanation_func, model, img_array, epsilon_min=1e-8):
    """
    Calculate Complexity metric for an explanation method.
    Pg(i) = |g(f,x)i| / ∑(j=1 to d) |g(f,x)j|
    Complexity = μC(f, g; x) = -∑(i=1 to d) Pg(i) ln(Pg(i)) where Pg(i) = |g(f,x)i| / ∑(j=1 to d) |g(f,x)j|

    Args:
        explanation_func: Function that takes (model, img_array) and returns explanation
        model: Trained model
        img_array: Preprocessed input image array (shape: 1, H, W, C)
        epsilon_min: Small constant to prevent log(0)

    Returns:
        complexity_score: Float value representing explanation complexity
    """

    # Get explanation
    explanation = explanation_func(model, img_array)

    # Normalize explanation to flat array
    def normalize_explanation(explanation):
        if explanation is None:
            return np.array([])
        if isinstance(explanation, tuple):
            explanation = explanation[0]
        explanation = np.array(explanation)
        if len(explanation.shape) == 4:
            explanation = explanation[0]
        elif len(explanation.shape) == 2:
            pass
        elif len(explanation.shape) == 3:
            if explanation.shape[2] == 3:
                explanation = np.mean(explanation, axis=2)
        return explanation.flatten()

    explanation_flat = normalize_explanation(explanation)

    if len(explanation_flat) == 0:
        print("Warning: Empty explanation")
        return float('inf')

    # Calculate absolute values
    abs_attributions = np.abs(explanation_flat)

    # Create probability distribution
    total_attribution = np.sum(abs_attributions)

    if total_attribution < epsilon_min:
        print("Warning: Near-zero total attribution")
        return float('inf')

    # Fractional contribution distribution
    prob_distribution = abs_attributions / total_attribution

    # Add small epsilon to prevent log(0)
    prob_distribution = np.where(prob_distribution < epsilon_min, epsilon_min, prob_distribution)

    # Calculate entropy (complexity)
    complexity = -np.sum(prob_distribution * np.log(prob_distribution))

    return complexity

import numpy as np
from scipy.stats import pearsonr

def calculate_faithfulness(explanation_func, model, img_array,
                           subset_fraction=0.5, num_samples=200,
                           preprocess_fn=None):
    """
    Bhatt et al. (2020) Definition 3 — faithfulness for image explainers.

    Key fixes vs the original:
    1. Baseline = preprocess_fn(np.zeros(img_shape)). For ResNet50 this is
       roughly [-104, -117, -124] per channel — what "black image" looks like
       after preprocessing. Setting features to literal 0.0 in preprocessed
       space was approximately the channel mean, which barely perturbs the
       model.
    2. subset_fraction (default 5%) replaces fixed subset_size, so the same
       fraction of input is masked regardless of explanation granularity.
    3. For coarse-grid explanations (e.g. GradCAM 7x7), each sampled grid
       cell is mapped back to its corresponding block of pixels in the input.

    Args:
        explanation_func: callable(model, img_array) -> attribution map
        model: keras model with sigmoid output (binary classifier)
        img_array: preprocessed input, shape (1, H, W, C)
        subset_fraction: fraction of the explanation to mask per trial
        num_samples: number of random subsets (Bhatt et al. used >= 50)
        preprocess_fn: e.g. tf.keras.applications.resnet50.preprocess_input.
                       If None, falls back to baseline_value=0.0 (use only
                       for models like EfficientNet whose preprocess is a
                       no-op).
    """


    def normalize_explanation(explanation):
        if explanation is None:
            return np.array([])
        if isinstance(explanation, tuple):
            explanation = explanation[0]
        explanation = np.array(explanation)
        if explanation.ndim == 4:
            explanation = explanation[0]
        elif explanation.ndim == 3 and explanation.shape[2] == 3:
            explanation = np.mean(explanation, axis=2)
        return explanation.flatten()

    # ---- compute the proper baseline image ---------------------------------
    if preprocess_fn is not None:
        baseline_img = preprocess_fn(
            np.zeros(img_array.shape, dtype=np.float32)
        )
    else:
        baseline_img = np.zeros_like(img_array)

    # ---- get original explanation and prediction ---------------------------
    explanation = explanation_func(model, img_array)
    explanation_flat = normalize_explanation(explanation)
    n_features = len(explanation_flat)

    if n_features == 0:
        return 0.0

    # subset size = fraction of features (at least 1, at most n_features // 2)
    subset_size = max(1, min(int(n_features * subset_fraction),
                             n_features // 2))

    original_output = float(model.predict(img_array, verbose=0)[0][0])

    # detect granularity: per-pixel vs coarse grid
    H, W = img_array.shape[1], img_array.shape[2]
    is_coarse_grid = (n_features != H * W)
    if is_coarse_grid:
        grid_side = int(np.sqrt(n_features))
        block_h, block_w = H // grid_side, W // grid_side

    attribution_sums = []
    output_differences = []

    for _ in range(num_samples):
        # 1. Randomly sample subset S
        subset_indices = np.random.choice(n_features, subset_size, replace=False)

        # 2. Sum of attributions in S
        attribution_sum = float(np.sum(explanation_flat[subset_indices]))

        # 3. Build x[x_S = baseline] — copy baseline values for masked positions
        modified_img = img_array.copy()
        if is_coarse_grid:
            # Coarse grid: each index is a grid cell; mask its corresponding
            # block of input pixels with the baseline.
            for idx in subset_indices:
                row, col = divmod(idx, grid_side)
                r0, c0 = row * block_h, col * block_w
                modified_img[0, r0:r0 + block_h, c0:c0 + block_w, :] = \
                    baseline_img[0, r0:r0 + block_h, c0:c0 + block_w, :]
        else:
            # Per-pixel: each index is a (row, col) location; mask all channels.
            for idx in subset_indices:
                row, col = divmod(idx, W)
                modified_img[0, row, col, :] = baseline_img[0, row, col, :]

        # 4. Output difference (P(class=1) drop)
        modified_output = float(model.predict(modified_img, verbose=0)[0][0])
        output_diff = original_output - modified_output

        attribution_sums.append(attribution_sum)
        output_differences.append(output_diff)

    # 5. Pearson correlation
    if (np.std(attribution_sums) == 0) or (np.std(output_differences) == 0):
        return 0.0
    correlation, _ = pearsonr(attribution_sums, output_differences)
    return float(correlation) if not np.isnan(correlation) else 0.0


def calculate_faithfulness1(explanation_func, model, img_array,
                                         subset_size=50, num_samples=75, baseline_value=0.0):
    """
    Faithful implementation of paper's faithfulness metric
    """
    def normalize_explanation(explanation):
        if explanation is None:
            return np.array([])
        if isinstance(explanation, tuple):
            explanation = explanation[0]
        explanation = np.array(explanation)
        if len(explanation.shape) == 4:
            explanation = explanation[0]
        elif len(explanation.shape) == 2:
            pass
        elif len(explanation.shape) == 3:
            if explanation.shape[2] == 3:
                explanation = np.mean(explanation, axis=2)
        return explanation.flatten()

    # Get explanation and original prediction
    explanation = explanation_func(model, img_array)
    explanation_flat = normalize_explanation(explanation)
    # print(f"Explanation shape: {explanation.shape}, flattened to {explanation_flat.shape}")
    original_output = model.predict(img_array, verbose=0)[0][0]

    attribution_sums = []
    output_differences = []

    for _ in range(num_samples):
        # 1. Randomly sample subset S of fixed size |S|
        subset_indices = np.random.choice(len(explanation_flat), subset_size, replace=False)

        # 2. Calculate sum of attributions for this subset
        attribution_sum = np.sum(explanation_flat[subset_indices])

        # 3. Create x[xs=x̄s] - set subset features to baseline
        modified_img = img_array.copy()
        modified_img_flat = modified_img.flatten()
        modified_img_flat[subset_indices] = baseline_value  # Consistent baseline
        modified_img = modified_img_flat.reshape(img_array.shape)

        # 4. Calculate output difference
        modified_output = model.predict(modified_img, verbose=0)[0][0]
        output_diff = original_output - modified_output

        attribution_sums.append(attribution_sum)
        output_differences.append(output_diff)

    # 5. Calculate correlation
    correlation, _ = pearsonr(attribution_sums, output_differences)
    return correlation if not np.isnan(correlation) else 0.0


def calculate_completeness_error(explanation_func, model, img_array, baseline=None):
    """
    Calculate completeness error: how well ∑attributions = F(x) - F(x⁰)
    Lower error = better completeness (0 = perfect completeness)
    """
    if baseline is None:
        baseline = np.zeros_like(img_array)

    # Get explanation
    explanation = explanation_func(model, img_array)
    explanation_flat = explanation.flatten()

    # Get model outputs
    f_x = model.predict(img_array, verbose=0)[0][0]
    f_x0 = model.predict(baseline, verbose=0)[0][0]

    # Calculate completeness error
    attribution_sum = np.sum(explanation_flat)
    expected_sum = f_x - f_x0

    completeness_error = abs(attribution_sum - expected_sum)

    return completeness_error


import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.util import img_as_float

def calculate_irof(explanation_func, model, img_array, n_segments=30, compactness=10, sigma=1):
    """
    Calculate IROF (Iterative Removal Of Features) metric for an explanation method.

    Args:
        explanation_func: Function that takes (model, img_array) and returns explanation
        model: Trained model
        img_array: Preprocessed input image array (shape: [1, H, W, C])
        n_segments: Number of segments for SLIC segmentation
        compactness: Compactness parameter for SLIC
        sigma: Sigma parameter for SLIC

    Returns:
        irof_score: Float value representing IROF score (higher is better)
    """
    # Get original explanation and prediction
    explanation = explanation_func(model, img_array)
    original_output = model.predict(img_array, verbose=0)[0][0]

    # Convert image to proper format for segmentation
    # Assuming img_array is preprocessed, we need to convert it back to [0,1] range
    img_for_segmentation = img_array[0].copy()

    # Normalize to [0,1] for segmentation (adjust based on your preprocessing)
    img_min = img_for_segmentation.min()
    img_max = img_for_segmentation.max()
    if img_max > img_min:
        img_normalized = (img_for_segmentation - img_min) / (img_max - img_min)
    else:
        img_normalized = img_for_segmentation

    # Convert to float for SLIC
    img_normalized = img_as_float(img_normalized)

    # Perform SLIC segmentation
    segments = slic(img_normalized, n_segments=n_segments, compactness=compactness,
                   sigma=sigma, start_label=1)

    # Handle different explanation shapes
    if len(explanation.shape) == 2:  # Saliency map (H, W)
        explanation_resized = cv2.resize(explanation, (img_array.shape[2], img_array.shape[1]))
        explanation_map = explanation_resized
    elif len(explanation.shape) == 3:  # (H, W, C)
        # Convert to grayscale if multichannel
        if explanation.shape[2] > 1:
            explanation_map = np.mean(explanation, axis=2)
        else:
            explanation_map = explanation[:, :, 0]
    else:  # (1, H, W, C) or (1, H, W)
        if len(explanation.shape) == 4:
            if explanation.shape[3] > 1:
                explanation_map = np.mean(explanation[0], axis=2)
            else:
                explanation_map = explanation[0, :, :, 0]
        else:  # (1, H, W)
            explanation_map = explanation[0]

    # Resize explanation to match segmentation if needed
    if explanation_map.shape != segments.shape:
        explanation_map = cv2.resize(explanation_map, (segments.shape[1], segments.shape[0]))

    # Calculate mean importance for each segment
    unique_segments = np.unique(segments)
    segment_importances = []

    for segment_id in unique_segments:
        if segment_id == 0:  # Skip background if exists
            continue
        segment_mask = segments == segment_id
        mean_importance = np.mean(np.abs(explanation_map[segment_mask]))
        segment_importances.append((segment_id, mean_importance))

    # Sort segments by importance (descending)
    segment_importances.sort(key=lambda x: x[1], reverse=True)

    # Calculate mean color for replacement
    mean_color = np.mean(img_array[0], axis=(0, 1))

    # Iteratively remove segments and track performance
    modified_img = img_array.copy()
    class_scores = [original_output]  # Start with original score

    for i, (segment_id, _) in enumerate(segment_importances):
        # Create mask for current segment
        segment_mask = segments == segment_id

        # Replace segment with mean color
        for channel in range(modified_img.shape[3]):
            modified_img[0, segment_mask, channel] = mean_color[channel]

        # Get prediction for modified image
        modified_output = model.predict(modified_img, verbose=0)[0][0]
        class_scores.append(modified_output)

    # Normalize scores to [0, 1] range
    normalized_scores = np.array(class_scores) / original_output if original_output != 0 else np.array(class_scores)

    # Calculate Area Over Curve (AOC)
    # This is the area between the curve and y=0
    aoc = np.trapz(normalized_scores, dx=1.0)

    # Normalize by the number of segments to get IROF score
    irof_score = aoc / len(segment_importances) if len(segment_importances) > 0 else 0.0

    return irof_score



def calculate_pixel_flipping_auc(explanation_func, model, img_array, num_samples=10, max_flip_percentage=30):
    """
    Calculate pixel flipping metric following Bach et al. (2015) LRP paper.

    Args:
        explanation_func: Function that takes (model, img_array) and returns explanation
        model: Trained model
        img_array: Preprocessed input image array (shape: 1, H, W, C)
        num_samples: Number of flip percentage steps to test
        max_flip_percentage: Maximum percentage of pixels to flip

    Returns:
        pixel_flipping_score: Float value (lower = better explanation)
    """

    # Get original explanation and prediction
    original_explanation = explanation_func(model, img_array)
    original_output = model.predict(img_array, verbose=0)

    # Handle different output formats
    if len(original_output.shape) > 1:
        if original_output.shape[1] == 1:  # Binary classification
            original_pred_score = original_output[0][0]
            original_prediction = int(original_pred_score > 0.5)
        else:  # Multi-class classification
            original_prediction = np.argmax(original_output, axis=1)[0]
            original_pred_score = original_output[0][original_prediction]
    else:
        original_pred_score = original_output[0]
        original_prediction = int(original_pred_score > 0.5)

    # Normalize explanation to consistent format
    def normalize_explanation(explanation):
        if explanation is None:
            return np.array([])

        if isinstance(explanation, tuple):
            explanation = explanation[0]

        explanation = np.array(explanation)

        # Handle different explanation shapes
        if len(explanation.shape) == 4:  # (1, H, W, C)
            explanation = explanation[0]
        elif len(explanation.shape) == 2:  # (H, W) - typical for GradCAM
            # Resize to match input image dimensions if needed
            if explanation.shape != (img_array.shape[1], img_array.shape[2]):
                explanation = cv2.resize(explanation, (img_array.shape[2], img_array.shape[1]))
            # Expand to match input channels
            explanation = np.stack([explanation] * img_array.shape[3], axis=-1)
        elif len(explanation.shape) == 3:  # (H, W, C)
            if explanation.shape[2] != img_array.shape[3]:
                # Average channels if mismatch
                explanation = np.mean(explanation, axis=2, keepdims=True)
                explanation = np.repeat(explanation, img_array.shape[3], axis=2)

        return explanation.flatten()

    # Get flattened explanation
    try:
        explanation_flat = normalize_explanation(original_explanation)
        img_flat = img_array.flatten()

        # Ensure same length
        min_len = min(len(explanation_flat), len(img_flat))
        explanation_flat = explanation_flat[:min_len]
        img_flat = img_flat[:min_len]

    except Exception as e:
        print(f"Warning: Error processing explanation shape: {e}")
        return float('inf')  # Return worst score if explanation processing fails

    # Sort pixels by importance (highest relevance first)
    sorted_indices = np.argsort(explanation_flat)[::-1]  # Descending order

    # Test different flip percentages
    flip_percentages = np.linspace(1, max_flip_percentage, num_samples)
    confidence_scores = [original_pred_score]  # Start with original confidence
    flip_points = [0]  # Start with 0% flipped

    for flip_percentage in flip_percentages:
        try:
            # Calculate how many pixels to flip
            num_pixels_to_flip = int((flip_percentage / 100.0) * len(explanation_flat))
            if num_pixels_to_flip == 0:
                continue

            pixels_to_flip = sorted_indices[:num_pixels_to_flip]

            # Create flipped image
            flipped_img_flat = img_flat.copy()

            # Flip pixels: flipped = pixel * (-1) as in paper
            flipped_img_flat[pixels_to_flip] = flipped_img_flat[pixels_to_flip] * (-1)

            # Reshape back to original image dimensions
            flipped_img = flipped_img_flat.reshape(img_array.shape)

            # Clip to valid range (same as your other functions)
            flipped_img = np.clip(flipped_img, -2.5, 2.5)

            # Get prediction on flipped image
            flipped_output = model.predict(flipped_img, verbose=0)

            # Extract confidence score for original predicted class
            if len(flipped_output.shape) > 1:
                if flipped_output.shape[1] == 1:  # Binary classification
                    if original_prediction == 1:
                        confidence = flipped_output[0][0]
                    else:
                        confidence = 1 - flipped_output[0][0]
                else:  # Multi-class classification
                    confidence = flipped_output[0][original_prediction]
            else:
                if original_prediction == 1:
                    confidence = flipped_output[0]
                else:
                    confidence = 1 - flipped_output[0]

            confidence_scores.append(confidence)
            flip_points.append(flip_percentage)

        except Exception as e:
            print(f"Warning: Error at flip percentage {flip_percentage:.1f}%: {e}")
            continue

    # Calculate AUC (Area Under Curve)
    if len(confidence_scores) < 2:
        print("Warning: Insufficient valid samples for pixel flipping calculation")
        return float('inf')

    try:
        # Normalize x-axis to [0,1] range
        x = np.array(flip_points) / max_flip_percentage
        y = np.array(confidence_scores)

        # Calculate AUC using trapezoidal rule
        auc = np.trapz(y, x)

        # Return log-transformed score (similar to your RIS calculation)
        # Lower AUC = better explanation, but log ensures positive values
        normalized_score = auc / max(confidence_scores)  # Normalize by max confidence

        return float(normalized_score)

    except Exception as e:
        print(f"Warning: Error calculating AUC: {e}")
        return float('inf')


def calculate_model_parameter_randomization(explanation_func, model, input_tensor,
                                               randomization_type='cascading',
                                               similarity_metrics=['spearman_abs', 'spearman_raw', 'ssim'],
                                               return_detailed=False,
                                               framework='auto'):
    """
    Calculate model parameter randomization test for both PyTorch and TensorFlow models.

    Args:
        explanation_func: Function that takes (model, input) and returns explanation
        model: The trained model (PyTorch or TensorFlow)
        input_tensor: Input tensor/array
        randomization_type: 'cascading' or 'independent'
        similarity_metrics: List of metrics to compute
        return_detailed: If True, return detailed results for each layer
        framework: 'pytorch', 'tensorflow', or 'auto' to detect

    Returns:
        dict: Results containing similarity scores and pass/fail assessment
    """

    # Auto-detect framework
    if framework == 'auto':
        if hasattr(model, 'eval') and hasattr(model, 'parameters'):
            framework = 'pytorch'
        elif hasattr(model, 'layers') or 'tensorflow' in str(type(model)):
            framework = 'tensorflow'
        else:
            raise ValueError("Could not detect framework. Please specify 'pytorch' or 'tensorflow'")

    def reinitialize_pytorch_layer(layer):
        """Reinitialize a PyTorch layer's parameters"""
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        else:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if hasattr(layer, 'weight') and layer.weight is not None:
                    nn.init.ones_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def reinitialize_tensorflow_layer(layer):
        """Reinitialize a TensorFlow layer's parameters"""
        if hasattr(layer, 'kernel') and layer.kernel is not None:
            # For Conv2D, Dense layers
            layer.kernel.assign(tf.keras.initializers.GlorotUniform()(layer.kernel.shape))
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.assign(tf.zeros_like(layer.bias))
        if hasattr(layer, 'gamma') and layer.gamma is not None:
            # For BatchNormalization
            layer.gamma.assign(tf.ones_like(layer.gamma))
        if hasattr(layer, 'beta') and layer.beta is not None:
            layer.beta.assign(tf.zeros_like(layer.beta))

    def get_model_layers(model, framework):
        """Get model layers based on framework"""
        if framework == 'pytorch':
            modules = list(model.named_modules())
            # Filter out the root module and any modules without parameters
            modules = [(name, module) for name, module in modules
                      if name != '' and any(p.numel() > 0 for p in module.parameters())]
            return list(reversed(modules))  # Top layers first

        elif framework == 'tensorflow':
            # For TensorFlow/Keras models
            layers = []
            for i, layer in enumerate(model.layers):
                if len(layer.get_weights()) > 0:  # Only layers with parameters
                    layers.append((f"layer_{i}_{layer.name}", layer))
            return list(reversed(layers))  # Top layers first

    def set_model_mode(model, framework, training=False):
        """Set model to training or evaluation mode"""
        if framework == 'pytorch':
            if training:
                model.train()
            else:
                model.eval()
        # TensorFlow models don't need explicit eval() mode for inference

    def copy_model(model, framework):
        """Create a copy of the model"""
        if framework == 'pytorch':
            return copy.deepcopy(model)
        elif framework == 'tensorflow':
            # For TensorFlow, we'll clone the model
            return tf.keras.models.clone_model(model)

    # def calculate_similarity(exp1, exp2, metrics):
    #     """Calculate similarity between two explanations"""
    #     results = {}

    #     # Convert to numpy arrays
    #     if hasattr(exp1, 'numpy'):
    #         exp1 = exp1.numpy()
    #     elif torch.is_tensor(exp1):
    #         exp1 = exp1.detach().cpu().numpy()

    #     if hasattr(exp2, 'numpy'):
    #         exp2 = exp2.numpy()
    #     elif torch.is_tensor(exp2):
    #         exp2 = exp2.detach().cpu().numpy()
    
    def calculate_similarity(exp1, exp2, metrics):
        """Calculate similarity between two explanations"""
        results = {}

        # Convert to numpy arrays
        if hasattr(exp1, 'numpy'):
            exp1 = exp1.numpy()
        elif _HAS_TORCH and torch.is_tensor(exp1):
            exp1 = exp1.detach().cpu().numpy()

        if hasattr(exp2, 'numpy'):
            exp2 = exp2.numpy()
        elif _HAS_TORCH and torch.is_tensor(exp2):
            exp2 = exp2.detach().cpu().numpy()

        # Flatten for correlation calculations
        exp1_flat = exp1.flatten()
        exp2_flat = exp2.flatten()

        if 'spearman_abs' in metrics:
            corr, _ = spearmanr(np.abs(exp1_flat), np.abs(exp2_flat))
            results['spearman_abs'] = corr if not np.isnan(corr) else 0.0

        if 'spearman_raw' in metrics:
            corr, _ = spearmanr(exp1_flat, exp2_flat)
            results['spearman_raw'] = corr if not np.isnan(corr) else 0.0

        if 'ssim' in metrics and len(exp1.shape) >= 2:
            # For SSIM, we need 2D images
            if len(exp1.shape) > 2:
                exp1_2d = exp1[0] if exp1.shape[0] < exp1.shape[-1] else exp1[..., 0]
                exp2_2d = exp2[0] if exp2.shape[0] < exp2.shape[-1] else exp2[..., 0]
            else:
                exp1_2d, exp2_2d = exp1, exp2

            # Normalize to [0, 1] for SSIM
            exp1_norm = (exp1_2d - exp1_2d.min()) / (exp1_2d.max() - exp1_2d.min() + 1e-8)
            exp2_norm = (exp2_2d - exp2_2d.min()) / (exp2_2d.max() - exp2_2d.min() + 1e-8)

            try:
                ssim_score = ssim(exp1_norm, exp2_norm, data_range=1.0)
                results['ssim'] = ssim_score
            except:
                results['ssim'] = 0.0

        return results

    # Get original explanation
    set_model_mode(model, framework, training=False)

    if framework == 'pytorch':
        with torch.no_grad():
            original_explanation = explanation_func(model, input_tensor)
    else:
        original_explanation = explanation_func(model, input_tensor)

    # Get list of layers to randomize
    named_layers = get_model_layers(model, framework)

    detailed_results = {}
    similarity_scores = {metric: [] for metric in similarity_metrics}

    if randomization_type == 'cascading':
        # Cascading randomization: progressively randomize from top to bottom
        if framework == 'tensorflow':
            # For TensorFlow, we need to copy weights and modify them
            model_copy = copy_model(model, framework)
            model_copy.set_weights(model.get_weights())  # Copy original weights
        else:
            model_copy = copy_model(model, framework)

        for i, (layer_name, layer) in enumerate(named_layers):
            # Randomize this layer
            if framework == 'pytorch':
                target_layer = dict(model_copy.named_modules())[layer_name]
                reinitialize_pytorch_layer(target_layer)
            else:
                # Find corresponding layer in copied model
                target_layer = model_copy.layers[int(layer_name.split('_')[1])]
                reinitialize_tensorflow_layer(target_layer)

            # Get explanation with partially randomized model
            set_model_mode(model_copy, framework, training=False)

            if framework == 'pytorch':
                with torch.no_grad():
                    randomized_explanation = explanation_func(model_copy, input_tensor)
            else:
                randomized_explanation = explanation_func(model_copy, input_tensor)

            # Calculate similarity
            similarities = calculate_similarity(original_explanation, randomized_explanation, similarity_metrics)
            detailed_results[f"after_randomizing_{layer_name}"] = similarities

            for metric in similarity_metrics:
                similarity_scores[metric].append(similarities.get(metric, 0.0))

    elif randomization_type == 'independent':
        # Independent randomization: randomize each layer independently
        for layer_name, layer in named_layers:
            if framework == 'tensorflow':
                model_copy = copy_model(model, framework)
                model_copy.set_weights(model.get_weights())  # Copy original weights
                target_layer = model_copy.layers[int(layer_name.split('_')[1])]
                reinitialize_tensorflow_layer(target_layer)
            else:
                model_copy = copy_model(model, framework)
                target_layer = dict(model_copy.named_modules())[layer_name]
                reinitialize_pytorch_layer(target_layer)

            # Get explanation with this single layer randomized
            set_model_mode(model_copy, framework, training=False)

            if framework == 'pytorch':
                with torch.no_grad():
                    randomized_explanation = explanation_func(model_copy, input_tensor)
            else:
                randomized_explanation = explanation_func(model_copy, input_tensor)

            # Calculate similarity
            similarities = calculate_similarity(original_explanation, randomized_explanation, similarity_metrics)
            detailed_results[f"randomized_{layer_name}"] = similarities

            for metric in similarity_metrics:
                similarity_scores[metric].append(similarities.get(metric, 0.0))

    # Calculate summary statistics
    summary_results = {}
    for metric in similarity_metrics:
        scores = similarity_scores[metric]
        if scores:
            summary_results[f'{metric}_mean'] = np.mean(scores)
            summary_results[f'{metric}_std'] = np.std(scores)
            summary_results[f'{metric}_min'] = np.min(scores)
            summary_results[f'{metric}_max'] = np.max(scores)

            # Sanity check assessment
            if metric == 'spearman_abs':
                summary_results[f'{metric}_passes_sanity_check'] = np.mean(scores) < 0.5
            elif metric == 'spearman_raw':
                summary_results[f'{metric}_passes_sanity_check'] = np.abs(np.mean(scores)) < 0.3
            elif metric == 'ssim':
                summary_results[f'{metric}_passes_sanity_check'] = np.mean(scores) < 0.5

    # Overall pass/fail
    passes_checks = [summary_results.get(f'{metric}_passes_sanity_check', False)
                    for metric in similarity_metrics
                    if f'{metric}_passes_sanity_check' in summary_results]

    summary_results['overall_passes_sanity_check'] = any(passes_checks) if passes_checks else False
    summary_results['randomization_type'] = randomization_type
    summary_results['framework'] = framework

    if return_detailed:
        return {
            'summary': summary_results,
            'detailed': detailed_results,
            'similarity_scores': similarity_scores
        }
    else:
        return summary_results


# ---------------------------------------------------------------------------
# Silence noisy third-party output BEFORE importing the libraries that emit it
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
XAI_METHODS = {
    'gradcam':  True,
    'lime':     True,
    'saliency': True,
}

model_path       = "resnet50_sk_detection_finetuned.h5"
malignant_folder = "skin-cancer/train/malignant"
benign_folder    = "skin-cancer/train/benign"

NUM_IMAGES = 10


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
model = load_model(model_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_image_files(folder_path, num_images=30):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    return image_files[:num_images]


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array_expanded)
    return img, img_preprocessed


# ---------------------------------------------------------------------------
# Explanation functions
# ---------------------------------------------------------------------------
def gradcam_explanation_func(model, img_array):
    last_conv_layer_name = "conv5_block3_out"
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Some saved models wrap the final output in a list — unwrap it.
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def lime_explanation_func(model, img_array):
    img_original = (img_array[0] + [103.939, 116.779, 123.68])
    img_original = np.clip(img_original, 0, 255)

    def model_predict_lime(images):
        images_preprocessed = np.array(
            [tf.keras.applications.resnet50.preprocess_input(im) for im in images]
        )
        return model.predict(images_preprocessed, verbose=0)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_original.astype('double'),
        model_predict_lime,
        top_labels=1,
        hide_color=0,
        num_samples=30,
    )

    top_pred_class = explanation.top_labels[0]
    _, mask = explanation.get_image_and_mask(
        top_pred_class,
        positive_only=True,
        num_features=5,
        hide_rest=False,
    )
    return mask.astype(float)


def saliency_explanation_func(model, img_array):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_score = predictions[0][0]

    gradients = tape.gradient(target_score, img_tensor).numpy()[0]
    saliency = np.max(np.abs(gradients), axis=-1)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency


# ---------------------------------------------------------------------------
# Metric calculation per image (assumes calculate_* funcs already in scope)
# ---------------------------------------------------------------------------
def calculate_metrics_for_method(method_name, explanation_func, explanation, model, img_preprocessed):
    metrics = {}


    # metrics['pixel_flipping_auc'] = calculate_pixel_flipping_auc(explanation_func, model, img_preprocessed)
    # metrics['completeness']       = calculate_completeness_error(explanation_func, model, img_preprocessed)
    # metrics['faithfulness_40_50']       = calculate_faithfulness(explanation_func, model, img_preprocessed, subset_size=40, num_samples=50, baseline_value=0.0)
    # metrics['infidelity']                          = calculate_infidelity(explanation, model, img_preprocessed)
    # metrics['max_sensitivity_constrained']         = calculate_max_sensitivity(explanation_func, model, img_preprocessed, preserve_prediction=True)
    # metrics['relative_input_stability']            = calculate_relative_input_stability(explanation_func, model, img_preprocessed)
    # metrics['relative_output_stability']           = calculate_relative_output_stability(explanation_func, model, img_preprocessed)
    # metrics['relative_representation_stability']   = calculate_rrs(explanation_func, model, img_preprocessed, normalize_representations=True)
    # metrics['complexity']                          = calculate_complexity(explanation_func, model, img_preprocessed)
    # metrics['irof']                                = calculate_irof(explanation_func, model, img_preprocessed)

    # mpr_results = calculate_model_parameter_randomization(explanation_func, model, img_preprocessed)
    # metrics['model_parameter_randomization_spearman_abs_mean'] = mpr_results.get('spearman_abs_mean', np.nan)

    metrics['faithfulness'] = calculate_faithfulness(
        explanation_func, model, img_preprocessed,
        preprocess_fn=tf.keras.applications.resnet50.preprocess_input,
    )

    return metrics


def process_images(image_files, class_name):
    all_results = []

    for i, img_path in enumerate(image_files):
        print(f"Processing {class_name} image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")

        img, img_preprocessed = preprocess_image(img_path)

        prediction      = model.predict(img_preprocessed, verbose=0)
        prob            = prediction[0][0]
        predicted_class = int(prob > 0.5)
        label           = "Malignant" if predicted_class == 1 else "Benign"
        confidence      = prob if predicted_class == 1 else 1 - prob

        image_results = {
            'image_path':      img_path,
            'true_class':      class_name,
            'predicted_class': label,
            'confidence':      confidence,
            'metrics':         {},
        }

        if XAI_METHODS['gradcam']:
            try:
                gradcam_explanation = gradcam_explanation_func(model, img_preprocessed)
                image_results['metrics']['gradcam'] = calculate_metrics_for_method(
                    'gradcam', gradcam_explanation_func, gradcam_explanation, model, img_preprocessed
                )
            except Exception as e:
                print(f"  GradCAM failed: {e}")

        if XAI_METHODS['lime']:
            try:
                lime_explanation = lime_explanation_func(model, img_preprocessed)
                image_results['metrics']['lime'] = calculate_metrics_for_method(
                    'lime', lime_explanation_func, lime_explanation, model, img_preprocessed
                )
            except Exception as e:
                print(f"  LIME failed: {e}")

        if XAI_METHODS['saliency']:
            try:
                saliency_explanation = saliency_explanation_func(model, img_preprocessed)
                image_results['metrics']['saliency'] = calculate_metrics_for_method(
                    'saliency', saliency_explanation_func, saliency_explanation, model, img_preprocessed
                )
            except Exception as e:
                print(f"  Saliency failed: {e}")

        all_results.append(image_results)

    return all_results


# ---------------------------------------------------------------------------
# Aggregation: mean ± std per method per metric
# ---------------------------------------------------------------------------
def calculate_metric_stats(results):
    """
    For each method, compute {metric_name: {'mean': ..., 'std': ..., 'n': ...}}
    averaged across the per-image metric dicts.
    """
    stats = {}

    for method in ['gradcam', 'lime', 'saliency']:
        if not XAI_METHODS[method]:
            continue

        method_metrics = [r['metrics'][method] for r in results if method in r['metrics']]
        if not method_metrics:
            continue

        stats[method] = {}
        metric_names = method_metrics[0].keys()

        for metric_name in metric_names:
            values = []
            for m in method_metrics:
                v = m.get(metric_name, None)
                if v is None:
                    continue
                if isinstance(v, (int, float, np.number)) and not (np.isnan(v) or np.isinf(v)):
                    values.append(float(v))

            if values:
                stats[method][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std':  float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'n':    len(values),
                }
            else:
                stats[method][metric_name] = {'mean': np.nan, 'std': np.nan, 'n': 0}

    return stats


def print_results(class_name, results, stats):
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {class_name.upper()} CLASS ({len(results)} images)")
    print(f"{'='*80}")

    for method in ['gradcam', 'lime', 'saliency']:
        if method not in stats:
            continue

        print(f"\n{method.upper()} — mean ± std (n samples used):")
        print("-" * 60)
        for metric_name, s in stats[method].items():
            label = metric_name.replace('_', ' ').title()
            if np.isnan(s['mean']):
                print(f"  {label}: NaN (n=0)")
            else:
                print(f"  {label}: {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")


def print_comparison(malignant_stats, benign_stats):
    print("\n" + "=" * 80)
    print("COMPARISON: MALIGNANT vs BENIGN  (mean ± std)")
    print("=" * 80)

    headline_metrics = [
        'infidelity', 'max_sensitivity_constrained',
        'relative_input_stability', 'relative_output_stability',
        'faithfulness',
    ]

    for method in ['gradcam', 'lime', 'saliency']:
        if method not in malignant_stats or method not in benign_stats:
            continue
        print(f"\n{method.upper()}")
        print("-" * 80)
        print(f"{'Metric':<35}{'Malignant':<22}{'Benign':<22}")
        for metric in headline_metrics:
            m = malignant_stats[method].get(metric)
            b = benign_stats[method].get(metric)
            if m is None or b is None:
                continue
            mal_str = "NaN" if np.isnan(m['mean']) else f"{m['mean']:.4f} ± {m['std']:.4f}"
            ben_str = "NaN" if np.isnan(b['mean']) else f"{b['mean']:.4f} ± {b['std']:.4f}"
            print(f"{metric.replace('_',' ').title():<35}{mal_str:<22}{ben_str:<22}")


# ---------------------------------------------------------------------------
# CSV saving
# ---------------------------------------------------------------------------
def save_results_to_csv(malignant_results, benign_results, malignant_stats, benign_stats):
    try:
        # Per-image rows
        rows = []
        for result in malignant_results + benign_results:
            row = {
                'image_path':      result['image_path'],
                'true_class':      result['true_class'],
                'predicted_class': result['predicted_class'],
                'confidence':      result['confidence'],
            }
            for method in ['gradcam', 'lime', 'saliency']:
                if method in result['metrics']:
                    for metric_name, value in result['metrics'][method].items():
                        row[f'{method}_{metric_name}'] = value
            rows.append(row)
        pd.DataFrame(rows).to_csv('xai_individual_results.csv', index=False)
        print("\nIndividual results saved to: xai_individual_results.csv")

        # Mean ± std summary rows
        summary_rows = []
        for class_name, stats in [('Malignant', malignant_stats), ('Benign', benign_stats)]:
            row = {'class': class_name}
            for method in ['gradcam', 'lime', 'saliency']:
                if method in stats:
                    for metric_name, s in stats[method].items():
                        row[f'{method}_{metric_name}_mean'] = s['mean']
                        row[f'{method}_{metric_name}_std']  = s['std']
                        row[f'{method}_{metric_name}_n']    = s['n']
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv('xai_summary_mean_std.csv', index=False)
        print("Summary (mean ± std) saved to: xai_summary_mean_std.csv")

    except Exception as e:
        print(f"Error saving to CSV: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("XAI Batch Processing Started")
    print(f"Selected methods: {[m for m, on in XAI_METHODS.items() if on]}")
    print(f"Processing {NUM_IMAGES} images from each class")

    malignant_files = get_image_files(malignant_folder, NUM_IMAGES)
    benign_files    = get_image_files(benign_folder, NUM_IMAGES)

    print(f"\nFound {len(malignant_files)} malignant images")
    print(f"Found {len(benign_files)} benign images")

    if not malignant_files or not benign_files:
        print("Error: No images found in one or both folders. Please check the paths.")
        return

    print("\n" + "=" * 80)
    print("PROCESSING MALIGNANT IMAGES")
    print("=" * 80)
    malignant_results = process_images(malignant_files, "Malignant")
    malignant_stats   = calculate_metric_stats(malignant_results)
    print_results("Malignant", malignant_results, malignant_stats)

    print("\n" + "=" * 80)
    print("PROCESSING BENIGN IMAGES")
    print("=" * 80)
    benign_results = process_images(benign_files, "Benign")
    benign_stats   = calculate_metric_stats(benign_results)
    print_results("Benign", benign_results, benign_stats)

    print_comparison(malignant_stats, benign_stats)

    save_results_to_csv(malignant_results, benign_results, malignant_stats, benign_stats)


if __name__ == "__main__":
    main()
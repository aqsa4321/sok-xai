#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image classification pipeline for skin_cancer and cell_cancer datasets.

Runs 4 pretrained CNNs (ResNet50, EfficientNetB0, VGG16, InceptionV3),
5 runs each, with per-run model save. Uses transfer learning:
Phase 1 trains classifier head with frozen base, Phase 2 fine-tunes
the top layers of the base model.

To switch datasets, change the DATASET constant below.
Both datasets should have a train/{class}/ and test/{class}/ layout.
(For cell_cancer, run preprocess/prep_images.py first to create the split.)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, InceptionV3, EfficientNetB0, ResNet50
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("TensorFlow version:", tf.__version__)
print("GPUs detected:", tf.config.list_physical_devices('GPU'))

# ---------------------------------------------------------------------------
# Configuration -- change DATASET to switch between skin_cancer and cell_cancer
# ---------------------------------------------------------------------------
DATASET = "skin_cancer"          # or "cell_cancer"

REPO_ROOT   = "/home/aqsa/xai-cdss"
BASE_DIR    = os.path.join(REPO_ROOT, "data", DATASET)
TRAIN_DIR   = os.path.join(BASE_DIR, "train")
TEST_DIR    = os.path.join(BASE_DIR, "test")
MODELS_DIR  = os.path.join(REPO_ROOT, "results", "image", f"models_{DATASET}")
RESULTS_CSV = os.path.join(REPO_ROOT, "results", "image", f"{DATASET}_models_info.csv")

IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 224, 224, 32
NUM_RUNS = 5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
print(f"Dataset: {DATASET}")
print(f"Models will be saved to: {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def create_generators(preprocess_func, seed=None):
    """Create train/val/test generators for a given preprocessing function."""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', subset='training', seed=seed)
    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', subset='validation', seed=seed)
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)
    return train_gen, val_gen, test_gen


def evaluate_model(model, test_gen):
    """Return (accuracy, precision, recall, f1) on the test set."""
    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    return (
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, average='weighted', zero_division=0),
        recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f1_score(y_true, y_pred, average='weighted', zero_division=0),
    )


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def _head(base_output, use_flatten=False):
    x = Flatten()(base_output) if use_flatten else GlobalAveragePooling2D()(base_output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    return Dense(1, activation='sigmoid')(x)


def build_vgg16():
    base = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    return Model(base.input, _head(base.output, use_flatten=True)), base


def build_inceptionv3():
    base = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    return Model(base.input, _head(base.output)), base


def build_efficientnet():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    return Model(base.input, _head(base.output)), base


def build_resnet50():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    return Model(base.input, _head(base.output)), base


# ---------------------------------------------------------------------------
# Fine-tuning strategies -- unfreeze the top-N layers of each backbone
# ---------------------------------------------------------------------------
def vgg16_unfreeze(base):
    for layer in base.layers:
        if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True


def inception_unfreeze(base):
    for layer in base.layers[-30:]:
        layer.trainable = True


def efficientnet_unfreeze(base):
    for layer in base.layers[-20:]:
        layer.trainable = True


def resnet_unfreeze(base):
    for layer in base.layers[-10:]:
        layer.trainable = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(model_name, build_func, preprocess_func, unfreeze_strategy,
                epochs_config, run_num):
    """Train one model for one run; save weights; return metrics."""
    print(f"  Run {run_num + 1}/{NUM_RUNS}...")
    tf.random.set_seed(42 + run_num)
    np.random.seed(42 + run_num)

    train_gen, val_gen, test_gen = create_generators(preprocess_func, seed=42 + run_num)
    model, base = build_func()

    # Phase 1: frozen base
    for layer in base.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=epochs_config[0], validation_data=val_gen, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    # Phase 2: fine-tune top layers
    unfreeze_strategy(base)
    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=epochs_config[1], validation_data=val_gen, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    metrics = evaluate_model(model, test_gen)
    save_path = os.path.join(MODELS_DIR, f"{model_name.lower()}_run_{run_num + 1}.h5")
    model.save(save_path)
    print(f"    Saved: {os.path.basename(save_path)}")
    return metrics


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
models_config = {
    "ResNet50": {
        "build_func": build_resnet50,
        "preprocess_func": resnet_preprocess,
        "unfreeze_strategy": resnet_unfreeze,
        "epochs": (10, 10),
    },
    "EfficientNetB0": {
        "build_func": build_efficientnet,
        "preprocess_func": efficientnet_preprocess,
        "unfreeze_strategy": efficientnet_unfreeze,
        "epochs": (10, 10),
    },
    "VGG16": {
        "build_func": build_vgg16,
        "preprocess_func": vgg_preprocess,
        "unfreeze_strategy": vgg16_unfreeze,
        "epochs": (10, 10),
    },
    "InceptionV3": {
        "build_func": build_inceptionv3,
        "preprocess_func": inception_preprocess,
        "unfreeze_strategy": inception_unfreeze,
        "epochs": (10, 10),
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_results = {}
    saved_models_info = []

    for model_name, config in models_config.items():
        print(f"\n{'=' * 50}")
        print(f"Training {model_name} - {NUM_RUNS} runs on {DATASET}")
        print('=' * 50)

        model_results = []
        best_accuracy, best_model_run = 0.0, 0

        for run in range(NUM_RUNS):
            metrics = train_model(
                model_name, config["build_func"], config["preprocess_func"],
                config["unfreeze_strategy"], config["epochs"], run,
            )
            model_results.append(metrics)
            acc = metrics[0]
            if acc > best_accuracy:
                best_accuracy, best_model_run = acc, run + 1

            saved_models_info.append({
                "dataset":   DATASET,
                "model":     model_name,
                "run":       run + 1,
                "filename":  f"{model_name.lower()}_run_{run + 1}.h5",
                "accuracy":  metrics[0],
                "precision": metrics[1],
                "recall":    metrics[2],
                "f1":        metrics[3],
            })
            print(f"    Acc={metrics[0]:.4f}  Prec={metrics[1]:.4f}  "
                  f"Rec={metrics[2]:.4f}  F1={metrics[3]:.4f}")

        arr = np.array(model_results)
        avg, std = arr.mean(axis=0), arr.std(axis=0)
        all_results[model_name] = {
            "avg_accuracy":  avg[0], "std_accuracy":  std[0],
            "avg_precision": avg[1], "std_precision": std[1],
            "avg_recall":    avg[2], "std_recall":    std[2],
            "avg_f1":        avg[3], "std_f1":        std[3],
            "best_accuracy": best_accuracy, "best_run": best_model_run,
        }

        print(f"\n{model_name} Summary ({NUM_RUNS} runs):")
        print(f"  Avg Accuracy:  {avg[0]:.4f} ± {std[0]:.4f}")
        print(f"  Avg Precision: {avg[1]:.4f} ± {std[1]:.4f}")
        print(f"  Avg Recall:    {avg[2]:.4f} ± {std[2]:.4f}")
        print(f"  Avg F1-Score:  {avg[3]:.4f} ± {std[3]:.4f}")
        print(f"  Best Accuracy: {best_accuracy:.4f} (Run {best_model_run})")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"FINAL RESULTS SUMMARY -- {DATASET.upper()} -- {NUM_RUNS} runs")
    print('=' * 70)
    print(f"{'Model':<16} {'Avg Accuracy':<18} {'Best Acc':<12} {'Best Run':<10}")
    print('-' * 70)
    best_avg_name, best_avg_acc = "", 0.0
    for name, r in all_results.items():
        print(f"{name:<16} {r['avg_accuracy']:.4f} ± {r['std_accuracy']:.4f}  "
              f"{r['best_accuracy']:.4f}       Run {r['best_run']}")
        if r['avg_accuracy'] > best_avg_acc:
            best_avg_acc, best_avg_name = r['avg_accuracy'], name
    print(f"\nBest Average Performance: {best_avg_name} ({best_avg_acc:.4f})")

    # Persist per-run info to CSV
    pd.DataFrame(saved_models_info).to_csv(RESULTS_CSV, index=False)
    print(f"\nPer-run info saved to: {RESULTS_CSV}")
    print(f"All model weights saved to: {MODELS_DIR}/")
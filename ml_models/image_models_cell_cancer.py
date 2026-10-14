import os
import tensorflow as tf
import numpy as np
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

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Model save directory
SAVE_DIR = 'models/'

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Models will be saved to: {SAVE_DIR}")

IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE = 224, 224, 32
NUM_RUNS = 5  # Number of times to run each model

def create_generators(preprocess_func, seed=None):
    """Create data generators for a specific preprocessing function"""
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', subset='training', seed=seed)

    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', subset='validation', seed=seed)

    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False)

    return train_gen, val_gen, test_gen

def evaluate_model(model, test_gen):
    """Evaluate model and return metrics"""
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    y_pred_probs = model.predict(test_gen, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

def build_vgg16():
    """Build VGG16 model"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(base_model.input, output), base_model

def build_inceptionv3():
    """Build InceptionV3 model"""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(base_model.input, output), base_model

def build_efficientnet():
    """Build EfficientNetB0 model"""
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(base_model.input, output), base_model

def build_resnet50():
    """Build ResNet50 model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(base_model.input, output), base_model

def train_model(model_name, build_func, preprocess_func, unfreeze_strategy, epochs_config, run_num):
    """Train a model and return metrics"""
    print(f"  Run {run_num + 1}/5...")

    # Set seed for reproducibility within each run
    tf.random.set_seed(42 + run_num)
    np.random.seed(42 + run_num)

    # Create generators
    train_gen, val_gen, test_gen = create_generators(preprocess_func, seed=42 + run_num)

    # Build model
    model, base_model = build_func()

    # Phase 1: Frozen training
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=epochs_config[0], validation_data=val_gen, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    # Phase 2: Fine-tuning
    unfreeze_strategy(base_model)
    model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, epochs=epochs_config[1], validation_data=val_gen, verbose=0,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    # Evaluate
    metrics = evaluate_model(model, test_gen)

    # Save ALL models with run number
    model_filename = f'{model_name.lower()}_run_{run_num + 1}.h5'
    model_path = os.path.join(SAVE_DIR, model_filename)
    model.save(model_path)
    print(f"    Saved: {model_filename}")

    return metrics, model

def vgg16_unfreeze(base_model):
    """VGG16 specific unfreezing strategy"""
    for layer in base_model.layers:
        if layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
            layer.trainable = True

def inception_unfreeze(base_model):
    """InceptionV3 specific unfreezing strategy"""
    for layer in base_model.layers[-30:]:
        layer.trainable = True

def efficientnet_unfreeze(base_model):
    """EfficientNetB0 specific unfreezing strategy"""
    for layer in base_model.layers[-20:]:
        layer.trainable = True

def resnet_unfreeze(base_model):
    """ResNet50 specific unfreezing strategy"""
    for layer in base_model.layers[-10:]:
        layer.trainable = True

# ---------------------------------------------------
# Model configurations
# ---------------------------------------------------
models_config = {


    "ResNet50": {
        "build_func": build_resnet50,
        "preprocess_func": resnet_preprocess,
        "unfreeze_strategy": resnet_unfreeze,
        "epochs": (10, 10)
    },
        "EfficientNetB0": {
        "build_func": build_efficientnet,
        "preprocess_func": efficientnet_preprocess,
        "unfreeze_strategy": efficientnet_unfreeze,
        "epochs": (10, 10)
    },
      "VGG16": {
        "build_func": build_vgg16,
        "preprocess_func": vgg_preprocess,
        "unfreeze_strategy": vgg16_unfreeze,
        "epochs": (10, 10)  # (frozen_epochs, finetune_epochs)
    },
    "InceptionV3": {
        "build_func": build_inceptionv3,
        "preprocess_func": inception_preprocess,
        "unfreeze_strategy": inception_unfreeze,
        "epochs": (10, 10)
    }
}

# ---------------------------------------------------
# Run all models multiple times
# ---------------------------------------------------
all_results = {}
saved_models_info = []

for model_name, config in models_config.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name} - {NUM_RUNS} runs")
    print('='*50)

    model_results = []
    best_accuracy = 0
    best_model_run = 0

    for run in range(NUM_RUNS):
        metrics, trained_model = train_model(
            model_name,
            config["build_func"],
            config["preprocess_func"],
            config["unfreeze_strategy"],
            config["epochs"],
            run
        )

        model_results.append(metrics)
        accuracy = metrics[0]

        # Keep track of best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_run = run + 1

        # Store model info
        saved_models_info.append({
            'model_name': model_name,
            'run': run + 1,
            'filename': f'{model_name.lower()}_run_{run + 1}.h5',
            'accuracy': accuracy,
            'precision': metrics[1],
            'recall': metrics[2],
            'f1_score': metrics[3]
        })

        print(f"    Accuracy: {accuracy:.4f}, Precision: {metrics[1]:.4f}, "
              f"Recall: {metrics[2]:.4f}, F1: {metrics[3]:.4f}")

    # Calculate averages and std
    model_results = np.array(model_results)
    avg_metrics = np.mean(model_results, axis=0)
    std_metrics = np.std(model_results, axis=0)

    all_results[model_name] = {
        'avg_accuracy': avg_metrics[0],
        'avg_precision': avg_metrics[1],
        'avg_recall': avg_metrics[2],
        'avg_f1': avg_metrics[3],
        'std_accuracy': std_metrics[0],
        'std_precision': std_metrics[1],
        'std_recall': std_metrics[2],
        'std_f1': std_metrics[3],
        'best_accuracy': best_accuracy,
        'best_run': best_model_run
    }

    print(f"\n{model_name} Summary ({NUM_RUNS} runs):")
    print(f"  Average Accuracy: {avg_metrics[0]:.4f} ± {std_metrics[0]:.4f}")
    print(f"  Average Precision: {avg_metrics[1]:.4f} ± {std_metrics[1]:.4f}")
    print(f"  Average Recall: {avg_metrics[2]:.4f} ± {std_metrics[2]:.4f}")
    print(f"  Average F1-Score: {avg_metrics[3]:.4f} ± {std_metrics[3]:.4f}")
    print(f"  Best Accuracy: {best_accuracy:.4f} (Run {best_model_run})")

# ---------------------------------------------------
# Final Results Summary
# ---------------------------------------------------
print(f"\n{'='*70}")
print(f"FINAL RESULTS SUMMARY - {NUM_RUNS} RUNS AVERAGE")
print('='*70)
print(f"{'Model':<15} {'Avg Accuracy':<15} {'Std':<10} {'Best Acc':<10} {'Best Run':<10}")
print('-'*70)

best_avg_model = ""
best_avg_accuracy = 0

for model_name, results in all_results.items():
    avg_acc = results['avg_accuracy']
    std_acc = results['std_accuracy']
    best_acc = results['best_accuracy']
    best_run = results['best_run']

    print(f"{model_name:<15} {avg_acc:.4f}         ±{std_acc:.4f}   {best_acc:.4f}    Run {best_run}")

    if avg_acc > best_avg_accuracy:
        best_avg_accuracy = avg_acc
        best_avg_model = model_name

print(f"\nBest Average Performance: {best_avg_model} ({best_avg_accuracy:.4f})")

# ---------------------------------------------------
# Save model information to CSV
# ---------------------------------------------------
import pandas as pd

models_df = pd.DataFrame(saved_models_info)
csv_path = os.path.join(SAVE_DIR, 'models_info.csv')
models_df.to_csv(csv_path, index=False)

print(f"\n{'='*70}")
print("SAVED MODELS SUMMARY")
print('='*70)
print(f"Total models saved: {len(saved_models_info)}")
print(f"Save location: {SAVE_DIR}")
print(f"Models info saved to: {csv_path}")

print(f"\nSaved model files:")
for info in saved_models_info:
    print(f"  {info['filename']} - Accuracy: {info['accuracy']:.4f}")

print(f"\nFile structure in {SAVE_DIR}:")
saved_files = os.listdir(SAVE_DIR)
for file in sorted(saved_files):
    if file.endswith('.h5'):
        print(f"  ✓ {file}")
    elif file.endswith('.csv'):
        print(f"  {file}")

print(f"\nTraining complete! All {len(models_config) * NUM_RUNS} models saved successfully.")
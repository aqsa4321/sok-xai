# Reproducibility Documentation

Documentation for reproducing the results in the paper. Everything below was extracted directly from the code and system used to generate the results.

## 1. Random seeds

All experiments use the fixed seed 42 as the base. Where multiple runs are performed per experiment, per-run seeds are derived deterministically:

- train_test_split(random_state=...): 42 (fixed across all runs)
- Classical ML random_state= in constructors (RF, DT, AdaBoost, GBM, XGBoost, LightGBM, EBM, SVM): 42
- Deep-learning training seeds (LSTM/GRU/CNN, tabular): tf.random.set_seed(i * 42) and np.random.seed(i * 42) per run i
- Image CNN training seeds (ResNet50/VGG16/InceptionV3/EfficientNetB0): tf.random.set_seed(42 + run_num), np.random.seed(42 + run_num)
- Bootstrap resampling (Task 1): random_state=42, n_resamples=10000
- XAI stability perturbation seeds: seed = j * 100 + 7 for the j-th perturbation

## 2. Train/test split ratios and procedure

- Split: 80% train / 20% test for every dataset
- Procedure: sklearn.model_selection.train_test_split(..., test_size=0.2, random_state=42, stratify=y)
- Stratification: yes, on the class label -- critical for MIMIC (imbalanced) and Parkinson's (imbalanced)
- Split reproducibility: split indices are saved (idx_train.npy, idx_test.npy) alongside each model so downstream XAI/metrics use the exact same test set
- DL internal validation split: an additional 20% of the training set is held out for early stopping (model.fit(..., validation_split=0.2))

## 3. Number of repetitions per experiment

- Model training (all classical ML + DL, both tabular and image): 5 runs each, mean +/- std reported
- XAI stability metrics (RIS/ROS/RRS, Max-Sensitivity): 5 perturbations per sample (STABILITY_M=5, n_samples=5 in image code)
- XAI consistency metric: 5 repeated explanations per sample (CONSISTENCY_M=5)
- XAI fidelity metric: 20 perturbations per sample (FIDELITY_M=20; paper cites 1000 but this was infeasible at our scale)
- Bootstrap CI (Task 1): 10,000 resamples per test

## 4. Model hyperparameters

### Classical ML (identical across breast, parkinson, mimic scripts)

- KNN: n_neighbors=5 (sklearn default)
- SVM: kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42
- Random Forest: n_estimators=100, random_state=42
- Decision Tree: random_state=42
- AdaBoost: n_estimators=100, random_state=42
- GBM: n_estimators=100, random_state=42
- XGBoost: n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0
- LightGBM: n_estimators=100, random_state=42, verbose=-1
- EBM: ExplainableBoostingClassifier(random_state=42) (all other hyperparameters default)

### Deep learning on tabular data (breast, parkinson, mimic)

- GRU: GRU(64, relu) -> Dropout(0.2) -> Dense(32, relu) -> Dropout(0.2) -> Dense(1, sigmoid)
- LSTM: LSTM(128, tanh) -> Dropout(0.2) -> Dense(32, relu) -> Dropout(0.2) -> Dense(1, sigmoid)
- 1D-CNN: Conv1D(128, kernel=3, relu) -> MaxPool(2) -> Flatten -> Dense(64, relu) -> Dropout(0.2) -> Dense(1, sigmoid)

Training:
- Optimizer: Adam with default 1e-3 learning rate
- Loss: binary_crossentropy
- Batch size: 32
- Max epochs: 100
- Callbacks: EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True), ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7)

### Image CNNs (skin cancer, cell cancer)

Base models: VGG16, InceptionV3, EfficientNetB0, ResNet50, all initialised with ImageNet weights, include_top=False.

Classifier head:
- VGG16: Flatten -> Dense(256, relu) -> Dropout(0.5) -> Dense(1, sigmoid)
- Others: GlobalAveragePooling2D -> Dense(256, relu) -> Dropout(0.5) -> Dense(1, sigmoid)

Two-phase training:
1. Frozen base: optimizer=Adam(1e-4), 10 epochs, EarlyStopping(patience=5, restore_best_weights=True)
2. Fine-tuning: optimizer=Adam(1e-5), 10 epochs, same early stopping. Unfrozen layers per backbone: VGG16 = last 3 conv blocks; InceptionV3 = last 30 layers; EfficientNetB0 = last 20 layers; ResNet50 = last 10 layers

Image size: 224 x 224, batch size 32, loss binary_crossentropy.

Data augmentation (training only): rotation +/- 20 deg, width/height shift 0.2, shear 0.2, zoom 0.2, horizontal flip, plus each backbone's own preprocess_input.

## 5. XAI method parameters

### SHAP
- Explainer: shap.KernelExplainer for opaque models (tabular). For deep models we used a wrapped-predict_proba closure.
- Background dataset: first bg_size rows of X_train (deterministic subset). Standard value: 100 rows.

### LIME
- Tabular: LimeTabularExplainer(training_data=X_train, mode="classification", discretize_continuous=True)
- Image: lime_image.LimeImageExplainer()
- num_samples for local surrogate fitting: default 5000 for tabular, 1000 for image (LIME defaults, not overridden)

### EBM
- Uses its own built-in local explanation via ebm_model.explain_local(x), which returns per-feature contributions from the additive model. No hyperparameters needed.

### Grad-CAM / Saliency Map (image)
- Grad-CAM: gradients w.r.t. final conv layer
- Saliency Map: absolute value of gradients w.r.t. input pixels

## 6. Metric computation parameters

### Fidelity
- FIDELITY_K = 5 -- top-k features to keep as "important"
- FIDELITY_M = 20 -- number of noisings per sample
- FIDELITY_SIGMA = 0.1 -- Gaussian noise std for perturbation

### Stability (RIS / ROS / RRS / Max-Sensitivity)
- STABILITY_M = 5 (tabular), n_samples = 5 (image)
- STABILITY_SIGMA = 0.1 (tabular). Image: noise_std=0.05 for RIS/ROS/RRS, noise_std=0.005 for MPRT, radius=0.1 for Max-Sensitivity
- epsilon_min = 1e-8 (or 1e-6) numerical stabiliser
- p_norm = 2

### Consistency
- CONSISTENCY_M = 5 -- repeated explanations of the same input

### Sparsity
- SPARSITY_T = 0.01 -- attribution magnitude threshold above which a feature is considered "significant"

### Pixel Flipping (image only)
- num_samples = 10 flip percentages, linearly spaced from 1% to 30%

### IROF (image only)
- SLIC segmentation: n_segments=30, compactness=10, sigma=1

### Model Parameter Randomization Test (MPRT)
- n_samples = 20, noise_std = 0.005, epsilon_min = 1e-8
- randomization_type = "full" (all layers randomized)

## 7. Library versions

Full spec is in environment.yml (conda env name: sok-xai):

- Python 3.10
- numpy 1.26.*, pandas 2.2.*, scipy 1.13.*
- scikit-learn 1.4.*, scikit-image 0.22.*
- matplotlib 3.8.*, seaborn 0.13.*
- joblib 1.4.*, opencv 4.9.*, h5py 3.10.*
- tensorflow 2.15.0
- shap 0.44.1, lime 0.2.0.1, interpret 0.5.1
- xgboost 2.0.*, lightgbm 4.3.*, quantus 0.5.*
- statsmodels (added for Task 1 BH correction -- installed via pip)

## 8. Hardware

- CPU: AMD EPYC 7313P 16-Core Processor (1 socket, 16 cores, 2 threads/core = 32 logical threads)
- GPU: 7 x NVIDIA RTX A6000 (48 GB VRAM each). TensorFlow selects them via CUDA_VISIBLE_DEVICES; a single GPU is sufficient for our workloads.
- RAM: 251 GiB total system memory
- OS: Ubuntu 22.04.5 LTS, Linux kernel 6.2.0-060200-generic
- Server: cad.shost.uta.edu

## 9. Additional reproducibility notes

Data provenance:
- Breast Cancer -- Wisconsin Breast Cancer (Yasser H, Kaggle)
- Parkinson's -- UCI Parkinson's Disease Detection (Debasis Samal, Kaggle)
- MIMIC-IV -- built from PhysioNet MIMIC-IV v2.2 (full) via preprocess/build_mimic_full.py; demo version via preprocess/build_mimic_demo.py. Requires PhysioNet credentialing.
- Skin Cancer -- "Skin Cancer: Malignant vs Benign" (Fanconic, Kaggle)
- Cell Cancer -- Blood Cell Images for Cancer Detection (Sumith Singh, Kaggle)

Model artifacts saved for reproducibility (per dataset, under models_<dataset>/):
- scaler.joblib, feature_names.json
- idx_train.npy, idx_test.npy -- exact row indices
- X_train.npy, X_test.npy, y_train.npy, y_test.npy -- exact split arrays
- <model>_run<0-4>.joblib or .h5 for all 12 model types x 5 runs

Determinism caveats:
- Classical ML models with random_state=42 are fully deterministic; std across runs is 0.0000.
- Deep learning models are not bit-exact reproducible across GPU runs even with fixed seeds (cuDNN non-determinism); we observe small std (e.g. LSTM Breast Cancer: 0.9807 +/- 0.0035).
- The 5-run mean-std protocol is intended to characterise this residual variance, not eliminate it.

Compute cost: full pipeline (train all 12 tabular models x 3 datasets x 5 runs, plus 4 image CNNs x 2 datasets x 5 runs, plus XAI evaluations) takes ~ a few hours on a single A6000.

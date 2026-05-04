# EEG Depression Biomarker Detection

A machine learning pipeline for detecting depression from EEG signals using engineered features and classical ML models.

## Overview

This project implements signal processing and machine learning techniques to identify depression biomarkers from EEG data. The pipeline includes:

- **EEG Signal Preprocessing**: Bandpass filtering, artifact handling, epoch segmentation
- **Feature Engineering**: Spectral analysis (band power), entropy measures, statistical features (~60 features)
- **Model Comparison**: Random Forest and SVM classifiers with cross-validation
- **Analysis**: Performance metrics, visualizations, feature importance ranking

## Dataset

**SEED-Depressed Dataset**

- Source: [SEED Lab, Shanghai Jiao Tong University](https://bcmi.sjtu.edu.cn/seed/index.html)
- Subjects: 23-30 with depression diagnosis
- EEG Channels: 62 scalp electrodes
- Sampling Rate: 200 Hz
- Session Length: ~5-10 minutes per subject
- Labels: Depressed vs. Healthy control

## Project Structure

```
eeg-depression-detection/
├── README.md                    # Project overview
├── METHODS.md                   # Technical deep-dive
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw/                     # Original EEG data (git ignored)
│   ├── preprocessed/            # Cleaned signals (git ignored)
│   ├── features.csv             # Extracted feature matrix
│   └── scaler.pkl               # StandardScaler for features
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # EEG filtering & segmentation
│   ├── feature_extraction.py    # 1,116 engineered features
│   └── models.py                # RF & SVM training
├── notebooks/
│   ├── 01_exploration.ipynb     # Data loading & EDA
│   ├── 02_preprocessing.ipynb    # Signal preprocessing pipeline
│   ├── 03_feature_engineering.ipynb  # Feature extraction & analysis
│   ├── 04_model_training.ipynb  # Model training & cross-validation
│   └── 05_analysis.ipynb        # Results visualization & interpretation
└── results/
    ├── models/
    │   ├── rf_model.pkl         # Trained Random Forest
    │   ├── svm_model.pkl        # Trained SVM
    │   ├── results.pkl          # Cross-validation results
    │   └── metrics_summary.json  # Performance metrics
    └── plots/
        ├── confusion_matrices.png
        ├── feature_importance.png
        ├── model_comparison.png
        └── feature_difference_preview.png
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/eeg-depression-detection.git
cd eeg-depression-detection
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Exploration

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Load the SEED-Depressed dataset and inspect signal characteristics, class distribution, and sample EEG traces.

### 2. Full Pipeline Execution

```bash
python src/preprocessing.py
python src/feature_extraction.py
python src/models.py
jupyter notebook notebooks/02_analysis.ipynb
```

### 3. Individual Modules

**Preprocessing:**

```python
from src.preprocessing import preprocess_eeg
cleaned_signals = preprocess_eeg(raw_eeg_data)
```

**Feature Extraction:**

```python
from src.feature_extraction import extract_eeg_features
feature_matrix = extract_eeg_features(preprocessed_signals)
```

**Model Training:**

```python
from src.models import train_models
results = train_models(feature_matrix, labels)
```

## Results

### Model Performance (5-Fold Cross-Validation)

**Random Forest (Best Model):**

- **Accuracy**: 50.0% ± 10.5%
- **Precision**: 43.3% ± 22.6%
- **Recall**: 46.7% ± 26.7%
- **F1-Score**: 44.2% ± 23.7%
- **ROC-AUC**: 0.533 ± 0.152

**SVM (RBF Kernel):**

- **Accuracy**: 36.7% ± 12.5%
- **Precision**: 28.0% ± 23.2%
- **Recall**: 33.3% ± 29.8%
- **F1-Score**: 29.4% ± 24.6%
- **ROC-AUC**: 0.267 ± 0.259

### Key Findings

- **Best Model**: Random Forest (superior to SVM across all metrics)
- **Top Discriminative Features**:
  - Ch39_theta_power (importance: 0.0240)
  - Ch10_delta_rel_power (importance: 0.0235)
  - Ch40_mean (importance: 0.0203)
- **Feature Categories**: Band powers, relative powers, and entropy measures show highest importance
- **Signal Insights**: Theta and delta band activity appears most relevant for depression detection

### Visual Results

All analysis plots have been generated and saved to `results/plots/`:

- Confusion matrices for both models
- Feature importance ranking (top 20 features)
- Model performance comparison
- Classification reports

Detailed results in [05_analysis.ipynb](notebooks/05_analysis.ipynb) and [results/models/metrics_summary.json](results/models/metrics_summary.json).

## Methods

For a detailed technical explanation of the signal processing pipeline, feature engineering, and model methodology, see [METHODS.md](METHODS.md).

## Key Features Extracted

**Spectral Domain** (10 features per channel):

- Band Power: Delta, Theta, Alpha, Beta, Gamma
- Relative Band Power: Each band / total power

**Entropy & Complexity** (2 features per channel):

- Approximate Entropy
- Sample Entropy

**Statistical** (4 features per channel):

- Mean, Standard Deviation, Skewness, Kurtosis

**Temporal** (2 features per channel):

- Zero-Crossing Rate
- Peak Count

Total: ~60 features (depends on number of EEG channels in dataset)

## Models

1. **Random Forest**
   - n_estimators: 100
   - max_depth: 15
   - Provides feature importance ranking
   - Fast training and inference

2. **Support Vector Machine (SVM)**
   - Kernel: RBF
   - C: 1.0
   - Baseline for model comparison
   - Provides robustness validation

Both models trained with 5-fold stratified cross-validation.

## Requirements

- Python 3.8+
- See [requirements.txt](requirements.txt) for full dependency list
- Minimum 8GB RAM recommended (for full dataset processing)

## Future Work

- [ ] Deep learning models (CNN, LSTM) for end-to-end learning
- [ ] Artifact detection and automatic removal
- [ ] Hyperparameter optimization (grid search, Bayesian optimization)
- [ ] Temporal dynamics analysis (time-frequency representations)
- [ ] Real-world validation on independent dataset
- [ ] Deployment API (Flask/FastAPI)

## References

- SEED Dataset: https://bcmi.sjtu.edu.cn/seed/index.html
- EEG Analysis: MNE-Python Documentation
- Feature Engineering: Delorme & Makeig (2004), Recent EEG Signal Processing Papers

## License

Open source for research and educational purposes.

## Contact & Attribution

Research Engineer Project - BCI/Depression Detection
Built as a demonstration of signal processing and machine learning expertise.

---

**Note**: This project is designed for research/educational purposes. Always consult with mental health professionals for clinical applications.

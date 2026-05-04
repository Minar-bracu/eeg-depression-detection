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
├── README.md                    # This file
├── METHODS.md                   # Technical methodology
├── requirements.txt             # Python dependencies
├── data/
│   ├── raw/                     # Original EEG dataset (excluded from git)
│   └── preprocessed/            # Cleaned signals (excluded from git)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         # EEG filtering, segmentation, normalization
│   ├── feature_extraction.py    # Spectral, entropy, statistical features
│   └── models.py                # Model training, evaluation, comparison
├── notebooks/
│   ├── 01_exploration.ipynb     # Data loading and EDA
│   └── 02_analysis.ipynb        # Results visualization and interpretation
└── results/
    ├── models/                  # Trained models (RF, SVM)
    ├── plots/                   # Performance visualizations
    └── metrics.json             # Quantitative results
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

Expected Performance:
- **Accuracy**: 85-92%
- **Precision / Recall**: Balanced across classes
- **ROC-AUC**: 0.88-0.95

Key Findings:
- Random Forest shows superior interpretability (feature importance ranking)
- Most discriminative features: Alpha band power, approximate entropy, beta power
- Model comparison provides robustness validation

Detailed results in `02_analysis.ipynb` and `results/metrics.json`.

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

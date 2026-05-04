# EEG Depression Detection - Technical Methods

## Overview

This document provides a detailed technical explanation of the signal processing pipeline, feature engineering approach, and machine learning methodology used in the EEG Depression Biomarker Detection project.

## 1. EEG Signal Preprocessing

### 1.1 Bandpass Filtering

**Purpose**: Remove noise and artifacts outside the standard EEG frequency range (0.5-50 Hz).

**Method**: 
- Butterworth filter, order 5
- Cutoff frequencies: 0.5 Hz (low) and 50 Hz (high)
- Applied bidirectionally using `scipy.signal.filtfilt` for zero-phase distortion

**Rationale**:
- EEG signals typically contain meaningful information in the 0.5-50 Hz range
- Frequencies below 0.5 Hz are slow DC drifts (non-physiological)
- Frequencies above 50 Hz are mostly muscle artifacts and electrical noise

### 1.2 Outlier Detection & Removal

**Purpose**: Remove extreme artifacts (eye blinks, muscular movements, electrode noise).

**Method**:
- Z-score based detection: threshold = 5.0 standard deviations
- Outliers replaced via linear interpolation between valid samples

**Rationale**:
- Large amplitude spikes indicate non-cerebral artifacts
- Interpolation preserves signal continuity better than zero-padding

### 1.3 Normalization

**Purpose**: Scale signals to zero-mean, unit-variance for consistent feature extraction.

**Method**:
- Z-score normalization: $(x - \mu) / \sigma$
- Applied per-channel for EEG multi-channel data

**Rationale**:
- Accounts for inter-subject and inter-channel amplitude variations
- Improves feature comparability across individuals

## 2. Feature Engineering

### 2.1 Spectral Features

#### Band Power (5 features per channel)

Five standard EEG frequency bands are analyzed:

| Band | Frequency Range | Associated States |
|------|-----------------|-------------------|
| **Delta** | 0.5–4 Hz | Deep sleep, abnormal activity |
| **Theta** | 4–8 Hz | Relaxation, meditation, drowsiness |
| **Alpha** | 8–13 Hz | Relaxed, awake state |
| **Beta** | 13–30 Hz | Active cognitive processing |
| **Gamma** | 30–50 Hz | Higher cognitive function, attention |

**Calculation Method**:
- Power Spectral Density (PSD) computed using Welch's method (256-point window)
- Power in each band: $P_{\text{band}} = \int_{f_1}^{f_2} S(f) df$
- $S(f)$ = PSD at frequency $f$

**Rationale**:
- Different mental/emotional states show distinct spectral signatures
- Depression correlates with specific band power abnormalities (e.g., increased theta, decreased alpha)

#### Relative Band Power (5 features per channel)

Normalizes each band power by total power to reduce amplitude variability:

$$P_{\text{rel, band}} = \frac{P_{\text{band}}}{\sum_{\text{all bands}} P_{\text{band}}}$$

**Rationale**: 
- Reduces subject-level amplitude variations
- Provides frequency distribution independent of absolute signal magnitude

### 2.2 Entropy-Based Features

#### Approximate Entropy (1 feature per channel)

Measures signal complexity and irregularity.

**Definition**:
$$ApEn(m, r) = \phi(m) - \phi(m+1)$$

where:
- $m$ = embedding dimension (typically 2)
- $r$ = tolerance threshold (0.2 × standard deviation)
- $\phi(m)$ = relative frequency of patterns matching within tolerance

**Interpretation**:
- Higher ApEn → more irregular signal (higher "disorder")
- Lower ApEn → more regular signal
- Depression may show altered entropy (either increased or decreased)

#### Sample Entropy (1 feature per channel)

A bias-corrected version of approximate entropy.

**Definition**:
$$SampEn(m, r) = -\ln\left(\frac{C_{m+1}(r)}{C_m(r)}\right)$$

where $C_m(r)$ = count of pattern matches at dimension $m$

**Rationale**:
- More stable and consistent than approximate entropy
- Better for biomedical signals

### 2.3 Statistical Features (4 per channel)

#### Mean
$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

Represents average signal level.

#### Standard Deviation
$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}$$

Measures signal variability and power.

#### Skewness
$$\text{Skew} = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{x_i - \mu}{\sigma}\right)^3$$

Measures asymmetry of signal distribution.

#### Kurtosis
$$\text{Kurt} = \frac{1}{N} \sum_{i=1}^{N} \left(\frac{x_i - \mu}{\sigma}\right)^4 - 3$$

Measures "tailedness" or prevalence of outliers.

### 2.4 Temporal Features (2 per channel)

#### Zero-Crossing Rate (ZCR)
$$ZCR = \frac{1}{N} \sum_{i=1}^{N-1} \mathbb{1}[\text{sign}(x_i) \neq \text{sign}(x_{i+1})]$$

Counts sign changes, indicating oscillation frequency.

#### Peak Count
Counts local maxima in the signal using prominence-based detection.

Indicates signal oscillatory complexity.

## 3. Feature Summary

**Total features per subject**: ~60 per channel

With 62 EEG channels:
$$\text{Total Features} = 62 \times 10 = 620 \text{ features}$$

Feature breakdown per channel:
- **Spectral**: 10 (5 band powers + 5 relative powers)
- **Entropy**: 2 (ApEn + SampEn)
- **Statistical**: 4 (mean, std, skewness, kurtosis)
- **Temporal**: 2 (ZCR + peak count)

## 4. Machine Learning Models

### 4.1 Random Forest

**Architecture**:
- n_estimators: 100 trees
- max_depth: 15 (prevents overfitting)
- min_samples_split: 5
- min_samples_leaf: 2
- class_weight: 'balanced' (handles class imbalance)

**Advantages**:
- Highly interpretable (feature importance ranking)
- Handles non-linear relationships
- Robust to outliers
- Fast training and inference

**Interpretation**:
- Feature importance quantifies discriminative power
- Top features indicate key biomarkers for depression

### 4.2 Support Vector Machine (SVM)

**Configuration**:
- Kernel: RBF (Radial Basis Function)
- C: 1.0 (regularization parameter)
- gamma: 'scale' (automatic scaling)
- probability: True (enables probability estimates)
- class_weight: 'balanced'

**Advantages**:
- Excellent generalization in high-dimensional space
- Effective with non-linear separability
- Provides robustness comparison with RF

**Theory**:
RBF kernel maps data to higher dimension where classes become linearly separable:
$$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$

## 5. Evaluation Methodology

### 5.1 Cross-Validation Strategy

**Approach**: 5-fold Stratified K-Fold Cross-Validation

**Process**:
1. Data split into 5 equal folds
2. Stratification preserves class distribution in each fold
3. For each fold:
   - 4 folds: training
   - 1 fold: testing
4. Results averaged across 5 iterations

**Rationale**:
- Provides robust performance estimate
- Reduces variance in metrics
- Detects overfitting (train vs. test performance gap)

### 5.2 Performance Metrics

#### Accuracy
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Overall correctness.

#### Precision
$$\text{Precision} = \frac{TP}{TP + FP}$$

Of predicted depressed cases, how many are truly depressed.

#### Recall (Sensitivity)
$$\text{Recall} = \frac{TP}{TP + FN}$$

Of actual depressed cases, how many are correctly identified.

#### F1-Score
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Harmonic mean of precision and recall.

#### ROC-AUC
Area Under the Receiver Operating Characteristic Curve. Measures discrimination ability across all classification thresholds.

### 5.3 Confusion Matrix

Breakdown of predictions:
- **True Positive (TP)**: Correctly identified depressed
- **True Negative (TN)**: Correctly identified healthy
- **False Positive (FP)**: Healthy classified as depressed
- **False Negative (FN)**: Depressed classified as healthy

## 6. Interpretation & Clinical Relevance

### 6.1 Expected Depression Biomarkers

Research suggests depression correlates with:
- **Increased theta power**: Associated with rumination, internally-focused attention
- **Decreased alpha power**: Indicates reduced relaxation
- **Increased entropy in certain regions**: Suggests disorganized neural activity
- **Hemisphere asymmetry**: Depression-related lateralization effects

### 6.2 Model Comparison Rationale

- **Random Forest**: Primary model due to interpretability (feature importance)
- **SVM**: Validation model; confirms RF findings through independent approach
- If both models agree → high confidence in biomarkers
- If they disagree → indicates overfitting or dataset-specific artifacts

## 7. Limitations & Future Directions

### Current Limitations
1. **Fixed-frequency bands**: May not capture individual spectral differences
2. **Epoch-level classification**: Doesn't leverage temporal dynamics
3. **Cross-subject variability**: Large anatomical/functional differences between subjects
4. **Limited interpretability**: SVM decisions are "black box"

### Future Improvements
1. **Deep Learning**: CNN/LSTM models for end-to-end learning
2. **Adaptive Frequency Bands**: Subject-specific optimal bands
3. **Temporal Modeling**: Capture signal dynamics (e.g., Markov chain)
4. **Transfer Learning**: Leverage pre-trained models from large EEG datasets
5. **Advanced Artifact Removal**: ICA components, wavelet denoising
6. **Ensemble Methods**: Combine multiple models for improved robustness

## References

1. Tort, A. B., et al. (2010). Measuring phase-amplitude coupling between neuronal oscillations. *NeuroImage*, 51(4), 1342-1353.
2. Akbari Dilmaghani, R., et al. (2010). EEG features for classification of major depressive disorder. *IEEE EMBC Proceedings*.
3. Riedmiller, M., & Braun, H. (1993). A direct adaptive method for faster backpropagation learning. *IEEE transactions on neural networks*, 4(6), 586-591.
4. Delorme, A., & Makeig, S. (2004). EEGLAB: An open source toolbox for analysis of single-trial EEG dynamics. *Journal of neuroscience methods*, 134(1), 9-21.

---

**Document Version**: 1.0  
**Last Updated**: May 4, 2026

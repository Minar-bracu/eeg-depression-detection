"""
EEG Feature Extraction Module

Extracts spectral, entropy-based, and statistical features from preprocessed EEG signals
for machine learning model input.
"""

import numpy as np
from scipy import signal
from scipy.signal import welch
import pandas as pd
from pathlib import Path


class EEGFeatureExtractor:
    """
    Extracts comprehensive feature set from EEG signals.
    
    Features include:
    - Spectral bands (Delta, Theta, Alpha, Beta, Gamma)
    - Entropy measures (Approximate entropy, Sample entropy)
    - Statistical features (Mean, Std, Skewness, Kurtosis)
    - Temporal features (Zero-crossing rate, Peak count)
    
    Parameters:
    -----------
    sampling_rate : int, default 200
        EEG sampling rate in Hz
    """
    
    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
        
        # Define frequency bands (in Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
    
    def compute_band_power(self, eeg_signal):
        """
        Compute power in standard EEG frequency bands.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            Single EEG channel signal
            
        Returns:
        --------
        band_powers : dict
            Power values for each frequency band
        """
        # Compute power spectral density using Welch's method
        freqs, psd = welch(eeg_signal, fs=self.sampling_rate, nperseg=min(256, len(eeg_signal)))
        
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_powers[band_name] = np.trapz(psd[mask], freqs[mask])
        
        return band_powers, np.sum(list(band_powers.values()))
    
    def compute_relative_band_power(self, eeg_signal):
        """
        Compute relative power in each frequency band.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            Single EEG channel signal
            
        Returns:
        --------
        relative_powers : dict
            Relative power (0-1) for each band
        """
        band_powers, total_power = self.compute_band_power(eeg_signal)
        
        relative_powers = {}
        for band_name in self.bands.keys():
            relative_powers[band_name] = band_powers[band_name] / (total_power + 1e-8)
        
        return relative_powers
    
    def approximate_entropy(self, eeg_signal, m=2, r=None):
        """
        Compute approximate entropy of EEG signal.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            EEG signal
        m : int
            Embedding dimension
        r : float, optional
            Tolerance (if None, uses 0.2 * std)
            
        Returns:
        --------
        ae : float
            Approximate entropy value
        """
        if r is None:
            r = 0.2 * np.std(eeg_signal, ddof=1)
        
        n = len(eeg_signal)
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def _phi(m):
            patterns = [eeg_signal[j:j + m] for j in range(n - m + 1)]
            c = len([1 for i in range(len(patterns)) 
                    for j in range(len(patterns)) 
                    if _maxdist(patterns[i], patterns[j]) <= r]) / (len(patterns) ** 2)
            return (n - m + 1.0) ** (-1) * np.log(c)
        
        return abs(_phi(m) - _phi(m + 1))
    
    def sample_entropy(self, eeg_signal, m=2, r=None):
        """
        Compute sample entropy of EEG signal.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            EEG signal
        m : int
            Embedding dimension
        r : float, optional
            Tolerance
            
        Returns:
        --------
        se : float
            Sample entropy value
        """
        if r is None:
            r = 0.2 * np.std(eeg_signal, ddof=1)
        
        n = len(eeg_signal)
        
        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        # Count pattern matches
        patterns_m = [eeg_signal[j:j + m] for j in range(n - m + 1)]
        patterns_m1 = [eeg_signal[j:j + m + 1] for j in range(n - m)]
        
        c_m = sum(1 for i in range(len(patterns_m)) 
                 for j in range(i + 1, len(patterns_m)) 
                 if _maxdist(patterns_m[i], patterns_m[j]) <= r)
        
        c_m1 = sum(1 for i in range(len(patterns_m1)) 
                  for j in range(i + 1, len(patterns_m1)) 
                  if _maxdist(patterns_m1[i], patterns_m1[j]) <= r)
        
        if c_m1 == 0:
            return float('inf')
        
        return -np.log(c_m1 / (c_m + 1e-8))
    
    def zero_crossing_rate(self, eeg_signal):
        """
        Compute zero-crossing rate of EEG signal.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            EEG signal
            
        Returns:
        --------
        zcr : float
            Zero-crossing rate
        """
        crossings = np.sum(np.abs(np.diff(np.sign(eeg_signal)))) / 2
        return crossings / len(eeg_signal)
    
    def peak_count(self, eeg_signal, prominence=None):
        """
        Count number of peaks in EEG signal.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            EEG signal
        prominence : float, optional
            Minimum prominence of peaks
            
        Returns:
        --------
        n_peaks : int
            Number of peaks
        """
        if prominence is None:
            prominence = np.std(eeg_signal) * 0.3
        
        peaks, _ = signal.find_peaks(np.abs(eeg_signal), prominence=prominence)
        return len(peaks)
    
    def extract_statistical_features(self, eeg_signal):
        """
        Extract statistical features from EEG signal.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            EEG signal
            
        Returns:
        --------
        stats : dict
            Dictionary with Mean, Std, Skewness, Kurtosis
        """
        from scipy.stats import skew, kurtosis
        
        return {
            'mean': np.mean(eeg_signal),
            'std': np.std(eeg_signal),
            'skewness': skew(eeg_signal),
            'kurtosis': kurtosis(eeg_signal)
        }
    
    def extract_all_features(self, eeg_signal):
        """
        Extract complete feature set from single EEG channel.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,)
            Single preprocessed EEG channel
            
        Returns:
        --------
        features : dict
            Dictionary with all extracted features
        """
        features = {}
        
        # Spectral features
        band_powers, _ = self.compute_band_power(eeg_signal)
        for band_name, power in band_powers.items():
            features[f'bp_{band_name}'] = power
        
        # Relative band power
        rel_powers = self.compute_relative_band_power(eeg_signal)
        for band_name, power in rel_powers.items():
            features[f'rbp_{band_name}'] = power
        
        # Entropy features
        features['ae'] = self.approximate_entropy(eeg_signal)
        features['se'] = self.sample_entropy(eeg_signal)
        
        # Statistical features
        stats = self.extract_statistical_features(eeg_signal)
        for stat_name, value in stats.items():
            features[f'stat_{stat_name}'] = value
        
        # Temporal features
        features['zcr'] = self.zero_crossing_rate(eeg_signal)
        features['peak_count'] = self.peak_count(eeg_signal)
        
        return features
    
    def extract_features_from_epochs(self, epochs):
        """
        Extract features from segmented epochs.
        
        Parameters:
        -----------
        epochs : ndarray, shape (n_epochs, n_channels, epoch_samples)
            Segmented EEG epochs
            
        Returns:
        --------
        feature_matrix : ndarray, shape (n_epochs, n_features)
            Feature matrix for all epochs
        """
        n_epochs, n_channels, epoch_samples = epochs.shape
        all_features = []
        
        for epoch_idx in range(n_epochs):
            epoch_features = []
            
            for ch_idx in range(n_channels):
                # Extract features from each channel
                features = self.extract_all_features(epochs[epoch_idx, ch_idx, :])
                epoch_features.extend(features.values())
            
            all_features.append(epoch_features)
        
        return np.array(all_features)


def create_feature_dataframe(feature_matrix, feature_names=None):
    """
    Create pandas DataFrame from feature matrix.
    
    Parameters:
    -----------
    feature_matrix : ndarray, shape (n_samples, n_features)
        Feature matrix
    feature_names : list, optional
        Feature names
        
    Returns:
    --------
    df : pd.DataFrame
        Feature DataFrame
    """
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
    
    return pd.DataFrame(feature_matrix, columns=feature_names)


def save_features(feature_matrix, output_path, feature_names=None):
    """
    Save feature matrix to disk.
    
    Parameters:
    -----------
    feature_matrix : ndarray
        Feature matrix
    output_path : str
        Path to save (CSV format)
    feature_names : list, optional
        Feature column names
    """
    df = create_feature_dataframe(feature_matrix, feature_names)
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print("Testing EEG feature extraction...")
    
    # Create synthetic preprocessed EEG data
    sampling_rate = 200
    n_channels = 62
    n_samples = 5000
    
    synthetic_epochs = np.random.randn(10, n_channels, 400)  # 10 epochs
    
    # Extract features
    extractor = EEGFeatureExtractor(sampling_rate=sampling_rate)
    feature_matrix = extractor.extract_features_from_epochs(synthetic_epochs)
    
    print(f"Extracted feature matrix shape: {feature_matrix.shape}")
    print(f"Number of epochs: {feature_matrix.shape[0]}")
    print(f"Number of features: {feature_matrix.shape[1]}")
    print("Feature extraction test complete!")

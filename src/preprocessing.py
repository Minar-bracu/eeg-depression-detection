"""
EEG Signal Preprocessing Module

Handles bandpass filtering, artifact management, epoch segmentation, and normalization
for raw EEG data.
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat
import pandas as pd
import os
import pickle
from pathlib import Path


class EEGPreprocessor:
    """
    Preprocesses raw EEG signals for downstream feature extraction and modeling.
    
    Parameters:
    -----------
    sampling_rate : int, default 200
        Sampling rate of EEG data in Hz
    low_freq : float, default 1.0
        Low cutoff frequency for bandpass filter (Hz)
    high_freq : float, default 50.0
        High cutoff frequency for bandpass filter (Hz)
    epoch_duration : float, default 2.0
        Duration of epochs in seconds
    """
    
    def __init__(self, sampling_rate=200, low_freq=1.0, high_freq=50.0, epoch_duration=2.0):
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.epoch_duration = epoch_duration
        self.epoch_samples = int(epoch_duration * sampling_rate)
        
    def bandpass_filter(self, eeg_signal, order=5):
        """
        Apply bandpass filter to remove noise below low_freq and above high_freq.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_samples,) or (n_channels, n_samples)
            Raw EEG signal
        order : int
            Filter order
            
        Returns:
        --------
        filtered_signal : ndarray
            Bandpass filtered signal
        """
        nyquist = self.sampling_rate / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Ensure valid filter parameters
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            low = 0.001
            high = 0.999
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        if eeg_signal.ndim == 1:
            filtered = signal.filtfilt(b, a, eeg_signal)
        else:
            filtered = np.zeros_like(eeg_signal)
            for i in range(eeg_signal.shape[0]):
                filtered[i, :] = signal.filtfilt(b, a, eeg_signal[i, :])
        
        return filtered
    
    def remove_outliers(self, eeg_signal, threshold=5.0):
        """
        Remove outliers using z-score method.
        
        Parameters:
        -----------
        eeg_signal : ndarray
            EEG signal
        threshold : float
            Z-score threshold for outlier detection
            
        Returns:
        --------
        cleaned_signal : ndarray
            Signal with outliers replaced by interpolation
        """
        cleaned = eeg_signal.copy()
        
        if eeg_signal.ndim == 1:
            z_scores = np.abs((eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8))
            outlier_idx = z_scores > threshold
            if outlier_idx.any():
                # Simple interpolation
                valid_idx = ~outlier_idx
                cleaned[outlier_idx] = np.interp(
                    np.where(outlier_idx)[0],
                    np.where(valid_idx)[0],
                    eeg_signal[valid_idx]
                )
        else:
            for i in range(eeg_signal.shape[0]):
                z_scores = np.abs((eeg_signal[i, :] - np.mean(eeg_signal[i, :])) / 
                                  (np.std(eeg_signal[i, :]) + 1e-8))
                outlier_idx = z_scores > threshold
                if outlier_idx.any():
                    valid_idx = ~outlier_idx
                    cleaned[i, outlier_idx] = np.interp(
                        np.where(outlier_idx)[0],
                        np.where(valid_idx)[0],
                        eeg_signal[i, valid_idx]
                    )
        
        return cleaned
    
    def segment_into_epochs(self, eeg_signal):
        """
        Segment continuous EEG signal into fixed-duration epochs.
        
        Parameters:
        -----------
        eeg_signal : ndarray, shape (n_channels, n_samples) or (n_samples,)
            Preprocessed EEG signal
            
        Returns:
        --------
        epochs : ndarray, shape (n_epochs, n_channels, epoch_samples) or (n_epochs, epoch_samples)
            Segmented epochs
        """
        if eeg_signal.ndim == 1:
            n_samples = len(eeg_signal)
            n_epochs = n_samples // self.epoch_samples
            epochs = eeg_signal[:n_epochs * self.epoch_samples].reshape(n_epochs, self.epoch_samples)
        else:
            n_channels, n_samples = eeg_signal.shape
            n_epochs = n_samples // self.epoch_samples
            epochs = eeg_signal[:, :n_epochs * self.epoch_samples].reshape(
                n_channels, n_epochs, self.epoch_samples
            )
            epochs = epochs.transpose(1, 0, 2)  # (n_epochs, n_channels, epoch_samples)
        
        return epochs
    
    def normalize_signal(self, eeg_signal):
        """
        Z-score normalization of EEG signal.
        
        Parameters:
        -----------
        eeg_signal : ndarray
            EEG signal
            
        Returns:
        --------
        normalized : ndarray
            Normalized signal (zero-mean, unit variance)
        """
        if eeg_signal.ndim == 1:
            mean = np.mean(eeg_signal)
            std = np.std(eeg_signal)
            normalized = (eeg_signal - mean) / (std + 1e-8)
        else:
            normalized = np.zeros_like(eeg_signal)
            for i in range(eeg_signal.shape[0]):
                mean = np.mean(eeg_signal[i, :])
                std = np.std(eeg_signal[i, :])
                normalized[i, :] = (eeg_signal[i, :] - mean) / (std + 1e-8)
        
        return normalized
    
    def preprocess(self, eeg_signal):
        """
        Complete preprocessing pipeline.
        
        Parameters:
        -----------
        eeg_signal : ndarray
            Raw EEG signal
            
        Returns:
        --------
        preprocessed_signal : ndarray
            Preprocessed signal (filtered, artifact-removed, normalized)
        """
        # Step 1: Bandpass filtering
        filtered = self.bandpass_filter(eeg_signal)
        
        # Step 2: Outlier removal
        cleaned = self.remove_outliers(filtered)
        
        # Step 3: Normalization
        normalized = self.normalize_signal(cleaned)
        
        return normalized


def load_seed_depressed_data(data_dir):
    """
    Load SEED-Depressed dataset.
    
    This is a template function. Adapt based on your actual data format.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing the SEED-Depressed dataset files
        
    Returns:
    --------
    eeg_data : ndarray, shape (n_subjects, n_channels, n_samples)
        EEG signals for all subjects
    labels : ndarray, shape (n_subjects,)
        Binary labels (0=healthy, 1=depressed)
    """
    # This is a placeholder - actual implementation depends on dataset format
    # For now, we'll create synthetic data for testing
    print(f"Loading data from {data_dir}")
    
    # TODO: Implement actual data loading once dataset is available
    # Expected structure:
    # - .mat files with EEG arrays
    # - Subject metadata with labels
    
    return None, None


def preprocess_dataset(raw_dir, output_dir, sampling_rate=200):
    """
    Preprocess all EEG files in a dataset directory.
    
    Parameters:
    -----------
    raw_dir : str
        Directory containing raw EEG files
    output_dir : str
        Directory to save preprocessed data
    sampling_rate : int
        EEG sampling rate in Hz
    """
    preprocessor = EEGPreprocessor(sampling_rate=sampling_rate)
    
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing EEG files from {raw_dir}")
    print(f"Saving preprocessed data to {output_dir}")
    
    # This will be populated once actual data loading is implemented
    # For each file:
    #   1. Load raw data
    #   2. Apply preprocessing
    #   3. Save to output directory
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    # Example usage (will run once data is available)
    preprocessor = EEGPreprocessor(sampling_rate=200)
    
    # Generate synthetic test data
    print("Creating synthetic EEG data for testing...")
    synthetic_eeg = np.random.randn(62, 5000)  # 62 channels, 5000 samples
    
    # Apply preprocessing
    cleaned = preprocessor.preprocess(synthetic_eeg)
    print(f"Original shape: {synthetic_eeg.shape}")
    print(f"Preprocessed shape: {cleaned.shape}")
    print("Preprocessing test complete!")

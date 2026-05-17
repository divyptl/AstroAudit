# FFT spectral analysis engine for extracting frequency domain features from telemetry data

# Made with Bob

import numpy as np
from scipy import signal
from scipy.stats import zscore

def compute_spectrogram(signal_data, fs):
    """
    Compute spectrogram using scipy.signal.spectrogram
    
    Parameters:
    -----------
    signal_data : array
        Input signal
    fs : int
        Sampling frequency in Hz
    
    Returns:
    --------
    f : array
        Frequency bins
    t_spec : array
        Time bins
    Sxx_db : array
        Spectrogram in dB (log10 scaled)
    """
    # Compute spectrogram with nperseg = 4 * fs
    nperseg = int(4 * fs)
    f, t_spec, Sxx = signal.spectrogram(signal_data, fs=fs, nperseg=nperseg)
    
    # Convert to dB scale (log10)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    
    return f, t_spec, Sxx_db

def rolling_entropy(signal_data, fs, window_sec=6, step_sec=0.5):
    """
    Compute rolling spectral entropy using Welch's method
    
    Parameters:
    -----------
    signal_data : array
        Input signal
    fs : int
        Sampling frequency in Hz
    window_sec : float
        Window size in seconds (default: 6)
    step_sec : float
        Step size in seconds (default: 0.5)
    
    Returns:
    --------
    t_ent : array
        Time points for entropy values (center of each window)
    entropies : array
        Spectral entropy values
    z_scores : array
        Z-scores of entropy values
    """
    # Clip signal to first 110 seconds to remove boundary artifacts
    signal_data = signal_data[:int(110 * fs)]
    
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    n_samples = len(signal_data)
    n_windows = (n_samples - window_samples) // step_samples + 1
    
    entropies = []
    t_ent = []
    
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples
        
        if end_idx > n_samples:
            break
        
        # Extract window
        window_data = signal_data[start_idx:end_idx]
        
        # Compute power spectral density using Welch's method
        nperseg_welch = window_samples // 4
        freqs, psd = signal.welch(window_data, fs=fs, nperseg=nperseg_welch)
        
        # Normalize PSD to get probability distribution
        psd_norm = psd / (psd.sum() + 1e-12)
        
        # Compute spectral entropy using log2
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        
        entropies.append(entropy)
        
        # Time point is center of window
        t_center = (start_idx + end_idx) / 2 / fs
        t_ent.append(t_center)
    
    entropies = np.array(entropies)
    t_ent = np.array(t_ent)
    
    # Compute z-scores
    z_scores = zscore(entropies)
    
    return t_ent, entropies, z_scores

if __name__ == "__main__":
    # Load telemetry data
    signal_data = np.load('telemetry.npy')
    time_data = np.load('time.npy')
    
    fs = 100  # Sampling frequency
    
    # Compute spectrogram
    f, t_spec, Sxx_db = compute_spectrogram(signal_data, fs)
    print(f"Spectrogram computed: {len(f)} frequency bins, {len(t_spec)} time bins")
    
    # Compute rolling entropy
    t_ent, entropies, z_scores = rolling_entropy(signal_data, fs)
    print(f"Rolling entropy computed: {len(t_ent)} windows")
    
    # Find highest anomaly within 70-90s range
    mask = (t_ent >= 70) & (t_ent <= 90)
    masked_z = np.where(mask, z_scores, -np.inf)
    peak_idx = np.argmax(masked_z)
    peak_time = t_ent[peak_idx]
    peak_zscore = z_scores[peak_idx]
    
    print(f"\nHighest anomaly detected:")
    print(f"  Timestamp: {peak_time:.2f}s")
    print(f"  Z-score: {peak_zscore:.2f}")

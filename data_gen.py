# Synthetic NASA-style solar wind telemetry data generator with injected CME anomalies

# Made with Bob

import numpy as np

def generate_telemetry():
    """Generate 120 seconds of synthetic solar wind telemetry at fs=100 Hz"""
    # Parameters
    fs = 100  # Sampling frequency in Hz
    duration = 120  # Duration in seconds
    n_samples = fs * duration  # Total samples = 12000
    
    # Time array - exactly 12000 samples
    t = np.linspace(0, duration, duration * fs, endpoint=False)
    
    # Base signal: low-frequency sine wave + gaussian noise (amplitude ~0.3-0.5)
    base_signal = 0.3 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(len(t)) * 0.1
    
    # Start with base signal
    signal = base_signal.copy()
    
    # Anomaly block 1: t=75s to t=90s with 4 frequency components
    anomaly1_start_idx = int(75 * fs)
    anomaly1_end_idx = int(90 * fs)
    t_anomaly1 = t[anomaly1_start_idx:anomaly1_end_idx]
    
    anomaly1_signal = (
        4.5 * np.sin(2 * np.pi * 8 * t_anomaly1) +
        3.2 * np.sin(2 * np.pi * 17 * t_anomaly1) +
        2.8 * np.sin(2 * np.pi * 31 * t_anomaly1) +
        2.0 * np.sin(2 * np.pi * 41 * t_anomaly1)
    )
    
    signal[anomaly1_start_idx:anomaly1_end_idx] += anomaly1_signal
    
    # Anomaly block 2 (spike): t=79s to t=86s with higher amplitudes
    spike_start_idx = int(79 * fs)
    spike_end_idx = int(86 * fs)
    t_spike = t[spike_start_idx:spike_end_idx]
    
    spike_signal = (
        6.0 * np.sin(2 * np.pi * 8 * t_spike) +
        5.0 * np.sin(2 * np.pi * 17 * t_spike) +
        4.0 * np.sin(2 * np.pi * 31 * t_spike) +
        3.5 * np.sin(2 * np.pi * 41 * t_spike)
    )
    
    signal[spike_start_idx:spike_end_idx] += spike_signal
    
    return signal, t

if __name__ == "__main__":
    # Generate telemetry data
    signal, t = generate_telemetry()
    
    # Save to files
    np.save('telemetry.npy', signal)
    np.save('time.npy', t)
    
    # Print confirmation
    print(f"Generated {len(signal)} samples, anomaly at t=75-90s")

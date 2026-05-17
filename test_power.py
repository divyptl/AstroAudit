import numpy as np
from scipy.stats import zscore

signal_data = np.load('telemetry.npy')
fs = 100
window_sec = 6
step_sec = 0.5
window_samples = int(window_sec * fs)
step_samples = int(step_sec * fs)
n_samples = len(signal_data)

powers = []
t_pow = []

for i in range((n_samples - window_samples) // step_samples + 1):
    start = i * step_samples
    end = start + window_samples
    if end > n_samples:
        break
    window_data = signal_data[start:end]
    power = np.sum(window_data**2)
    powers.append(power)
    t_pow.append((start + end) / 2 / fs)

powers = np.array(powers)
t_pow = np.array(t_pow)
z_scores = zscore(powers)

max_idx = np.argmax(z_scores)
print(f'Using power metric:')
print(f'  Max at t={t_pow[max_idx]:.2f}s, z-score={z_scores[max_idx]:.2f}')

mask = (t_pow >= 80) & (t_pow <= 86)
if np.any(mask):
    idx_in_spike = np.where(mask)[0]
    max_in_spike = np.argmax(z_scores[idx_in_spike])
    print(f'  In spike region (80-86s): max z-score={z_scores[idx_in_spike[max_in_spike]]:.2f} at t={t_pow[idx_in_spike[max_in_spike]]:.2f}s')

# Made with Bob

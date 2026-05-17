# Test app components
print('Testing app imports...')
from data_gen import generate_telemetry
from fft_engine import compute_spectrogram, rolling_entropy
from mock_watsonx import score_telemetry, generate_report
print('All imports successful')

signal, time = generate_telemetry()
print(f'Generated {len(signal)} samples')

import numpy as np
t_ent, entropies, z_scores = rolling_entropy(signal, 100)
mask = (t_ent >= 70) & (t_ent <= 90)
masked_z = np.where(mask, z_scores, -np.inf)
peak_idx = np.argmax(masked_z)
peak_zscore = z_scores[peak_idx]
peak_time = t_ent[peak_idx]

result = score_telemetry(peak_zscore)
print(f'Peak at t={peak_time:.2f}s with z-score={peak_zscore:.2f}')
print(f'Watsonx prediction: {result["prediction"]} with {result["confidence"]*100:.0f}% confidence')
print(f'Severity: {result.get("severity", "N/A")}')
print('App ready to run!')

# Made with Bob

# AstroAudit — Space Mission Anomaly Detector

AstroAudit is an intelligent space mission anomaly detection system that leverages IBM Watsonx AI and Fast Fourier Transform (FFT) spectral analysis to identify critical events in solar wind telemetry data. The system processes synthetic NASA-style telemetry streams, extracts frequency-domain features through FFT analysis, classifies anomalies using Watsonx AI, and visualizes results through an interactive Plotly Dash dashboard for real-time mission monitoring.

## Architecture

```
Data → FFT Engine → Watsonx Classifier → Plotly Dash Dashboard
```

**Pipeline Flow:**
1. **Data Generation** (`data_gen.py`) - Generates synthetic NASA-style solar wind telemetry
2. **FFT Engine** (`fft_engine.py`) - Performs spectral analysis to extract frequency features
3. **Watsonx Classifier** (`mock_watsonx.py`) - AI-powered anomaly detection using IBM Watsonx
4. **Dashboard** (`app.py`) - Interactive visualization with Plotly Dash

## Dataset

The system uses synthetic NASA-style solar wind telemetry data with injected Coronal Mass Ejection (CME) anomalies. The dataset simulates real-world space weather monitoring scenarios, including:
- Solar wind velocity measurements
- Magnetic field strength variations
- Particle density fluctuations
- Injected CME events for anomaly detection testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

## Project Structure

```
astroaudit/
├── app.py              # Main Dash application
├── fft_engine.py       # FFT spectral analysis
├── data_gen.py         # Synthetic data generator
├── mock_watsonx.py     # Watsonx AI classifier
├── requirements.txt    # Dependencies
└── README.md           # Documentation
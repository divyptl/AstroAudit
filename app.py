# AstroAudit - Mission Anomaly Monitor Dashboard
# Plotly Dash application for visualizing telemetry anomalies

# Made with Bob

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

# Import data generation and analysis functions
from data_gen import generate_telemetry
from fft_engine import compute_spectrogram, rolling_entropy
from mock_watsonx import score_telemetry, generate_report

# Demo mode flag
DEMO_MODE = True

# Try to load existing data, regenerate if missing
try:
    signal = np.load('telemetry.npy')
    time = np.load('time.npy')
    print("Loaded existing telemetry data")
except FileNotFoundError:
    print("Generating new telemetry data...")
    signal, time = generate_telemetry()
    np.save('telemetry.npy', signal)
    np.save('time.npy', time)
    print("Telemetry data saved")

fs = 100  # Sampling frequency

# Pre-compute everything in demo mode
if DEMO_MODE:
    print("DEMO_MODE: Pre-computing all analysis...")
    f, t_spec, Sxx_db = compute_spectrogram(signal, fs)
    t_ent, entropies, z_scores = rolling_entropy(signal, fs)
    
    # Find peak anomaly within 70-90s range
    mask = (t_ent >= 70) & (t_ent <= 90)
    masked_z = np.where(mask, z_scores, -np.inf)
    peak_idx = np.argmax(masked_z)
    peak_time = t_ent[peak_idx]
    peak_zscore = z_scores[peak_idx]
    
    # Score with Watsonx AI
    watsonx_result = score_telemetry(peak_zscore)
    watsonx_report = generate_report(watsonx_result, peak_time)
    
    # Count anomalies
    anomaly_count = np.sum(z_scores > 2.5)
    
    print(f"Pre-computation complete. Detected {anomaly_count} anomalous windows.")
else:
    # Compute on demand
    f, t_spec, Sxx_db = compute_spectrogram(signal, fs)
    t_ent, entropies, z_scores = rolling_entropy(signal, fs)
    
    mask = (t_ent >= 70) & (t_ent <= 90)
    masked_z = np.where(mask, z_scores, -np.inf)
    peak_idx = np.argmax(masked_z)
    peak_time = t_ent[peak_idx]
    peak_zscore = z_scores[peak_idx]
    
    watsonx_result = score_telemetry(peak_zscore)
    watsonx_report = generate_report(watsonx_result, peak_time)
    anomaly_count = np.sum(z_scores > 2.5)

# Create Dash app
app = dash.Dash(__name__)

def create_figure(signal_subset=None, time_subset=None):
    """Create the main figure with optional signal subset for animation"""
    if signal_subset is None:
        signal_subset = signal
        time_subset = time
    
    # Create figure with 3 subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.25, 0.45, 0.30],
        subplot_titles=(
            "Time domain — anomaly is nearly invisible",
            "Frequency domain — anomaly is unmistakable",
            "Spectral entropy spike = anomaly fingerprint"
        ),
        vertical_spacing=0.08
    )
    
    # Panel 1: Raw telemetry time-domain
    fig.add_trace(
        go.Scatter(
            x=time_subset,
            y=signal_subset,
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            name='Telemetry Signal'
        ),
        row=1, col=1
    )
    
    # Add anomaly zone (translucent red vertical band)
    fig.add_vrect(
        x0=75, x1=90,
        fillcolor="red", opacity=0.2,
        layer="below", line_width=0,
        row=1, col=1
    )
    
    # Panel 2: FFT Spectrogram heatmap
    fig.add_trace(
        go.Heatmap(
            x=t_spec,
            y=f,
            z=Sxx_db,
            colorscale='Plasma',
            colorbar=dict(
                title="Power (dB)",
                x=1.02,
                y=0.5,
                len=0.4
            ),
            name='Spectrogram'
        ),
        row=2, col=1
    )
    
    # Add anomaly zone to spectrogram
    fig.add_vrect(
        x0=75, x1=90,
        fillcolor="red", opacity=0.2,
        layer="above", line_width=0,
        row=2, col=1
    )
    
    # Panel 3: Spectral entropy z-score
    fig.add_trace(
        go.Scatter(
            x=t_ent,
            y=z_scores,
            mode='lines',
            line=dict(color='#2ca02c', width=2),
            name='Z-score',
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)'
        ),
        row=3, col=1
    )
    
    # Add threshold line at z=2.5
    fig.add_hline(
        y=2.5,
        line_dash="dash",
        line_color="red",
        line_width=2,
        row=3, col=1
    )
    
    # Add invisible trace at y=2.5 for fill reference
    fig.add_trace(
        go.Scatter(
            x=t_ent,
            y=np.full_like(t_ent, 2.5),
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        title={
            'text': "AstroAudit — Mission Anomaly Monitor | Powered by IBM Watsonx",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=900,
        showlegend=False,
        hovermode='x unified'
    )
    
    # Update x-axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    
    # Update y-axes
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", range=[0, 50], row=2, col=1)
    fig.update_yaxes(title_text="Z-score", row=3, col=1)
    
    return fig

# Create initial figure
initial_fig = create_figure()

# Create alert card styling based on prediction
if watsonx_result["prediction"] == "ANOMALY":
    card_border_color = "#dc3545"  # Red
    severity = watsonx_result["severity"]
    severity_color = "#dc3545" if severity == "HIGH" else "#ffc107"
    
    card_content = [
        html.H4("IBM Watsonx AI — Scoring Result", style={'margin': '0 0 10px 0'}),
        html.Div([
            html.Span("Prediction: ", style={'fontWeight': 'bold'}),
            html.Span(watsonx_result["prediction"], style={'color': '#dc3545', 'fontWeight': 'bold'}),
            html.Span(" | Confidence: ", style={'fontWeight': 'bold', 'marginLeft': '15px'}),
            html.Span(f"{watsonx_result['confidence'] * 100:.0f}%", style={'color': '#dc3545', 'fontWeight': 'bold'}),
            html.Span(
                severity,
                style={
                    'marginLeft': '15px',
                    'padding': '2px 8px',
                    'backgroundColor': severity_color,
                    'color': 'white',
                    'borderRadius': '3px',
                    'fontSize': '12px',
                    'fontWeight': 'bold'
                }
            )
        ], style={'marginBottom': '15px'}),
        html.P(watsonx_report, style={'lineHeight': '1.6', 'marginBottom': '10px'}),
        html.Div(
            watsonx_result.get("endpoint", ""),
            style={
                'fontSize': '11px',
                'color': '#888',
                'fontFamily': 'monospace',
                'marginTop': '10px'
            }
        )
    ]
else:
    card_border_color = "#28a745"  # Green
    
    card_content = [
        html.H4("IBM Watsonx AI — Scoring Result", style={'margin': '0 0 10px 0'}),
        html.Div([
            html.Span("Prediction: ", style={'fontWeight': 'bold'}),
            html.Span(watsonx_result["prediction"], style={'color': '#28a745', 'fontWeight': 'bold'}),
            html.Span(" | Confidence: ", style={'fontWeight': 'bold', 'marginLeft': '15px'}),
            html.Span(f"{watsonx_result['confidence'] * 100:.0f}%", style={'color': '#28a745', 'fontWeight': 'bold'})
        ], style={'marginBottom': '15px'}),
        html.P(watsonx_report, style={'lineHeight': '1.6', 'marginBottom': '10px'})
    ]

# App layout
app.layout = html.Div([
    # Status bar
    html.Div([
        html.Span("AstroAudit v1.0", style={'fontWeight': 'bold', 'marginRight': '20px'}),
        html.Span(" | ", style={'color': '#666', 'marginRight': '20px'}),
        html.Span("Dataset: NASA-style Solar Wind", style={'marginRight': '20px'}),
        html.Span(" | ", style={'color': '#666', 'marginRight': '20px'}),
        html.Span("IBM Watsonx: ", style={'marginRight': '5px'}),
        html.Span("Connected", style={'color': '#28a745', 'fontWeight': 'bold', 'marginRight': '20px'}),
        html.Span(" | ", style={'color': '#666', 'marginRight': '20px'}),
        html.Span(f"Anomalies detected: {anomaly_count}", style={'color': '#dc3545', 'fontWeight': 'bold'})
    ], style={
        'backgroundColor': '#1a1a1a',
        'padding': '10px 20px',
        'color': '#e0e0e0',
        'fontSize': '14px',
        'borderBottom': '1px solid #333'
    }),
    
    # Demo button
    html.Div([
        html.Button(
            "Run Demo Sequence",
            id='demo-button',
            n_clicks=0,
            style={
                'backgroundColor': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '10px 20px',
                'fontSize': '14px',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontWeight': 'bold'
            }
        )
    ], style={'padding': '15px 20px', 'backgroundColor': '#0d0d0d'}),
    
    # Interval for animation
    dcc.Interval(
        id='demo-interval',
        interval=100,  # milliseconds
        n_intervals=0,
        disabled=True
    ),
    
    # Store for demo state
    dcc.Store(id='demo-state', data={'running': False, 'current_sample': 0}),
    
    # Main graph
    dcc.Graph(
        id='anomaly-dashboard',
        figure=initial_fig,
        style={'height': '900px'}
    ),
    
    # Alert card
    html.Div(
        card_content,
        style={
            'backgroundColor': '#1e1e1e',
            'padding': '20px',
            'marginTop': '20px',
            'borderLeft': f'5px solid {card_border_color}',
            'borderRadius': '5px',
            'color': '#e0e0e0'
        }
    )
], style={'padding': '0', 'backgroundColor': '#0d0d0d', 'margin': '0'})

# Callback for demo button
@app.callback(
    [Output('demo-interval', 'disabled'),
     Output('demo-state', 'data')],
    [Input('demo-button', 'n_clicks')],
    [State('demo-state', 'data')]
)
def start_demo(n_clicks, demo_state):
    if n_clicks == 0:
        raise PreventUpdate
    
    # Toggle demo
    if demo_state['running']:
        # Stop demo
        return True, {'running': False, 'current_sample': 0}
    else:
        # Start demo
        return False, {'running': True, 'current_sample': 0}

# Callback for animation
@app.callback(
    [Output('anomaly-dashboard', 'figure'),
     Output('demo-state', 'data', allow_duplicate=True),
     Output('demo-interval', 'disabled', allow_duplicate=True)],
    [Input('demo-interval', 'n_intervals')],
    [State('demo-state', 'data')],
    prevent_initial_call=True
)
def update_demo(n_intervals, demo_state):
    if not demo_state['running']:
        raise PreventUpdate
    
    # Update current sample
    current_sample = demo_state['current_sample'] + 200
    
    # Check if demo is complete
    if current_sample >= len(signal):
        # Show full signal and stop
        fig = create_figure()
        return fig, {'running': False, 'current_sample': len(signal)}, True
    
    # Create figure with subset
    signal_subset = signal[:current_sample]
    time_subset = time[:current_sample]
    fig = create_figure(signal_subset, time_subset)
    
    return fig, {'running': True, 'current_sample': current_sample}, False

if __name__ == '__main__':
    app.run(debug=True, port=8050)

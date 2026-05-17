# Mock IBM Watsonx AI scoring functions for anomaly detection

# Made with Bob

def score_telemetry(z_score_value):
    """
    Score telemetry data using mock Watsonx AI model
    
    Parameters:
    -----------
    z_score_value : float
        Z-score of spectral entropy
    
    Returns:
    --------
    dict : Prediction result with confidence, model info, and severity
    """
    if z_score_value > 2.5:
        confidence = round(min(0.99, 0.71 + z_score_value * 0.09), 2)
        severity = "HIGH" if z_score_value > 3.5 else "MEDIUM"
        
        return {
            "prediction": "ANOMALY",
            "confidence": confidence,
            "model": "ibm/autoai-anomaly-v2",
            "endpoint": "https://us-south.ml.cloud.ibm.com/ml/v4/deployments/astroaudit/predictions",
            "channel": "solar_wind_Bz",
            "severity": severity
        }
    else:
        return {
            "prediction": "NOMINAL",
            "confidence": 0.94,
            "model": "ibm/autoai-anomaly-v2"
        }

def generate_report(watsonx_result, anomaly_time):
    """
    Generate human-readable report from Watsonx scoring result
    
    Parameters:
    -----------
    watsonx_result : dict
        Result from score_telemetry function
    anomaly_time : float
        Timestamp of detected anomaly in seconds
    
    Returns:
    --------
    str : Three-sentence report with detection details and recommendations
    """
    if watsonx_result["prediction"] == "ANOMALY":
        severity = watsonx_result["severity"]
        confidence = watsonx_result["confidence"] * 100
        channel = watsonx_result.get("channel", "unknown")
        
        report = (
            f"IBM Watsonx AI detected a {severity} severity anomaly in the {channel} telemetry channel "
            f"with {confidence:.0f}% confidence. "
            f"The anomaly signature was identified at timestamp t={anomaly_time:.2f}s, "
            f"exhibiting spectral characteristics consistent with a Coronal Mass Ejection (CME) event. "
            f"Recommended action: Isolate affected channel for detailed analysis and alert mission control "
            f"for potential spacecraft system impacts."
        )
    else:
        confidence = watsonx_result["confidence"] * 100
        report = (
            f"IBM Watsonx AI analysis indicates nominal telemetry behavior with {confidence:.0f}% confidence. "
            f"No significant anomalies detected in the spectral entropy profile. "
            f"Recommended action: Continue routine monitoring of all telemetry channels."
        )
    
    return report

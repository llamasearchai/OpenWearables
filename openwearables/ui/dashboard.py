import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional

def create_ecg_chart(ecg_data: List[float], timestamps: List[float], sampling_rate: int = 250) -> Dict[str, Any]:
    """
    Create an ECG visualization chart.
    
    Args:
        ecg_data: ECG signal values
        timestamps: Corresponding timestamps
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Plotly figure as JSON
    """
    # Create time axis in seconds
    time_axis = np.arange(len(ecg_data)) / sampling_rate
    
    # Create figure
    fig = go.Figure()
    
    # Add ECG trace
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=ecg_data,
        mode='lines',
        name='ECG',
        line=dict(color='rgba(0, 122, 255, 1)', width=1.5)
    ))
    
    # Customize layout
    fig.update_layout(
        title="ECG Signal",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    
    return fig.to_dict()

def create_heart_rate_chart(heart_rates: List[float], timestamps: List[float]) -> Dict[str, Any]:
    """
    Create a heart rate trend chart.
    
    Args:
        heart_rates: Heart rate values
        timestamps: Corresponding timestamps
        
    Returns:
        Plotly figure as JSON
    """
    # Convert timestamps to datetime
    import datetime
    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Create figure
    fig = go.Figure()
    
    # Add heart rate trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=heart_rates,
        mode='lines+markers',
        name='Heart Rate',
        line=dict(color='rgba(255, 69, 58, 1)', width=2),
        marker=dict(size=6)
    ))
    
    # Add reference ranges
    fig.add_hrect(
        y0=60, y1=100,
        fillcolor="rgba(0, 255, 0, 0.1)",
        line_width=0,
        annotation_text="Normal Range",
        annotation_position="right"
    )
    
    # Customize layout
    fig.update_layout(
        title="Heart Rate Trend",
        xaxis_title="Time",
        yaxis_title="Heart Rate (bpm)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)',
            range=[40, 160]
        )
    )
    
    return fig.to_dict()

def create_spo2_chart(spo2_values: List[float], timestamps: List[float]) -> Dict[str, Any]:
    """
    Create a blood oxygen (SpO2) trend chart.
    
    Args:
        spo2_values: SpO2 values
        timestamps: Corresponding timestamps
        
    Returns:
        Plotly figure as JSON
    """
    # Convert timestamps to datetime
    import datetime
    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Create figure
    fig = go.Figure()
    
    # Add SpO2 trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=spo2_values,
        mode='lines+markers',
        name='SpO2',
        line=dict(color='rgba(52, 199, 89, 1)', width=2),
        marker=dict(size=6)
    ))
    
    # Add reference ranges
    fig.add_hrect(
        y0=95, y1=100,
        fillcolor="rgba(0, 255, 0, 0.1)",
        line_width=0,
        annotation_text="Normal Range",
        annotation_position="right"
    )
    
    fig.add_hrect(
        y0=90, y1=95,
        fillcolor="rgba(255, 204, 0, 0.1)",
        line_width=0,
        annotation_text="Caution",
        annotation_position="right"
    )
    
    fig.add_hrect(
        y0=0, y1=90,
        fillcolor="rgba(255, 69, 58, 0.1)",
        line_width=0,
        annotation_text="Low",
        annotation_position="right"
    )
    
    # Customize layout
    fig.update_layout(
        title="Blood Oxygen Saturation (SpO2)",
        xaxis_title="Time",
        yaxis_title="SpO2 (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)',
            range=[85, 100]
        )
    )
    
    return fig.to_dict()

def create_activity_chart(activities: List[str], durations: List[float]) -> Dict[str, Any]:
    """
    Create an activity distribution chart.
    
    Args:
        activities: List of activity types
        durations: Corresponding durations in minutes
        
    Returns:
        Plotly figure as JSON
    """
    # Create data frame
    df = pd.DataFrame({
        'Activity': activities,
        'Duration': durations
    })
    
    # Set colors for activities
    color_map = {
        'resting': 'rgba(175, 82, 222, 0.8)',
        'walking': 'rgba(90, 200, 250, 0.8)',
        'running': 'rgba(255, 149, 0, 0.8)',
        'other': 'rgba(142, 142, 147, 0.8)'
    }
    
    colors = [color_map.get(a.lower(), 'rgba(142, 142, 147, 0.8)') for a in activities]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=activities,
        y=durations,
        marker_color=colors,
        text=durations,
        textposition='auto'
    ))
    
    # Customize layout
    fig.update_layout(
        title="Activity Distribution",
        xaxis_title="Activity Type",
        yaxis_title="Duration (minutes)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    
    return fig.to_dict()

def create_stress_gauge(stress_level: float) -> Dict[str, Any]:
    """
    Create a stress level gauge.
    
    Args:
        stress_level: Stress level (0-100)
        
    Returns:
        Plotly figure as JSON
    """
    # Create figure
    fig = go.Figure()
    
    # Add gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stress_level,
        title={'text': "Stress Level"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'rgba(255, 255, 255, 0.5)'},
            'bar': {'color': "rgba(0, 0, 0, 0)"},
            'steps': [
                {'range': [0, 30], 'color': 'rgba(52, 199, 89, 0.6)'},
                {'range': [30, 70], 'color': 'rgba(255, 204, 0, 0.6)'},
                {'range': [70, 100], 'color': 'rgba(255, 69, 58, 0.6)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': stress_level
            }
        }
    ))
    
    # Customize layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=250,
        font=dict(color="white")
    )
    
    return fig.to_dict()

def create_health_score_gauge(health_score: float) -> Dict[str, Any]:
    """
    Create a health score gauge.
    
    Args:
        health_score: Overall health score (0-100)
        
    Returns:
        Plotly figure as JSON
    """
    # Create figure
    fig = go.Figure()
    
    # Add gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "Health Score"},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'rgba(255, 255, 255, 0.5)'},
            'bar': {'color': "rgba(0, 122, 255, 0.8)"},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(30, 30, 30, 0.4)'},
                {'range': [50, 75], 'color': 'rgba(40, 40, 40, 0.4)'},
                {'range': [75, 100], 'color': 'rgba(50, 50, 50, 0.4)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': health_score
            }
        }
    ))
    
    # Customize layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20),
        height=250,
        font=dict(color="white")
    )
    
    return fig.to_dict()

def create_hrv_chart(hrv_values: List[Dict[str, float]], timestamps: List[float]) -> Dict[str, Any]:
    """
    Create a heart rate variability trend chart.
    
    Args:
        hrv_values: List of HRV metrics dictionaries
        timestamps: Corresponding timestamps
        
    Returns:
        Plotly figure as JSON
    """
    # Convert timestamps to datetime
    import datetime
    dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    
    # Extract SDNN and RMSSD values
    sdnn_values = [hrv.get('SDNN', 0) for hrv in hrv_values]
    rmssd_values = [hrv.get('RMSSD', 0) for hrv in hrv_values]
    
    # Create figure
    fig = go.Figure()
    
    # Add SDNN trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=sdnn_values,
        mode='lines+markers',
        name='SDNN',
        line=dict(color='rgba(94, 92, 230, 1)', width=2),
        marker=dict(size=6)
    ))
    
    # Add RMSSD trace
    fig.add_trace(go.Scatter(
        x=dates,
        y=rmssd_values,
        mode='lines+markers',
        name='RMSSD',
        line=dict(color='rgba(255, 159, 10, 1)', width=2),
        marker=dict(size=6)
    ))
    
    # Customize layout
    fig.update_layout(
        title="Heart Rate Variability",
        xaxis_title="Time",
        yaxis_title="HRV (ms)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            showline=True,
            linecolor='rgba(255, 255, 255, 0.2)'
        )
    )
    
    return fig.to_dict()
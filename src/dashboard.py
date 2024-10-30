# src/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from signal_processor import PowerSignalProcessor, generate_test_data
from ml_models import PowerConsumptionPredictor, AnomalyDetector
import time

def init_dashboard():
    st.set_page_config(page_title="Power Grid Analyzer", layout="wide", theme="dark")
    st.title("Smart Power Grid Analysis")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    duration = st.sidebar.slider("Signal Duration (seconds)", 1, 30, 10)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    
    # Initialize components
    processor = PowerSignalProcessor()
    
    # Generate data with noise
    data = generate_test_data(duration_seconds=duration)
    data['voltage'] += np.random.normal(0, noise_level, len(data))
    
    # Process signal
    features = processor.process_signal(data['voltage'])
    
    # Metrics with improved styling
    cols = st.columns(3)
    with cols[0]:
        st.metric("RMS Value", f"{features['rms']:.2f} V", 
                 delta=f"{(features['rms']-1):.2f}")
    with cols[1]:
        st.metric("THD", f"{features['thd']*100:.2f}%", 
                 delta=f"{(features['thd']-0.05)*100:.2f}%")
    with cols[2]:
        quality = "Good" if features['thd'] < 0.05 else "Poor"
        st.metric("Signal Quality", quality)
    
    # Interactive signal plot
    st.subheader("Power Signal Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['voltage'],
        name="Voltage",
        line=dict(color='#00ff00', width=1)
    ))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Spectrogram with improved styling
    st.subheader("Frequency Analysis")
    freq_data = features['frequency_components']
    fig2 = go.Figure(data=go.Heatmap(
        z=10 * np.log10(freq_data['power']),
        x=freq_data['times'],
        y=freq_data['frequencies'],
        colorscale='Viridis'
    ))
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    init_dashboard()
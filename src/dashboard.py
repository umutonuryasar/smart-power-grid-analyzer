# src/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from signal_processor import PowerSignalProcessor, generate_test_data
from ml_models import PowerConsumptionPredictor, AnomalyDetector
from data_simulator import PowerDataSimulator
from datetime import datetime, timedelta
import time

def init_dashboard():
    st.set_page_config(page_title="Power Grid Analyzer", layout="wide")
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: white;
        }
        .metric-card {
            background-color: #1E2530;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .quality-high { color: #00ff00; }
        .quality-medium { color: #ffaa00; }
        .quality-low { color: #ff0000; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Smart Power Grid Analysis")

    # Initialize session state
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame(columns=['timestamp', 'load'])
        st.session_state.anomalies = []
        st.session_state.simulator = PowerDataSimulator()
        st.session_state.predictor = PowerConsumptionPredictor()
        st.session_state.anomaly_detector = AnomalyDetector()
        st.session_state.signal_processor = PowerSignalProcessor()
        st.session_state.is_model_trained = False
        st.session_state.quality_history = []
        st.session_state.last_update = datetime.now()

    # Sidebar Configuration
    with st.sidebar:
        st.header("Settings")
        
        # Simulation Settings
        st.subheader("Simulation Settings")
        base_load = st.slider("Base Load (W)", 500, 2000, 1000)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
        anomaly_prob = st.slider("Anomaly Probability", 0.0, 0.2, 0.05)
        
        # Update simulator settings
        st.session_state.simulator.base_load = base_load
        st.session_state.simulator.noise_std = noise_level * 50
        st.session_state.simulator.anomaly_probability = anomaly_prob

        # ML Model Settings
        st.subheader("ML Model Settings")
        sequence_length = st.slider("Prediction Window (hours)", 1, 24, 6)
        forecast_horizon = st.slider("Forecast Horizon (hours)", 1, 12, 3)
        
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                if len(st.session_state.historical_data) > sequence_length:
                    # Train prediction model
                    data = st.session_state.historical_data['load'].values
                    st.session_state.predictor = PowerConsumptionPredictor(
                        sequence_length=sequence_length,
                        forecast_horizon=forecast_horizon
                    )
                    training_history = st.session_state.predictor.train(data)
                    
                    # Train anomaly detector
                    st.session_state.anomaly_detector.train(data)
                    
                    st.session_state.is_model_trained = True
                    st.success("Models trained successfully!")
                    
                    # Show training metrics
                    metrics = st.session_state.predictor.get_metrics()
                    if metrics:
                        st.write("Training Metrics:")
                        st.write(f"Final MAE: {metrics['final_mae']:.2f}")
                        st.write(f"Validation MAE: {metrics['val_mae']:.2f}")
                else:
                    st.warning("Not enough data for training. Please collect more data.")

    # Main Dashboard Area
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_load = st.session_state.historical_data['load'].iloc[-1] if len(st.session_state.historical_data) > 0 else 0
        st.metric("Current Load", f"{current_load:.2f} W", 
                 delta=f"{current_load - base_load:.2f} W")
    
    with col2:
        avg_load = st.session_state.historical_data['load'].mean() if len(st.session_state.historical_data) > 0 else 0
        st.metric("Average Load", f"{avg_load:.2f} W")
    
    with col3:
        anomaly_count = len(st.session_state.anomalies)
        st.metric("Detected Anomalies", anomaly_count)
    
    with col4:
        if st.session_state.quality_history:
            latest_quality = st.session_state.quality_history[-1]
            quality_color = 'quality-high' if latest_quality > 80 else 'quality-medium' if latest_quality > 60 else 'quality-low'
            st.markdown(f"<div class='metric-card {quality_color}'>Power Quality Score: {latest_quality:.1f}%</div>", 
                       unsafe_allow_html=True)

    # Main Visualization Area
    st.subheader("Power Consumption Monitor")
    
    # Create main power consumption plot
    fig = go.Figure()
    
    if len(st.session_state.historical_data) > 0:
        # Plot actual data
        fig.add_trace(go.Scatter(
            x=st.session_state.historical_data['timestamp'],
            y=st.session_state.historical_data['load'],
            name="Actual Load",
            line=dict(color='#00ff00', width=1)
        ))
        
        # Plot predictions if model is trained
        if st.session_state.is_model_trained:
            recent_data = st.session_state.historical_data['load'].values[-sequence_length:]
            if len(recent_data) >= sequence_length:
                predictions = st.session_state.predictor.predict(recent_data)[0]
                prediction_times = [
                    st.session_state.historical_data['timestamp'].iloc[-1] + timedelta(hours=i+1)
                    for i in range(len(predictions))
                ]
                
                fig.add_trace(go.Scatter(
                    x=prediction_times,
                    y=predictions,
                    name="Prediction",
                    line=dict(color='#ff8c00', width=1, dash='dash')
                ))
        
        # Plot anomalies
        if st.session_state.anomalies:
            anomaly_df = pd.DataFrame(st.session_state.anomalies)
            fig.add_trace(go.Scatter(
                x=anomaly_df['timestamp'],
                y=anomaly_df['load'],
                mode='markers',
                name="Anomalies",
                marker=dict(color='red', size=10, symbol='x')
            ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Power Quality Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Power Quality Analysis")
        if len(st.session_state.historical_data) > 0:
            recent_data = st.session_state.historical_data['load'].values[-1000:]  # Analyze last 1000 points
            quality_features = st.session_state.signal_processor.process_signal(recent_data)
            
            # Display quality metrics
            quality_metrics = {
                "THD": f"{quality_features['thd']*100:.2f}%",
                "Crest Factor": f"{quality_features['crest_factor']:.2f}",
                "RMS Deviation": f"{quality_features['rms_deviation']*100:.2f}%"
            }
            
            for metric, value in quality_metrics.items():
                st.write(f"{metric}: {value}")
            
            # Display quality issues if any
            if quality_features['quality_issues']:
                st.warning("Quality Issues Detected:")
                for issue in quality_features['quality_issues']:
                    st.write(f"- {issue}")
            
            # Store quality score in history
            st.session_state.quality_history.append(quality_features['quality_score'])
    
    with col2:
        st.subheader("Frequency Analysis")
        if len(st.session_state.historical_data) > 0:
            # Create frequency spectrum plot
            freq_data = quality_features['frequency_components']
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=freq_data['frequencies'],
                y=freq_data['magnitudes'],
                name="Frequency Spectrum",
                line=dict(color='#00ffff')
            ))
            
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    title="Frequency (Hz)",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    title="Magnitude",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                height=300
            )
            
            st.plotly_chart(fig2, use_container_width=True)

    # Data Generation Controls
    st.subheader("Data Generation")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Generate New Data Point'):
            timestamp, load, is_anomaly = st.session_state.simulator.generate_realtime_data(noise_level)
            
            # Add to historical data
            new_data = pd.DataFrame({'timestamp': [timestamp], 'load': [load]})
            st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_data])
            
            # Check for anomalies if model is trained
            if st.session_state.is_model_trained and is_anomaly:
                st.session_state.anomalies.append({
                    'timestamp': timestamp,
                    'load': load
                })
                st.warning(f"⚠️ Anomaly detected at {timestamp}!")
    
    with col2:
        if st.button('Generate Batch Data'):
            with st.spinner('Generating batch data...'):
                batch_data = st.session_state.simulator.generate_batch_data(
                    hours=6,
                    noise_level=noise_level
                )
                st.session_state.historical_data = pd.concat(
                    [st.session_state.historical_data, batch_data]
                )
                st.success('Batch data generated successfully!')

    # Recent Anomalies Table
    if st.session_state.anomalies:
        st.subheader("Recent Anomalies")
        anomaly_df = pd.DataFrame(st.session_state.anomalies[-5:])  # Show last 5 anomalies
        st.dataframe(anomaly_df)

if __name__ == "__main__":
    init_dashboard()
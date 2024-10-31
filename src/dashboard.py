# src/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from signal_processor import PowerSignalProcessor, generate_test_data
from ml_models import PowerConsumptionPredictor, AnomalyDetector
from data_simulator import PowerDataSimulator
from datetime import datetime, timedelta

def init_dashboard():
    st.set_page_config(page_title="Power Grid Analyzer", layout="wide")
    st.markdown("""
       <style>
       .stApp {
           background-color: #0E1117;
           color: white;
       }
       </style>
    """, unsafe_allow_html=True)
    
    st.title("Smart Power Grid Analysis")
    
    # Initialize session state for data storage
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = pd.DataFrame(columns=['timestamp', 'load'])
        st.session_state.anomalies = []
        st.session_state.simulator = PowerDataSimulator()
        st.session_state.predictor = PowerConsumptionPredictor()
        st.session_state.anomaly_detector = AnomalyDetector()
        st.session_state.is_model_trained = False

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        simulation_speed = st.slider("Simulation Speed (seconds)", 1, 10, 2)
        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)
        anomaly_threshold = st.slider("Anomaly Detection Threshold", 2.0, 5.0, 3.0)
        
        # Training controls
        st.subheader("ML Model Settings")
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                if len(st.session_state.historical_data) > 24:  # Minimum data required
                    # Train prediction model
                    data = st.session_state.historical_data['load'].values
                    st.session_state.predictor.train(data)
                    
                    # Train anomaly detector
                    st.session_state.anomaly_detector = AnomalyDetector(threshold=anomaly_threshold)
                    st.session_state.anomaly_detector.train(data)
                    
                    st.session_state.is_model_trained = True
                    st.success("Models trained successfully!")
                else:
                    st.warning("Not enough data for training. Please collect more data.")

    # Main dashboard area
    col1, col2, col3 = st.columns(3)
    
    # Real-time metrics
    with col1:
        if len(st.session_state.historical_data) > 0:
            current_load = st.session_state.historical_data['load'].iloc[-1]
            st.metric("Current Load", f"{current_load:.2f} W")
    
    with col2:
        if len(st.session_state.historical_data) > 0:
            avg_load = st.session_state.historical_data['load'].mean()
            st.metric("Average Load", f"{avg_load:.2f} W")
    
    with col3:
        if len(st.session_state.historical_data) > 0:
            anomaly_count = len(st.session_state.anomalies)
            st.metric("Detected Anomalies", anomaly_count)

    # Real-time power consumption plot
    st.subheader("Power Consumption Monitor")
    
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
        if st.session_state.is_model_trained and len(st.session_state.historical_data) > 24:
            recent_data = st.session_state.historical_data['load'].values[-24:]
            prediction = st.session_state.predictor.predict(recent_data)[0][0]
            
            last_timestamp = st.session_state.historical_data['timestamp'].iloc[-1]
            next_timestamp = last_timestamp + timedelta(seconds=simulation_speed)
            
            fig.add_trace(go.Scatter(
                x=[last_timestamp, next_timestamp],
                y=[st.session_state.historical_data['load'].iloc[-1], prediction],
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
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Update data button
    if st.button('Generate New Data Point'):
        # Generate new data point
        timestamp, load = st.session_state.simulator.generate_realtime_data()
        
        # Add to historical data
        new_data = pd.DataFrame({'timestamp': [timestamp], 'load': [load]})
        st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_data])
        
        # Detect anomalies if model is trained
        if st.session_state.is_model_trained:
            is_anomaly = st.session_state.anomaly_detector.detect_anomalies(
                np.array([load])
            )[0]
            
            if is_anomaly:
                st.session_state.anomalies.append({
                    'timestamp': timestamp,
                    'load': load
                })
                st.warning(f"Anomaly detected at {timestamp}!")

    # Display recent anomalies
    if st.session_state.anomalies:
        st.subheader("Recent Anomalies")
        anomaly_df = pd.DataFrame(st.session_state.anomalies[-5:])  # Show last 5 anomalies
        st.dataframe(anomaly_df)

if __name__ == "__main__":
    init_dashboard()
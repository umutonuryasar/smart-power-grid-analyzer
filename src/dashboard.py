# src/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from signal_processor import PowerSignalProcessor, generate_test_data
from ml_models import PowerConsumptionPredictor, AnomalyDetector

def init_dashboard():
   st.set_page_config(page_title="Power Grid Analyzer", layout="wide")
   st.title("Smart Power Grid Analysis")
   
   # Initialize components
   processor = PowerSignalProcessor()
   predictor = PowerConsumptionPredictor()
   detector = AnomalyDetector()
   
   # Simulated data
   data = generate_test_data(duration_seconds=10)
   
   # Process signal
   features = processor.process_signal(data['voltage'])
   
   # Layout
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("RMS Value", f"{features['rms']:.2f} V")
   with col2:
       st.metric("THD", f"{features['thd']:.2%}")
   with col3:
       st.metric("Signal Quality", "Good" if features['thd'] < 0.05 else "Poor")

   # Signal plot
   st.subheader("Power Signal Analysis")
   fig = go.Figure()
   fig.add_trace(go.Scatter(x=data['timestamp'], y=data['voltage'], name="Voltage"))
   st.plotly_chart(fig)

   # Spectrogram
   st.subheader("Frequency Analysis")
   freq_data = features['frequency_components']
   fig2 = go.Figure(data=go.Heatmap(
       z=10 * np.log10(freq_data['power']),
       x=freq_data['times'],
       y=freq_data['frequencies'],
       colorscale='Viridis'
   ))
   st.plotly_chart(fig2)

if __name__ == "__main__":
   init_dashboard()
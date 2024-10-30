import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def init_dashboard():
    st.set_page_config(page_title="Power Grid Analyzer", layout="wide")
    st.title("Smart Power Grid Analysis")
    
    # Sidebar
    st.sidebar.header("Controls")
    timeframe = st.sidebar.selectbox("Timeframe", ["24h", "7d", "30d"])
    
    # Main layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Load", "2.4 MW", "1.2%")
    with col2:
        st.metric("Power Quality", "98.5%", "0.5%")
    with col3:
        st.metric("Active Alerts", "2", "-1")

if __name__ == "__main__":
    init_dashboard()
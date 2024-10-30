# Smart Power Grid Analyzer

Real-time power grid analysis system with machine learning capabilities for power quality monitoring and anomaly detection.

## Features

- Power signal processing and spectral analysis
- Real-time monitoring dashboard using Streamlit
- RMS and THD calculations
- Frequency domain visualization
- Machine learning-based load prediction
- Anomaly detection

## Tech Stack

- Python 3.11+
- Streamlit
- NumPy/SciPy
- TensorFlow
- Plotly
- Pandas

## Installation

```bash
git clone https://github.com/yourusername/smart-power-grid-analyzer.git
cd smart-power-grid-analyzer
pip install -r requirements.txt
```

## Usage

```bash
streamlit run src/main.py
```

## Project Structure

smart-power-grid-analyzer/
├── src/
│   ├── signal_processor.py    # Signal processing module
│   ├── ml_models.py          # Machine learning models
│   ├── dashboard.py          # Streamlit dashboard
│   ├── data_simulator.py     # Real-time data simulation
│   └── main.py              # Main application entry
├── tests/                    # Unit tests
├── data/                     # Data storage
└── docs/                     # Documentation

## LICENSE

MIT

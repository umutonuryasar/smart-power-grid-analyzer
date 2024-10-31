# Smart Power Grid Analyzer

A real-time power grid monitoring and analysis system with ML-powered predictions and anomaly detection.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
- [Example Use Cases](#example-use-cases)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Features

- Real-time power signal monitoring
- Power quality analysis and metrics
- ML-based consumption prediction
- Anomaly detection
- Interactive dashboard
- Configurable data simulation
- Frequency analysis and visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Packages

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
streamlit>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
plotly>=5.3.0
scipy>=1.7.0
tensorflow>=2.8.0
scikit-learn>=0.24.0
```

### Project Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/smart-power-grid-analyzer.git
cd smart-power-grid-analyzer
```

2. Create and activate virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

Run the following command in your terminal:

```bash
streamlit run src/main.py
```

### Basic Operations

1. **Data Generation**:
   - Use "Generate New Data Point" for single data points
   - Use "Generate Batch Data" for multiple points
   - Adjust simulation parameters in the sidebar

2. **Model Training**:
   - Generate sufficient data points (minimum 24 points)
   - Click "Train Models" in the sidebar
   - Wait for training completion confirmation

3. **Monitoring**:
   - Watch real-time power consumption
   - Monitor quality metrics
   - Check for anomalies
   - View frequency analysis

### Example Workflows

#### 1. Basic Monitoring Setup

```python
from dashboard import init_dashboard
import streamlit as st

# Initialize and run dashboard
init_dashboard()
```

#### 2. Custom Data Simulation

```python
from data_simulator import PowerDataSimulator

# Initialize simulator with custom parameters
simulator = PowerDataSimulator(base_load=1000)

# Generate real-time data point
timestamp, load, is_anomaly = simulator.generate_realtime_data(noise_level=0.1)

# Generate batch data
batch_data = simulator.generate_batch_data(hours=24, noise_level=0.1)
```

#### 3. Signal Processing

```python
from signal_processor import PowerSignalProcessor

# Initialize processor
processor = PowerSignalProcessor(sampling_rate=1000)

# Process signal data
features = processor.process_signal(data)

# Access quality metrics
thd = features['thd']
crest_factor = features['crest_factor']
quality_score = features['quality_score']
```

#### 4. ML Model Usage

```python
from ml_models import PowerConsumptionPredictor, AnomalyDetector

# Initialize predictive model
predictor = PowerConsumptionPredictor(sequence_length=24)

# Train model
predictor.train(historical_data)

# Make predictions
predictions = predictor.predict(recent_data)

# Initialize anomaly detector
detector = AnomalyDetector(threshold=3)

# Train detector
detector.train(historical_data)

# Detect anomalies
anomalies = detector.detect_anomalies(new_data)
```

## Components

### 1. Dashboard (`dashboard.py`)

The main interface component providing:

- Real-time visualization
- Interactive controls
- Quality metrics display
- Model training interface

### 2. Signal Processor (`signal_processor.py`)

Handles signal analysis including:

- Basic metrics calculation
- Frequency analysis
- Power quality assessment
- Transient detection

### 3. Data Simulator (`data_simulator.py`)

Generates realistic power grid data with:

- Configurable base load
- Daily patterns
- Noise simulation
- Anomaly injection

### 4. ML Models (`ml_models.py`)

Provides predictive capabilities:

- Power consumption prediction
- Anomaly detection
- Model training and evaluation
- Prediction visualization

## Configuration

### Simulator Settings

- `base_load`: Base power consumption (default: 1000W)
- `noise_level`: Amount of noise (0.0 to 1.0)
- `anomaly_probability`: Chance of anomaly generation

### Signal Processor Settings

- `sampling_rate`: Data sampling rate (Hz)
- `quality_thresholds`: Quality assessment thresholds
- `freq_bands`: Frequency band definitions

### ML Model Parameters

- `sequence_length`: Input sequence length
- `forecast_horizon`: Prediction horizon
- `threshold`: Anomaly detection threshold

## Troubleshooting

### Common Issues

1. **Insufficient Data Error**

```
Warning: Not enough data for training. Please collect more data.
```

Solution: Generate more data points before training models.

2. **Model Training Issues**

```
Error: Model must be trained before making predictions.
```

Solution: Ensure model training is completed successfully.

3. **Memory Issues**

If the dashboard becomes slow:

- Reduce the amount of historical data
- Lower the batch generation size
- Clear session state if needed

### Best Practices

1. **Data Generation**:

- Start with small batch sizes
- Gradually increase complexity
- Monitor system resources

2. **Model Training**:

- Ensure sufficient data quality
- Adjust parameters gradually
- Monitor training metrics

3. **Performance**:

- Regular cache clearing
- Limit historical data size
- Optimize visualization settings

## Example Use Cases

### 1. Power Quality Monitoring

```python
# Initialize components
processor = PowerSignalProcessor()
simulator = PowerDataSimulator()

# Generate and analyze data
data = simulator.generate_batch_data(hours=1)
quality_metrics = processor.process_signal(data['load'].values)

# Monitor quality
print(f"THD: {quality_metrics['thd']*100:.2f}%")
print(f"Quality Score: {quality_metrics['quality_score']}")
```

### 2. Predictive Maintenance

```python
# Setup prediction system
predictor = PowerConsumptionPredictor()
detector = AnomalyDetector()

# Train models
predictor.train(historical_data)
detector.train(historical_data)

# Monitor and predict
predictions = predictor.predict(recent_data)
anomalies = detector.detect_anomalies(new_data)

# Act on predictions
if anomalies.any():
    print("Maintenance may be required!")
```

### 3. Load Forecasting

```python
# Initialize predictor with custom parameters
predictor = PowerConsumptionPredictor(
    sequence_length=24,
    forecast_horizon=12
)

# Train and forecast
predictor.train(historical_data)
future_load = predictor.predict(recent_data)

# Use forecasts
print(f"Predicted load: {future_load[0]} W")

```

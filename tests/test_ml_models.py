# tests/test_ml_models.py
import pytest
import numpy as np
from src.ml_models import PowerConsumptionPredictor, AnomalyDetector

def test_predictor_initialization():
    predictor = PowerConsumptionPredictor(sequence_length=24)
    assert predictor.sequence_length == 24
    
def test_predictor_data_preparation():
    predictor = PowerConsumptionPredictor(sequence_length=3)
    data = np.array([1, 2, 3, 4, 5])
    X, y = predictor.prepare_sequences(data)
    assert X.shape[1] == 3
    assert len(y) == len(X)

def test_anomaly_detector():
    detector = AnomalyDetector(threshold=2)
    data = np.random.normal(0, 1, 100)
    detector.train(data, epochs=1)
    anomalies = detector.detect_anomalies(data)
    assert len(anomalies) == len(data)
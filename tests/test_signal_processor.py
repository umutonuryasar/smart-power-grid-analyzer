# tests/test_signal_processor.py
import pytest
import numpy as np
from src.signal_processor import PowerSignalProcessor, generate_test_data

def test_signal_processor_initialization():
    processor = PowerSignalProcessor(sampling_rate=1000)
    assert processor.sampling_rate == 1000

def test_rms_calculation():
    processor = PowerSignalProcessor()
    test_data = generate_test_data()
    result = processor._calculate_rms(test_data['voltage'])
    assert isinstance(result, float)
    assert result > 0
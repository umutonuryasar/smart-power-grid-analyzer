# src/signal_processor.py
import numpy as np
from scipy import signal
from scipy.fft import fft
import pandas as pd

class PowerSignalProcessor:
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        
    def process_signal(self, data):
        data_array = np.array(data, dtype=float)  # Convert to numpy array
        features = {
            'rms': self._calculate_rms(data_array),
            'thd': self._calculate_thd(data_array),
            'frequency_components': self._frequency_analysis(data_array)
        }
        return features
    
    def _calculate_rms(self, data):
        return np.sqrt(np.mean(data**2))
    
    def _calculate_thd(self, data):
        spectrum = np.abs(fft(data.astype(np.float64)))  # Ensure float64 type
        fundamental = max(spectrum[1:len(spectrum)//2])  # Get max component as fundamental
        harmonics = spectrum[2:10]
        thd = np.sqrt(np.sum(harmonics**2))/fundamental if fundamental > 0 else 0
        return thd
    
    def _frequency_analysis(self, data):
        frequencies, times, Sxx = signal.spectrogram(data, self.sampling_rate)
        return {'frequencies': frequencies, 'times': times, 'power': Sxx}

def generate_test_data(duration_seconds=1):
    t = np.linspace(0, duration_seconds, 1000)
    signal_data = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
    return pd.DataFrame({
        'timestamp': t,
        'voltage': signal_data
    })
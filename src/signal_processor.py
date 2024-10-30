# src/signal_processor.py
import numpy as np
from scipy import signal
from scipy.fft import fft
import pandas as pd

class PowerSignalProcessor:
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        
    def process_signal(self, data):
        features = {
            'rms': self._calculate_rms(data),
            'thd': self._calculate_thd(data),
            'frequency_components': self._frequency_analysis(data)
        }
        return features
    
    def _calculate_rms(self, data):
        return np.sqrt(np.mean(np.array(data)**2))
    
    def _calculate_thd(self, data):
        spectrum = np.abs(fft(data))
        fundamental = spectrum[1]
        harmonics = spectrum[2:10]
        return np.sqrt(np.sum(harmonics**2))/fundamental
    
    def _frequency_analysis(self, data):
        frequencies, times, Sxx = signal.spectrogram(data, self.sampling_rate)
        return {'frequencies': frequencies, 'times': times, 'power': Sxx}

# Test data generator for development
def generate_test_data(duration_seconds=1):
    t = np.linspace(0, duration_seconds, 1000)
    # Fundamental frequency (50 Hz) + harmonics
    signal_data = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*100*t)
    return pd.DataFrame({'timestamp': t, 'voltage': signal_data})
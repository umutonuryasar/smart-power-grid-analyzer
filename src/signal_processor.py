# src/signal_processor.py
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
from typing import Dict, Any, List, Tuple

class PowerSignalProcessor:
    def __init__(self, sampling_rate: int = 1000):
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate // 2
        
        self.freq_bands = {
            'power_line': (45, 65),
            'harmonics': (100, 300),
            'transients': (500, 1000)
        }
        
        self.quality_thresholds = {
            'thd': 0.05,
            'rms_deviation': 0.1,
            'crest_factor': 1.5
        }
    
    def process_signal(self, data: np.ndarray) -> Dict[str, Any]:
        """Process power signal and extract features."""
        data_array = np.array(data, dtype=float)
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(data_array)
        
        # Frequency domain analysis
        freq_features = self._analyze_frequency_components(data_array)
        
        # Power quality metrics
        quality_metrics = self._assess_power_quality(data_array, freq_features)
        
        return {
            **basic_metrics,
            **freq_features,
            **quality_metrics,
            'sampling_rate': self.sampling_rate
        }
    
    def _calculate_basic_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic time-domain metrics."""
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))
        crest_factor = peak / rms if rms > 0 else float('inf')
        
        return {
            'rms': rms,
            'peak': peak,
            'crest_factor': crest_factor,
            'mean': np.mean(data),
            'std': np.std(data)
        }
    
    def _analyze_frequency_components(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform frequency domain analysis."""
        # Compute FFT
        n_samples = len(data)
        yf = fft(data)
        xf = fftfreq(n_samples, 1/self.sampling_rate)
        
        # Get positive frequencies
        positive_freq_mask = xf > 0
        frequencies = xf[positive_freq_mask]
        magnitudes = np.abs(yf[positive_freq_mask])
        
        # Normalize magnitudes
        magnitudes = magnitudes / n_samples
        
        # Find fundamental frequency (highest magnitude)
        fundamental_idx = np.argmax(magnitudes)
        fundamental_freq = frequencies[fundamental_idx]
        fundamental_magnitude = magnitudes[fundamental_idx]
        
        # Calculate THD
        harmonics = []
        for i in range(2, 11):  # Up to 10th harmonic
            harmonic_freq = fundamental_freq * i
            idx = np.argmin(np.abs(frequencies - harmonic_freq))
            harmonics.append(magnitudes[idx])
        
        harmonics = np.array(harmonics)
        thd = np.sqrt(np.sum(harmonics**2)) / fundamental_magnitude if fundamental_magnitude > 0 else 0
        
        # Convert numpy arrays to lists for JSON serialization
        frequencies_list = frequencies.tolist()
        magnitudes_list = magnitudes.tolist()
        
        return {
            'frequency_components': {
                'frequencies': frequencies_list,
                'magnitudes': magnitudes_list,
            },
            'thd': thd,
            'fundamental_frequency': float(fundamental_freq),
        }
    
    def _assess_power_quality(self, data: np.ndarray, freq_features: Dict) -> Dict[str, Any]:
        """Assess power quality metrics."""
        quality_issues = []
        
        # Basic calculations
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))
        crest_factor = peak / rms if rms > 0 else float('inf')
        rms_deviation = abs(1 - rms)
        
        # Check thresholds
        if freq_features['thd'] > self.quality_thresholds['thd']:
            quality_issues.append('High harmonic distortion')
            
        if rms_deviation > self.quality_thresholds['rms_deviation']:
            quality_issues.append('Voltage deviation')
            
        if crest_factor > self.quality_thresholds['crest_factor']:
            quality_issues.append('High crest factor')
        
        # Calculate quality score
        quality_score = 100
        quality_score -= len(quality_issues) * 20  # Reduce score for each issue
        quality_score = max(0, min(100, quality_score))  # Ensure between 0 and 100
        
        return {
            'quality_score': quality_score,
            'quality_issues': quality_issues,
            'rms_deviation': rms_deviation,
            'crest_factor': crest_factor
        }

def generate_test_data(duration_seconds: float = 1, 
                      base_frequency: float = 50,
                      sampling_rate: int = 1000,
                      noise_level: float = 0.1) -> pd.DataFrame:
    """Generate test power signal data."""
    t = np.linspace(0, duration_seconds, int(duration_seconds * sampling_rate))
    
    # Generate fundamental component
    signal_data = np.sin(2 * np.pi * base_frequency * t)
    
    # Add some harmonics
    signal_data += 0.1 * np.sin(2 * np.pi * base_frequency * 2 * t)  # 2nd harmonic
    signal_data += 0.05 * np.sin(2 * np.pi * base_frequency * 3 * t)  # 3rd harmonic
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(t))
    signal_data += noise
    
    return pd.DataFrame({
        'timestamp': t,
        'voltage': signal_data
    })
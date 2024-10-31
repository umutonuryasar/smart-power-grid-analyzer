# src/signal_processor.py
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
from typing import Dict, Any, List, Tuple
import warnings

class PowerSignalProcessor:
    def __init__(self, sampling_rate: int = 1000):
        """
        Initialize the power signal processor.
        
        Args:
            sampling_rate (int): Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate // 2
        
        # Configuration parameters
        self.freq_bands = {
            'power_line': (45, 65),      # Power line frequency (around 50/60 Hz)
            'harmonics': (100, 300),      # Harmonic frequencies
            'transients': (500, 1000)     # High-frequency transients
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'thd': 0.05,          # Total Harmonic Distortion threshold
            'rms_deviation': 0.1,  # Maximum allowable RMS deviation
            'crest_factor': 1.5    # Maximum allowable crest factor
        }
    
    def process_signal(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Process power signal and extract comprehensive features.
        
        Args:
            data: Input voltage/current signal
            
        Returns:
            Dictionary containing various signal features and quality metrics
        """
        data_array = np.array(data, dtype=float)
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(data_array)
        
        # Frequency domain analysis
        freq_features = self._analyze_frequency_components(data_array)
        
        # Power quality metrics
        quality_metrics = self._assess_power_quality(data_array, freq_features)
        
        # Combine all features
        features = {
            **basic_metrics,
            **freq_features,
            **quality_metrics,
            'sampling_rate': self.sampling_rate
        }
        
        return features
    
    def _calculate_basic_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate basic time-domain signal metrics."""
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
        """Perform detailed frequency domain analysis."""
        # Compute FFT
        n_samples = len(data)
        yf = fft(data)
        xf = fftfreq(n_samples, 1/self.sampling_rate)
        
        # Get positive frequencies only
        positive_freq_mask = xf > 0
        frequencies = xf[positive_freq_mask]
        magnitudes = np.abs(yf[positive_freq_mask])
        
        # Normalize magnitudes
        magnitudes = magnitudes / n_samples
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(magnitudes, height=np.max(magnitudes)*0.1)[0]
        dominant_freqs = [(frequencies[i], magnitudes[i]) for i in peak_indices]
        dominant_freqs.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate THD
        fundamental_idx = np.argmax(magnitudes)
        fundamental_freq = frequencies[fundamental_idx]
        fundamental_magnitude = magnitudes[fundamental_idx]
        
        harmonics_mask = np.zeros_like(frequencies, dtype=bool)
        for i in range(2, 11):  # Consider up to 10th harmonic
            harmonic_freq = fundamental_freq * i
            freq_range = (harmonic_freq - 5, harmonic_freq + 5)  # 5 Hz tolerance
            harmonics_mask |= (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        
        harmonic_magnitudes = magnitudes[harmonics_mask]
        thd = np.sqrt(np.sum(harmonic_magnitudes**2)) / fundamental_magnitude if fundamental_magnitude > 0 else 0
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(data, self.sampling_rate, 
                                     nperseg=min(256, len(data)),
                                     scaling='spectrum')
        
        return {
            'frequency_components': {
                'frequencies': frequencies.tolist(),
                'magnitudes': magnitudes.tolist(),
                'dominant_frequencies': dominant_freqs[:5],  # Top 5 dominant frequencies
            },
            'thd': thd,
            'fundamental_frequency': fundamental_freq,
            'spectrogram': {
                'frequencies': f,
                'times': t,
                'power': Sxx
            }
        }
    
    def _assess_power_quality(self, data: np.ndarray, freq_features: Dict) -> Dict[str, Any]:
        """Assess power quality metrics."""
        # Initialize quality flags
        quality_issues = []
        
        # Check THD
        if freq_features['thd'] > self.quality_thresholds['thd']:
            quality_issues.append('High harmonic distortion')
        
        # Check RMS deviation from nominal
        rms = np.sqrt(np.mean(data**2))
        rms_deviation = abs(1 - rms)  # Assuming normalized to 1
        if rms_deviation > self.quality_thresholds['rms_deviation']:
            quality_issues.append('Voltage deviation')
        
        # Check crest factor
        peak = np.max(np.abs(data))
        crest_factor = peak / rms if rms > 0 else float('inf')
        if crest_factor > self.quality_thresholds['crest_factor']:
            quality_issues.append('High crest factor')
        
        # Analyze frequency bands
        for band_name, (low_freq, high_freq) in self.freq_bands.items():
            band_power = self._calculate_band_power(
                freq_features['frequency_components']['frequencies'],
                freq_features['frequency_components']['magnitudes'],
                low_freq, high_freq
            )
            if band_power > 0.1:  # If band power is more than 10% of total
                quality_issues.append(f'High {band_name} content')
        
        return {
            'quality_score': max(0, 100 - len(quality_issues) * 20),  # Simple scoring
            'quality_issues': quality_issues,
            'rms_deviation': rms_deviation,
            'band_powers': {
                band: self._calculate_band_power(
                    freq_features['frequency_components']['frequencies'],
                    freq_features['frequency_components']['magnitudes'],
                    low_freq, high_freq
                )
                for band, (low_freq, high_freq) in self.freq_bands.items()
            }
        }
    
    def _calculate_band_power(self, frequencies: np.ndarray, magnitudes: np.ndarray,
                            low_freq: float, high_freq: float) -> float:
        """Calculate power in a specific frequency band."""
        mask = (frequencies >= low_freq) & (frequencies <= high_freq)
        return np.sum(magnitudes[mask]**2)
    
    def analyze_transients(self, data: np.ndarray, window_size: int = 100) -> List[Dict]:
        """
        Detect and analyze transient events in the signal.
        
        Args:
            data: Input signal
            window_size: Size of the sliding window for detection
            
        Returns:
            List of detected transients with their characteristics
        """
        transients = []
        
        # Calculate rolling statistics
        rolling_mean = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        rolling_std = np.array([np.std(data[i:i+window_size]) 
                              for i in range(len(data)-window_size+1)])
        
        # Detect sudden changes
        threshold = np.mean(rolling_std) + 3 * np.std(rolling_std)
        potential_transients = np.where(rolling_std > threshold)[0]
        
        # Group consecutive points into events
        if len(potential_transients) > 0:
            event_starts = [potential_transients[0]]
            event_ends = []
            
            for i in range(1, len(potential_transients)):
                if potential_transients[i] - potential_transients[i-1] > window_size:
                    event_ends.append(potential_transients[i-1])
                    event_starts.append(potential_transients[i])
            event_ends.append(potential_transients[-1])
            
            # Analyze each transient event
            for start, end in zip(event_starts, event_ends):
                event_data = data[start:end+window_size]
                
                transients.append({
                    'start_index': start,
                    'end_index': end + window_size,
                    'duration': (end - start + window_size) / self.sampling_rate,
                    'magnitude': np.max(np.abs(event_data)),
                    'rise_time': self._calculate_rise_time(event_data)
                })
        
        return transients
    
    def _calculate_rise_time(self, event_data: np.ndarray) -> float:
        """Calculate rise time of a transient event."""
        peak_idx = np.argmax(np.abs(event_data))
        pre_event = event_data[:peak_idx]
        
        if len(pre_event) < 2:
            return 0
        
        # Find time from 10% to 90% of peak
        peak_value = np.abs(event_data[peak_idx])
        thresh_10 = 0.1 * peak_value
        thresh_90 = 0.9 * peak_value
        
        t_10 = np.where(np.abs(pre_event) >= thresh_10)[0]
        t_90 = np.where(np.abs(pre_event) >= thresh_90)[0]
        
        if len(t_10) > 0 and len(t_90) > 0:
            return (t_90[0] - t_10[0]) / self.sampling_rate
        return 0

def generate_test_data(duration_seconds: float = 1, 
                      base_frequency: float = 50,
                      sampling_rate: int = 1000,
                      noise_level: float = 0.1,
                      harmonics: List[Tuple[int, float]] = [(2, 0.1), (3, 0.05)],
                      add_transient: bool = False) -> pd.DataFrame:
    """
    Generate test power signal data with configurable parameters.
    
    Args:
        duration_seconds: Duration of the signal
        base_frequency: Fundamental frequency (Hz)
        sampling_rate: Sampling rate (Hz)
        noise_level: Amount of noise to add
        harmonics: List of (harmonic_number, amplitude) tuples
        add_transient: Whether to add a transient event
        
    Returns:
        DataFrame with timestamp and voltage columns
    """
    t = np.linspace(0, duration_seconds, int(duration_seconds * sampling_rate))
    
    # Generate fundamental component
    signal_data = np.sin(2 * np.pi * base_frequency * t)
    
    # Add harmonics
    for harmonic_num, amplitude in harmonics:
        signal_data += amplitude * np.sin(2 * np.pi * base_frequency * harmonic_num * t)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(t))
    signal_data += noise
    
    # Add transient if requested
    if add_transient:
        # Add a spike at 70% of the duration
        transient_idx = int(0.7 * len(t))
        signal_data[transient_idx:transient_idx+10] += 2.0
    
    return pd.DataFrame({
        'timestamp': t,
        'voltage': signal_data
    })
# src/data_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PowerDataSimulator:
    def __init__(self, base_load=1000):
        """
        Initialize the power data simulator with configurable parameters.
        
        Args:
            base_load (float): Base power load in watts
        """
        self.base_load = base_load
        self.time_index = 0
        
        # Configuration parameters
        self.daily_pattern_amplitude = 200  # Daily variation amplitude
        self.noise_std = 50  # Standard deviation of random noise
        self.anomaly_probability = 0.05  # 5% chance of anomaly
        self.anomaly_magnitude = 400  # Maximum magnitude of anomalies
        
        # Peak hours configuration (higher consumption periods)
        self.peak_hours = {
            'morning': (7, 9),    # Morning peak (7 AM - 9 AM)
            'evening': (18, 22)   # Evening peak (6 PM - 10 PM)
        }
        
        # Weekly pattern (weekend vs weekday)
        self.weekend_load_factor = 0.8  # 80% of weekday load on weekends
        
    def _is_peak_hour(self, hour):
        """Check if current hour is during peak consumption period."""
        return (self.peak_hours['morning'][0] <= hour < self.peak_hours['morning'][1] or
                self.peak_hours['evening'][0] <= hour < self.peak_hours['evening'][1])
    
    def _get_weekly_factor(self, timestamp):
        """Get load factor based on day of week."""
        return self.weekend_load_factor if timestamp.weekday() >= 5 else 1.0
    
    def _generate_anomaly(self):
        """Generate an anomaly if probability threshold is met."""
        if np.random.random() < self.anomaly_probability:
            return np.random.uniform(-self.anomaly_magnitude, self.anomaly_magnitude)
        return 0
    
    def generate_realtime_data(self, noise_level=1.0):
        """
        Generate a single data point of power consumption.
        
        Args:
            noise_level (float): Factor to adjust noise level (0.0 to 1.0)
            
        Returns:
            tuple: (timestamp, load, is_anomaly)
        """
        timestamp = datetime.now() + timedelta(seconds=self.time_index)
        hour = timestamp.hour + timestamp.minute / 60
        
        # Base daily pattern (sinusoidal with 24-hour period)
        daily_pattern = self.daily_pattern_amplitude * np.sin(2 * np.pi * (hour - 6) / 24)
        
        # Add peak hour boost
        peak_boost = 300 if self._is_peak_hour(hour) else 0
        
        # Weekly pattern adjustment
        weekly_factor = self._get_weekly_factor(timestamp)
        
        # Generate base load with patterns
        base_value = (self.base_load + daily_pattern + peak_boost) * weekly_factor
        
        # Add noise
        noise = np.random.normal(0, self.noise_std * noise_level)
        
        # Generate potential anomaly
        anomaly = self._generate_anomaly()
        is_anomaly = anomaly != 0
        
        # Combine all components
        load = base_value + noise + anomaly
        
        # Ensure load is never negative
        load = max(0, load)
        
        self.time_index += 1
        return timestamp, load, is_anomaly
    
    def generate_batch_data(self, hours=24, noise_level=1.0):
        """
        Generate a batch of power consumption data.
        
        Args:
            hours (int): Number of hours of data to generate
            noise_level (float): Factor to adjust noise level (0.0 to 1.0)
            
        Returns:
            pandas.DataFrame: DataFrame with timestamp, load, and is_anomaly columns
        """
        data = []
        start_time = datetime.now()
        
        for i in range(hours * 3600):  # Generate data points for each second
            timestamp = start_time + timedelta(seconds=i)
            self.time_index = i
            _, load, is_anomaly = self.generate_realtime_data(noise_level)
            data.append({
                'timestamp': timestamp,
                'load': load,
                'is_anomaly': is_anomaly
            })
        
        return pd.DataFrame(data)
    
    def reset(self):
        """Reset the simulator's time index."""
        self.time_index = 0
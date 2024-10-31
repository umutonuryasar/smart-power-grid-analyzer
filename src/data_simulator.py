# src/data_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PowerDataSimulator:
    def __init__(self):
        self.base_load = 1000  # Base load in watts
        self.time_index = 0
        
    def generate_realtime_data(self):
        timestamp = datetime.now() + timedelta(seconds=self.time_index)
        noise = np.random.normal(0, 50)
        daily_pattern = 200 * np.sin(2 * np.pi * self.time_index / (24 * 3600))
        load = self.base_load + daily_pattern + noise
        self.time_index += 1
        return timestamp, load
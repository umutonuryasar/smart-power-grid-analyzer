# src/ml_models.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class PowerConsumptionPredictor:
   def __init__(self, sequence_length=24):
       self.sequence_length = sequence_length
       self.scaler = StandardScaler()
       self.model = self._build_lstm_model()
       
   def _build_lstm_model(self):
       model = tf.keras.Sequential([
           tf.keras.layers.LSTM(64, input_shape=(self.sequence_length, 1), return_sequences=True),
           tf.keras.layers.LSTM(32),
           tf.keras.layers.Dense(16, activation='relu'),
           tf.keras.layers.Dense(1)
       ])
       model.compile(optimizer='adam', loss='mse', metrics=['mae'])
       return model
   
   def prepare_sequences(self, data):
       scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
       X, y = [], []
       for i in range(len(scaled_data) - self.sequence_length):
           X.append(scaled_data[i:i + self.sequence_length])
           y.append(scaled_data[i + self.sequence_length])
       return np.array(X), np.array(y)
   
   def train(self, data, epochs=50, validation_split=0.2):
       X, y = self.prepare_sequences(data)
       return self.model.fit(X, y, epochs=epochs, validation_split=validation_split)
   
   def predict(self, data):
       scaled_data = self.scaler.transform(data.reshape(-1, 1))
       X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
       prediction = self.model.predict(X)
       return self.scaler.inverse_transform(prediction)

class AnomalyDetector:
   def __init__(self, threshold=3):
       self.threshold = threshold
       self.model = self._build_autoencoder()
       self.scaler = StandardScaler()
       
   def _build_autoencoder(self):
       model = tf.keras.Sequential([
           tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
           tf.keras.layers.Dense(16, activation='relu'),
           tf.keras.layers.Dense(32, activation='relu'),
           tf.keras.layers.Dense(1)
       ])
       model.compile(optimizer='adam', loss='mse')
       return model
       
   def train(self, data, epochs=50):
       scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
       self.model.fit(scaled_data, scaled_data, epochs=epochs)
       
   def detect_anomalies(self, data):
       scaled_data = self.scaler.transform(data.reshape(-1, 1))
       reconstructed = self.model.predict(scaled_data)
       mse = np.mean(np.power(scaled_data - reconstructed, 2), axis=1)
       return mse > self.threshold * np.std(mse)
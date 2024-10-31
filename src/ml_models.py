# src/ml_models.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
from datetime import datetime
import pandas as pd

class PowerConsumptionPredictor:
    def __init__(self, sequence_length=24, forecast_horizon=6):
        """
        Initialize the power consumption predictor.
        
        Args:
            sequence_length (int): Number of time steps to use for prediction
            forecast_horizon (int): Number of time steps to predict into the future
        """
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        self.model = self._build_lstm_model()
        self.is_trained = False
        self.training_history = None
        
    def _build_lstm_model(self):
        """Build and compile the LSTM model."""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.LSTM(128, input_shape=(self.sequence_length, 1), 
                               return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            
            # Hidden layers
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu'),
            
            # Output layer
            tf.keras.layers.Dense(self.forecast_horizon)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def prepare_sequences(self, data):
        """Prepare data sequences for training/prediction."""
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        X, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length - self.forecast_horizon + 1):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[(i + self.sequence_length):(i + self.sequence_length + self.forecast_horizon)])
        
        return np.array(X), np.array(y)
    
    def train(self, data, epochs=50, validation_split=0.2, batch_size=32):
        """
        Train the model on historical power consumption data.
        
        Args:
            data: numpy array of power consumption values
            epochs: number of training epochs
            validation_split: fraction of data to use for validation
            batch_size: training batch size
        
        Returns:
            Training history
        """
        X, y = self.prepare_sequences(data)
        
        if len(X) < 2:  # Check if we have enough data
            raise ValueError("Not enough data for training. Need at least "
                           f"{self.sequence_length + self.forecast_horizon + 2} points.")
        
        # Add early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        self.training_history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.is_trained = True
        return self.training_history
    
    def predict(self, data):
        """Generate predictions for the next forecast_horizon time steps."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions.")
        
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        X = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        scaled_predictions = self.model.predict(X)
        
        return self.scaler.inverse_transform(scaled_predictions)
    
    def get_metrics(self):
        """Return training metrics if model is trained."""
        if not self.is_trained or not self.training_history:
            return None
            
        return {
            'final_loss': self.training_history.history['loss'][-1],
            'final_mae': self.training_history.history['mae'][-1],
            'val_loss': self.training_history.history['val_loss'][-1],
            'val_mae': self.training_history.history['val_mae'][-1]
        }

class AnomalyDetector:
    def __init__(self, threshold=3, contamination=0.1):
        """
        Initialize the anomaly detector with multiple detection methods.
        
        Args:
            threshold: Standard deviations for statistical detection
            contamination: Expected proportion of anomalies (for Isolation Forest)
        """
        self.threshold = threshold
        self.statistical_scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        self.is_trained = False
        self.training_timestamp = None
        
        # Store historical statistics
        self.historical_mean = None
        self.historical_std = None
        
    def train(self, data):
        """
        Train the anomaly detection models.
        
        Args:
            data: numpy array of power consumption values
        """
        # Reshape data if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        # Train statistical model
        self.statistical_scaler.fit(data)
        self.historical_mean = np.mean(data)
        self.historical_std = np.std(data)
        
        # Train Isolation Forest
        self.isolation_forest.fit(data)
        
        self.is_trained = True
        self.training_timestamp = datetime.now()
        
    def detect_anomalies(self, data, method='ensemble'):
        """
        Detect anomalies using specified method.
        
        Args:
            data: numpy array of values to check for anomalies
            method: 'statistical', 'isolation_forest', or 'ensemble'
            
        Returns:
            Array of boolean values indicating anomalies
        """
        if not self.is_trained:
            raise RuntimeError("Models must be trained before detecting anomalies.")
            
        # Reshape data if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        if method == 'statistical':
            scaled_data = self.statistical_scaler.transform(data)
            return np.abs(scaled_data) > self.threshold
            
        elif method == 'isolation_forest':
            # Isolation Forest returns 1 for inliers and -1 for outliers
            return self.isolation_forest.predict(data) == -1
            
        elif method == 'ensemble':
            # Combine both methods (logical OR)
            statistical_anomalies = np.abs(self.statistical_scaler.transform(data)) > self.threshold
            isolation_anomalies = self.isolation_forest.predict(data) == -1
            return np.logical_or(statistical_anomalies, isolation_anomalies)
            
        else:
            raise ValueError("Invalid method. Choose 'statistical', 'isolation_forest', or 'ensemble'.")
            
    def get_anomaly_score(self, data):
        """
        Get anomaly scores for the data points.
        
        Args:
            data: numpy array of values
            
        Returns:
            Dictionary with statistical and isolation forest scores
        """
        if not self.is_trained:
            raise RuntimeError("Models must be trained before calculating scores.")
            
        # Reshape data if needed
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        statistical_scores = np.abs(self.statistical_scaler.transform(data))
        isolation_scores = -self.isolation_forest.score_samples(data)  # Negative of the scores
        
        return {
            'statistical_scores': statistical_scores,
            'isolation_scores': isolation_scores
        }
        
    def save_models(self, path):
        """Save trained models to disk."""
        if not self.is_trained:
            raise RuntimeError("Models must be trained before saving.")
            
        model_data = {
            'statistical_scaler': self.statistical_scaler,
            'isolation_forest': self.isolation_forest,
            'threshold': self.threshold,
            'historical_mean': self.historical_mean,
            'historical_std': self.historical_std,
            'training_timestamp': self.training_timestamp
        }
        
        joblib.dump(model_data, path)
        
    def load_models(self, path):
        """Load trained models from disk."""
        model_data = joblib.load(path)
        
        self.statistical_scaler = model_data['statistical_scaler']
        self.isolation_forest = model_data['isolation_forest']
        self.threshold = model_data['threshold']
        self.historical_mean = model_data['historical_mean']
        self.historical_std = model_data['historical_std']
        self.training_timestamp = model_data['training_timestamp']
        self.is_trained = True
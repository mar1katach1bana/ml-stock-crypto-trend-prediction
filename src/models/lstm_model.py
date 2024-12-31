import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM model for time series prediction."""
    
    def __init__(self, config_path: str):
        """Initialize LSTM model with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.history = None
        self.build_model()
    
    def build_model(self):
        """Build LSTM model architecture."""
        try:
            model_config = self.config['lstm_model']
            
            # Create sequential model
            self.model = Sequential()
            
            # Add layers according to configuration
            input_shape = tuple(model_config['architecture']['input_shape'])
            
            for i, layer in enumerate(model_config['architecture']['layers']):
                if layer['type'] == 'LSTM':
                    if i == 0:  # First layer
                        self.model.add(LSTM(
                            units=layer['units'],
                            return_sequences=layer['return_sequences'],
                            input_shape=input_shape
                        ))
                    else:
                        self.model.add(LSTM(
                            units=layer['units'],
                            return_sequences=layer['return_sequences']
                        ))
                    self.model.add(Dropout(layer['dropout']))
                
                elif layer['type'] == 'Dense':
                    self.model.add(Dense(
                        units=layer['units'],
                        activation=layer['activation']
                    ))
            
            # Compile model
            optimizer = Adam(learning_rate=model_config['training']['optimizer']['learning_rate'])
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            logger.info("LSTM model built successfully")
            self.model.summary()
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    def create_callbacks(self, model_dir: str) -> List[tf.keras.callbacks.Callback]:
        """Create training callbacks."""
        callbacks = []
        model_config = self.config['lstm_model']['training']
        
        # Create model directory if it doesn't exist
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Model checkpoint
        if "ModelCheckpoint" in model_config['callbacks']:
            checkpoint_path = model_path / "best_model.h5"
            callbacks.append(ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
            ))
        
        # Early stopping
        if model_config.get('early_stopping'):
            callbacks.append(EarlyStopping(
                monitor=model_config['early_stopping']['monitor'],
                patience=model_config['early_stopping']['patience'],
                restore_best_weights=model_config['early_stopping']['restore_best_weights'],
                verbose=1
            ))
        
        # Learning rate reduction
        if "ReduceLROnPlateau" in model_config['callbacks']:
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ))
        
        # TensorBoard
        if "TensorBoard" in model_config['callbacks']:
            log_dir = model_path / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks.append(TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1
            ))
        
        return callbacks
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             model_dir: str) -> Dict[str, List[float]]:
        """Train the LSTM model."""
        try:
            model_config = self.config['lstm_model']['training']
            
            # Create callbacks
            callbacks = self.create_callbacks(model_dir)
            
            # Train model
            self.history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=model_config['epochs'],
                batch_size=model_config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info("Model training completed successfully")
            return self.history.history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model has not been built or loaded")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance."""
        try:
            if self.model is None:
                raise ValueError("Model has not been built or loaded")
            
            # Get evaluation metrics
            metrics = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Create metrics dictionary
            metrics_dict = {
                'loss': metrics[0],
                'mae': metrics[1]
            }
            
            logger.info("Model evaluation completed successfully")
            return metrics_dict
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save(self, model_path: str):
        """Save the trained model."""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            # Create directory if it doesn't exist
            save_path = Path(model_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save(str(save_path))
            logger.info(f"Model saved successfully to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, model_path: str):
        """Load a trained model."""
        try:
            # Check if model file exists
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            self.model = load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            if self.history is None:
                raise ValueError("No training history available")
            
            # Create figure
            plt.figure(figsize=(12, 4))
            
            # Plot training & validation loss
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot training & validation MAE
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['mae'], label='Training MAE')
            plt.plot(self.history.history['val_mae'], label='Validation MAE')
            plt.title('Model MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Training history plot saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    model = LSTMModel('config/model_config.yaml')
    
    # Generate dummy data for testing
    X_train = np.random.random((100, 60, 32))  # 100 samples, 60 timesteps, 32 features
    y_train = np.random.random((100, 1))
    X_val = np.random.random((20, 60, 32))
    y_val = np.random.random((20, 1))
    
    # Train model
    history = model.train(X_train, y_train, X_val, y_val, 'models/lstm')
    
    # Make predictions
    predictions = model.predict(X_val)
    
    # Evaluate model
    metrics = model.evaluate(X_val, y_val)
    print(f"Evaluation metrics: {metrics}")
    
    # Plot and save training history
    model.plot_training_history('models/lstm/training_history.png')
    
    # Save model
    model.save('models/lstm/final_model.h5')

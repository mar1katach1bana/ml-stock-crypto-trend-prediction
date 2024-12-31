import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import logging
from pathlib import Path
import joblib

from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

logger = logging.getLogger(__name__)

class EnsembleModel:
    """Ensemble model combining LSTM, Random Forest, and XGBoost predictions."""
    
    def __init__(self, config_path: str, task: str = 'regression'):
        """
        Initialize Ensemble model.
        
        Args:
            config_path (str): Path to configuration file
            task (str): 'regression' for price prediction or 'classification' for trend prediction
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.task = task
        self.models = {}
        self.weights = None
        self.threshold = None
        self.voting = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize individual models."""
        try:
            ensemble_config = self.config['ensemble']
            
            # Initialize models based on configuration
            if 'lstm' in ensemble_config['models']:
                self.models['lstm'] = LSTMModel(self.config_path)
            
            if 'random_forest' in ensemble_config['models']:
                self.models['random_forest'] = RandomForestModel(self.config_path, task=self.task)
            
            if 'xgboost' in ensemble_config['models']:
                self.models['xgboost'] = XGBoostModel(self.config_path, task=self.task)
            
            # Set ensemble parameters
            self.weights = dict(zip(ensemble_config['models'], ensemble_config['weights']))
            self.voting = ensemble_config['voting']
            self.threshold = ensemble_config['threshold']
            
            logger.info("Ensemble model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ensemble model: {e}")
            raise
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None):
        """Train all models in the ensemble."""
        try:
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                
                if name == 'lstm':
                    model.train(X_train, y_train, X_val, y_val, f'models/{name}')
                else:
                    model.train(X_train, y_train, X_val, y_val)
            
            logger.info("Ensemble training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble of models."""
        try:
            predictions = {}
            
            # Get predictions from each model
            for name, model in self.models.items():
                predictions[name] = model.predict(X)
            
            # Combine predictions based on task and voting method
            if self.task == 'regression':
                # Weighted average for regression
                ensemble_pred = np.zeros_like(predictions[list(predictions.keys())[0]])
                for name, pred in predictions.items():
                    ensemble_pred += pred * self.weights[name]
                
            else:  # classification
                if self.voting == 'soft':
                    # Weighted probability average for soft voting
                    proba_predictions = {}
                    for name, model in self.models.items():
                        proba_predictions[name] = model.predict_proba(X)
                    
                    ensemble_proba = np.zeros_like(proba_predictions[list(proba_predictions.keys())[0]])
                    for name, proba in proba_predictions.items():
                        ensemble_proba += proba * self.weights[name]
                    
                    ensemble_pred = (ensemble_proba[:, 1] > self.threshold).astype(int)
                    
                else:  # hard voting
                    # Weighted vote counting for hard voting
                    weighted_votes = np.zeros_like(predictions[list(predictions.keys())[0]])
                    for name, pred in predictions.items():
                        weighted_votes += pred * self.weights[name]
                    
                    ensemble_pred = (weighted_votes > self.threshold).astype(int)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for classification."""
        try:
            if self.task != 'classification':
                raise ValueError("Probability predictions only available for classification task")
            
            # Get probability predictions from each model
            proba_predictions = {}
            for name, model in self.models.items():
                proba_predictions[name] = model.predict_proba(X)
            
            # Weighted average of probabilities
            ensemble_proba = np.zeros_like(proba_predictions[list(proba_predictions.keys())[0]])
            for name, proba in proba_predictions.items():
                ensemble_proba += proba * self.weights[name]
            
            return ensemble_proba
            
        except Exception as e:
            logger.error(f"Error getting ensemble probability predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Evaluate all models and the ensemble."""
        try:
            metrics = {}
            
            # Evaluate individual models
            for name, model in self.models.items():
                metrics[name] = model.evaluate(X_test, y_test)
            
            # Evaluate ensemble
            ensemble_pred = self.predict(X_test)
            metrics['ensemble'] = {}
            
            if self.task == 'regression':
                metrics['ensemble']['mse'] = mean_squared_error(y_test, ensemble_pred)
                metrics['ensemble']['rmse'] = np.sqrt(metrics['ensemble']['mse'])
                metrics['ensemble']['mae'] = mean_absolute_error(y_test, ensemble_pred)
                metrics['ensemble']['r2'] = r2_score(y_test, ensemble_pred)
            else:  # classification
                metrics['ensemble']['accuracy'] = accuracy_score(y_test, ensemble_pred)
                metrics['ensemble']['precision'] = precision_score(y_test, ensemble_pred, average='weighted')
                metrics['ensemble']['recall'] = recall_score(y_test, ensemble_pred, average='weighted')
                metrics['ensemble']['f1'] = f1_score(y_test, ensemble_pred, average='weighted')
                
                # Calculate ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    ensemble_proba = self.predict_proba(X_test)[:, 1]
                    metrics['ensemble']['roc_auc'] = roc_auc_score(y_test, ensemble_proba)
            
            logger.info("Ensemble evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating ensemble model: {e}")
            raise
    
    def save(self, model_dir: str):
        """Save all models in the ensemble."""
        try:
            model_path = Path(model_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save individual models
            for name, model in self.models.items():
                model_save_path = model_path / f"{name}_model"
                model.save(str(model_save_path))
            
            # Save ensemble configuration
            config = {
                'weights': self.weights,
                'voting': self.voting,
                'threshold': self.threshold,
                'task': self.task
            }
            
            config_path = model_path / 'ensemble_config.joblib'
            joblib.dump(config, config_path)
            
            logger.info(f"Ensemble model saved successfully to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving ensemble model: {e}")
            raise
    
    def load(self, model_dir: str):
        """Load all models in the ensemble."""
        try:
            model_path = Path(model_dir)
            
            # Load individual models
            for name, model in self.models.items():
                model_load_path = model_path / f"{name}_model"
                model.load(str(model_load_path))
            
            # Load ensemble configuration
            config_path = model_path / 'ensemble_config.joblib'
            config = joblib.load(config_path)
            
            self.weights = config['weights']
            self.voting = config['voting']
            self.threshold = config['threshold']
            self.task = config['task']
            
            logger.info(f"Ensemble model loaded successfully from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {e}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage for regression
    ensemble_reg = EnsembleModel('config/model_config.yaml', task='regression')
    
    # Generate dummy regression data
    X_train = np.random.random((100, 20))
    y_train = np.random.random(100)
    X_val = np.random.random((20, 20))
    y_val = np.random.random(20)
    
    # Train and evaluate regression ensemble
    ensemble_reg.train(X_train, y_train, X_val, y_val)
    reg_metrics = ensemble_reg.evaluate(X_val, y_val)
    print(f"Regression metrics: {reg_metrics}")
    
    # Example usage for classification
    ensemble_clf = EnsembleModel('config/model_config.yaml', task='classification')
    
    # Generate dummy classification data
    y_train_clf = np.random.randint(0, 2, 100)
    y_val_clf = np.random.randint(0, 2, 20)
    
    # Train and evaluate classification ensemble
    ensemble_clf.train(X_train, y_train_clf, X_val, y_val_clf)
    clf_metrics = ensemble_clf.evaluate(X_val, y_val_clf)
    print(f"Classification metrics: {clf_metrics}")
    
    # Save models
    ensemble_reg.save('models/ensemble/regression')
    ensemble_clf.save('models/ensemble/classification')

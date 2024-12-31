import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import logging
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple

logger = logging.getLogger(__name__)

class XGBoostModel:
    """XGBoost model for market prediction."""
    
    def __init__(self, config_path: str, task: str = 'regression'):
        """
        Initialize XGBoost model.
        
        Args:
            config_path (str): Path to configuration file
            task (str): 'regression' for price prediction or 'classification' for trend prediction
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.task = task
        self.model = None
        self.feature_importance = None
        self.build_model()
    
    def build_model(self):
        """Build XGBoost model."""
        try:
            model_config = self.config['xgboost']
            
            if self.task == 'regression':
                self.model = xgb.XGBRegressor(
                    objective=model_config['objective'],
                    max_depth=model_config['max_depth'],
                    learning_rate=model_config['learning_rate'],
                    n_estimators=model_config['n_estimators'],
                    min_child_weight=model_config['min_child_weight'],
                    subsample=model_config['subsample'],
                    colsample_bytree=model_config['colsample_bytree'],
                    gamma=model_config['gamma'],
                    reg_alpha=model_config['reg_alpha'],
                    reg_lambda=model_config['reg_lambda'],
                    random_state=model_config['random_state'],
                    n_jobs=model_config['n_jobs']
                )
            else:  # classification
                self.model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    max_depth=model_config['max_depth'],
                    learning_rate=model_config['learning_rate'],
                    n_estimators=model_config['n_estimators'],
                    min_child_weight=model_config['min_child_weight'],
                    subsample=model_config['subsample'],
                    colsample_bytree=model_config['colsample_bytree'],
                    gamma=model_config['gamma'],
                    reg_alpha=model_config['reg_alpha'],
                    reg_lambda=model_config['reg_lambda'],
                    random_state=model_config['random_state'],
                    n_jobs=model_config['n_jobs']
                )
            
            logger.info(f"XGBoost {self.task} model built successfully")
            
        except Exception as e:
            logger.error(f"Error building XGBoost model: {e}")
            raise
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None):
        """Train the XGBoost model."""
        try:
            # Prepare evaluation set if provided
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            # Train model
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                eval_metric=['rmse'] if self.task == 'regression' else ['logloss', 'auc'],
                early_stopping_rounds=10,
                verbose=True
            )
            
            # Get feature importance
            importance_type = 'weight'  # Can be 'weight', 'gain', or 'cover'
            self.feature_importance = pd.DataFrame(
                self.model.get_booster().get_score(importance_type=importance_type).items(),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained")
            
            predictions = self.model.predict(X)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions for classification."""
        try:
            if self.task != 'classification':
                raise ValueError("Probability predictions only available for classification task")
            
            if self.model is None:
                raise ValueError("Model has not been trained")
            
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            logger.error(f"Error getting probability predictions: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the model performance."""
        try:
            if self.model is None:
                raise ValueError("Model has not been trained")
            
            predictions = self.predict(X_test)
            metrics = {}
            
            if self.task == 'regression':
                metrics['mse'] = mean_squared_error(y_test, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y_test, predictions)
                metrics['r2'] = r2_score(y_test, predictions)
            else:  # classification
                metrics['accuracy'] = accuracy_score(y_test, predictions)
                metrics['precision'] = precision_score(y_test, predictions, average='weighted')
                metrics['recall'] = recall_score(y_test, predictions, average='weighted')
                metrics['f1'] = f1_score(y_test, predictions, average='weighted')
                
                # Calculate ROC AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    prob_predictions = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, prob_predictions)
            
            logger.info("Model evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Get feature importance ranking."""
        try:
            if self.feature_importance is None:
                raise ValueError("Model has not been trained")
            
            importance_df = self.feature_importance.copy()
            
            if feature_names is not None:
                importance_df['feature'] = feature_names
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise
    
    def plot_feature_importance(self, 
                              feature_names: Optional[List[str]] = None,
                              top_n: int = 10,
                              save_path: Optional[str] = None):
        """Plot feature importance."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            importance_df = self.get_feature_importance(feature_names)
            
            # Plot top N features
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x='importance',
                y='feature',
                data=importance_df.head(top_n)
            )
            plt.title(f'Top {top_n} Most Important Features')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            else:
                plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")
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
            self.model.save_model(str(save_path))
            
            # Save feature importance if available
            if self.feature_importance is not None:
                importance_path = save_path.parent / 'feature_importance.csv'
                self.feature_importance.to_csv(importance_path, index=False)
            
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
            self.model = xgb.XGBRegressor() if self.task == 'regression' else xgb.XGBClassifier()
            self.model.load_model(model_path)
            
            # Load feature importance if available
            importance_path = Path(model_path).parent / 'feature_importance.csv'
            if importance_path.exists():
                self.feature_importance = pd.read_csv(importance_path)
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage for regression
    xgb_reg = XGBoostModel('config/model_config.yaml', task='regression')
    
    # Generate dummy regression data
    X_train = np.random.random((100, 20))
    y_train = np.random.random(100)
    X_val = np.random.random((20, 20))
    y_val = np.random.random(20)
    
    # Train and evaluate regression model
    xgb_reg.train(X_train, y_train, X_val, y_val)
    reg_metrics = xgb_reg.evaluate(X_val, y_val)
    print(f"Regression metrics: {reg_metrics}")
    
    # Plot and save feature importance
    feature_names = [f"Feature_{i}" for i in range(20)]
    xgb_reg.plot_feature_importance(
        feature_names=feature_names,
        save_path='models/xgboost/feature_importance_reg.png'
    )
    
    # Example usage for classification
    xgb_clf = XGBoostModel('config/model_config.yaml', task='classification')
    
    # Generate dummy classification data
    y_train_clf = np.random.randint(0, 2, 100)
    y_val_clf = np.random.randint(0, 2, 20)
    
    # Train and evaluate classification model
    xgb_clf.train(X_train, y_train_clf, X_val, y_val_clf)
    clf_metrics = xgb_clf.evaluate(X_val, y_val_clf)
    print(f"Classification metrics: {clf_metrics}")
    
    # Save models
    xgb_reg.save('models/xgboost/regression_model.json')
    xgb_clf.save('models/xgboost/classification_model.json')

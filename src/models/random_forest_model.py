import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import yaml
import logging
from pathlib import Path
from typing import Dict, Union, Optional, List, Tuple

logger = logging.getLogger(__name__)

class RandomForestModel:
    """Random Forest model for market prediction."""
    
    def __init__(self, config_path: str, task: str = 'regression'):
        """
        Initialize Random Forest model.
        
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
        """Build Random Forest model."""
        try:
            model_config = self.config['random_forest']
            
            if self.task == 'regression':
                self.model = RandomForestRegressor(
                    n_estimators=model_config['n_estimators'],
                    max_depth=model_config['max_depth'],
                    min_samples_split=model_config['min_samples_split'],
                    min_samples_leaf=model_config['min_samples_leaf'],
                    max_features=model_config['max_features'],
                    bootstrap=model_config['bootstrap'],
                    n_jobs=model_config['n_jobs'],
                    random_state=model_config['random_state']
                )
            else:  # classification
                self.model = RandomForestClassifier(
                    n_estimators=model_config['n_estimators'],
                    max_depth=model_config['max_depth'],
                    min_samples_split=model_config['min_samples_split'],
                    min_samples_leaf=model_config['min_samples_leaf'],
                    max_features=model_config['max_features'],
                    bootstrap=model_config['bootstrap'],
                    n_jobs=model_config['n_jobs'],
                    class_weight=model_config['class_weight'],
                    random_state=model_config['random_state']
                )
            
            logger.info(f"Random Forest {self.task} model built successfully")
            
        except Exception as e:
            logger.error(f"Error building Random Forest model: {e}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the Random Forest model."""
        try:
            self.model.fit(X_train, y_train)
            
            # Calculate feature importance
            self.feature_importance = pd.DataFrame(
                self.model.feature_importances_,
                index=range(X_train.shape[1]),
                columns=['importance']
            ).sort_values('importance', ascending=False)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {e}")
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
                if len(feature_names) != len(importance_df):
                    raise ValueError("Length of feature_names does not match number of features")
                importance_df.index = feature_names
            
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
                y=importance_df.index[:top_n],
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
            joblib.dump(self.model, str(save_path))
            
            # Save feature importance if available
            if self.feature_importance is not None:
                importance_path = save_path.parent / 'feature_importance.csv'
                self.feature_importance.to_csv(importance_path)
            
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
            self.model = joblib.load(model_path)
            
            # Load feature importance if available
            importance_path = Path(model_path).parent / 'feature_importance.csv'
            if importance_path.exists():
                self.feature_importance = pd.read_csv(importance_path, index_col=0)
            
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
    rf_reg = RandomForestModel('config/model_config.yaml', task='regression')
    
    # Generate dummy regression data
    X_train = np.random.random((100, 20))
    y_train = np.random.random(100)
    X_test = np.random.random((20, 20))
    y_test = np.random.random(20)
    
    # Train and evaluate regression model
    rf_reg.train(X_train, y_train)
    reg_metrics = rf_reg.evaluate(X_test, y_test)
    print(f"Regression metrics: {reg_metrics}")
    
    # Plot and save feature importance
    feature_names = [f"Feature_{i}" for i in range(20)]
    rf_reg.plot_feature_importance(
        feature_names=feature_names,
        save_path='models/random_forest/feature_importance_reg.png'
    )
    
    # Example usage for classification
    rf_clf = RandomForestModel('config/model_config.yaml', task='classification')
    
    # Generate dummy classification data
    y_train_clf = np.random.randint(0, 2, 100)
    y_test_clf = np.random.randint(0, 2, 20)
    
    # Train and evaluate classification model
    rf_clf.train(X_train, y_train_clf)
    clf_metrics = rf_clf.evaluate(X_test, y_test_clf)
    print(f"Classification metrics: {clf_metrics}")
    
    # Save models
    rf_reg.save('models/random_forest/regression_model.joblib')
    rf_clf.save('models/random_forest/classification_model.joblib')

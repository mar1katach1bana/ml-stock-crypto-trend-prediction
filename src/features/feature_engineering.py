import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering class for creating technical indicators and derived features."""
    
    def __init__(self, config_path: str):
        """Initialize FeatureEngineer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scalers = {}
        self._initialize_scalers()
        
    def _initialize_scalers(self):
        """Initialize scalers based on configuration."""
        scaling_method = self.config['data_processing']['scaling']['method']
        feature_range = tuple(self.config['data_processing']['scaling']['feature_range'])
        
        if scaling_method == 'min_max':
            self.scalers['price'] = MinMaxScaler(feature_range=feature_range)
            self.scalers['volume'] = MinMaxScaler(feature_range=feature_range)
            self.scalers['technical'] = MinMaxScaler(feature_range=feature_range)
        elif scaling_method == 'standard':
            self.scalers['price'] = StandardScaler()
            self.scalers['volume'] = StandardScaler()
            self.scalers['technical'] = StandardScaler()
        elif scaling_method == 'robust':
            self.scalers['price'] = RobustScaler()
            self.scalers['volume'] = RobustScaler()
            self.scalers['technical'] = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the given dataframe."""
        result_df = df.copy()
        
        # Get configuration for technical indicators
        indicators_config = self.config['data_processing']['features']['technical_indicators']
        
        for indicator in indicators_config:
            try:
                if indicator['name'] == 'RSI':
                    result_df['RSI'] = talib.RSI(
                        result_df['close'],
                        timeperiod=indicator['parameters']['window']
                    )
                
                elif indicator['name'] == 'MACD':
                    macd, signal, hist = talib.MACD(
                        result_df['close'],
                        fastperiod=indicator['parameters']['fast'],
                        slowperiod=indicator['parameters']['slow'],
                        signalperiod=indicator['parameters']['signal']
                    )
                    result_df['MACD'] = macd
                    result_df['MACD_Signal'] = signal
                    result_df['MACD_Hist'] = hist
                
                elif indicator['name'] == 'BB':
                    upper, middle, lower = talib.BBANDS(
                        result_df['close'],
                        timeperiod=indicator['parameters']['window'],
                        nbdevup=indicator['parameters']['std'],
                        nbdevdn=indicator['parameters']['std']
                    )
                    result_df['BB_Upper'] = upper
                    result_df['BB_Middle'] = middle
                    result_df['BB_Lower'] = lower
                
                elif indicator['name'] == 'EMA':
                    for window in indicator['parameters']['windows']:
                        result_df[f'EMA_{window}'] = talib.EMA(
                            result_df['close'],
                            timeperiod=window
                        )
                
            except Exception as e:
                logger.error(f"Error calculating {indicator['name']}: {e}")
                continue
        
        return result_df

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features like returns and volatility."""
        result_df = df.copy()
        
        # Calculate returns
        if 'returns' in self.config['data_processing']['features']['derived_features']:
            result_df['returns'] = result_df['close'].pct_change()
        
        # Calculate log returns
        if 'log_returns' in self.config['data_processing']['features']['derived_features']:
            result_df['log_returns'] = np.log(result_df['close']).diff()
        
        # Calculate volatility
        if 'volatility' in self.config['data_processing']['features']['derived_features']:
            result_df['volatility'] = result_df['returns'].rolling(window=20).std()
        
        return result_df

    def create_sequence_features(self, df: pd.DataFrame, target_col: str = 'close') -> tuple:
        """Create sequence features for time series models."""
        sequence_length = self.config['data_processing']['sequence_length']
        prediction_horizon = self.config['data_processing']['prediction_horizon']
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(df) - sequence_length - prediction_horizon + 1):
            sequence = df.iloc[i:(i + sequence_length)]
            target = df[target_col].iloc[i + sequence_length + prediction_horizon - 1]
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)

    def scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale features using initialized scalers."""
        result_df = df.copy()
        
        if is_training:
            # Scale price features
            price_cols = self.config['data_processing']['features']['price']
            if price_cols:
                result_df[price_cols] = self.scalers['price'].fit_transform(df[price_cols])
            
            # Scale volume
            if 'volume' in df.columns:
                result_df[['volume']] = self.scalers['volume'].fit_transform(df[['volume']])
            
            # Scale technical indicators
            technical_cols = [col for col in df.columns if col not in price_cols + ['volume']]
            if technical_cols:
                result_df[technical_cols] = self.scalers['technical'].fit_transform(df[technical_cols])
        else:
            # Transform using pre-fitted scalers
            price_cols = self.config['data_processing']['features']['price']
            if price_cols:
                result_df[price_cols] = self.scalers['price'].transform(df[price_cols])
            
            if 'volume' in df.columns:
                result_df[['volume']] = self.scalers['volume'].transform(df[['volume']])
            
            technical_cols = [col for col in df.columns if col not in price_cols + ['volume']]
            if technical_cols:
                result_df[technical_cols] = self.scalers['technical'].transform(df[technical_cols])
        
        return result_df

    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepare all features for model training or prediction."""
        # Calculate technical indicators
        df_with_technicals = self.calculate_technical_indicators(df)
        
        # Calculate derived features
        df_with_features = self.calculate_derived_features(df_with_technicals)
        
        # Remove rows with NaN values
        df_cleaned = df_with_features.dropna()
        
        # Scale features
        df_scaled = self.scale_features(df_cleaned, is_training)
        
        return df_scaled

    def create_target_labels(self, df: pd.DataFrame, horizon: Optional[int] = None) -> pd.DataFrame:
        """Create target labels for prediction."""
        if horizon is None:
            horizon = self.config['data_processing']['prediction_horizon']
        
        df_copy = df.copy()
        
        # Calculate future returns
        df_copy['future_returns'] = df_copy['close'].pct_change(periods=horizon).shift(-horizon)
        
        # Create binary labels (1 for positive returns, 0 for negative)
        df_copy['target'] = (df_copy['future_returns'] > 0).astype(int)
        
        return df_copy

    def save_scalers(self, output_dir: str):
        """Save fitted scalers for later use."""
        import joblib
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, scaler in self.scalers.items():
            scaler_path = output_path / f"{name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved {name} scaler to {scaler_path}")

    def load_scalers(self, input_dir: str):
        """Load pre-fitted scalers."""
        import joblib
        
        input_path = Path(input_dir)
        
        for name in self.scalers.keys():
            scaler_path = input_path / f"{name}_scaler.joblib"
            if scaler_path.exists():
                self.scalers[name] = joblib.load(scaler_path)
                logger.info(f"Loaded {name} scaler from {scaler_path}")
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    engineer = FeatureEngineer('config/model_config.yaml')
    
    # Load sample data
    data_path = Path('data/raw/AAPL.parquet')
    if data_path.exists():
        df = pd.read_parquet(data_path)
        
        # Prepare features
        df_features = engineer.prepare_features(df, is_training=True)
        
        # Create target labels
        df_with_targets = engineer.create_target_labels(df_features)
        
        # Create sequences for deep learning
        sequences, targets = engineer.create_sequence_features(df_with_targets)
        
        # Save scalers
        engineer.save_scalers('models/scalers')
        
        logger.info("Feature engineering completed successfully")

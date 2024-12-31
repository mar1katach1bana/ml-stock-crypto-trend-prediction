# Stock/Crypto Trend Prediction ML Model

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-black)]()

A sophisticated machine learning framework for predicting stock and cryptocurrency price trends using historical data and advanced feature engineering. This project combines traditional technical analysis with modern deep learning approaches to forecast market movements.

## 🚀 Features

- Automated data collection from multiple sources (Alpha Vantage, Binance, Yahoo Finance)
- Comprehensive data preprocessing and cleaning pipeline
- Advanced feature engineering including technical indicators and sentiment analysis
- Multiple ML model implementations (LSTM, Random Forest, XGBoost)
- Hyperparameter optimization and model evaluation
- Interactive visualizations and performance metrics
- Backtesting framework for strategy validation

## 📋 Prerequisites

```python
python>=3.8
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=0.24.0
tensorflow>=2.6.0  # or pytorch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
ta>=0.7.0  # Technical Analysis library
yfinance>=0.1.63
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-crypto-prediction.git
cd stock-crypto-prediction
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Data Collection

The project supports multiple data sources:

### Yahoo Finance
```python
from data.collectors.yahoo_finance import YahooDataCollector

collector = YahooDataCollector()
data = collector.fetch_data("AAPL", start_date="2020-01-01", end_date="2023-12-31")
```

### Binance (Crypto)
```python
from data.collectors.binance_api import BinanceDataCollector

collector = BinanceDataCollector()
data = collector.fetch_data("BTCUSDT", interval="1d")
```

## 🔧 Feature Engineering

The project implements various technical indicators and features:

- Moving Averages (SMA, EMA, WMA)
- Momentum Indicators (RSI, MACD, Stochastic Oscillator)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators (OBV, Volume Profile)
- Custom Features (Price Patterns, Support/Resistance)

Example usage:
```python
from features.technical_indicators import TechnicalFeatureGenerator

feature_generator = TechnicalFeatureGenerator(data)
enriched_data = feature_generator.generate_all_features()
```

## 🤖 Models

### Available Models

1. Deep Learning
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - Transformer-based models

2. Traditional ML
   - Random Forest
   - XGBoost
   - Support Vector Machines

Example model training:
```python
from models.lstm_model import LSTMModel
from models.random_forest import RandomForestModel

# LSTM
lstm_model = LSTMModel(
    input_dims=len(feature_columns),
    sequence_length=30,
    hidden_units=[64, 32]
)
lstm_model.train(X_train, y_train, epochs=100, batch_size=32)

# Random Forest
rf_model = RandomForestModel(n_estimators=100)
rf_model.train(X_train, y_train)
```

## 📈 Performance Evaluation

The project includes comprehensive evaluation metrics:

- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Directional Accuracy
- Trading Performance Metrics (Sharpe Ratio, Max Drawdown)

## 🔄 Backtesting

Includes a robust backtesting framework to validate trading strategies:

```python
from evaluation.backtesting import TradingSimulator

simulator = TradingSimulator(
    initial_capital=10000,
    commission_rate=0.001
)
results = simulator.run_backtest(predictions, test_data)
```

## 📊 Visualization

The project provides various visualization tools:

- Price and prediction plots
- Technical indicator visualization
- Performance metrics dashboards
- Trading signals and entry/exit points

## 📝 Project Structure

```
ml-stock-crypto-trend-prediction/
├── data/
│ ├── raw/                    # Raw historical data files
│ ├── processed/              # Processed data with features
│ ├── external/               # External data sources
│ ├── collectors/
│ │ ├── yahoo_finance.py
│ │ ├── binance_api.py
│ │ ├── alpha_vantage.py
│ │ └── base_collector.py
│ └── processors/
│   ├── data_cleaner.py
│   ├── data_validator.py
│   └── feature_scaler.py
├── features/
│ ├── technical_indicators.py
│ └── external_features.py
├── models/
│ ├── lstm_model.py
│ ├── random_forest.py
│ ├── xgboost_model.py
│ └── model_utils.py
├── evaluation/
│ ├── performance_metrics.py
│ ├── cross_validation.py
│ └── backtesting.py
├── tuning/
│ ├── hyperparameter_tuning.py
│ └── tuning_results/
├── visualization/
│ ├── trend_visualization.py
│ ├── feature_correlation.py
│ └── prediction_vs_actual.py
├── strategies/
│ ├── base_strategy.py
│ ├── trend_following.py
│ ├── mean_reversion.py
│ └── ml_enhanced.py
├── utils/
│ ├── logger.py
│ ├── decorators.py
│ ├── config_parser.py
│ └── validation.py
├── docs/
│ ├── api/
│ ├── models/
│ ├── features/
│ ├── examples/
│ └── maintenance/
├── scripts/
│ ├── setup.py
│ ├── data_download.py
│ ├── train_models.py
│ └── backtest_run.py
├── monitoring/
│ ├── performance_tracker.py
│ ├── drift_detection.py
│ └── alerts.py
├── experiments/
│ ├── mlflow/
│ ├── tensorboard/
│ └── model_registry/
├── web/
│ ├── api/
│ ├── dashboard/
│ └── static/
├── notebooks/
│ ├── data_preprocessing.ipynb
│ ├── feature_engineering.ipynb
│ ├── model_training.ipynb
│ └── backtesting_analysis.ipynb
├── config/
│ ├── model_config.yaml
│ ├── data_sources.yaml
│ ├── feature_config.yaml
│ ├── training_config.yaml
│ ├── backtest_config.yaml
│ └── logging_config.yaml
├── tests/
│ ├── integration/
│ ├── performance/
│ ├── stress/
│ ├── test_data_processing.py
│ ├── test_feature_engineering.py
│ ├── test_models.py
│ └── test_backtesting.py
├── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped with the development
- Special thanks to the maintainers of the data source APIs
- Inspiration from various trading and ML communities

## ⚠️ Disclaimer

This project is for educational purposes only. Trading cryptocurrencies and stocks carries significant risks. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
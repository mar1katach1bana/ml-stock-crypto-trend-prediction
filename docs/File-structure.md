# Project File Structure

## Project File Structure
```
ml-stock-crypto-trend-prediction/
├── config/                  # Configuration files
│   ├── data_sources.yaml    # Data source configurations
│   └── model_config.yaml    # Model configurations
├── data/                    # Data management
│   ├── collectors/          # Data collection modules
│   ├── external/            # External data sources
│   ├── processed/           # Processed data
│   ├── raw/                 # Raw collected data
│   └── processors/          # Data processing modules
├── docs/                    # Documentation
├── evaluation/              # Model evaluation
├── experiments/             # Experiment tracking
│   ├── mlflow/              # MLflow tracking
│   ├── model_registry/      # Model registry
│   └── tensorboard/         # TensorBoard logs
├── features/                # Feature engineering
├── models/                  # Model implementations
├── monitoring/              # Monitoring tools
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
│   ├── data/                # Data loading utilities
│   ├── features/            # Feature engineering
│   └── models/              # Model implementations
├── strategies/              # Trading strategies
├── tests/                   # Unit and integration tests
├── tuning/                  # Hyperparameter tuning
│   └── tuning_results/      # Tuning results
├── visualization/           # Visualization utilities
└── web/                     # Web interface
    ├── api/                 # REST API
    ├── dashboard/           # Web dashboard
    └── static/              # Static assets
```

## Key Files and Their Roles
1. **config/data_sources.yaml**: Contains configuration for data sources (Binance, Yahoo Finance)
2. **data/collectors/binance_api.py**: Implements Binance API data collection
3. **data/collectors/yahoo_finance.py**: Implements Yahoo Finance data collection
4. **src/models/lstm_model.py**: LSTM model implementation
5. **src/models/random_forest_model.py**: Random Forest model implementation
6. **src/models/xgboost_model.py**: XGBoost model implementation
7. **src/models/ensemble_model.py**: Ensemble model implementation
8. **web/api/**: Contains REST API implementation for model predictions
9. **web/dashboard/**: Contains web dashboard implementation
10. **experiments/mlflow/**: Stores MLflow experiment tracking data
11. **experiments/tensorboard/**: Stores TensorBoard logs

## Component Connections
- **Data Flow**: Data collectors → Raw data → Processors → Processed data → Feature engineering → Models
- **Model Lifecycle**: Model training → Evaluation → Tuning → Registry → Deployment → Monitoring
- **Web Interface**: Models → API → Dashboard → Visualization
- **Experiment Tracking**: Models → MLflow/TensorBoard → Registry

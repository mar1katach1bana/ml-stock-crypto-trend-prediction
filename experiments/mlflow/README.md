# MLflow Experiment Tracking

This directory contains configuration and tracking data for MLflow experiments.

## Setup

1. Install MLflow:
```bash
pip install mlflow
```

2. Start the MLflow tracking server:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./artifacts \
    --host 0.0.0.0
```

3. Access the MLflow UI at: http://localhost:5000

## Usage

### Tracking Experiments

In your Python code, use the following pattern to track experiments:

```python
import mlflow

# Start an experiment
mlflow.set_experiment("market-prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("learning_rate", 0.001)
    
    # Log metrics
    mlflow.log_metric("rmse", 0.05)
    mlflow.log_metric("mae", 0.03)
    
    # Log artifacts
    mlflow.log_artifact("model.h5")
```

### Viewing Results

1. Open the MLflow UI in your browser
2. Browse experiments and runs
3. Compare different runs
4. View logged parameters, metrics, and artifacts

## Directory Structure

- `mlflow.db`: SQLite database for storing experiment metadata
- `artifacts/`: Directory for storing experiment artifacts
- `models/`: Directory for storing trained models

## Best Practices

1. Use descriptive experiment names
2. Log all relevant parameters and metrics
3. Save important artifacts (models, plots, etc.)
4. Use tags to organize runs
5. Document experiments in the notes section

## Integration with Models

The LSTM model implementation automatically logs metrics and artifacts to MLflow. Other models can be similarly integrated using the MLflow API.

## Troubleshooting

- If the UI doesn't load, ensure the MLflow server is running
- If data isn't appearing, check the database connection
- If artifacts are missing, verify the artifact root directory exists

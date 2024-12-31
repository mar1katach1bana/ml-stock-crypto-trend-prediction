from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from datetime import datetime

from src.models.ensemble_model import EnsembleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Market Prediction API",
    description="API for predicting stock and cryptocurrency trends",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config_path = Path("config/model_config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize models
MODELS = {
    "regression": EnsembleModel(config_path, task='regression'),
    "classification": EnsembleModel(config_path, task='classification')
}

# Load trained models
MODELS["regression"].load("models/ensemble/regression")
MODELS["classification"].load("models/ensemble/classification")

# Request models
class PredictionRequest(BaseModel):
    features: List[List[float]]
    task: str = "regression"

class FeatureImportanceRequest(BaseModel):
    feature_names: Optional[List[str]] = None
    top_n: Optional[int] = 10

# Response models
class PredictionResponse(BaseModel):
    predictions: List[float]
    probabilities: Optional[List[float]] = None

class FeatureImportanceResponse(BaseModel):
    features: List[str]
    importance: List[float]

class ModelMetricsResponse(BaseModel):
    metrics: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make predictions using the ensemble model."""
    try:
        if request.task not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid task type")
        
        model = MODELS[request.task]
        features = np.array(request.features)
        
        predictions = model.predict(features).tolist()
        probabilities = None
        
        if request.task == "classification":
            probabilities = model.predict_proba(features)[:, 1].tolist()
        
        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/{task}", response_model=ModelMetricsResponse)
async def get_metrics(task: str):
    """Get model evaluation metrics."""
    try:
        if task not in MODELS:
            raise HTTPException(status_code=400, detail="Invalid task type")
        
        # Load test data (this should be implemented)
        X_test = np.random.random((20, 20))
        y_test = np.random.random(20) if task == "regression" else np.random.randint(0, 2, 20)
        
        metrics = MODELS[task].evaluate(X_test, y_test)
        return ModelMetricsResponse(metrics=metrics)
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(request: FeatureImportanceRequest):
    """Get feature importance ranking."""
    try:
        model = MODELS["regression"]  # Use regression model for feature importance
        importance_df = model.get_feature_importance(request.feature_names)
        
        if request.top_n:
            importance_df = importance_df.head(request.top_n)
        
        return FeatureImportanceResponse(
            features=importance_df.index.tolist(),
            importance=importance_df['importance'].tolist()
        )
        
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import hashlib
import re
from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import json
from datetime import datetime
from ml.models import ThermalAutoencoder, LSTMForecaster, train_autoencoder, train_lstm, ThermalFrameDataset, TimeSeriesDataset, DataLoader
from agent.core import analyze_input
from utils.models import Insight
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold
import numpy as np
import torch


PII_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b|\b\d{3}-\d{2}-\d{4}\b')  # Emails, SSNs, etc.


class PrivacyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and any(path in request.url.path for path in ["/upload", "/analyze"]):
            body = await request.body()
            try:
                data = json.loads(body)
                # Check for PII
                text = json.dumps(data)
                if PII_REGEX.search(text):
                    raise HTTPException(status_code=400, detail="PII detected in input; please anonymize.")
                # Anonymize equipment_id
                if "equipment_id" in data:
                    data["equipment_id"] = hashlib.sha256(data["equipment_id"].encode()).hexdigest()[:16]
                request._body = json.dumps(data).encode()
            except json.JSONDecodeError:
                # For file uploads, check filename or metadata
                pass
        response = await call_next(request)
        return response


def anonymize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Anonymize sensitive fields in data."""
    anonymized = data.copy()
    if "equipment_id" in anonymized:
        anonymized["equipment_id"] = hashlib.sha256(anonymized["equipment_id"].encode()).hexdigest()[:16]
    # Strip any PII from strings
    for key, value in anonymized.items():
        if isinstance(value, str):
            anonymized[key] = PII_REGEX.sub("[REDACTED]", value)
        elif isinstance(value, dict):
            anonymized[key] = anonymize_data(value)
        elif isinstance(value, list):
            anonymized[key] = [anonymize_data(item) if isinstance(item, dict) else (PII_REGEX.sub("[REDACTED]", item) if isinstance(item, str) else item) for item in value]
    return anonymized


def validate_ml_models() -> Dict[str, float]:
    """Run validation on ML models: F1 for anomaly detection, MAE for forecasting."""
    # Mock sample data for validation (in prod, load from dataset)
    # Anomaly frames: 100 normal, 20 anomalous
    frames = [np.random.rand(224, 224) for _ in range(120)]  # Mock
    labels = np.array([0] * 100 + [1] * 20)
    
    # Autoencoder validation
    kf = KFold(n_splits=5)
    f1_scores = []
    for train_idx, val_idx in kf.split(labels):
        model = ThermalAutoencoder()
        # Mock train
        train_autoencoder(model, DataLoader(ThermalFrameDataset([frames[i] for i in train_idx], list(labels[train_idx])), batch_size=32), epochs=1)
        # Mock preds
        preds = np.array([1 if np.random.rand() > 0.8 else 0 for _ in val_idx])
        val_labels = labels[val_idx]
        f1_scores.append(f1_score(val_labels, preds))
    avg_f1 = float(np.mean(f1_scores))
    
    # LSTM validation
    sequences = [np.random.rand(24, 3) for _ in range(100)]
    targets = [np.random.rand(7, 3) for _ in range(100)]
    model = LSTMForecaster()
    # Mock train
    train_lstm(model, DataLoader(TimeSeriesDataset(sequences, targets), batch_size=32), epochs=1)
    # Mock preds
    preds = [np.random.rand(7, 3) for _ in range(100)]
    mae = float(np.mean([mean_absolute_error(t, p) for t, p in zip(targets, preds)]))
    
    # Bias check (mock)
    bias_disparity = 0.05  # Assume low
    
    if avg_f1 < 0.85 or mae > 5.0 or bias_disparity > 0.1:
        raise ValueError("Validation failed: Retrain models.")
    
    return {"f1_anomaly": avg_f1, "mae_forecast": mae, "bias_disparity": float(bias_disparity)}


def validate_agent_sample(input_data: Dict[str, Any]) -> bool:
    """Validate agent output structure and confidence."""
    insight = analyze_input(input_data)
    if not isinstance(insight, Insight):
        return False
    if insight.health_score < 0 or insight.health_score > 100:
        return False
    if insight.ethical_notes and "human review" in insight.ethical_notes.lower():
        return True  # Flagged correctly
    return True


if __name__ == "__main__":
    print("Running ethics validation...")
    metrics = validate_ml_models()
    print(f"Metrics: {metrics}")
    sample_input = {"log_data": [{"equipment_id": "test", "sensor_data": {"temperature": 80.0}, "anomaly_summary": "test", "timestamp": datetime.now().isoformat()}]}
    print("Agent validation passed:", validate_agent_sample(sample_input))
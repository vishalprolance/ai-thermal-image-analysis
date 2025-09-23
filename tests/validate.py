import pytest
from utils.ethics import validate_ml_models, validate_agent_sample, anonymize_data
from typing import Dict, Any


def test_anonymize_data():
    data = {"equipment_id": "MOTOR-001", "email": "test@example.com", "notes": "Contact john.doe@company.com"}
    anonymized = anonymize_data(data)
    assert anonymized["equipment_id"] != "MOTOR-001"  # Hashed
    assert "[REDACTED]" in anonymized["email"]
    assert "[REDACTED]" in anonymized["notes"]


def test_validate_ml_models():
    metrics = validate_ml_models()
    assert "f1_anomaly" in metrics
    assert metrics["f1_anomaly"] >= 0.0
    assert "mae_forecast" in metrics
    assert metrics["mae_forecast"] >= 0.0
    assert "bias_disparity" in metrics
    assert metrics["bias_disparity"] >= 0.0


def test_validate_agent_sample():
    sample_input = {"log_data": [{"equipment_id": "test", "sensor_data": {"temperature": 80.0}, "anomaly_summary": "test", "timestamp": "2023-01-01T00:00:00"}]}
    assert validate_agent_sample(sample_input) is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
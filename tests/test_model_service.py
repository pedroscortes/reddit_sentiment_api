# tests/test_model_service.py
import pytest
from src.api.model_service import ModelService

def test_model_service_initialization():
    model_service = ModelService()
    assert model_service is not None

def test_metrics_initialization():
    from src.monitoring.metrics import MetricsTracker
    metrics = MetricsTracker()
    assert metrics is not None
    assert isinstance(metrics.prediction_history, list)
    assert isinstance(metrics.confidence_history, list)
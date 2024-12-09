# tests/test_monitoring.py

import pytest
from src.monitoring.metrics_manager import MetricsManager
from src.monitoring.model_monitor import ModelPerformanceMonitor
from prometheus_client import CollectorRegistry, REGISTRY
import time
import psutil

@pytest.fixture(autouse=True)
def cleanup_prometheus():
    """Clean up Prometheus registry after each test."""
    yield
    from prometheus_client import REGISTRY
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

@pytest.fixture
def metrics_manager():
    """Create a metrics manager with clean registry."""
    registry = CollectorRegistry()
    manager = MetricsManager()
    manager.registry = registry
    return manager

@pytest.fixture
def model_monitor():
    """Create a model monitor with clean registry."""
    registry = CollectorRegistry()
    return ModelPerformanceMonitor(registry=registry)

def test_metrics_manager_request_tracking(metrics_manager):
    """Test request tracking functionality."""
    metrics_manager.track_request("/predict", "POST")
    metrics_manager.track_request("/predict/batch", "POST")
    
    metrics = metrics_manager.get_metrics()
    assert metrics["requests"]["total"] > 0

def test_metrics_manager_system_metrics(metrics_manager):
    """Test system metrics collection."""
    metrics = metrics_manager.get_metrics()
    assert "system" in metrics
    assert metrics["system"]["memory_mb"] > 0
    assert isinstance(metrics["system"]["cpu_percent"], (int, float))

def test_model_monitor_prediction_tracking(model_monitor):
    """Test prediction tracking."""
    model_monitor.record_prediction(
        prediction="positive",
        confidence=0.95,
        model_version="test_v1"
    )
    
    metrics = model_monitor.get_performance_metrics(model_version="test_v1")
    assert metrics["total_predictions"] == 1
    assert metrics["avg_confidence"] == 0.95

def test_model_monitor_timing(model_monitor):
    """Test inference timing tracking."""
    start_time = model_monitor.start_prediction()
    time.sleep(0.1)  # Simulate processing
    duration = model_monitor.end_prediction(start_time, "test_v1")
    assert duration >= 0.1
    
    # Verify metrics in registry
    metrics = model_monitor.get_performance_metrics("test_v1")
    assert metrics is not None

def test_model_monitor_error_tracking(model_monitor):
    """Test error tracking."""
    model_monitor.record_error("test_v1", "ValueError")
    metrics = model_monitor.get_performance_metrics("test_v1")
    assert metrics["error_count"] == 1
    assert metrics["errors_by_type"]["ValueError"] == 1

def test_metrics_manager_prediction_distribution(metrics_manager):
    """Test sentiment prediction distribution tracking."""
    # Record multiple predictions
    metrics_manager.track_prediction("positive")
    metrics_manager.track_prediction("negative")
    metrics_manager.track_prediction("neutral")
    
    metrics = metrics_manager.get_metrics()
    assert "predictions" in metrics
    assert all(sentiment in metrics["predictions"] 
              for sentiment in ["positive", "negative", "neutral"])

def test_model_monitor_confidence_distribution(model_monitor):
    """Test confidence score distribution."""
    confidences = [0.7, 0.8, 0.9]
    for conf in confidences:
        model_monitor.record_prediction(
            prediction="positive",
            confidence=conf,
            model_version="test_v1"
        )
    
    metrics = model_monitor.get_performance_metrics("test_v1")
    assert metrics["avg_confidence"] == pytest.approx(0.8, 0.01)
    assert metrics["min_confidence"] == 0.7
    assert metrics["max_confidence"] == 0.9

def test_model_monitor_historical_tracking(model_monitor):
    """Test historical metrics tracking."""
    predictions = ["positive", "negative", "neutral"]
    confidences = [0.8, 0.85, 0.9]
    
    for pred, conf in zip(predictions, confidences):
        model_monitor.record_prediction(
            prediction=pred,
            confidence=conf,
            model_version="test_v1"
        )
    
    metrics = model_monitor.get_performance_metrics("test_v1")
    assert len(metrics["prediction_history"]) == 3
    assert "prediction_distribution" in metrics
    assert metrics["prediction_distribution"]["positive"] == 1

def test_system_resource_monitoring(metrics_manager):
    """Test detailed system resource monitoring."""
    metrics_manager.update_system_metrics()
    metrics = metrics_manager.get_metrics()
    
    assert "system" in metrics
    assert isinstance(metrics["system"]["memory_mb"], float)
    assert isinstance(metrics["system"]["cpu_percent"], float)
    assert metrics["system"]["memory_mb"] > 0
    assert 0 <= metrics["system"]["cpu_percent"] <= 100

def test_metrics_aggregation(metrics_manager):
    """Test metrics aggregation over time."""
    # Record some requests
    for _ in range(3):
        metrics_manager.track_request("/predict", "POST")
        time.sleep(0.1)
    
    metrics = metrics_manager.get_metrics()
    assert metrics["requests"]["total"] >= 3
# src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge
from typing import Dict
import numpy as np
import time

# Request metrics
REQUEST_COUNT = Counter(
    'sentiment_request_total',
    'Total number of sentiment analysis requests',
    ['endpoint']
)

REQUEST_LATENCY = Histogram(
    'sentiment_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint'],
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

# Prediction metrics
PREDICTION_DISTRIBUTION = Counter(
    'sentiment_predictions_total',
    'Distribution of sentiment predictions',
    ['sentiment']
)

CONFIDENCE_SCORE = Histogram(
    'sentiment_confidence_score',
    'Distribution of confidence scores',
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

# System metrics
MODEL_LOAD_TIME = Gauge(
    'sentiment_model_load_time_seconds',
    'Time taken to load the model'
)

MEMORY_USAGE = Gauge(
    'sentiment_memory_usage_bytes',
    'Memory usage of the application'
)

class MetricsTracker:
    def __init__(self):
        self.prediction_history = []
        self.confidence_history = []
    
    def track_request(self, endpoint: str):
        """Track API request."""
        REQUEST_COUNT.labels(endpoint=endpoint).inc()
    
    def track_latency(self, endpoint: str, start_time: float):
        """Track request latency."""
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
    
    def track_prediction(self, sentiment: str, confidence: float):
        """Track prediction and confidence."""
        PREDICTION_DISTRIBUTION.labels(sentiment=sentiment).inc()
        CONFIDENCE_SCORE.observe(confidence)
        
        # Store for drift detection
        self.prediction_history.append(sentiment)
        self.confidence_history.append(confidence)
    
    def track_system_metrics(self, memory_usage: float):
        """Track system metrics."""
        MEMORY_USAGE.set(memory_usage)
    
    def calculate_drift_metrics(self) -> Dict:
        """Calculate drift metrics based on recent predictions."""
        if len(self.prediction_history) < 100:
            return {}
            
        # Get last 100 predictions
        recent = self.prediction_history[-100:]
        confidence = self.confidence_history[-100:]
        
        # Calculate metrics
        sentiment_distribution = {
            'positive': recent.count('positive') / len(recent),
            'negative': recent.count('negative') / len(recent),
            'neutral': recent.count('neutral') / len(recent)
        }
        
        avg_confidence = np.mean(confidence)
        confidence_std = np.std(confidence)
        
        return {
            'sentiment_distribution': sentiment_distribution,
            'average_confidence': avg_confidence,
            'confidence_std': confidence_std
        }
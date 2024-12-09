# src/monitoring/model_monitor.py

import time
from datetime import datetime
from typing import Dict, List
import numpy as np
from prometheus_client import Histogram, Counter, Gauge, CollectorRegistry

class ModelPerformanceMonitor:
    def __init__(self):
        # Create a custom registry for this instance
        self.registry = CollectorRegistry()
        
        # Initialize metrics with the custom registry
        self.inference_time = Histogram(
            'model_inference_time_seconds',
            'Time spent on model inference',
            ['model_version'],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Confidence scores of predictions',
            ['model_version', 'prediction'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        self.predictions_total = Counter(
            'model_predictions_total',
            'Total number of predictions',
            ['model_version', 'prediction'],
            registry=self.registry
        )
        
        self.model_errors = Counter(
            'model_errors_total',
            'Total number of model errors',
            ['model_version', 'error_type'],
            registry=self.registry
        )
        
        # Performance tracking
        self.performance_window = 1000
        self._predictions = []
        self._inference_times = []
        
    def start_prediction(self) -> float:
        """Start timing a prediction."""
        return time.time()
    
    def end_prediction(self, start_time: float, model_version: str) -> float:
        """End timing a prediction and record it."""
        duration = time.time() - start_time
        self.inference_time.labels(model_version=model_version).observe(duration)
        self._inference_times.append(duration)
        if len(self._inference_times) > self.performance_window:
            self._inference_times.pop(0)
        return duration
    
    def record_prediction(self, 
                         prediction: str,
                         confidence: float,
                         model_version: str):
        """Record a model prediction and its confidence."""
        self.prediction_confidence.labels(
            model_version=model_version,
            prediction=prediction
        ).observe(confidence)
        
        self.predictions_total.labels(
            model_version=model_version,
            prediction=prediction
        ).inc()
        
        self._predictions.append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version
        })
        
        if len(self._predictions) > self.performance_window:
            self._predictions.pop(0)
    
    def record_error(self, model_version: str, error_type: str):
        """Record a model error."""
        self.model_errors.labels(
            model_version=model_version,
            error_type=error_type
        ).inc()
    
    def get_performance_metrics(self, model_version: str = None) -> Dict:
        """Get performance metrics for a specific model version."""
        predictions = (
            self._predictions if model_version is None else
            [p for p in self._predictions if p['model_version'] == model_version]
        )
        
        if not predictions:
            return {}
        
        confidences = [p['confidence'] for p in predictions]
        prediction_counts = {}
        for p in predictions:
            prediction_counts[p['prediction']] = prediction_counts.get(p['prediction'], 0) + 1
            
        return {
            'total_predictions': len(predictions),
            'avg_confidence': np.mean(confidences),
            'median_confidence': np.median(confidences),
            'prediction_distribution': prediction_counts,
            'avg_inference_time': np.mean(self._inference_times),
            'p95_inference_time': np.percentile(self._inference_times, 95)
        }
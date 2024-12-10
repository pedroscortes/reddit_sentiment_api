# src/monitoring/model_monitor.py

import time
from datetime import datetime
from typing import Dict, List
import numpy as np
from prometheus_client import Histogram, Counter, Gauge, CollectorRegistry

class ModelPerformanceMonitor:
    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()
        
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
        
        self.error_counter = Counter(
            'model_errors_total',
            'Total number of model errors',
            ['model_version', 'error_type'],
            registry=self.registry
        )
        
        self.performance_window = 1000
        self._predictions = []
        self._confidence_scores = []
        self._errors = {}
        self._start_times = {}

    def start_prediction(self) -> float:
        """Start timing a prediction."""
        start_time = time.time()
        return start_time

    def end_prediction(self, start_time: float, model_version: str) -> float:
        """End timing a prediction and record it."""
        duration = time.time() - start_time
        self.inference_time.labels(model_version=model_version).observe(duration)
        return duration

    def record_prediction(self, prediction: str, confidence: float, model_version: str):
        """Record a prediction and its confidence."""
        self._predictions.append({
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        })
        self._confidence_scores.append(confidence)
        
        if len(self._predictions) > self.performance_window:
            self._predictions.pop(0)
            self._confidence_scores.pop(0)
        
        self.prediction_confidence.labels(
            model_version=model_version,
            prediction=prediction
        ).observe(confidence)

    def record_error(self, model_version: str, error_type: str):
        """Record a model error."""
        self.error_counter.labels(
            model_version=model_version,
            error_type=error_type
        ).inc()
        
        if error_type not in self._errors:
            self._errors[error_type] = 0
        self._errors[error_type] += 1

    def get_performance_metrics(self, model_version: str = None) -> Dict:
        """Get performance metrics."""
        metrics = {
            'total_predictions': len(self._predictions),
            'error_count': sum(self._errors.values()),
            'errors_by_type': self._errors.copy()
        }
        
        if self._confidence_scores:
            metrics.update({
                'avg_confidence': float(np.mean(self._confidence_scores)),
                'median_confidence': float(np.median(self._confidence_scores)),
                'min_confidence': float(np.min(self._confidence_scores)),
                'max_confidence': float(np.max(self._confidence_scores))
            })
        else:
            metrics.update({
                'avg_confidence': 0.0,
                'median_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            })
            
        if self._predictions:
            prediction_counts = {}
            for p in self._predictions:
                pred = p['prediction']
                if pred not in prediction_counts:
                    prediction_counts[pred] = 0
                prediction_counts[pred] += 1
            metrics['prediction_distribution'] = prediction_counts
            metrics['prediction_history'] = self._predictions[-10:]  
            
        return metrics
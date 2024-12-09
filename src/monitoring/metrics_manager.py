# src/monitoring/metrics_manager.py

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import psutil

class MetricsManager:
    def __init__(self):
        self.registry = CollectorRegistry()
        
        # Initialize request metrics
        self.request_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'api_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint'],
            registry=self.registry
        )
        
        # Initialize prediction metrics
        self.prediction_counter = Counter(
            'sentiment_predictions_total',
            'Total predictions by sentiment',
            ['sentiment'],
            registry=self.registry
        )
        
        # Initialize system metrics
        self.memory_gauge = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_gauge = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
    
    def track_request(self, endpoint: str, method: str):
        """Track API request."""
        self.request_counter.labels(endpoint=endpoint, method=method).inc()
    
    def track_latency(self, endpoint: str, duration: float):
        """Track request latency."""
        self.request_latency.labels(endpoint=endpoint).observe(duration)
    
    def track_prediction(self, sentiment: str):
        """Track prediction."""
        self.prediction_counter.labels(sentiment=sentiment).inc()
    
    def update_system_metrics(self):
        """Update system metrics."""
        self.memory_gauge.set(psutil.Process().memory_info().rss)
        self.cpu_gauge.set(psutil.cpu_percent())
    
    def get_metrics(self):
        """Get current metrics."""
        self.update_system_metrics()
        return {
            "predictions": {
                sentiment: self.prediction_counter.labels(sentiment=sentiment)._value.get()
                for sentiment in ['positive', 'negative', 'neutral']
            },
            "system": {
                "memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
                "cpu_percent": psutil.cpu_percent()
            },
            "requests": {
                "total": sum(
                    self.request_counter.labels(endpoint="/predict", method="POST")._value.get(),
                    self.request_counter.labels(endpoint="/predict/batch", method="POST")._value.get()
                )
            }
        }

# Create a singleton instance
metrics_manager = MetricsManager()
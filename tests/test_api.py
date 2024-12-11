# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.model_service import PredictionResponse
import asyncio
from unittest.mock import Mock, AsyncMock, patch

@pytest.fixture
def client(mock_reddit_analyzer):
    from src.api.main import app
    return TestClient(app)

@pytest.fixture
def mock_model_service():
    """Create a mock model service."""
    service = Mock()
    
    service.predict_batch.return_value = [
        {
            "sentiment": "positive",
            "confidence": 0.95,
            "probabilities": {"positive": 0.95, "negative": 0.05}
        },
        {
            "sentiment": "negative",
            "confidence": 0.85,
            "probabilities": {"positive": 0.15, "negative": 0.85}
        },
        {
            "sentiment": "positive",
            "confidence": 0.75,
            "probabilities": {"positive": 0.75, "negative": 0.25}
        }
    ]
    
    return service

@pytest.fixture(autouse=True)
def mock_reddit_analyzer(monkeypatch):
    """Mock RedditAnalyzer for all tests"""
    mock_analyzer = Mock()
    mock_analyzer.analyze_url = AsyncMock(return_value={
        "comments": [{"confidence": 0.98, "is_op": False, "score": 7, "sentiment": "negative"}],
        "comments_analyzed": 1,
        "overall_sentiment": {"negative": 81.25, "positive": 18.75},
        "post": {"confidence": 0.96, "score": 14, "sentiment": "negative", "title": "Test Post"}
    })
    mock_analyzer.analyze_subreddit = AsyncMock(return_value={
        "posts": [],
        "sentiment_distribution": {"positive": 50, "negative": 50},
        "average_confidence": 0.9
    })
    mock_analyzer.analyze_user = AsyncMock(return_value={
        "comments": [],
        "sentiment_distribution": {"positive": 50, "negative": 50},
        "average_confidence": 0.9
    })
    
    monkeypatch.setattr("src.api.main.RedditAnalyzer", Mock(return_value=mock_analyzer))
    return mock_analyzer

def test_analyze_url_endpoint(client):
    test_input = {"url": "https://www.reddit.com/r/python/comments/123abc/test_post"}
    response = client.post("/analyze/url", json=test_input)
    assert response.status_code == 200

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Reddit Sentiment Analysis API", "version": "1.0.0", "status": "active"}

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_endpoint(client, mock_model_service):
    """Test single prediction endpoint."""
    test_input = {"text": "This is a test message"}
    
    mock_model_service.predict.return_value = {
        "sentiment": "positive",
        "confidence": 0.9,
        "probabilities": {"positive": 0.9, "negative": 0.1}
    }
    
    app.state.model_service = mock_model_service
    
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    assert "sentiment" in response.json()

def test_predict_batch_endpoint(client):
    test_input = {"texts": ["This is test 1", "This is test 2"]}
    response = client.post("/predict/batch", json=test_input)
    assert response.status_code == 200

def test_analyze_subreddit_endpoint(client):
    test_input = {"subreddit": "python", "time_filter": "week", "post_limit": 10}
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 200

def test_analyze_user_endpoint(client):
    test_input = {"username": "test_user", "limit": 10}
    response = client.post("/analyze/user", json=test_input)
    assert response.status_code == 200

def test_analyze_trend_endpoint(client):
    test_input = {
        "keyword": "python",
        "subreddits": ["programming", "learnpython"],
        "time_filter": "week",
        "limit": 10
    }
    response = client.post("/analyze/trend", json=test_input)
    assert response.status_code == 200

def test_predict_empty_text(client):
    test_input = {"text": ""}
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422

def test_predict_long_text(client):
    test_input = {"text": "a" * 5001}  
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422

def test_invalid_subreddit_request(client):
    test_input = {"subreddit": "python", "time_filter": "invalid"}
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 422

def test_predict_long_batch(client):
    """Test batch prediction with too many texts."""
    test_input = {
        "texts": ["Test message"] * 101  
    }
    response = client.post("/predict/batch", json=test_input)
    assert response.status_code == 422

def test_predict_batch_empty_texts(client):
    test_input = {"texts": []}
    response = client.post("/predict/batch", json=test_input)
    assert response.status_code == 422

def test_analyze_subreddit_invalid_timefilter(client):
    test_input = {"subreddit": "python", "time_filter": "invalid", "post_limit": 10}
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 422

def test_analyze_subreddit_invalid_limit(client):
    test_input = {"subreddit": "python", "time_filter": "week", "post_limit": 0}
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 422

def test_analyze_user_invalid_limit(client):
    """Test user analysis with invalid limit."""
    test_input = {
        "username": "test_user",
        "limit": 201  
    }
    response = client.post("/analyze/user", json=test_input)
    assert response.status_code == 422

def test_analyze_trend_invalid_subreddits(client):
    test_input = {
        "keyword": "python",
        "subreddits": [],
        "time_filter": "week",
        "limit": 10
    }
    response = client.post("/analyze/trend", json=test_input)
    assert response.status_code == 422

def test_analyze_trend_invalid_keyword(client):
    """Test trend analysis with empty keyword."""
    test_input = {
        "keyword": "",
        "subreddits": ["python"],
        "time_filter": "week",
        "limit": 100
    }
    response = client.post("/analyze/trend", json=test_input)
    assert response.status_code == 422

def test_health_check_no_model(client, app_with_mocks, mock_model_service):
    mock_model_service.model = None
    response = client.get("/health")
    assert response.status_code == 503

def test_metrics_endpoint(client, monkeypatch):
    """Test metrics endpoint."""
    mock_metrics = {
        "predictions": {
            "positive": 10,
            "negative": 5,
            "neutral": 3
        },
        "system": {
            "memory_mb": 100.0,
            "cpu_percent": 50.0
        },
        "requests": {
            "total": 18
        }
    }
    
    class MockMetricsManager:
        def get_metrics(self):
            return mock_metrics
    
    monkeypatch.setattr("src.api.main.metrics_manager", MockMetricsManager())
    
    response = client.get("/monitoring/metrics")
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert "system" in data
    assert "requests" in data
    assert data["predictions"]["positive"] == 10
    assert data["system"]["memory_mb"] > 0
    assert isinstance(data["requests"]["total"], (int, float))

def test_batch_prediction_mixed_content(client, mock_model_service):
    """Test batch prediction with mixed content types."""
    test_texts = ["This is great!", "This is normal"]
    
    mock_responses = [
        {
            "sentiment": "positive",
            "confidence": 0.9,
            "probabilities": {"positive": 0.9, "negative": 0.1}
        },
        {
            "sentiment": "neutral",
            "confidence": 0.6,
            "probabilities": {"positive": 0.4, "negative": 0.6}
        }
    ]
    
    mock_model_service.predict_batch.return_value = mock_responses
    
    if hasattr(app.state, 'model_service'):
        delattr(app.state, 'model_service')
    app.state.model_service = mock_model_service

    response = client.post("/predict/batch", json={"texts": test_texts})
    assert response.status_code == 200
    
    data = response.json()
    predictions = data["predictions"]
    assert len(predictions) == len(test_texts)
    assert predictions[0]["sentiment"] == "positive"
    assert predictions[1]["sentiment"] == "neutral"
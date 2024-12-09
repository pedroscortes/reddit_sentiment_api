# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import asyncio
from unittest.mock import Mock, AsyncMock

pytestmark = pytest.mark.asyncio

@pytest.fixture
def client():
    """Create test client with mocked dependencies."""
    # Mock the dependencies before creating client
    app.state.model_service = Mock()
    app.state.model_service.model = Mock()
    app.state.model_service.predict.return_value = {
        "sentiment": "positive",
        "confidence": 0.95,
        "probabilities": {"positive": 0.95, "negative": 0.05}
    }

    app.state.reddit_analyzer = Mock()
    app.state.reddit_analyzer.analyze_url = AsyncMock(return_value={
        "sentiment": "positive",
        "confidence": 0.85,
        "text": "Test comment",
        "metadata": {
            "post_title": "Test Post",
            "subreddit": "python",
            "created_utc": "2024-01-01"
        }
    })

    with TestClient(app) as client:
        yield client

@pytest.fixture
def mock_model_service():
    """Create a mock model service."""
    service = Mock()
    
    # Mock batch prediction response
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

@pytest.fixture
def mock_reddit_analyzer():
    """Create a mock Reddit analyzer."""
    analyzer = Mock()
    
    async def mock_analyze_url(url):
        return {
            "sentiment": "positive",
            "confidence": 0.85,
            "text": "Test comment",
            "metadata": {
                "post_title": "Test Post",
                "subreddit": "python",
                "created_utc": "2024-01-01"
            }
        }
    
    analyzer.analyze_url = mock_analyze_url
    return analyzer

@pytest.mark.asyncio
async def test_analyze_url_endpoint(client, monkeypatch):
    """Test the URL analysis endpoint."""
    mock_response = {
        "comments": [
            {
                "confidence": 0.98,
                "is_op": False,
                "score": 7,
                "sentiment": "negative"
            }
        ],
        "comments_analyzed": 1,
        "overall_sentiment": {
            "negative": 81.25,
            "positive": 18.75
        },
        "post": {
            "confidence": 0.96,
            "score": 14,
            "sentiment": "negative",
            "title": "Test Post"
        }
    }

    # Create mock analyzer
    mock_analyzer = Mock()
    mock_analyzer.analyze_url = AsyncMock(return_value=mock_response)
    
    # Replace the reddit_analyzer instance in the app
    app.state.reddit_analyzer = mock_analyzer

    test_input = {
        "url": "https://www.reddit.com/r/python/comments/123abc/test_post"
    }
    response = client.post("/analyze/url", json=test_input)
    
    assert response.status_code == 200
    result = response.json()
    
    # Updated assertions to match actual response structure
    assert "comments" in result
    assert "comments_analyzed" in result
    assert "overall_sentiment" in result
    assert "post" in result
    assert result["post"]["sentiment"] in ["positive", "negative"]
    assert isinstance(result["comments_analyzed"], int)
    assert all(key in result["overall_sentiment"] for key in ["positive", "negative"])

def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Reddit Sentiment Analysis API",
        "version": "1.0.0",
        "status": "active"
    }

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(client):
    """Test the prediction endpoint."""
    test_input = {"text": "This is a test message"}
    response = client.post("/predict", json=test_input)
    assert response.status_code == 200
    result = response.json()
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result

def test_predict_batch_endpoint(client, mock_model_service, monkeypatch):
    """Test the batch prediction endpoint."""
    # Patch the service in the app
    monkeypatch.setattr("src.api.main.model_service", mock_model_service)
    
    test_input = {"texts": ["Message 1", "Message 2", "Message 3"]}
    response = client.post("/predict/batch", json=test_input)
    
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 3

def test_analyze_subreddit_endpoint(client):
    """Test the subreddit analysis endpoint."""
    test_input = {
        "subreddit": "python",
        "time_filter": "week",
        "post_limit": 10
    }
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 200
    result = response.json()
    assert "sentiment_distribution" in result

def test_analyze_user_endpoint(client):
    """Test the user analysis endpoint."""
    test_input = {
        "username": "test_user",
        "limit": 10
    }
    response = client.post("/analyze/user", json=test_input)
    assert response.status_code == 200

def test_analyze_trend_endpoint(client):
    """Test the trend analysis endpoint."""
    test_input = {
        "keyword": "python",
        "subreddits": ["programming", "learnpython"],
        "time_filter": "week",
        "limit": 10
    }
    response = client.post("/analyze/trend", json=test_input)
    assert response.status_code == 200

# Error cases
def test_predict_empty_text(client):
    """Test prediction with empty text."""
    test_input = {"text": ""}
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422  # Validation error

def test_predict_long_text(client):
    """Test prediction with text exceeding length limit."""
    test_input = {"text": "a" * 5001}  # Exceeds 5000 char limit
    response = client.post("/predict", json=test_input)
    assert response.status_code == 422

def test_invalid_subreddit_request(client):
    """Test invalid subreddit analysis request."""
    test_input = {
        "subreddit": "",
        "time_filter": "invalid",
        "post_limit": -1
    }
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 422

# tests/test_api.py (add these new test functions)

def test_predict_long_batch(client):
    """Test batch prediction with too many texts."""
    test_input = {
        "texts": ["Test message"] * 101  # Over the 100 limit
    }
    response = client.post("/predict/batch", json=test_input)
    assert response.status_code == 422

def test_predict_batch_empty_texts(client):
    """Test batch prediction with empty texts list."""
    test_input = {"texts": []}
    response = client.post("/predict/batch", json=test_input)
    assert response.status_code == 422

def test_analyze_subreddit_invalid_timefilter(client):
    """Test subreddit analysis with invalid time filter."""
    test_input = {
        "subreddit": "python",
        "time_filter": "invalid_time",
        "post_limit": 10
    }
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 422

def test_analyze_subreddit_invalid_limit(client):
    """Test subreddit analysis with invalid post limit."""
    test_input = {
        "subreddit": "python",
        "time_filter": "week",
        "post_limit": 501  # Over the 500 limit
    }
    response = client.post("/analyze/subreddit", json=test_input)
    assert response.status_code == 422

def test_analyze_user_invalid_limit(client):
    """Test user analysis with invalid limit."""
    test_input = {
        "username": "test_user",
        "limit": 201  # Over the 200 limit
    }
    response = client.post("/analyze/user", json=test_input)
    assert response.status_code == 422

def test_analyze_trend_invalid_subreddits(client):
    """Test trend analysis with too many subreddits."""
    test_input = {
        "keyword": "python",
        "subreddits": ["sub1", "sub2", "sub3", "sub4", "sub5", "sub6"],  # Over 5 limit
        "time_filter": "week",
        "limit": 100
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

def test_health_check_no_model(client, monkeypatch):
    """Test health check when model is not loaded."""
    # Mock model service without a model
    class MockModelService:
        model = None
        
    monkeypatch.setattr("src.api.main.model_service", MockModelService())
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"

def test_metrics_endpoint(client, monkeypatch):
    """Test metrics endpoint."""
    # Mock metrics manager
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

# Add more complex scenario tests
def test_batch_prediction_mixed_content(client, monkeypatch):
    """Test batch prediction with mixed content types."""
    # Mock the model service
    def mock_predict_batch(texts):
        results = []
        for text in texts:
            # Simulate different sentiments based on content
            if "great" in text.lower():
                sentiment = "positive"
            elif "bad" in text.lower():
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            results.append({
                "sentiment": sentiment,
                "confidence": 0.9,
                "probabilities": {"positive": 0.9, "negative": 0.1}
            })
        return results
    
    # Create mock service
    mock_service = Mock()
    mock_service.predict_batch = mock_predict_batch
    
    # Patch the service
    monkeypatch.setattr("src.api.main.model_service", mock_service)
    
    test_input = {
        "texts": [
            "This is great!",
            "This is bad.",
            "This is neutral."
        ]
    }
    response = client.post("/predict/batch", json=test_input)
    
    assert response.status_code == 200
    results = response.json()["predictions"]
    assert len(results) == 3
    assert results[0]["sentiment"] == "positive"
    assert results[1]["sentiment"] == "negative"
    assert results[2]["sentiment"] == "neutral"
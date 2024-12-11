# tests/conftest.py
import pytest
import os
import torch
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from src.api.main import app

@pytest.fixture(autouse=True)
def mock_mlflow_tracking():
    """Mock MLflow tracking for all tests"""
    with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
        yield

@pytest.fixture(autouse=True)
def mock_env():
    """Set up test environment variables"""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['REDDIT_CLIENT_ID'] = 'test_client_id'
    os.environ['REDDIT_CLIENT_SECRET'] = 'test_client_secret'
    os.environ['REDDIT_USER_AGENT'] = 'test_user_agent'

@pytest.fixture(autouse=True)
def mock_reddit():
    """Mock Reddit API for all tests"""
    mock = Mock()
    mock_subreddit = Mock()
    mock_subreddit.hot.return_value = [
        Mock(
            title="Test Post",
            selftext="Test Content",
            score=100,
            num_comments=10,
            created_utc=1234567890,
            comments=[
                Mock(
                    body="Test comment",
                    score=5,
                    created_utc=1234567890
                )
            ]
        )
    ]
    mock.subreddit.return_value = mock_subreddit

    mock_user = Mock()
    mock_user.comments.new.return_value = [
        Mock(
            body="Test comment",
            score=5,
            created_utc=1234567890
        )
    ]
    mock.redditor.return_value = mock_user

    with patch('praw.Reddit', return_value=mock), \
         patch('asyncpraw.Reddit', return_value=AsyncMock(return_value=mock)):
        yield mock

@pytest.fixture
def mock_model_service():
    mock = Mock()
    mock.model = Mock()
    mock.predict.return_value = {"sentiment": "positive", "confidence": 0.95}
    mock.predict_batch.return_value = [{"sentiment": "positive", "confidence": 0.95}]
    return mock

@pytest.fixture
def mock_reddit_analyzer():
    mock = Mock()
    mock.analyze_url = AsyncMock(return_value={
        "comments": [{"sentiment": "positive", "confidence": 0.95}],
        "overall_sentiment": {"positive": 75, "negative": 25},
        "comments_analyzed": 1
    })
    mock.analyze_trend = AsyncMock(return_value={
        "trend_data": [],
        "overall_sentiment": {"positive": 60, "negative": 40}
    })
    mock.analyze_subreddit = AsyncMock(return_value={
        "posts": [],
        "sentiment_distribution": {"positive": 60, "negative": 40}
    })
    mock.analyze_user = AsyncMock(return_value={
        "comments": [],
        "sentiment_distribution": {"positive": 60, "negative": 40}
    })
    return mock

@pytest.fixture
def app_with_mocks(mock_model_service, mock_reddit_analyzer):
    from src.api.main import app, get_model_service, get_reddit_analyzer
    
    def override_get_model_service():
        return mock_model_service
        
    def override_get_reddit_analyzer():
        return mock_reddit_analyzer
    
    app.dependency_overrides[get_model_service] = override_get_model_service
    app.dependency_overrides[get_reddit_analyzer] = override_get_reddit_analyzer
    return app


@pytest.fixture
def client(app_with_mocks):
    from fastapi.testclient import TestClient
    with TestClient(app_with_mocks) as test_client:
        yield test_client

@pytest.fixture(autouse=True)
def mock_torch_device():
    """Mock torch device to always use CPU"""
    with patch('torch.cuda.is_available', return_value=False):
        yield

@pytest.fixture(autouse=True)
def mock_nltk_downloads():
    """Mock NLTK downloads for tests"""
    with patch('nltk.download'):
        yield

@pytest.fixture
def test_app(mock_model_service, mock_reddit_analyzer):
    """Test app with mocked dependencies"""
    app.state.model_service = mock_model_service
    app.state.reddit_analyzer = mock_reddit_analyzer
    return app
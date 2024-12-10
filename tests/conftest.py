# tests/conftest.py
import pytest
import os
import torch
from unittest.mock import patch

@pytest.fixture(autouse=True)
def mock_mlflow_tracking():
    """Mock MLflow tracking for all tests"""
    with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
        yield

@pytest.fixture(autouse=True)
def mock_env():
    """Set up test environment variables"""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  
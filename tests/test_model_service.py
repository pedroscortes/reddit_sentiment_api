# tests/test_model_service.py

import pytest
import os
from src.api.model_service import ModelService
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from prometheus_client import REGISTRY

@pytest.fixture(autouse=True)
def cleanup():
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

@pytest.fixture
def model_service():
    service = ModelService()
    try:
        models_dir = Path('models')
        model_dirs = [d for d in os.listdir(models_dir) 
                     if d.startswith('sentiment_model_20241207')]
        
        if not model_dirs:
            pytest.skip("No trained model found in models directory")
            
        latest_model = max(model_dirs)
        model_path = os.path.join('models', latest_model, 'final_model')
        
        print(f"Loading model from: {model_path}")
        
        service.model_registry.register_model(
            model_path=model_path,
            model_name="sentiment_analyzer",
            version="1.0",
            metrics={"accuracy": 0.90},
            description="Sentiment analysis model"
        )
        
        service.load_model(model_path)
        
        service.current_model_id = latest_model
        
        return service
    except Exception as e:
        pytest.skip(f"Could not load model: {str(e)}")

def test_model_initialization(model_service):
    """Test if model is properly initialized"""
    assert model_service.model is not None
    assert model_service.tokenizer is not None
    assert model_service.current_model_id is not None

def test_single_prediction(model_service):
    text = "This is a great day!"
    result = model_service.predict(text)
    
    assert result is not None
    assert hasattr(result, 'sentiment')
    assert hasattr(result, 'confidence')
    assert hasattr(result, 'probabilities')
    assert result.confidence >= 0 and result.confidence <= 1
    assert result.sentiment in ['positive', 'negative']  
    assert len(result.probabilities) == 2  

def test_batch_prediction(model_service):
    """Test batch prediction"""
    texts = [
        "This is wonderful!",
        "I'm feeling sad today",
        "The weather is okay"
    ]
    results = model_service.predict_batch(texts)
    
    assert len(results) == 3
    for result in results:
        assert hasattr(result, 'sentiment')
        assert hasattr(result, 'confidence')
        assert result.confidence >= 0 and result.confidence <= 1

def test_model_performance_metrics(model_service):
    """Test performance monitoring"""
    texts = ["Great!", "Terrible!", "Okay."]
    for text in texts:
        model_service.predict(text)
    
    metrics = model_service.get_model_performance()
    assert isinstance(metrics, dict)
    
    if metrics:
        assert 'total_predictions' in metrics
        assert metrics['total_predictions'] >= len(texts)

def test_model_info(model_service):
    info = model_service.get_model_info()
    assert isinstance(info, dict)
    assert 'name' in info
    assert info['name'] == 'sentiment_analyzer'
    assert info['version'] == '1.0'

def test_error_handling(model_service):
    with pytest.raises(ValueError):
        model_service.predict("")

def test_device_handling(model_service):
    """Test correct device assignment"""
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert str(model_service.device) == expected_device

@pytest.mark.parametrize("text,expected_sentiment", [
    ("This is absolutely amazing!", "positive"),
    ("This is terrible and horrible.", "negative"),
    ("The weather is normal today.", "positive")  
])
def test_sentiment_classification(model_service, text, expected_sentiment):
    result = model_service.predict(text)
    assert result.sentiment == expected_sentiment
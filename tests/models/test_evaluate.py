# tests/models/test_evaluate.py
import pytest
import torch
import numpy as np
import pandas as pd
import json
import os
from unittest.mock import Mock, patch, MagicMock
from transformers import BatchEncoding, AutoTokenizer, AutoConfig
from src.models.evaluate import ModelEvaluator
from src.models.training import SentimentClassifier

@pytest.fixture
def mock_model():
    class MockOutput:
        def __init__(self, batch_size):
            self.logits = torch.randn(batch_size, 3)

    class MockModel(Mock):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.eval = Mock(return_value=None)
            self.to = Mock(return_value=self)

        def __call__(self, **kwargs):
            batch_size = kwargs['input_ids'].shape[0]
            return MockOutput(batch_size)

    return MockModel()

@pytest.fixture
def mock_tokenizer():
    class MockTokenizer:
        def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors=None):
            if isinstance(texts, str):
                texts = [texts]
            if not texts:
                raise ValueError("Empty input is not allowed")
            
            batch_size = len(texts)

            class BatchEncoding(dict):
                def to(self, device):
                    return self

            return BatchEncoding({
                'input_ids': torch.ones(batch_size, max_length, dtype=torch.long),
                'attention_mask': torch.ones(batch_size, max_length, dtype=torch.long)
            })

        def save_pretrained(self, path):
            pass

    return MockTokenizer()

@pytest.fixture
def model_evaluator(tmp_path, mock_model, mock_tokenizer):
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    torch.save({}, model_dir / "pytorch_model.bin")
    
    config = {
        "model_type": "distilbert",
        "architectures": ["DistilBertForSequenceClassification"],
        "num_labels": 3
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)

    with patch('transformers.AutoConfig.from_pretrained'), \
         patch('src.models.training.SentimentClassifier.from_pretrained', return_value=mock_model), \
         patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        return ModelEvaluator(str(model_dir))

def test_evaluator_initialization(model_evaluator):
    assert model_evaluator.device in [torch.device('cuda'), torch.device('cpu')]
    assert model_evaluator.model is not None
    assert model_evaluator.tokenizer is not None

def test_predict_batch(model_evaluator):
    texts = ['test text 1', 'test text 2']
    numerical_features = torch.randn(2, 3)
    predictions = model_evaluator.predict_batch(texts, numerical_features)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 2

@patch('pandas.read_csv')
def test_evaluate_model(mock_read_csv, model_evaluator, tmp_path):
    test_data = pd.DataFrame({
        'text': ['text1', 'text2', 'text3'],  
        'processed_text': ['text1', 'text2', 'text3'],
        'sentiment': ['positive', 'negative', 'neutral'],  
        'tfidf_word_1': [0.5, 0.3, 0.4],
        'sentiment_score_1': [0.8, -0.6, 0.0],
    })
    mock_read_csv.return_value = test_data
    
    metrics = model_evaluator.evaluate_model(str(tmp_path / "test_data.csv"))
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics

def test_evaluator_edge_cases(model_evaluator):
    with pytest.raises(ValueError):
        model_evaluator.predict_batch([], torch.randn(0, 3))

def test_evaluator_device_handling(model_evaluator):
    texts = ['test text']
    numerical_features = torch.randn(1, 3)
    predictions = model_evaluator.predict_batch(texts, numerical_features)
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == 1
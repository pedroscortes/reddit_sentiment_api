# tests/models/test_training.py
import pytest
import torch
import pandas as pd
import numpy as np
import json
import os
import mlflow
from unittest.mock import Mock, patch, MagicMock, create_autospec
from datasets import Dataset
from transformers import TrainingArguments, AutoTokenizer, BatchEncoding, AutoModel, PretrainedConfig
from src.models.training import SentimentClassifier, SentimentModelTrainer

@pytest.fixture
def mock_bert_config():
    config = Mock(spec=PretrainedConfig)
    config.hidden_size = 768
    return config

@pytest.fixture
def mock_bert_output():
    output = Mock()
    output.last_hidden_state = torch.randn(2, 128, 768)
    return output

@pytest.fixture
def mock_bert(mock_bert_config, mock_bert_output):
    model = Mock(spec=AutoModel)
    model.config = mock_bert_config
    model.return_value = mock_bert_output
    return model

@pytest.fixture
def sample_data():
    samples_per_class = 5
    data = {
        'processed_text': [f"text_{i}" for i in range(samples_per_class * 3)],
        'sentiment': (['positive'] * samples_per_class +
                     ['negative'] * samples_per_class +
                     ['neutral'] * samples_per_class),
        'tfidf_word_1': [0.5] * (samples_per_class * 3),
        'sentiment_score_1': [0.8] * (samples_per_class * 3),
        'score_metric_1': [4.5] * (samples_per_class * 3)
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_mlflow_run():
    mock_run = MagicMock()
    mock_run.info = MagicMock()
    mock_run.info.run_id = "test_run_id"
    return mock_run

@pytest.fixture
def mock_mlflow():
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.set_tracking_uri') as mock_set_uri, \
         patch('mlflow.set_experiment') as mock_set_exp, \
         patch('mlflow.active_run') as mock_active_run, \
         patch('mlflow._ml_flow', create=True) as mock_ml_flow:
        
        mock_run = MagicMock()
        mock_run.info = MagicMock()
        mock_run.info.run_id = "test_run_id"
        
        mock_active_run.return_value = mock_run
        mock_start_run.return_value.__enter__.return_value = mock_run
        mock_ml_flow.active_run = mock_active_run
        mock_ml_flow.start_run = mock_start_run
        mock_ml_flow.is_tracking_uri_set = Mock(return_value=True)
        mock_ml_flow.get_tracking_uri = Mock(return_value="mock://")
        
        yield mock_run

@pytest.fixture
def mock_tokenizer():
    class MockTokenizer:
        def __init__(self):
            self.save_pretrained = Mock()
            
        def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors=None):
            if isinstance(texts, str):
                texts = [texts]
            batch_size = len(texts)
            input_ids = torch.ones(batch_size, max_length, dtype=torch.long)
            attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
            
            class EncodingDict(dict):
                def to(self, device):
                    return self
            
            return EncodingDict({
                'input_ids': input_ids,
                'attention_mask': attention_mask
            })

    return MockTokenizer()

@pytest.fixture
def model_trainer(mock_tokenizer, mock_mlflow):
    with patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer):
        trainer = SentimentModelTrainer(
            model_name="distilbert-base-uncased",
            experiment_name="test_experiment"
        )
        return trainer

@pytest.fixture
def mock_trainer():
    trainer = Mock()
    trainer.train = Mock(return_value={'loss': 0.1, 'steps': 100})
    return trainer

@pytest.fixture
def mock_dataset():
    class MockDataset:
        def __init__(self):
            self.features = {
                'input_ids': torch.ones(4, 128, dtype=torch.long),  
                'attention_mask': torch.ones(4, 128, dtype=torch.long),  
                'numerical_features': torch.randn(4, 3),
                'label': torch.tensor([0, 1, 1, 2], dtype=torch.long)  
            }
            
        def __len__(self):
            return 4
            
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.features.items()}
            
    return MockDataset()

@pytest.fixture
def sample_state_dict():
    return {
        'numerical_layer.0.weight': torch.randn(128, 3),
        'numerical_layer.0.bias': torch.randn(128),
        'numerical_layer.3.weight': torch.randn(64, 128),
        'numerical_layer.3.bias': torch.randn(64),
        'classifier.0.weight': torch.randn(256, 832),
        'classifier.0.bias': torch.randn(256),
        'classifier.3.weight': torch.randn(3, 256),
        'classifier.3.bias': torch.randn(3)
    }

def test_sentiment_classifier_initialization(mock_bert):
    with patch('transformers.AutoModel.from_pretrained', return_value=mock_bert):
        model = SentimentClassifier(model_name="distilbert-base-uncased", num_numerical_features=3)
        assert isinstance(model, torch.nn.Module)
        assert model.numerical_layer[0].in_features == 3
        assert model.numerical_layer[-1].out_features == 64
        assert model.classifier[-1].out_features == 3

def test_sentiment_classifier_forward(mock_bert):
    with patch('transformers.AutoModel.from_pretrained', return_value=mock_bert):
        model = SentimentClassifier(model_name="distilbert-base-uncased", num_numerical_features=3)
        
        batch_size = 2
        seq_length = 128
        num_features = 3
        
        inputs = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
            'attention_mask': torch.ones((batch_size, seq_length)),
            'numerical_features': torch.randn((batch_size, num_features)),
            'labels': torch.tensor([0, 1])
        }
        
        outputs = model(**inputs)
        assert outputs.loss is not None
        assert outputs.logits.shape == (batch_size, 3)

def test_sentiment_classifier_save_load(tmp_path, sample_state_dict, mock_bert):
    with patch('transformers.AutoModel.from_pretrained', return_value=mock_bert):
        model = SentimentClassifier(model_name="distilbert-base-uncased", num_numerical_features=3)
        save_path = tmp_path / "test_model"
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        
        loaded_model = SentimentClassifier.from_pretrained(save_path)
        assert isinstance(loaded_model, SentimentClassifier)

@patch('pandas.read_csv')
def test_prepare_data(mock_read_csv, model_trainer, sample_data, mock_tokenizer):
    mock_read_csv.return_value = sample_data
    
    def mock_map(func, *args, **kwargs):
        processed = {
            'input_ids': torch.ones(len(sample_data), 128),
            'attention_mask': torch.ones(len(sample_data), 128),
            'label': torch.zeros(len(sample_data)),
            'numerical_features': torch.ones(len(sample_data), 3)
        }
        return Dataset.from_dict(processed)
    
    with patch('datasets.Dataset.map', side_effect=mock_map):
        train_dataset, test_dataset, num_features = model_trainer.prepare_data("dummy_path")
        
        assert isinstance(train_dataset, Dataset)
        assert isinstance(test_dataset, Dataset)
        assert num_features == 3

@patch('src.models.training.Trainer')  
@patch('mlflow.start_run')
def test_train_model(mock_mlflow_start_run, mock_trainer_class, model_trainer, mock_dataset, tmp_path, mock_mlflow):
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_mlflow_start_run.return_value.__enter__.return_value = mock_run

    mock_save = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = {'loss': 0.5}
    mock_trainer.save_model = mock_save
    mock_trainer_class.return_value = mock_trainer

    with patch.multiple(
        'mlflow',
        log_metrics=Mock(),
        log_params=Mock(),
        set_tracking_uri=Mock(),
        set_experiment=Mock(),
        active_run=Mock(return_value=mock_run)
    ), patch.object(model_trainer.tokenizer, 'save_pretrained'), \
       patch('torch.save'):

        model, returned_trainer, model_path = model_trainer.train_model(
            mock_dataset,
            mock_dataset,
            num_numerical_features=3,
            output_dir=str(tmp_path)
        )

        mock_trainer_class.assert_called_once()
        assert mock_trainer.train.called
        assert isinstance(model, SentimentClassifier)
        assert model_path.endswith('final_model')

        trainer_call_kwargs = mock_trainer_class.call_args[1]
        assert 'model' in trainer_call_kwargs
        assert 'train_dataset' in trainer_call_kwargs
        assert 'eval_dataset' in trainer_call_kwargs
# src/models/training.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn as nn
from datasets import Dataset
import logging
import os
from datetime import datetime
import mlflow
from accelerate import Accelerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_numerical_features):
        super().__init__()
        # Load base BERT model without classification head
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Layer for numerical features
        self.numerical_layer = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None):
        # Get BERT outputs
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get CLS token output
        cls_output = bert_outputs.last_hidden_state[:, 0]
        
        # Process numerical features
        numerical_output = self.numerical_layer(numerical_features)
        
        # Combine features
        combined_features = torch.cat([cls_output, numerical_output], dim=1)
        
        # Get logits
        logits = self.classifier(combined_features)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=bert_outputs.hidden_states,
            attentions=bert_outputs.attentions
        )

class SentimentModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased", experiment_name="sentiment_analysis"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()
        logger.info(f"Using device: {self.device}")
        
        # Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, data_path):
        """Load and prepare data for training with engineered features."""
        logger.info("Loading and preparing data...")
        
        df = pd.read_csv(data_path)
        
        # Separate features
        text_column = 'processed_text'
        numerical_features = [col for col in df.columns if col.startswith(('tfidf_', 'sentiment_', 'score_'))]
        
        logger.info(f"Number of numerical features: {len(numerical_features)}")
        
        # Convert sentiment to numeric labels
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = df['sentiment'].map(sentiment_map)
        
        # Split data
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['label'], 
            random_state=42
        )
        
        # Normalize numerical features
        scaler = StandardScaler()
        train_numerical = scaler.fit_transform(train_df[numerical_features].fillna(0))
        test_numerical = scaler.transform(test_df[numerical_features].fillna(0))
        
        # Convert to torch tensors
        train_numerical = torch.FloatTensor(train_numerical)
        test_numerical = torch.FloatTensor(test_numerical)
        
        # Create datasets
        train_dataset = Dataset.from_dict({
            'text': train_df[text_column].tolist(),
            'label': train_df['label'].tolist(),
            'numerical_features': train_numerical.tolist()
        })
        
        test_dataset = Dataset.from_dict({
            'text': test_df[text_column].tolist(),
            'label': test_df['label'].tolist(),
            'numerical_features': test_numerical.tolist()
        })
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )
        
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Train set size: {len(train_dataset)}")
        logger.info(f"Test set size: {len(test_dataset)}")
        logger.info(f"Number of numerical features: {len(numerical_features)}")
        
        return train_dataset, test_dataset, len(numerical_features)
    
    def train_model(self, train_dataset, test_dataset, num_numerical_features, output_dir="models"):
        """Train model with both text and numerical features."""
        logger.info("Starting model training...")
        
        # Create model directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f'models/sentiment_model_{timestamp}'
        os.makedirs(model_dir, exist_ok=True)
        
        with mlflow.start_run() as run:
            # Initialize model
            model = SentimentClassifier(
                model_name=self.model_name,
                num_numerical_features=num_numerical_features
            ).to(self.device)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=model_dir,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(model_dir, 'logs'),
                logging_steps=100,
                eval_steps=500,
                save_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                gradient_accumulation_steps=2,
                fp16=torch.cuda.is_available(),
                report_to="mlflow"
            )
            
            # Log parameters
            mlflow.log_params({
                "model_name": self.model_name,
                "num_epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "num_numerical_features": num_numerical_features
            })
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset
            )
            
            logger.info("Training started...")
            train_result = trainer.train()
            logger.info("Training completed!")
            
            # Save final model
            final_model_path = os.path.join(model_dir, 'final_model')
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            # Log metrics
            mlflow.log_metrics({
                "training_loss": train_result.metrics["train_loss"],
                "training_runtime": train_result.metrics["train_runtime"]
            })
            
            return model, trainer, final_model_path

def main():
    try:
        # Initialize trainer
        trainer = SentimentModelTrainer()
        
        # Get latest engineered features file
        data_files = [f for f in os.listdir('data/processed') if f.startswith('engineered_features_')]
        latest_file = max(data_files)
        data_path = os.path.join('data/processed', latest_file)
        
        logger.info(f"Using data file: {data_path}")
        
        # Prepare data
        train_dataset, test_dataset, num_numerical_features = trainer.prepare_data(data_path)
        
        # Train model
        model, trainer_obj, model_path = trainer.train_model(
            train_dataset,
            test_dataset,
            num_numerical_features
        )
        
        logger.info(f"""
        Training Complete!
        Model saved to: {model_path}
        """)
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
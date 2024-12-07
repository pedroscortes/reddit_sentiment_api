# src/models/training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
)
import torch
from datasets import Dataset
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, data_path):
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")
        
        df = pd.read_csv(data_path)
        sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['label'] = df['sentiment'].map(sentiment_map)
        
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        
        logger.info(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
        logger.info(f"Label distribution in train set: \n{train_df['sentiment'].value_counts()}")
        
        train_dataset = Dataset.from_pandas(train_df[['processed_text', 'label']])
        test_dataset = Dataset.from_pandas(test_df[['processed_text', 'label']])
        
        return train_dataset, test_dataset, sentiment_map
        
    def tokenize_data(self, dataset):
        return dataset.map(
            lambda x: self.tokenizer(
                x['processed_text'],
                truncation=True,
                padding='max_length',
                max_length=128
            ),
            batched=True
        )
        
    def train_model(self, train_dataset, test_dataset, output_dir="models"):
        """Train the model and save checkpoints."""
        logger.info("Starting model training...")
        
        # Create model directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f'models/sentiment_model_{timestamp}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        ).to(self.device)
        
        # Define training arguments
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
        )
        
        # Prepare datasets
        train_dataset = self.tokenize_data(train_dataset)
        test_dataset = self.tokenize_data(test_dataset)
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        
        # Train model
        logger.info("Training started...")
        train_result = trainer.train()
        logger.info("Training completed!")
        
        # Log training metrics
        logger.info(f"Training metrics: {train_result.metrics}")
        
        # Evaluate model
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_results}")
        
        # Save final model and tokenizer
        final_model_path = os.path.join(model_dir, 'final_model')
        os.makedirs(final_model_path, exist_ok=True)
        
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        logger.info(f"Model and tokenizer saved to {final_model_path}")
        
        # Save training args
        with open(os.path.join(model_dir, 'training_args.txt'), 'w') as f:
            f.write(str(training_args))
        
        return model, trainer, final_model_path

def main():
    # Initialize trainer
    trainer = SentimentModelTrainer()
    
    # Get latest processed data
    data_files = [f for f in os.listdir('data/processed') if f.startswith('processed_large_dataset_')]
    latest_file = max(data_files)
    data_path = os.path.join('data/processed', latest_file)
    
    logger.info(f"Using data file: {data_path}")
    
    # Prepare data
    train_dataset, test_dataset, sentiment_map = trainer.prepare_data(data_path)
    
    # Train model
    model, trainer_obj, model_path = trainer.train_model(train_dataset, test_dataset)
    
    logger.info(f"""
    Training Complete!
    Model saved to: {model_path}
    
    To use this model for predictions:
    1. Load the model: AutoModelForSequenceClassification.from_pretrained('{model_path}')
    2. Load the tokenizer: AutoTokenizer.from_pretrained('{model_path}')
    """)

if __name__ == "__main__":
    main()
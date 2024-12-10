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
import json  
from datetime import datetime
import mlflow
from accelerate import Accelerator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_numerical_features):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.numerical_layer = nn.Sequential(
            nn.Linear(num_numerical_features, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 64)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 64, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 3)
        )

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, numerical_features=None, labels=None):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        cls_output = bert_outputs.last_hidden_state[:, 0]

        numerical_output = self.numerical_layer(numerical_features)

        combined_features = torch.cat([cls_output, numerical_output], dim=1)

        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(cls, model_path):
        """Load a trained model from a directory."""
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                config = json.load(f)
            num_numerical_features = config.get("num_numerical_features", 1006)
        else:
            num_numerical_features = 1006

        model = cls("distilbert-base-uncased", num_numerical_features)

        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)

        return model

    def save_pretrained(self, save_path):
        """Save the model to a directory."""
        os.makedirs(save_path, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))

        config = {
            "num_numerical_features": self.numerical_layer[0].in_features,
            "model_type": "sentiment_classifier",
            "base_model": "distilbert-base-uncased",
        }

        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)


class SentimentModelTrainer:
    def __init__(self, model_name="distilbert-base-uncased", experiment_name="sentiment_analysis"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()
        logger.info(f"Using device: {self.device}")

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(experiment_name)

    def prepare_data(self, data_path):
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")

        df = pd.read_csv(data_path)

        text_column = "processed_text"
        numerical_features = [col for col in df.columns if col.startswith(("tfidf_", "sentiment_", "score_"))]

        logger.info(f"Number of numerical features: {len(numerical_features)}")

        sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
        df["label"] = df["sentiment"].map(sentiment_map)

        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

        scaler = StandardScaler()
        train_numerical = scaler.fit_transform(train_df[numerical_features].fillna(0))
        test_numerical = scaler.transform(test_df[numerical_features].fillna(0))

        def create_dataset(texts, labels, numerical_feats):
            return Dataset.from_dict({"text": texts, "label": labels, "numerical_features": numerical_feats.tolist()})

        train_dataset = create_dataset(train_df[text_column].tolist(), train_df["label"].tolist(), train_numerical)

        test_dataset = create_dataset(test_df[text_column].tolist(), test_df["label"].tolist(), test_numerical)

        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        train_dataset = train_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)

        logger.info(f"Train set size: {len(train_dataset)}")
        logger.info(f"Test set size: {len(test_dataset)}")
        logger.info(f"Number of numerical features: {len(numerical_features)}")

        return train_dataset, test_dataset, len(numerical_features)

    def train_model(self, train_dataset, test_dataset, num_numerical_features, output_dir="models"):
        """Train model with both text and numerical features."""
        logger.info("Starting model training...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"models/sentiment_model_{timestamp}"
        os.makedirs(model_dir, exist_ok=True)

        with mlflow.start_run() as run:
            model = SentimentClassifier(model_name=self.model_name, num_numerical_features=num_numerical_features).to(
                self.device
            )

            training_args = TrainingArguments(
                output_dir=model_dir,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(model_dir, "logs"),
                logging_steps=100,
                eval_steps=500,
                save_steps=500,
                evaluation_strategy="steps",
                save_strategy="steps",
                load_best_model_at_end=True,
                gradient_accumulation_steps=2,
                fp16=torch.cuda.is_available(),
                report_to="mlflow",
            )

            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

            logger.info("Training started...")
            train_result = trainer.train()
            logger.info("Training completed!")

            final_model_path = os.path.join(model_dir, "final_model")
            model.save_pretrained(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)

            return model, trainer, final_model_path


def main():
    try:
        trainer = SentimentModelTrainer()

        data_files = [f for f in os.listdir("data/processed") if f.startswith("engineered_features_")]
        latest_file = max(data_files)
        data_path = os.path.join("data/processed", latest_file)

        logger.info(f"Using data file: {data_path}")

        train_dataset, test_dataset, num_numerical_features = trainer.prepare_data(data_path)

        model, trainer_obj, model_path = trainer.train_model(train_dataset, test_dataset, num_numerical_features)

        logger.info(
            f"""
        Training Complete!
        Model saved to: {model_path}
        """
        )

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()

# src/api/model_service.py

import torch
import numpy as np
from transformers import AutoTokenizer
import logging
import os
from src.models.training import SentimentClassifier
from typing import Dict, List
import joblib

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}

    def load_model(self, model_path: str):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = SentimentClassifier.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Load scaler if available
            scaler_path = os.path.join(model_path, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, text: str):
        """Make a single prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Tokenize input
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)

            # Create dummy numerical features
            numerical_features = torch.zeros((1, 1006)).to(self.device)

            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs, numerical_features=numerical_features)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)
                confidence = torch.max(probabilities).item()

            # Convert probabilities to dict
            prob_dict = {self.id2label[i]: prob.item() for i, prob in enumerate(probabilities[0])}

            return type(
                "PredictionResult",
                (),
                {"sentiment": self.id2label[prediction.item()], "confidence": confidence, "probabilities": prob_dict},
            )

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, texts: List[str]):
        """Make batch predictions."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Tokenize all texts
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)

            # Create dummy numerical features
            numerical_features = torch.zeros((len(texts), 1006)).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs, numerical_features=numerical_features)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1).values

            # Convert to response format
            responses = []
            for pred, conf, probs in zip(predictions, confidences, probabilities):
                prob_dict = {self.id2label[i]: prob.item() for i, prob in enumerate(probs)}
                responses.append(
                    type(
                        "PredictionResult",
                        (),
                        {"sentiment": self.id2label[pred.item()], "confidence": conf.item(), "probabilities": prob_dict},
                    )
                )

            return responses

        except Exception as e:
            logger.error(f"Error making batch prediction: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")

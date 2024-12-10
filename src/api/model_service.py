# src/api/model_service.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import logging
from pydantic import BaseModel
import os
from src.models.model_registry import ModelRegistry
from src.monitoring.model_monitor import ModelPerformanceMonitor
from src.models.ab_testing import ABTestingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = ["negative", "positive"]
        
        self.model_registry = ModelRegistry()
        self.model_monitor = ModelPerformanceMonitor()
        self.ab_testing = ABTestingManager()
        self.current_model_id = None

    def load_model(self, model_path: str):
        """Load model from path or use pretrained model."""
        try:
            logger.info(f"Loading model from {model_path}")
            start_time = self.model_monitor.start_prediction()

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            except:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            self.model.to(self.device)
            self.model.eval()

            load_time = self.model_monitor.end_prediction(start_time, "model_loading")

            metrics = {
                "load_time": load_time,
                "device": str(self.device)
            }

            self.current_model_id = self.model_registry.register_model(
                model_path=model_path,
                model_name="sentiment_analyzer",
                version=self._get_model_version(model_path),
                metrics=metrics,
                description="Sentiment analysis model based on DistilBERT"
            )

            logger.info(f"Model loaded successfully. Model ID: {self.current_model_id}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_monitor.record_error("model_loading", str(type(e).__name__))
            raise

    def _get_model_version(self, model_path: str) -> str:
        """Extract version from model path or generate a default one."""
        try:
            dir_name = os.path.basename(os.path.dirname(model_path))
            if dir_name.startswith("sentiment_model_"):
                return dir_name.split("_")[-1]
        except:
            pass
        return "1.0.0"

    def predict(self, text: str) -> PredictionResponse:
        if not text.strip():
            raise ValueError("Empty text provided")
            
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        start_time = self.model_monitor.start_prediction()

        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            probs = probabilities[0].cpu().numpy()
            pred_class = int(torch.argmax(probabilities, dim=1)[0])
            confidence = float(probs[pred_class])
            sentiment = self.labels[pred_class]

            response = PredictionResponse(
                sentiment=sentiment,
                confidence=confidence,
                probabilities={label: float(prob) for label, prob in zip(self.labels, probs)}
            )

            self.model_monitor.end_prediction(start_time, self.current_model_id)
            self.model_monitor.record_prediction(
                prediction=sentiment,
                confidence=confidence,
                model_version=self.current_model_id
            )

            return response

        except Exception as e:
            self.model_monitor.record_error(self.current_model_id, str(type(e).__name__))
            raise

    def predict_batch(self, texts: List[str]) -> List[PredictionResponse]:
        """Make batch predictions with monitoring."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Please load the model first.")

        start_time = self.model_monitor.start_prediction()
        responses = []

        try:
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(batch_texts, 
                                      return_tensors="pt", 
                                      truncation=True, 
                                      max_length=512, 
                                      padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

                batch_probs = probabilities.cpu().numpy()
                pred_classes = torch.argmax(probabilities, dim=1).cpu().numpy()

                for j in range(len(batch_texts)):
                    probs = batch_probs[j]
                    pred_class = pred_classes[j]
                    sentiment = self.labels[pred_class]
                    confidence = float(probs[pred_class])

                    response = PredictionResponse(
                        sentiment=sentiment,
                        confidence=confidence,
                        probabilities={label: float(prob) for label, prob in zip(self.labels, probs)}
                    )
                    responses.append(response)

                    self.model_monitor.record_prediction(
                        prediction=sentiment,
                        confidence=confidence,
                        model_version=self.current_model_id
                    )

            self.model_monitor.end_prediction(start_time, self.current_model_id)
            
            return responses

        except Exception as e:
            self.model_monitor.record_error(self.current_model_id, str(type(e).__name__))
            logger.error(f"Error during batch prediction: {str(e)}")
            raise

    def get_model_performance(self) -> Dict:
        """Get model performance metrics."""
        return self.model_monitor.get_performance_metrics(self.current_model_id)

    def get_model_info(self) -> Dict:
        """Get current model information."""
        if not self.current_model_id:
            return {"status": "No model loaded"}
            
        return self.model_registry.get_model(self.current_model_id) or {
            "name": "sentiment_analyzer",
            "version": "1.0",
            "status": "loaded",
            "model_id": self.current_model_id
        }
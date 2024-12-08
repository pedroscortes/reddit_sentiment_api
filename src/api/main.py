# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
import numpy as np
from transformers import AutoTokenizer
import logging
import os
from src.models.training import SentimentClassifier
from sklearn.preprocessing import StandardScaler
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)

class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class ModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.scaler = None
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
    def load_model(self, model_path: str):
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = SentimentClassifier.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load scaler if available
            scaler_path = os.path.join(model_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, text: str) -> PredictionResponse:
        """Make a single prediction."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
            # Create dummy numerical features
            numerical_features = torch.zeros((1, 1006)).to(self.device)  # Adjust size based on your model
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs, numerical_features=numerical_features)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1)
                confidence = torch.max(probabilities).item()
            
            # Convert probabilities to dict
            prob_dict = {
                self.id2label[i]: prob.item()
                for i, prob in enumerate(probabilities[0])
            }
            
            return PredictionResponse(
                sentiment=self.id2label[prediction.item()],
                confidence=confidence,
                probabilities=prob_dict
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def predict_batch(self, texts: List[str]) -> List[PredictionResponse]:
        """Make batch predictions."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
            
        try:
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            
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
                prob_dict = {
                    self.id2label[i]: prob.item()
                    for i, prob in enumerate(probs)
                }
                responses.append(
                    PredictionResponse(
                        sentiment=self.id2label[pred.item()],
                        confidence=conf.item(),
                        probabilities=prob_dict
                    )
                )
            
            return responses
            
        except Exception as e:
            logger.error(f"Error making batch prediction: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Reddit Sentiment Analysis API",
    description="API for predicting sentiment in text using a fine-tuned BERT model",
    version="1.0.0"
)

# Initialize model service
model_service = ModelService()

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        # Get latest model
        model_dirs = [d for d in os.listdir('models') if d.startswith('sentiment_model_')]
        latest_model = max(model_dirs)
        model_path = os.path.join('models', latest_model, 'final_model')
        
        # Load model
        model_service.load_model(model_path)
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise RuntimeError(f"Failed to start application: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Reddit Sentiment Analysis API",
        "version": "1.0.0",
        "status": "active"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: TextInput):
    """Predict sentiment for a single text."""
    try:
        return model_service.predict(input_data.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchInput):
    """Predict sentiment for multiple texts."""
    try:
        predictions = model_service.predict_batch(input_data.texts)
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
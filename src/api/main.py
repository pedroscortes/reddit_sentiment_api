# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import os
from dotenv import load_dotenv
from .model_service import ModelService
from .reddit_analyzer import RedditAnalyzer
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from src.monitoring.metrics import MetricsTracker
from src.monitoring.metrics_manager import metrics_manager
import psutil
import time
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge
from starlette.responses import Response


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


class SubredditRequest(BaseModel):
    subreddit: str
    time_filter: str = Field(default="week", pattern="^(hour|day|week|month|year|all)$")
    post_limit: int = Field(default=100, ge=1, le=500)


class RedditURLRequest(BaseModel):
    url: str = Field(..., pattern=r"^https?://(?:www\.)?reddit\.com/.*$")


class UserRequest(BaseModel):
    username: str
    limit: int = Field(default=50, ge=1, le=200)


class TrendRequest(BaseModel):
    keyword: str = Field(..., min_length=1, max_length=100)
    subreddits: List[str] = Field(..., min_items=1, max_items=5)
    time_filter: str = Field(default="week", pattern="^(hour|day|week|month|year|all)$")
    limit: int = Field(default=100, ge=1, le=500)


# Initialize FastAPI app
app = FastAPI(title="Reddit Sentiment Analysis API")

# Initialize services
model_service = ModelService()
reddit_analyzer = None
metrics = MetricsTracker()


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(metrics_manager.registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/monitoring/metrics")
async def monitoring_metrics():
    """Get current monitoring metrics."""
    return metrics_manager.get_metrics()


@app.on_event("startup")
async def startup_event():
    """Load model and initialize services on startup."""
    global reddit_analyzer
    try:
        # Get latest model
        model_dirs = [d for d in os.listdir("models") if d.startswith("sentiment_model_")]
        latest_model = max(model_dirs)
        model_path = os.path.join("models", latest_model, "final_model")

        # Load model
        model_service.load_model(model_path)

        # Initialize Reddit analyzer
        reddit_analyzer = RedditAnalyzer(model_service)

        # Load environment variables
        load_dotenv()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise RuntimeError(f"Failed to start application: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {"message": "Reddit Sentiment Analysis API", "version": "1.0.0", "status": "active"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model_service.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict")
async def predict(input_data: TextInput):
    """Predict sentiment for a single text."""
    start_time = time.time()

    try:
        metrics_manager.track_request("/predict", "POST")
        result = model_service.predict(input_data.text)
        metrics_manager.track_prediction(result.sentiment)

        duration = time.time() - start_time
        metrics_manager.track_latency("/predict", duration)

        return result
    except Exception as e:
        metrics_manager.track_latency("/predict_error", time.time() - start_time)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchInput):
    """Predict sentiment for multiple texts."""
    try:
        predictions = model_service.predict_batch(input_data.texts)
        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/subreddit")
async def analyze_subreddit(request: SubredditRequest):
    """Analyze sentiment patterns in a subreddit."""
    try:
        return await reddit_analyzer.analyze_subreddit(request.subreddit, request.time_filter, request.post_limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/url")
async def analyze_url(request: RedditURLRequest):
    """Analyze sentiment from a Reddit URL."""
    try:
        return await reddit_analyzer.analyze_url(request.url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/user")
async def analyze_user(request: UserRequest):
    """Analyze sentiment patterns of a user's comments."""
    try:
        return await reddit_analyzer.analyze_user(request.username, request.limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/trend")
async def analyze_trend(request: TrendRequest):
    """Analyze sentiment trends around specific keywords."""
    try:
        logger.info(f"Received trend analysis request for keyword: {request.keyword} " f"in subreddits: {request.subreddits}")

        result = await reddit_analyzer.analyze_trend(request.keyword, request.subreddits, request.time_filter, request.limit)
        return result
    except Exception as e:
        logger.error(f"Error in analyze_trend endpoint: {str(e)}", exc_info=True)
        # Return a more detailed error response
        return JSONResponse(
            status_code=500, content={"error": "Internal Server Error", "detail": str(e), "path": "/analyze/trend"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# Reddit Sentiment Analysis API

A production-ready API service that analyzes sentiment in Reddit content using state-of-the-art NLP models with comprehensive MLOps infrastructure.

## 🌟 Features

- **Real-time Sentiment Analysis**
  - Single text prediction
  - Batch prediction support
  - Subreddit analysis
  - User comment analysis
  - URL-based analysis
  - Cross-subreddit trend analysis

- **Advanced ML Pipeline**
  - BERT-based sentiment classifier
  - Custom feature engineering
  - MLflow experiment tracking
  - Model versioning and registry
  - Automated model evaluation

- **Production-Ready Infrastructure**
  - FastAPI service
  - Docker containerization
  - Prometheus metrics
  - Grafana dashboards
  - Comprehensive testing
  - Extensive monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.12
- Docker & Docker Compose
- Reddit API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/pedroscortes/sentiment-analysis-api.git
cd sentiment-analysis-api
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your Reddit API credentials and other settings
```

### Running with Docker

```bash
docker-compose up -d
```

The API will be available at `http://localhost:8000`

### Running Locally

1. Start the API service:
```bash
python -m src.api.main
```

2. Visit `http://localhost:8000/docs` for the Swagger UI documentation

## 📊 Monitoring

- Metrics dashboard: `http://localhost:3000` (Grafana)
- Prometheus metrics: `http://localhost:9090`

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## 🔧 API Endpoints

### Sentiment Analysis
- `POST /api/v1/analyze/text` - Analyze single text
- `POST /api/v1/analyze/batch` - Batch text analysis
- `POST /api/v1/analyze/subreddit` - Analyze subreddit
- `POST /api/v1/analyze/user` - Analyze user comments
- `POST /api/v1/analyze/url` - Analyze URL content
- `GET /api/v1/analyze/trends` - Get sentiment trends

### Monitoring
- `GET /health` - Service health check
- `GET /metrics` - Prometheus metrics

## 📁 Project Structure

```
sentiment-analysis-api/
├── src/
│   ├── api/             # FastAPI service
│   ├── data/            # Data processing
│   ├── models/          # ML models
│   └── monitoring/      # Metrics & monitoring
├── tests/               # Test suites
├── docker/              # Docker configurations
├── notebooks/           # Development notebooks
└── mlruns/             # MLflow experiments
```

## 📈 Monitoring & Metrics

- Request metrics
  - Request count by endpoint
  - Latency distribution
  - Error rates
- Model metrics
  - Sentiment distribution
  - Confidence scores
  - Prediction timings
- System metrics
  - Memory usage
  - CPU utilization
  - Model load times

## 🛠️ Tech Stack

- **API Framework**: FastAPI
- **ML Framework**: PyTorch, Transformers
- **Data Processing**: NLTK
- **Monitoring**: Prometheus, Grafana
- **Experiment Tracking**: MLflow
- **Testing**: pytest
- **Container**: Docker
- **Documentation**: FastAPI Swagger UI

Feel free for opening PRs or giving me any suggestions!

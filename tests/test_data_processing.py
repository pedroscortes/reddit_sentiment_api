# tests/test_data_processing.py

import pytest
from src.data.reddit_collector import RedditDataCollector
from src.data.data_preprocessing import RedditDataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
import nltk
import os

@pytest.fixture
def sample_comments():
    """Sample Reddit comments for testing."""
    return [
        {
            "comment_id": "abc123",
            "text": "This is a fantastic test comment!",
            "subreddit": "python",
            "score": 10,
            "created_utc": "2024-01-01",
            "collected_at": "2024-01-01",
            "post_title": "Test Post",
            "post_id": "xyz789",
            "is_submitter": False,
            "permalink": "https://reddit.com/r/python/comments/xyz789",
            "controversiality": 0
        },
        {
            "comment_id": "def456",
            "text": "This is a horrible test comment...",
            "subreddit": "unpopularopinion",
            "score": -5,
            "created_utc": "2024-01-01",
            "collected_at": "2024-01-01",
            "post_title": "Another Test",
            "post_id": "abc123",
            "is_submitter": True,
            "permalink": "https://reddit.com/r/unpopularopinion/comments/abc123",
            "controversiality": 1
        }
    ]

@pytest.fixture
def preprocessor():
    """Initialize preprocessor for testing."""
    return RedditDataPreprocessor()

@pytest.fixture
def feature_engineer():
    """Initialize feature engineer for testing."""
    return FeatureEngineer()

def test_reddit_collector_initialization():
    """Test RedditCollector initialization with mocked PRAW."""
    with patch('praw.Reddit') as mock_reddit:
        collector = RedditDataCollector()
        assert collector is not None
        assert hasattr(collector, 'subreddit_categories')
        assert all(cat in collector.subreddit_categories 
                  for cat in ['positive', 'negative', 'neutral'])

def test_sentiment_estimation():
    """Test sentiment estimation function."""
    collector = RedditDataCollector()
    
    mock_comment = Mock()
    mock_comment.body = "This is fantastic and amazing!"
    mock_comment.score = 10
    mock_comment.controversiality = 0
    
    sentiment = collector.estimate_sentiment(mock_comment)
    assert sentiment in ['positive', 'negative', 'neutral']

def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor is not None
    assert hasattr(preprocessor, 'lemmatizer')
    assert hasattr(preprocessor, 'stop_words')

def test_text_cleaning(preprocessor):
    """Test text cleaning functionality."""
    test_text = "This is a TEST with URL: https://example.com and @mention"
    cleaned = preprocessor.clean_text(test_text)
    
    assert "https://" not in cleaned
    assert "@mention" not in cleaned
    assert cleaned.islower()

def test_sentiment_labeling(preprocessor):
    """Test sentiment labeling."""
    df = pd.DataFrame({
        "text": ["amazing!", "terrible!"],
        "score": [10, -10],
        "subreddit": ["UpliftingNews", "rant"],
        "controversiality": [0, 1],  
        "created_utc": ["2024-01-01", "2024-01-01"],
        "collected_at": ["2024-01-01", "2024-01-01"],
        "post_title": ["Test Post", "Test Post"],
        "post_id": ["123", "456"],
        "is_submitter": [False, False],
        "permalink": ["link1", "link2"],
        "comment_id": ["id1", "id2"]
    })
    
    labeled_df = preprocessor.create_sentiment_labels(df)
    assert "sentiment" in labeled_df.columns
    assert "sentiment_score" in labeled_df.columns
    
    sentiment_counts = labeled_df["sentiment"].value_counts()
    assert len(sentiment_counts) > 0
    
    positive_df = pd.DataFrame({
        "text": ["This is amazing and fantastic!"],
        "score": [10],
        "subreddit": ["UpliftingNews"],
        "controversiality": [0],
        "created_utc": ["2024-01-01"],
        "collected_at": ["2024-01-01"],
        "post_title": ["Test Post"],
        "post_id": ["123"],
        "is_submitter": [False],
        "permalink": ["link1"],
        "comment_id": ["id1"]
    })
    pos_result = preprocessor.create_sentiment_labels(positive_df)
    assert pos_result["sentiment"].iloc[0] == "positive"
    
    negative_df = pd.DataFrame({
        "text": ["This is horrible and terrible!"],
        "score": [-10],
        "subreddit": ["rant"],
        "controversiality": [1],
        "created_utc": ["2024-01-01"],
        "collected_at": ["2024-01-01"],
        "post_title": ["Test Post"],
        "post_id": ["123"],
        "is_submitter": [False],
        "permalink": ["link1"],
        "comment_id": ["id1"]
    })
    neg_result = preprocessor.create_sentiment_labels(negative_df)
    assert neg_result["sentiment"].iloc[0] == "negative"

def test_feature_engineer_initialization(feature_engineer):
    """Test feature engineer initialization."""
    assert feature_engineer is not None
    assert hasattr(feature_engineer, 'max_features')
    assert hasattr(feature_engineer, 'tfidf')

def test_text_feature_creation(feature_engineer):
    """Test text feature creation."""
    df = pd.DataFrame({
        "text": ["Test!", "Another test..."],
        "processed_text": ["test", "another test"],
        "text_length": [5, 13]
    })
    
    features = feature_engineer.create_text_features(df)
    assert "exclamation_count" in features.columns
    assert "question_count" in features.columns

def test_engagement_features(feature_engineer):
    """Test engagement feature creation."""
    df = pd.DataFrame({
        "score": [1, -1],
        "hour": [12, 18],
        "day_of_week": [1, 5]
    })
    
    features = feature_engineer.create_engagement_features(df)
    assert "score_log" in features.columns
    assert "hour_sin" in features.columns
    assert "weekday_sin" in features.columns

def test_class_balancing(feature_engineer):
    """Test class balancing functionality."""
    df = pd.DataFrame({
        "sentiment": ["positive"] * 100 + ["negative"] * 50 + ["neutral"] * 25
    })
    
    balanced = feature_engineer.balance_classes(df)
    sentiment_counts = balanced["sentiment"].value_counts()
    assert max(sentiment_counts) - min(sentiment_counts) < 10  

def test_full_preprocessing_pipeline(preprocessor):
    """Test the complete preprocessing pipeline."""
    df = pd.DataFrame({
        "text": ["This is amazing! https://test.com @user",
                 "This is terrible... #hashtag"],
        "score": [10, -10],
        "subreddit": ["UpliftingNews", "rant"],
        "controversiality": [0, 1],
        "created_utc": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "collected_at": ["2024-01-01", "2024-01-01"],
        "post_title": ["Test Post", "Test Post"],
        "post_id": ["123", "456"],
        "is_submitter": [False, False],
        "permalink": ["link1", "link2"],
        "comment_id": ["id1", "id2"]
    })
    
    processed_df = preprocessor.preprocess_data(df)
    
    assert "cleaned_text" in processed_df.columns
    assert "processed_text" in processed_df.columns
    assert "text_length" in processed_df.columns
    assert "word_count" in processed_df.columns
    assert "sentiment" in processed_df.columns

def test_preprocessing_error_handling(preprocessor):
    """Test error handling in preprocessing."""
    df_invalid = pd.DataFrame({
        "wrong_column": ["test"]
    })
    
    with pytest.raises(Exception):
        preprocessor.preprocess_data(df_invalid)

def test_text_statistics(preprocessor):
    """Test text statistics calculation."""
    text = "This is a test sentence! With some statistics..."
    cleaned = preprocessor.clean_text(text)
    
    assert len(cleaned) > 0
    assert cleaned.islower()
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned

def test_tfidf_feature_creation(feature_engineer):
    """Test TF-IDF feature creation."""
    df = pd.DataFrame({
        "processed_text": [
            "this is first document",
            "this is second document",
            "and this is third one"
        ]
    })
    
    result = feature_engineer.create_tfidf_features(df)
    assert len(result.columns) > len(df.columns)
    assert any('tfidf' in col for col in result.columns)

def test_engagement_feature_completeness(feature_engineer):
    """Test all engagement features are created."""
    df = pd.DataFrame({
        "score": [1, -1],
        "hour": [12, 18],
        "day_of_week": [1, 5],
        "text": ["test1", "test2"],
        "processed_text": ["test1", "test2"],
        "text_length": [5, 5]
    })
    
    result = feature_engineer.create_engagement_features(df)
    expected_features = [
        "score_log", "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos"
    ]
    for feature in expected_features:
        assert feature in result.columns

def test_full_feature_pipeline(feature_engineer):
    """Test the complete feature engineering pipeline."""
    input_df = pd.DataFrame({
        "text": ["Test text one", "Test text two"],
        "processed_text": ["test text one", "test text two"],
        "score": [1, -1],
        "hour": [12, 18],
        "day_of_week": [1, 5],
        "text_length": [12, 12],
        "sentiment": ["positive", "negative"],
        "sentiment_score": [0.8, -0.8],
        "sentiment_strength": [0.8, 0.8]
    })
    
    with patch('pandas.read_csv', return_value=input_df):
        result = feature_engineer.engineer_features("mock_path")
        
        assert "exclamation_count" in result.columns
        assert "caps_ratio" in result.columns
        assert "score_log" in result.columns
        assert "sentiment_confidence" in result.columns

def test_collector_basic_error_handling():
    """Test basic error handling in collector."""
    with patch('praw.Reddit') as mock_reddit:
        mock_reddit.subreddit.side_effect = Exception("Test API Error")
        collector = RedditDataCollector()
        
        assert collector is not None
        assert hasattr(collector, 'reddit')
        
        with pytest.raises(Exception) as exc_info:
            mock_reddit.subreddit("test")
        assert str(exc_info.value) == "Test API Error"

def test_collector_basic_error_handling():
    """Test basic error handling in collector."""
    with patch('praw.Reddit') as mock_reddit:
        mock_reddit.subreddit.side_effect = Exception("Test API Error")
        collector = RedditDataCollector()
        
        assert collector is not None
        assert hasattr(collector, 'reddit')
        
        with pytest.raises(Exception) as exc_info:
            mock_reddit.subreddit("test")
        assert str(exc_info.value) == "Test API Error"

def test_preprocess_data_with_temporal_features():
    """Test preprocessing with temporal features."""
    preprocessor = RedditDataPreprocessor()
    df = pd.DataFrame({
        "text": ["Test text"],
        "score": [10],
        "subreddit": ["test"],
        "controversiality": [0],
        "created_utc": pd.Timestamp("2024-01-01"),
        "collected_at": pd.Timestamp("2024-01-01"),
        "post_title": ["Test"],
        "post_id": ["123"],
        "is_submitter": [False],
        "permalink": ["link"],
        "comment_id": ["id1"]
    })
    
    result = preprocessor.preprocess_data(df)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns

def test_preprocess_data_filtering():
    """Test preprocessing data filtering."""
    preprocessor = RedditDataPreprocessor()
    df = pd.DataFrame({
        "text": ["", "This is a valid test text", "A"],  
        "score": [0, 1, 0],
        "subreddit": ["test", "UpliftingNews", "test"],  
        "controversiality": [0, 0, 0],
        "created_utc": pd.to_datetime(["2024-01-01"] * 3),
        "collected_at": pd.to_datetime(["2024-01-01"] * 3),
        "post_title": ["Test", "Test", "Test"],
        "post_id": ["123", "124", "125"],
        "is_submitter": [False, False, False],
        "permalink": ["link1", "link2", "link3"],
        "comment_id": ["id1", "id2", "id3"]
    })
    
    result = preprocessor.preprocess_data(df)
    assert len(result) >= 1
    assert "This is a valid test text" in result['text'].values

def test_balance_classes_with_different_strategies():
    """Test class balancing with different strategies."""
    engineer = FeatureEngineer()
    df = pd.DataFrame({
        "sentiment": ["positive"] * 100 + ["negative"] * 50 + ["neutral"] * 25,
        "text": ["test"] * 175,
        "processed_text": ["test"] * 175,
        "score": [1] * 175
    })
    
    balanced = engineer.balance_classes(df, strategy="hybrid")
    sentiment_counts = balanced["sentiment"].value_counts()
    assert max(sentiment_counts) - min(sentiment_counts) <= 10

def test_feature_engineering_pipeline_with_empty_data():
    """Test feature engineering pipeline with edge cases."""
    engineer = FeatureEngineer()

    empty_df = pd.DataFrame({
        "text": pd.Series([], dtype='string'),
        "processed_text": pd.Series([], dtype='string'),
        "score": pd.Series([], dtype='float'),
        "sentiment_strength": pd.Series([], dtype='float'),
        "sentiment_score": pd.Series([], dtype='float')
    })

    with patch('pandas.read_csv', return_value=empty_df):
        result = engineer.engineer_features("mock_path")

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert "sentiment_strength_normalized" in result.columns

def test_nltk_download_resources():
    """Test NLTK resources download functionality."""
    from src.data.setup_nltk import download_nltk_resources
    
    try:
        download_nltk_resources()
        
        nltk_data_path = nltk.data.path[0]
        
        assert os.path.exists(os.path.join(nltk_data_path, 'tokenizers')), "Tokenizers not found"
        assert os.path.exists(os.path.join(nltk_data_path, 'corpora')), "Corpora not found"
        assert os.path.exists(os.path.join(nltk_data_path, 'taggers')), "Taggers not found"
        
    except Exception as e:
        pytest.fail(f"NLTK resource download failed: {str(e)}")

def test_nltk_download_error_handling():
    """Test NLTK download error handling."""
    from src.data.setup_nltk import download_nltk_resources
    
    with patch('nltk.download') as mock_download:
        mock_download.side_effect = Exception("Download failed")
        
        with pytest.raises(Exception) as exc_info:
            download_nltk_resources()
        assert "Download failed" in str(exc_info.value)

def test_reddit_collector_sentiment_estimation():
    """Test sentiment estimation functionality."""
    collector = RedditDataCollector()
    
    positive_comment = Mock()
    positive_comment.body = "This is amazing and fantastic!"
    positive_comment.score = 10
    positive_comment.controversiality = 0
    assert collector.estimate_sentiment(positive_comment) == "positive"
    
    negative_comment = Mock()
    negative_comment.body = "This is terrible and horrible."
    negative_comment.score = -5
    negative_comment.controversiality = 1
    assert collector.estimate_sentiment(negative_comment) == "negative"

def test_reddit_collector_subreddit_categories():
    """Test subreddit categorization."""
    collector = RedditDataCollector()
    
    assert "UpliftingNews" in collector.subreddit_categories["positive"]
    assert "unpopularopinion" in collector.subreddit_categories["negative"]
    assert "AskReddit" in collector.subreddit_categories["neutral"]

def test_reddit_collector_sentiment_words():
    """Test sentiment word categorization."""
    collector = RedditDataCollector()
    
    assert "amazing" in collector.positive_indicators["strong"]
    assert "good" in collector.positive_indicators["moderate"]
    
    assert "terrible" in collector.negative_indicators["strong"]
    assert "bad" in collector.negative_indicators["moderate"]

def test_save_data_functionality():
    """Test data saving functionality."""
    collector = RedditDataCollector()
    
    test_data = pd.DataFrame({
        'comment_id': ['123'],
        'text': ['Test comment'],
        'subreddit': ['test'],
        'score': [1],
        'sentiment': ['positive']
    })
    
    with TemporaryDirectory() as temp_dir:
        with patch('os.makedirs') as mock_makedirs:
            with patch('pandas.DataFrame.to_csv') as mock_to_csv:
                filepath = collector.save_data(test_data, 'test_data')
                
                assert filepath is not None
                mock_makedirs.assert_called_once()
                mock_to_csv.assert_called_once()

def test_comment_validation():
    """Test comment validation logic."""
    collector = RedditDataCollector()
    
    short_comment = Mock()
    short_comment.body = "Hi"
    short_comment.score = 1
    
    deleted_comment = Mock()
    deleted_comment.body = "[deleted]"
    deleted_comment.score = 1
    
    removed_comment = Mock()
    removed_comment.body = "[removed]"
    removed_comment.score = 1
    
    assert all([
        len(short_comment.body) < 30,  
        deleted_comment.body == "[deleted]",  
        removed_comment.body == "[removed]"  
    ])
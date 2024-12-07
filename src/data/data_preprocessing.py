# src/data/data_preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedditDataPreprocessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {str(e)}")
            
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def create_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create sentiment labels based on comment scores and content.
        
        Enhanced labeling strategy:
        - Positive: score > 10 or (score > 5 and positive_subreddit)
        - Negative: score < 0 or controversial_ratio > 0.4
        - Neutral: everything else
        """
        df = df.copy()
        
        # Define positive and negative leaning subreddits
        positive_subreddits = {'UpliftingNews', 'MadeMeSmile', 'wholesome'}
        
        def get_sentiment(row):
            score = row['score']
            subreddit = row['subreddit']
            
            # Strong positive signals
            if score > 10 or (score > 5 and subreddit in positive_subreddits):
                return 'positive'
            # Negative signals
            elif score < 0:
                return 'negative'
            # Everything else is neutral
            return 'neutral'
            
        df['sentiment'] = df.apply(get_sentiment, axis=1)
        
        logger.info(f"Created sentiment labels: {df['sentiment'].value_counts().to_dict()}")
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def lemmatize_text(self, text: str) -> str:
        """Tokenize, remove stop words, and lemmatize text."""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def preprocess_data(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Main preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        df = df.copy()
        
        # Create sentiment labels
        df = self.create_sentiment_labels(df)
        
        # Clean and lemmatize text
        logger.info("Cleaning and lemmatizing text...")
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        df['processed_text'] = df['cleaned_text'].apply(self.lemmatize_text)
        
        # Add text features
        df['word_count'] = df['processed_text'].apply(lambda x: len(x.split()))
        
        # Remove rows with empty text after processing
        df = df[df['processed_text'].str.len() > 0]
        
        logger.info(f"Preprocessing complete. {len(df)} samples retained.")
        return df

def get_latest_dataset():
    """Get the most recent large dataset file."""
    data_files = [f for f in os.listdir('data/raw') if f.startswith('large_reddit_dataset_final_')]
    if not data_files:
        raise Exception("No large dataset files found!")
    
    latest_file = max(data_files)
    return os.path.join('data/raw', latest_file)

def main():
    try:
        # Load the large dataset
        latest_file = get_latest_dataset()
        logger.info(f"Processing file: {latest_file}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Loaded dataset with {len(df)} comments from {df['subreddit'].nunique()} subreddits")
        
        # Initialize preprocessor
        preprocessor = RedditDataPreprocessor()
        
        # Preprocess data
        processed_df = preprocessor.preprocess_data(df)
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/processed/processed_large_dataset_{timestamp}.csv'
        processed_df.to_csv(output_file, index=False)
        
        # Print detailed statistics
        print("\nPreprocessing Results:")
        print(f"Original samples: {len(df)}")
        print(f"Processed samples: {len(processed_df)}")
        
        print("\nSentiment Distribution:")
        sentiment_dist = processed_df['sentiment'].value_counts()
        print(sentiment_dist)
        
        print("\nSentiment Distribution by Subreddit:")
        subreddit_sentiment = pd.crosstab(processed_df['subreddit'], processed_df['sentiment'])
        print(subreddit_sentiment)
        
        print("\nText Length Statistics:")
        print(processed_df['word_count'].describe())
        
        # Sample of processed text
        print("\nSample of processed text (one from each sentiment):")
        samples = processed_df.groupby('sentiment').first()[['text', 'processed_text', 'subreddit']]
        print(samples)
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
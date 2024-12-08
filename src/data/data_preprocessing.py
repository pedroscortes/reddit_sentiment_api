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
import emoji

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
        # Keep sentiment-bearing stopwords
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_stop_words = {'not', 'no', 'nor', 'neither', 'never', 'none'}
        self.stop_words = self.stop_words - self.sentiment_stop_words
        
        # Initialize sentiment word lists
        self.init_sentiment_words()
        
    def init_sentiment_words(self):
        """Initialize sentiment word dictionaries"""
        self.sentiment_words = {
            'positive_strong': {
                'amazing', 'excellent', 'fantastic', 'wonderful', 'brilliant',
                'outstanding', 'perfect', 'exceptional', 'incredible', 'awesome'
            },
            'positive_moderate': {
                'good', 'great', 'nice', 'happy', 'glad', 'pleased', 'enjoy',
                'helpful', 'appreciate', 'thanks', 'thank', 'interesting'
            },
            'negative_strong': {
                'horrible', 'terrible', 'awful', 'disgusting', 'pathetic',
                'disaster', 'hate', 'worst', 'garbage', 'useless'
            },
            'negative_moderate': {
                'bad', 'poor', 'disappointing', 'annoying', 'frustrated',
                'difficult', 'wrong', 'problem', 'issues', 'unfortunately'
            }
        }
    
    def create_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced sentiment labeling using multiple signals"""
        df = df.copy()
        
        # Define subreddit categories
        subreddit_categories = {
            'positive': {'UpliftingNews', 'MadeMeSmile', 'wholesome', 'happy', 'aww'},
            'negative': {'rant', 'complaints', 'unpopularopinion', 'TrueOffMyChest'},
            'neutral': {'AskReddit', 'explainlikeimfive', 'NoStupidQuestions'}
        }
        
        def calculate_sentiment_score(row):
            text = row['text'].lower()
            score = row['score']
            subreddit = row['subreddit']
            
            # Base score from comment karma
            sentiment_score = np.tanh(score / 10)  # Normalize score
            
            # Add subreddit bias
            if subreddit in subreddit_categories['positive']:
                sentiment_score += 0.5
            elif subreddit in subreddit_categories['negative']:
                sentiment_score -= 0.5
                
            # Add word-based sentiment
            for word in self.sentiment_words['positive_strong']:
                if word in text:
                    sentiment_score += 2
            for word in self.sentiment_words['positive_moderate']:
                if word in text:
                    sentiment_score += 1
            for word in self.sentiment_words['negative_strong']:
                if word in text:
                    sentiment_score -= 2
            for word in self.sentiment_words['negative_moderate']:
                if word in text:
                    sentiment_score -= 1
                    
            # Consider controversiality
            if row.get('controversiality', 0) > 0:
                sentiment_score -= 1
                
            return sentiment_score
        
        df['sentiment_score'] = df.apply(calculate_sentiment_score, axis=1)
        
        # Convert scores to labels
        df['sentiment'] = pd.cut(
            df['sentiment_score'],
            bins=[-np.inf, -1, 1, np.inf],
            labels=['negative', 'neutral', 'positive']
        )
        
        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        return df
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Handle contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove Reddit-specific patterns
        text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # Remove [deleted], [removed], etc.
        text = re.sub(r'/s|/S', '', text)     # Remove sarcasm tags
        text = re.sub(r'r/\w+', '', text)     # Remove subreddit references
        text = re.sub(r'u/\w+', '', text)     # Remove user references
        
        # Keep important punctuation for sentiment
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_data(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Enhanced preprocessing pipeline with error handling and comprehensive feature generation.
        
        Args:
            df (pd.DataFrame): Input dataframe with Reddit comments
            text_column (str): Name of the column containing text to process
            
        Returns:
            pd.DataFrame: Processed dataframe with additional features
        """
        logger.info("Starting data preprocessing...")
        
        df = df.copy()
        
        try:
            # Create sentiment labels
            df = self.create_sentiment_labels(df)
            
            # Clean and lemmatize text
            logger.info("Cleaning and lemmatizing text...")
            
            # Clean text
            df['cleaned_text'] = df[text_column].apply(self.clean_text)
            
            # Process text with error handling
            def process_text(text):
                try:
                    # Basic tokenization
                    tokens = text.split()
                    
                    # Remove stopwords and lemmatize
                    processed_tokens = [
                        self.lemmatizer.lemmatize(token.lower())
                        for token in tokens
                        if token.lower() not in self.stop_words
                    ]
                    
                    return ' '.join(processed_tokens)
                except Exception as e:
                    logger.warning(f"Error processing text: {str(e)}")
                    return text
            
            df['processed_text'] = df['cleaned_text'].apply(process_text)
            
            # Add text features
            logger.info("Generating text features...")
            
            # Basic text statistics
            df['text_length'] = df['cleaned_text'].str.len()
            df['word_count'] = df['processed_text'].str.split().str.len()
            df['avg_word_length'] = df['cleaned_text'].apply(
                lambda x: np.mean([len(word) for word in x.split()] or [0])
            )
            
            # Add temporal features if available
            if 'created_utc' in df.columns:
                logger.info("Adding temporal features...")
                df['created_utc'] = pd.to_datetime(df['created_utc'])
                df['hour'] = df['created_utc'].dt.hour
                df['day_of_week'] = df['created_utc'].dt.dayofweek
            
            # Add metadata features
            logger.info("Adding metadata features...")
            if 'score' in df.columns:
                df['score_normalized'] = df['score'].apply(lambda x: np.tanh(x/10))
            
            if 'controversiality' in df.columns:
                df['is_controversial'] = df['controversiality'].apply(lambda x: 1 if x > 0 else 0)
            
            # Calculate sentiment strength
            df['sentiment_strength'] = abs(df['sentiment_score'])
            
            # Remove empty or very short processed texts
            logger.info("Filtering out invalid samples...")
            initial_count = len(df)
            df = df[df['processed_text'].str.len() > 0]
            df = df[df['word_count'] >= 3]  # Remove extremely short comments
            
            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} samples with invalid or too short text")
            
            # Final statistics
            logger.info(f"Preprocessing complete. {len(df)} samples retained.")
            logger.info("\nText length statistics:")
            logger.info(df['text_length'].describe())
            logger.info("\nWord count statistics:")
            logger.info(df['word_count'].describe())
            
            # Verify sentiment distribution
            sentiment_dist = df['sentiment'].value_counts()
            logger.info("\nFinal sentiment distribution:")
            logger.info(sentiment_dist)
            
            return df
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

def main():
    try:
        # Load the dataset
        latest_file = os.path.join('data/raw', 'reddit_comments_balanced_20241207_195437.csv')
        logger.info(f"Processing file: {latest_file}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Loaded dataset with {len(df)} comments from {df['subreddit'].nunique()} subreddits")
        
        # Initialize and run preprocessor
        preprocessor = RedditDataPreprocessor()
        processed_df = preprocessor.preprocess_data(df)
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/processed/processed_reddit_comments_{timestamp}.csv'
        processed_df.to_csv(output_file, index=False)
        
        # Print statistics
        print("\nPreprocessing Results:")
        print(f"Original samples: {len(df)}")
        print(f"Processed samples: {len(processed_df)}")
        print("\nSentiment Distribution:")
        print(processed_df['sentiment'].value_counts())
        print("\nAverage Text Length by Sentiment:")
        print(processed_df.groupby('sentiment')['text_length'].mean())
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")

if __name__ == "__main__":
    main()
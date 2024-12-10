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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RedditDataPreprocessor:
    def __init__(self):
        try:
            nltk.download("punkt")
            nltk.download("stopwords")
            nltk.download("wordnet")
            nltk.download("averaged_perceptron_tagger")
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {str(e)}")

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_stop_words = {"not", "no", "nor", "neither", "never", "none"}
        self.stop_words = self.stop_words - self.sentiment_stop_words

        self.init_sentiment_words()

    def init_sentiment_words(self):
        """Initialize sentiment word dictionaries"""
        self.sentiment_words = {
            "positive_strong": {
                "amazing",
                "excellent",
                "fantastic",
                "wonderful",
                "brilliant",
                "outstanding",
                "perfect",
                "exceptional",
                "incredible",
                "awesome",
            },
            "positive_moderate": {
                "good",
                "great",
                "nice",
                "happy",
                "glad",
                "pleased",
                "enjoy",
                "helpful",
                "appreciate",
                "thanks",
                "thank",
                "interesting",
            },
            "negative_strong": {
                "horrible",
                "terrible",
                "awful",
                "disgusting",
                "pathetic",
                "disaster",
                "hate",
                "worst",
                "garbage",
                "useless",
            },
            "negative_moderate": {
                "bad",
                "poor",
                "disappointing",
                "annoying",
                "frustrated",
                "difficult",
                "wrong",
                "problem",
                "issues",
                "unfortunately",
            },
        }

    def create_sentiment_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced sentiment labeling using multiple signals"""
        df = df.copy()

        subreddit_categories = {
            "positive": {"UpliftingNews", "MadeMeSmile", "wholesome", "happy", "aww"},
            "negative": {"rant", "complaints", "unpopularopinion", "TrueOffMyChest"},
            "neutral": {"AskReddit", "explainlikeimfive", "NoStupidQuestions"},
        }

        def calculate_sentiment_score(row):
            text = row["text"].lower()
            score = row["score"]
            subreddit = row["subreddit"]

            sentiment_score = np.tanh(score / 10)  

            if subreddit in subreddit_categories["positive"]:
                sentiment_score += 0.5
            elif subreddit in subreddit_categories["negative"]:
                sentiment_score -= 0.5

            for word in self.sentiment_words["positive_strong"]:
                if word in text:
                    sentiment_score += 2
            for word in self.sentiment_words["positive_moderate"]:
                if word in text:
                    sentiment_score += 1
            for word in self.sentiment_words["negative_strong"]:
                if word in text:
                    sentiment_score -= 2
            for word in self.sentiment_words["negative_moderate"]:
                if word in text:
                    sentiment_score -= 1

            if row.get("controversiality", 0) > 0:
                sentiment_score -= 1

            return sentiment_score

        df["sentiment_score"] = df.apply(calculate_sentiment_score, axis=1)

        df["sentiment"] = pd.cut(
            df["sentiment_score"], bins=[-np.inf, -1, 1, np.inf], labels=["negative", "neutral", "positive"]
        )

        logger.info(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        return df

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not isinstance(text, str):
            return ""

        text = text.lower()

        text = emoji.demojize(text)

        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'d", " would", text)

        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        text = re.sub(r"\[.*?\]|\(.*?\)", "", text)  
        text = re.sub(r"/s|/S", "", text)  
        text = re.sub(r"r/\w+", "", text)  
        text = re.sub(r"u/\w+", "", text)  

        text = re.sub(r"[^\w\s!?.,]", " ", text)

        text = " ".join(text.split())

        return text

    def preprocess_data(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
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
            df = self.create_sentiment_labels(df)

            logger.info("Cleaning and lemmatizing text...")

            df["cleaned_text"] = df[text_column].apply(self.clean_text)

            def process_text(text):
                try:
                    tokens = text.split()

                    processed_tokens = [
                        self.lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in self.stop_words
                    ]

                    return " ".join(processed_tokens)
                except Exception as e:
                    logger.warning(f"Error processing text: {str(e)}")
                    return text

            df["processed_text"] = df["cleaned_text"].apply(process_text)

            logger.info("Generating text features...")

            df["text_length"] = df["cleaned_text"].str.len()
            df["word_count"] = df["processed_text"].str.split().str.len()
            df["avg_word_length"] = df["cleaned_text"].apply(lambda x: np.mean([len(word) for word in x.split()] or [0]))

            if "created_utc" in df.columns:
                logger.info("Adding temporal features...")
                df["created_utc"] = pd.to_datetime(df["created_utc"])
                df["hour"] = df["created_utc"].dt.hour
                df["day_of_week"] = df["created_utc"].dt.dayofweek

            logger.info("Adding metadata features...")
            if "score" in df.columns:
                df["score_normalized"] = df["score"].apply(lambda x: np.tanh(x / 10))

            if "controversiality" in df.columns:
                df["is_controversial"] = df["controversiality"].apply(lambda x: 1 if x > 0 else 0)

            df["sentiment_strength"] = abs(df["sentiment_score"])

            logger.info("Filtering out invalid samples...")
            initial_count = len(df)
            df = df[df["processed_text"].str.len() > 0]
            df = df[df["word_count"] >= 3] 

            removed_count = initial_count - len(df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} samples with invalid or too short text")

            logger.info(f"Preprocessing complete. {len(df)} samples retained.")
            logger.info("\nText length statistics:")
            logger.info(df["text_length"].describe())
            logger.info("\nWord count statistics:")
            logger.info(df["word_count"].describe())

            sentiment_dist = df["sentiment"].value_counts()
            logger.info("\nFinal sentiment distribution:")
            logger.info(sentiment_dist)

            return df

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise


def main():
    try:
        latest_file = os.path.join("data/raw", "reddit_comments_balanced_20241207_195437.csv")
        logger.info(f"Processing file: {latest_file}")

        df = pd.read_csv(latest_file)
        logger.info(f"Loaded dataset with {len(df)} comments from {df['subreddit'].nunique()} subreddits")

        preprocessor = RedditDataPreprocessor()
        processed_df = preprocessor.preprocess_data(df)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"data/processed/processed_reddit_comments_{timestamp}.csv"
        processed_df.to_csv(output_file, index=False)

        print("\nPreprocessing Results:")
        print(f"Original samples: {len(df)}")
        print(f"Processed samples: {len(processed_df)}")
        print("\nSentiment Distribution:")
        print(processed_df["sentiment"].value_counts())
        print("\nAverage Text Length by Sentiment:")
        print(processed_df.groupby("sentiment")["text_length"].mean())

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")


if __name__ == "__main__":
    main()

# src/data/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging
from typing import Tuple, List
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), stop_words="english")
        self.scaler = StandardScaler()

    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced text-based features."""
        logger.info("Creating text features...")

        df = df.copy()

        df["exclamation_count"] = df["text"].str.count("!")
        df["question_count"] = df["text"].str.count(r"\?")
        df["punctuation_count"] = df["text"].str.count("[.,!?;:]")

        df["caps_count"] = df["text"].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
        if "text_length" in df.columns:
            df["caps_ratio"] = df["caps_count"] / df["text_length"]
        else:
            df["caps_ratio"] = 0

        df["avg_word_length"] = df["processed_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

        return df

    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from engagement metrics."""
        logger.info("Creating engagement features...")

        df = df.copy()

        df["score_log"] = np.log1p(df["score"].abs()) * np.sign(df["score"])

        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        else:
            df["hour_sin"] = 0
            df["hour_cos"] = 0

        if "day_of_week" in df.columns:
            df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        else:
            df["weekday_sin"] = 0
            df["weekday_cos"] = 0

        return df

    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment-specific features."""
        logger.info("Creating sentiment features...")

        df = df.copy()
        
        if df.empty:
            df["sentiment_strength_normalized"] = pd.Series(dtype='float')
            return df
            
        df["sentiment_strength_normalized"] = self.scaler.fit_transform(df[["sentiment_strength"]])
        df["sentiment_confidence"] = abs(df["sentiment_score"])
        
        return df

    def create_tfidf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create TF-IDF features from processed text."""
        logger.info("Creating TF-IDF features...")

        if df.empty:
            return df

        tfidf_matrix = self.tfidf.fit_transform(df["processed_text"])

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])

        return pd.concat([df, tfidf_df], axis=1)

    def balance_classes(self, df: pd.DataFrame, strategy: str = "hybrid") -> pd.DataFrame:
        """Balance classes using specified strategy."""
        logger.info(f"Balancing classes using {strategy} strategy...")

        if strategy == "hybrid":
            if "sentiment" in df.columns:
                neutral_count = len(df[df["sentiment"] == "neutral"])
                positive_count = len(df[df["sentiment"] == "positive"])
                negative_count = len(df[df["sentiment"] == "negative"])

                target_count = int(np.mean([neutral_count, positive_count, negative_count]))

                balanced_dfs = []
                for sentiment in ["negative", "neutral", "positive"]:
                    sentiment_df = df[df["sentiment"] == sentiment]
                    if len(sentiment_df) > target_count:
                        balanced_df = sentiment_df.sample(n=target_count, random_state=42)
                    else:
                        balanced_df = sentiment_df.sample(n=target_count, replace=True, random_state=42)
                    balanced_dfs.append(balanced_df)

                return pd.concat(balanced_dfs, axis=0).reset_index(drop=True)
            else:
                return df

        return df

    def engineer_features(self, input_path: str, balance_strategy: str = "hybrid") -> pd.DataFrame:
        """Main feature engineering pipeline."""
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)

        df = self.create_text_features(df)
        df = self.create_engagement_features(df)
        df = self.create_sentiment_features(df)
        df = self.create_tfidf_features(df)

        df = self.balance_classes(df, strategy=balance_strategy)

        logger.info("\nFeature Engineering Results:")
        logger.info(f"Total features created: {len(df.columns)}")

        if "sentiment" in df.columns:
            logger.info("\nClass distribution after balancing:")
            logger.info(df["sentiment"].value_counts())
        else:
            logger.info("No 'sentiment' column found in the data.")

        return df


def main():
    try:
        processed_dir = "data/processed"
        latest_file = max([f for f in os.listdir(processed_dir) if f.startswith("processed_reddit_comments_")])
        input_path = os.path.join(processed_dir, latest_file)

        engineer = FeatureEngineer()

        df_engineered = engineer.engineer_features(input_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/processed/engineered_features_{timestamp}.csv"
        df_engineered.to_csv(output_path, index=False)

        logger.info(f"\nFeatures saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise


if __name__ == "__main__":
    main()

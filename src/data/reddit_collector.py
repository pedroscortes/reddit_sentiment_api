import praw
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple
import logging
from logging.handlers import RotatingFileHandler
import time


def setup_logging():
    os.makedirs("logs", exist_ok=True)

    file_handler = RotatingFileHandler(
        f'logs/collection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', maxBytes=10000000, backupCount=5
    )

    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()


class RedditDataCollector:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

        self.subreddit_categories = {
            "positive": [
                "UpliftingNews",
                "MadeMeSmile",
                "wholesome",
                "happy",
                "CasualConversation",
                "aww",
                "CongratsLikeImFive",
            ],
            "negative": [
                "unpopularopinion",
                "rant",
                "TrueOffMyChest",
                "AmItheAsshole",
                "relationship_advice",
                "complaints",
                "mildlyinfuriating",
                "ChoosingBeggars",
                "tifu",
                "pettyrevenge",
                "BestofRedditorUpdates",
                "relationship_advice",
                "JUSTNOMIL",
                "raisedbynarcissists",
                "talesfromretail",
                "entitledparents",
                "careerguidance",
            ],
            "neutral": [
                "AskReddit",
                "explainlikeimfive",
                "NoStupidQuestions",
                "todayilearned",
                "OutOfTheLoop",
                "movies",
                "gaming",
            ],
        }

        self.negative_indicators = {
            "strong": {
                "hate",
                "terrible",
                "awful",
                "horrible",
                "disgusting",
                "furious",
                "angry",
                "disappointed",
                "frustrating",
                "worst",
                "pathetic",
            },
            "moderate": {
                "bad",
                "poor",
                "annoying",
                "difficult",
                "upset",
                "worried",
                "concerned",
                "tired",
                "stressed",
                "disagree",
                "unfortunately",
            },
        }

        self.positive_indicators = {
            "strong": {
                "amazing",
                "excellent",
                "fantastic",
                "wonderful",
                "brilliant",
                "perfect",
                "incredible",
                "outstanding",
                "delighted",
                "blessed",
            },
            "moderate": {
                "good",
                "nice",
                "happy",
                "glad",
                "pleased",
                "thanks",
                "helpful",
                "appreciate",
                "enjoyed",
                "like",
                "cool",
            },
        }

    def estimate_sentiment(self, comment) -> str:
        """Enhanced sentiment estimation with weighted indicators"""
        text = comment.body.lower()
        words = set(text.split())

        sentiment_score = 0

        sentiment_score -= sum(2.0 for word in self.negative_indicators["strong"] if word in words)
        sentiment_score -= sum(1.0 for word in self.negative_indicators["moderate"] if word in words)

        sentiment_score += sum(2.0 for word in self.positive_indicators["strong"] if word in words)
        sentiment_score += sum(1.0 for word in self.positive_indicators["moderate"] if word in words)

        score_factor = min(max(comment.score, -10), 10) / 10
        sentiment_score += score_factor

        if hasattr(comment, "controversiality") and comment.controversiality > 0:
            sentiment_score -= 1.5

        if sentiment_score > 2:
            return "positive"
        elif sentiment_score < -1:  
            return "negative"
        return "neutral"

    def collect_balanced_comments(
        self, target_per_class: int = 5000, min_comment_length: int = 30, max_per_subreddit: int = 1000
    ) -> pd.DataFrame:
        """Collect balanced dataset with improved negative sentiment collection"""
        all_comments = []
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        subreddit_counts = {}
        start_time = time.time()

        for category in self.subreddit_categories.values():
            for subreddit in category:
                subreddit_counts[subreddit] = 0

        category_order = ["negative", "neutral", "positive"]

        for category in category_order:
            subreddits = self.subreddit_categories[category]
            logger.info(f"Collecting from {category} subreddits")

            while any(count < target_per_class for count in sentiment_counts.values()):
                for subreddit_name in subreddits:
                    if subreddit_counts[subreddit_name] >= max_per_subreddit:
                        continue

                    try:
                        subreddit = self.reddit.subreddit(subreddit_name)

                        if category == "negative":
                            sort_methods = ["controversial", "new", "hot"]
                        else:
                            sort_methods = ["hot", "new", "top"]

                        for sort_method in sort_methods:
                            try:
                                if sort_method == "hot":
                                    posts = subreddit.hot(limit=15)
                                elif sort_method == "controversial":
                                    posts = subreddit.controversial(time_filter="month", limit=15)
                                elif sort_method == "top":
                                    posts = subreddit.top(time_filter="month", limit=15)
                                else:
                                    posts = subreddit.new(limit=15)

                                for post in posts:
                                    if post.over_18:
                                        continue

                                    post.comments.replace_more(limit=0)

                                    for comment in post.comments.list():
                                        if (
                                            not hasattr(comment, "body")
                                            or len(comment.body) < min_comment_length
                                            or comment.body in ["[deleted]", "[removed]"]
                                        ):
                                            continue

                                        sentiment = self.estimate_sentiment(comment)

                                        if (
                                            sentiment_counts[sentiment] < target_per_class
                                            and subreddit_counts[subreddit_name] < max_per_subreddit
                                        ):
                                            all_comments.append(
                                                {
                                                    "comment_id": comment.id,
                                                    "text": comment.body,
                                                    "subreddit": subreddit_name,
                                                    "score": comment.score,
                                                    "sentiment": sentiment,
                                                    "created_utc": datetime.fromtimestamp(
                                                        comment.created_utc, tz=timezone.utc
                                                    ),
                                                    "collected_at": datetime.now(timezone.utc),
                                                    "post_title": post.title,
                                                    "post_id": post.id,
                                                    "is_submitter": comment.is_submitter,
                                                    "permalink": f"https://reddit.com{comment.permalink}",
                                                    "controversiality": getattr(comment, "controversiality", 0),
                                                }
                                            )

                                            sentiment_counts[sentiment] += 1
                                            subreddit_counts[subreddit_name] += 1

                                            if len(all_comments) % 100 == 0:
                                                elapsed = time.time() - start_time
                                                rate = len(all_comments) / elapsed
                                                logger.info(
                                                    f"Progress after {elapsed:.1f}s "
                                                    f"(Rate: {rate:.1f} comments/s) - "
                                                    f"Positive: {sentiment_counts['positive']}, "
                                                    f"Negative: {sentiment_counts['negative']}, "
                                                    f"Neutral: {sentiment_counts['neutral']}"
                                                )

                            except Exception as e:
                                logger.error(f"Error in {sort_method} posts for {subreddit_name}: {str(e)}")
                                continue

                            time.sleep(2)  

                    except Exception as e:
                        logger.error(f"Error accessing subreddit {subreddit_name}: {str(e)}")
                        continue

        df = pd.DataFrame(all_comments)

        if not df.empty:
            df["text_length"] = df["text"].str.len()
            df["word_count"] = df["text"].str.split().str.len()

            logger.info("\nCollection Complete!")
            logger.info(f"Total time: {time.time() - start_time:.1f}s")
            logger.info(f"Total samples: {len(df)}")
            logger.info("\nSentiment distribution:")
            logger.info(df["sentiment"].value_counts())
            logger.info("\nSubreddit distribution:")
            logger.info(df["subreddit"].value_counts())

        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        """Save the collected data to a CSV file with timestamp."""
        if df.empty:
            logger.warning("No data to save!")
            return

        os.makedirs("data/raw", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"data/raw/{filename}_{timestamp}.csv"

        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath


def main():
    collector = RedditDataCollector()

    df = collector.collect_balanced_comments(target_per_class=5000, min_comment_length=30, max_per_subreddit=1000)

    if not df.empty:
        collector.save_data(df, "reddit_comments_balanced")


if __name__ == "__main__":
    main()

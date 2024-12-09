# src/data/collect_large_dataset.py
import praw
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import logging
from typing import List, Dict
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LargeRedditCollector:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT"),
        )

    def collect_from_subreddit(self, subreddit_name: str, post_limit: int, comment_limit: int) -> List[Dict]:
        """Collect comments from a single subreddit with error handling and rate limiting."""
        comments = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            posts_processed = 0

            # Collect from different post categories for diversity
            for category in ["hot", "top", "new", "controversial"]:
                if len(comments) >= comment_limit:
                    break

                try:
                    if category == "top":
                        posts = subreddit.top(limit=post_limit, time_filter="month")
                    elif category == "controversial":
                        posts = subreddit.controversial(limit=post_limit, time_filter="month")
                    elif category == "new":
                        posts = subreddit.new(limit=post_limit)
                    else:
                        posts = subreddit.hot(limit=post_limit)

                    for post in posts:
                        if len(comments) >= comment_limit:
                            break

                        if post.over_18:  # Skip NSFW content
                            continue

                        try:
                            post.comments.replace_more(limit=0)
                            for comment in post.comments.list():
                                if len(comments) >= comment_limit:
                                    break

                                if (
                                    hasattr(comment, "body")
                                    and len(comment.body) > 20
                                    and comment.body != "[deleted]"
                                    and comment.body != "[removed]"
                                ):
                                    comment_data = {
                                        "comment_id": comment.id,
                                        "text": comment.body,
                                        "subreddit": subreddit_name,
                                        "score": comment.score,
                                        "created_utc": datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                                        "post_title": post.title,
                                        "post_id": post.id,
                                        "is_submitter": comment.is_submitter,
                                        "permalink": f"https://reddit.com{comment.permalink}",
                                        "category": category,
                                    }
                                    comments.append(comment_data)

                            posts_processed += 1
                            if posts_processed % 5 == 0:
                                logger.info(f"Processed {posts_processed} posts from r/{subreddit_name} ({category})")
                                time.sleep(1)  # Rate limiting

                        except Exception as e:
                            logger.warning(f"Error processing post in r/{subreddit_name}: {str(e)}")
                            continue

                except Exception as e:
                    logger.warning(f"Error processing category {category} in r/{subreddit_name}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error accessing subreddit {subreddit_name}: {str(e)}")

        return comments


def main():
    # Subreddits chosen for sentiment diversity
    subreddits_config = {
        # Technology and Science
        "technology": {"limit": 500},
        "science": {"limit": 500},
        "programming": {"limit": 500},
        # Entertainment
        "movies": {"limit": 500},
        "gaming": {"limit": 500},
        "music": {"limit": 500},
        # Discussion and Opinion
        "politics": {"limit": 500},
        "worldnews": {"limit": 500},
        "AskReddit": {"limit": 500},
        # Positive-leaning
        "UpliftingNews": {"limit": 500},
        "MadeMeSmile": {"limit": 500},
        "wholesome": {"limit": 500},
        # Critical Discussion
        "unpopularopinion": {"limit": 500},
        "changemyview": {"limit": 500},
        "TrueOffMyChest": {"limit": 500},
    }

    collector = LargeRedditCollector()
    all_comments = []

    for subreddit, config in subreddits_config.items():
        logger.info(f"Collecting from r/{subreddit}")
        comments = collector.collect_from_subreddit(
            subreddit_name=subreddit,
            post_limit=50,  # Posts per category (hot, top, new, controversial)
            comment_limit=config["limit"],
        )
        all_comments.extend(comments)
        logger.info(f"Collected {len(comments)} comments from r/{subreddit}")

        # Save intermediate results
        df = pd.DataFrame(all_comments)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = f"data/raw/large_reddit_dataset_intermediate_{timestamp}.csv"
        df.to_csv(intermediate_file, index=False)
        logger.info(f"Saved intermediate dataset with {len(df)} comments")

        time.sleep(2)  # Rate limiting between subreddits

    # Save final dataset
    final_df = pd.DataFrame(all_comments)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"data/raw/large_reddit_dataset_final_{timestamp}.csv"
    final_df.to_csv(final_file, index=False)

    logger.info(
        f"""
    Data Collection Complete:
    - Total comments collected: {len(final_df)}
    - Subreddits processed: {len(subreddits_config)}
    - Data saved to: {final_file}
    
    Comments per subreddit:
    {final_df['subreddit'].value_counts().to_dict()}
    """
    )


if __name__ == "__main__":
    main()

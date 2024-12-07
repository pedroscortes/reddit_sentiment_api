# src/data/reddit_collector.py
import praw
import pandas as pd
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedditDataCollector:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
    def collect_comments(self, 
                        subreddits: List[str], 
                        post_limit: int = 10,
                        comment_limit: int = 100,
                        min_comment_length: int = 20) -> pd.DataFrame:
        """
        Collect comments from specified subreddits with various filters.
        
        Args:
            subreddits: List of subreddit names to collect from
            post_limit: Number of posts to collect from each subreddit
            comment_limit: Maximum number of comments to collect per subreddit
            min_comment_length: Minimum length of comments to include
            
        Returns:
            DataFrame containing collected comments and metadata
        """
        all_comments = []
        
        for subreddit_name in subreddits:
            try:
                logger.info(f"Collecting from r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Collect from top posts
                for post in subreddit.hot(limit=post_limit):
                    try:
                        # Skip posts marked as NSFW
                        if post.over_18:
                            continue
                            
                        post.comments.replace_more(limit=0)  # Fetch all comments
                        
                        for comment in post.comments.list():
                            if len(all_comments) >= comment_limit:
                                break
                                
                            # Skip short or deleted comments
                            if (not hasattr(comment, 'body') or 
                                len(comment.body) < min_comment_length or 
                                comment.body == '[deleted]' or 
                                comment.body == '[removed]'):
                                continue
                            
                            comment_data = {
                                'comment_id': comment.id,
                                'text': comment.body,
                                'subreddit': subreddit_name,
                                'score': comment.score,
                                'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc),
                                'post_title': post.title,
                                'post_id': post.id,
                                'is_submitter': comment.is_submitter,
                                'permalink': f"https://reddit.com{comment.permalink}"
                            }
                            
                            all_comments.append(comment_data)
                            
                    except Exception as e:
                        logger.error(f"Error processing post: {str(e)}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error accessing subreddit {subreddit_name}: {str(e)}")
                continue
                
        df = pd.DataFrame(all_comments)
        
        if not df.empty:
            # Add basic sentiment indicators for potential labeling
            df['controversial_score'] = df['score'].apply(lambda x: 1 if x < 0 else 0)
            
            # Basic preprocessing
            df['text_length'] = df['text'].str.len()
            
            logger.info(f"Collected {len(df)} comments from {len(subreddits)} subreddits")
        else:
            logger.warning("No comments were collected!")
            
        return df
    
    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save the collected data to a CSV file with timestamp.
        
        Args:
            df: DataFrame to save
            filename: Base filename to use
        """
        if df.empty:
            logger.warning("No data to save!")
            return
            
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Add timestamp to filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"data/raw/{filename}_{timestamp}.csv"
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        return filepath

def main():
    # Example usage
    collector = RedditDataCollector()
    
    # List of subreddits to collect from
    subreddits = [
        'technology', 
        'politics',
        'movies',
        'games',
        'science'
    ]
    
    # Collect the data
    df = collector.collect_comments(
        subreddits=subreddits,
        post_limit=15,
        comment_limit=200,
        min_comment_length=30
    )
    
    # Save the collected data
    if not df.empty:
        collector.save_data(df, 'reddit_comments')

if __name__ == "__main__":
    main()
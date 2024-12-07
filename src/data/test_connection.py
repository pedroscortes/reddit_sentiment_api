import praw
from dotenv import load_dotenv
import os

load_dotenv()

def test_reddit_connection():
    try:

        reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        
        subreddit = reddit.subreddit("python")
        print(f"Successfully connected to Reddit API!")
        print(f"Accessing subreddit: r/{subreddit.display_name}")
        print(f"Title: {subreddit.title}")
        
        return True
    except Exception as e:
        print(f"Error connecting to Reddit API: {str(e)}")
        return False

if __name__ == "__main__":
    test_reddit_connection()
# src/data/test_collector.py
from reddit_collector import RedditDataCollector


def test_collection():
    collector = RedditDataCollector()

    test_subreddits = ["python"]

    df = collector.collect_comments(
        subreddits=test_subreddits, post_limit=5, comment_limit=50, min_comment_length=20  
    )

    if not df.empty:
        print(f"\nCollected {len(df)} comments")
        print("\nSample of collected data:")
        print(df[["subreddit", "text_length", "score"]].head())

        collector.save_data(df, "test_reddit_comments")
    else:
        print("No data was collected!")


if __name__ == "__main__":
    test_collection()

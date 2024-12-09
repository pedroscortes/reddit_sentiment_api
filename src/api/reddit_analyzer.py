# src/api/reddit_analyzer.py

import praw
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from urllib.parse import urlparse
from collections import Counter
import asyncio
from prawcore import NotFound
import logging
import os
from dotenv import load_dotenv
from .model_service import ModelService

# Set up logging
logger = logging.getLogger(__name__)

class RedditAnalyzer:
    def __init__(self, model_service: ModelService):
        # Ensure environment variables are loaded
        load_dotenv()
        
        # Get credentials from environment
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT")
        
        # Validate credentials
        if not all([client_id, client_secret, user_agent]):
            raise ValueError(
                "Missing Reddit API credentials. Please ensure REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set in .env file"
            )
        
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.model_service = model_service
        logger.info("Reddit Analyzer initialized successfully")

    async def analyze_subreddit(self, 
                              subreddit_name: str, 
                              time_filter: str = "week",
                              post_limit: int = 100) -> Dict:
        """Analyze sentiment trends in a subreddit."""
        try:
            logger.info(f"Analyzing subreddit: r/{subreddit_name}")
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Verify subreddit exists
            try:
                _ = subreddit.id
            except:
                raise ValueError(f"Subreddit r/{subreddit_name} not found")
            
            # Get posts based on time filter
            posts = subreddit.top(time_filter=time_filter, limit=post_limit)
            
            comments = []
            sentiment_stats = {"positive": 0, "neutral": 0, "negative": 0}
            total_comments = 0
            
            for post in posts:
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if hasattr(comment, 'body') and len(comment.body) > 20:
                        prediction = self.model_service.predict(comment.body)
                        sentiment_stats[prediction.sentiment] += 1
                        total_comments += 1
                        
                        comments.append({
                            "text": comment.body,
                            "sentiment": prediction.sentiment,
                            "confidence": prediction.confidence,
                            "score": comment.score,
                            "created_utc": datetime.fromtimestamp(comment.created_utc),
                            "post_title": post.title
                        })
            
            if total_comments == 0:
                return {
                    "subreddit": subreddit_name,
                    "message": "No comments found for analysis",
                    "total_comments_analyzed": 0
                }
            
            # Calculate sentiment distribution
            sentiment_distribution = {
                k: round(v/total_comments * 100, 2)
                for k, v in sentiment_stats.items()
            }
            
            # Time series analysis
            df = pd.DataFrame(comments)
            df.set_index('created_utc', inplace=True)
            
            # Daily sentiment trends
            daily_sentiment = df.resample('D')['sentiment'].value_counts().unstack().fillna(0)
            
            # Top posts analysis
            post_sentiment = df.groupby('post_title')['sentiment'].value_counts().unstack().fillna(0)
            
            return {
                "subreddit": subreddit_name,
                "analysis_period": time_filter,
                "total_comments_analyzed": total_comments,
                "sentiment_distribution": sentiment_distribution,
                "daily_trends": daily_sentiment.to_dict(),
                "top_posts_sentiment": post_sentiment.head().to_dict(),
                "sample_comments": [
                    {
                        "text": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
                        "sentiment": c["sentiment"],
                        "confidence": c["confidence"],
                        "score": c["score"]
                    }
                    for c in sorted(comments, key=lambda x: x["score"], reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing subreddit {subreddit_name}: {str(e)}")
            raise

    async def analyze_url(self, url: str) -> Dict:
        """Analyze sentiment from a Reddit URL."""
        try:
            logger.info(f"Analyzing URL: {url}")
            parsed_url = urlparse(url)
            
            if 'reddit.com' not in parsed_url.netloc:
                raise ValueError("Not a valid Reddit URL")
            
            # Extract post ID and optional comment ID
            path_parts = parsed_url.path.strip('/').split('/')
            
            if 'comments' in path_parts:
                post_id_index = path_parts.index('comments') + 1
                submission_id = path_parts[post_id_index]
                comment_id = path_parts[post_id_index + 2] if len(path_parts) > post_id_index + 2 else None
            else:
                raise ValueError("Invalid Reddit URL format")
            
            submission = self.reddit.submission(id=submission_id)
            
            # Analyze post content
            post_text = submission.selftext if submission.selftext else submission.title
            post_analysis = self.model_service.predict(post_text)
            
            comments_data = []
            
            if comment_id:
                # Analyze specific comment
                comment = self.reddit.comment(comment_id)
                comment_analysis = self.model_service.predict(comment.body)
                comments_data = [{
                    "text": comment.body,
                    "sentiment": comment_analysis.sentiment,
                    "confidence": comment_analysis.confidence,
                    "score": comment.score,
                    "is_op": comment.is_submitter
                }]
            else:
                # Analyze top comments
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]:
                    if hasattr(comment, 'body') and len(comment.body) > 20:
                        analysis = self.model_service.predict(comment.body)
                        comments_data.append({
                            "text": comment.body,
                            "sentiment": analysis.sentiment,
                            "confidence": analysis.confidence,
                            "score": comment.score,
                            "is_op": comment.is_submitter
                        })
            
            # Calculate overall sentiment
            sentiment_counts = Counter(c["sentiment"] for c in comments_data)
            total_comments = len(comments_data)
            
            overall_sentiment = {
                sentiment: round(count/total_comments * 100, 2)
                for sentiment, count in sentiment_counts.items()
            } if total_comments > 0 else {}
            
            return {
                "post": {
                    "title": submission.title,
                    "sentiment": post_analysis.sentiment,
                    "confidence": post_analysis.confidence,
                    "score": submission.score,
                    "upvote_ratio": submission.upvote_ratio
                },
                "comments_analyzed": len(comments_data),
                "overall_sentiment": overall_sentiment,
                "comments": comments_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing URL: {str(e)}")
            raise

    async def analyze_user(self, username: str, limit: int = 50) -> Dict:
        """Analyze sentiment patterns of a user's comments."""
        try:
            logger.info(f"Analyzing user: u/{username}")
            user = self.reddit.redditor(username)
            
            # Verify user exists
            try:
                _ = user.id
            except:
                raise ValueError(f"User u/{username} not found")
            
            comments_data = []
            subreddit_stats = Counter()
            
            for comment in user.comments.new(limit=limit):
                if hasattr(comment, 'body') and len(comment.body) > 20:
                    analysis = self.model_service.predict(comment.body)
                    
                    comments_data.append({
                        "text": comment.body,
                        "sentiment": analysis.sentiment,
                        "confidence": analysis.confidence,
                        "subreddit": comment.subreddit.display_name,
                        "score": comment.score,
                        "created_utc": datetime.fromtimestamp(comment.created_utc)
                    })
                    
                    subreddit_stats[comment.subreddit.display_name] += 1
            
            if not comments_data:
                return {
                    "username": username,
                    "message": "No comments found for analysis",
                    "total_comments_analyzed": 0
                }
            
            # Create DataFrame for analysis
            df = pd.DataFrame(comments_data)
            df['created_utc'] = pd.to_datetime(df['created_utc'])
            
            # Time series analysis
            sentiment_over_time = df.set_index('created_utc').resample('D')['sentiment'].value_counts().unstack().fillna(0)
            
            # Calculate sentiment distribution
            sentiment_counts = Counter(c["sentiment"] for c in comments_data)
            total_comments = len(comments_data)
            
            sentiment_distribution = {
                sentiment: {
                    "count": count,
                    "percentage": round(count/total_comments * 100, 2)
                }
                for sentiment, count in sentiment_counts.items()
            }
            
            return {
                "username": username,
                "total_comments_analyzed": total_comments,
                "sentiment_distribution": sentiment_distribution,
                "most_active_subreddits": dict(subreddit_stats.most_common(5)),
                "sentiment_over_time": sentiment_over_time.to_dict(),
                "recent_comments": sorted(
                    comments_data,
                    key=lambda x: x["created_utc"],
                    reverse=True
                )[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user {username}: {str(e)}")
            raise

    async def analyze_trend(self, 
                       keyword: str, 
                       subreddits: List[str],
                       time_filter: str = "week",
                       limit: int = 100) -> Dict:
        """Analyze sentiment trends around specific keywords."""
        try:
            logger.info(f"Starting trend analysis for keyword: {keyword}")
            trend_data = []
            
            for subreddit_name in subreddits:
                try:
                    logger.info(f"Analyzing subreddit: r/{subreddit_name}")
                    subreddit = self.reddit.subreddit(subreddit_name)
                    search_results = list(subreddit.search(
                        keyword,
                        time_filter=time_filter,
                        limit=limit,
                        sort='relevance'
                    ))
                    logger.info(f"Found {len(search_results)} results in r/{subreddit_name}")
                    
                    for submission in search_results:
                        submission.comments.replace_more(limit=0)
                        relevant_comments = [
                            comment for comment in submission.comments.list()
                            if hasattr(comment, 'body') and 
                            keyword.lower() in comment.body.lower() and 
                            len(comment.body) > 20
                        ]
                        
                        for comment in relevant_comments:
                            analysis = self.model_service.predict(comment.body)
                            trend_data.append({
                                "subreddit": subreddit_name,
                                "text": comment.body,
                                "sentiment": analysis.sentiment,
                                "confidence": analysis.confidence,
                                "score": comment.score,
                                "created_utc": datetime.fromtimestamp(comment.created_utc).isoformat(),
                                "post_title": submission.title
                            })
                            
                except Exception as e:
                    logger.error(f"Error processing subreddit {subreddit_name}: {str(e)}")
                    continue
            
            if not trend_data:
                return {
                    "keyword": keyword,
                    "message": f"No mentions found for '{keyword}' in specified subreddits",
                    "subreddits_analyzed": subreddits,
                    "total_mentions": 0
                }
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(trend_data)
            
            # Calculate sentiment counts per subreddit
            sentiment_counts = {}
            for subreddit in df['subreddit'].unique():
                subreddit_data = df[df['subreddit'] == subreddit]
                sentiment_counts[subreddit] = {
                    'positive': int(sum(subreddit_data['sentiment'] == 'positive')),
                    'neutral': int(sum(subreddit_data['sentiment'] == 'neutral')),
                    'negative': int(sum(subreddit_data['sentiment'] == 'negative'))
                }
            
            # Calculate engagement metrics
            engagement_metrics = {}
            for subreddit in df['subreddit'].unique():
                subreddit_data = df[df['subreddit'] == subreddit]
                engagement_metrics[subreddit] = {
                    'avg_score': float(subreddit_data['score'].mean()),
                    'total_score': int(subreddit_data['score'].sum()),
                    'comment_count': int(len(subreddit_data))
                }
            
            # Prepare sample mentions
            sample_mentions = [
                {
                    "text": item["text"][:200] + "..." if len(item["text"]) > 200 else item["text"],
                    "sentiment": item["sentiment"],
                    "confidence": float(item["confidence"]),
                    "score": int(item["score"]),
                    "subreddit": item["subreddit"],
                    "created_at": item["created_utc"]
                }
                for item in sorted(trend_data, key=lambda x: x["score"], reverse=True)[:5]
            ]
            
            result = {
                "keyword": keyword,
                "total_mentions": len(trend_data),
                "subreddits_analyzed": list(subreddits),
                "sentiment_distribution": sentiment_counts,
                "engagement_metrics": engagement_metrics,
                "sample_mentions": sample_mentions
            }
            
            logger.info(f"Successfully analyzed trend for '{keyword}' with {len(trend_data)} mentions")
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_trend: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to analyze trend: {str(e)}")

    def _calculate_sentiment_stats(self, items: List[Dict]) -> Dict:
        """Calculate sentiment statistics from a list of analyzed items."""
        if not items:
            return {}
            
        total = len(items)
        sentiment_counts = Counter(item['sentiment'] for item in items)
        
        return {
            sentiment: {
                'count': count,
                'percentage': round(count/total * 100, 2)
            }
            for sentiment, count in sentiment_counts.items()
        }
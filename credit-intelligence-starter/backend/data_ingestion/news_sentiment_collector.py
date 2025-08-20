import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, NewsRaw
from typing import List, Optional, Dict
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)

class NewsSentimentCollector:
    """Collects news data and performs sentiment analysis"""
    
    def __init__(self, news_api_key: str = None):
        self.news_api_key = news_api_key or os.getenv('NEWS_API_KEY')
        self.session = next(get_db())
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # News API endpoints
        self.news_api_url = 'https://newsapi.org/v2/everything'
        
        if not self.news_api_key:
            logger.warning("News API key not provided. Using free news sources.")
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to sentiment label
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': polarity,
                'label': label,
                'subjectivity': subjectivity,
                'method': 'textblob'
            }
        except Exception as e:
            logger.error(f"TextBlob sentiment analysis failed: {str(e)}")
            return {'score': 0.0, 'label': 'neutral', 'subjectivity': 0.5, 'method': 'textblob'}
    
    def analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores['compound']  # -1 to 1
            
            # Convert to sentiment label
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            
            return {
                'score': compound,
                'label': label,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'method': 'vader'
            }
        except Exception as e:
            logger.error(f"VADER sentiment analysis failed: {str(e)}")
            return {'score': 0.0, 'label': 'neutral', 'positive': 0.33, 'negative': 0.33, 'neutral': 0.33, 'method': 'vader'}
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text
    
    def collect_news_from_api(self, company_name: str, ticker: str, days_back: int = 7) -> List[Dict]:
        """Collect news from News API"""
        if not self.news_api_key:
            return []
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Search query
            query = f'"{company_name}" OR "{ticker}"'
            
            params = {
                'q': query,
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': 'en',
                'pageSize': 100,
                'apiKey': self.news_api_key
            }
            
            response = requests.get(self.news_api_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = []
            for article in data.get('articles', []):
                if article['title'] and article['publishedAt']:
                    articles.append({
                        'title': article['title'],
                        'content': article.get('description', '') or article.get('content', ''),
                        'source': article['source']['name'],
                        'url': article['url'],
                        'published_at': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    })
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting news from API for {ticker}: {str(e)}")
            return []
    
    def collect_free_news_sources(self, company_name: str, ticker: str) -> List[Dict]:
        """Collect news from free sources (RSS feeds, etc.)"""
        # This is a placeholder for free news sources
        # You can implement RSS feed parsing, web scraping of financial news sites, etc.
        
        articles = []
        
        # Example: Yahoo Finance RSS (this would need proper implementation)
        try:
            # Placeholder for free news collection
            # You could implement:
            # - RSS feed parsing from Yahoo Finance, Google News, etc.
            # - Web scraping from financial news websites
            # - Social media sentiment (Twitter API, Reddit, etc.)
            
            logger.info(f"Free news collection for {ticker} not implemented yet")
            
        except Exception as e:
            logger.error(f"Error collecting free news for {ticker}: {str(e)}")
        
        return articles
    
    def process_and_store_news(self, company_id: int, articles: List[Dict]) -> int:
        """Process articles and store in database with sentiment analysis"""
        stored_count = 0
        
        for article in articles:
            try:
                # Check if article already exists
                existing = self.session.query(NewsRaw).filter(
                    NewsRaw.company_id == company_id,
                    NewsRaw.url == article['url']
                ).first()
                
                if existing:
                    continue
                
                # Clean and combine title and content for sentiment analysis
                full_text = f"{article['title']} {article['content']}"
                clean_text = self.clean_text(full_text)
                
                if not clean_text:
                    continue
                
                # Perform sentiment analysis using both methods
                textblob_sentiment = self.analyze_sentiment_textblob(clean_text)
                vader_sentiment = self.analyze_sentiment_vader(clean_text)
                
                # Use VADER as primary sentiment (generally better for social media/news)
                primary_sentiment = vader_sentiment
                
                # Create news record
                news_record = NewsRaw(
                    company_id=company_id,
                    title=article['title'][:500],  # Limit title length
                    content=article['content'][:2000] if article['content'] else None,  # Limit content length
                    source=article['source'][:100],  # Limit source length
                    published_at=article['published_at'],
                    url=article['url'],
                    sentiment_score=primary_sentiment['score'],
                    sentiment_label=primary_sentiment['label'],
                    processed=True
                )
                
                self.session.add(news_record)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error processing article: {str(e)}")
                continue
        
        try:
            self.session.commit()
            logger.info(f"Stored {stored_count} news articles")
        except Exception as e:
            logger.error(f"Error committing news articles: {str(e)}")
            self.session.rollback()
            stored_count = 0
        
        return stored_count
    
    def collect_company_news(self, ticker: str, days_back: int = 7) -> Dict:
        """Collect and analyze news for a specific company"""
        try:
            # Get company from database
            company = self.session.query(Company).filter(Company.ticker == ticker).first()
            if not company:
                logger.error(f"Company {ticker} not found in database")
                return {'success': False, 'message': f'Company {ticker} not found'}
            
            # Collect news from various sources
            articles = []
            
            # Try News API first
            if self.news_api_key:
                api_articles = self.collect_news_from_api(company.name, ticker, days_back)
                articles.extend(api_articles)
                time.sleep(1)  # Rate limiting
            
            # Fallback to free sources
            if not articles:
                free_articles = self.collect_free_news_sources(company.name, ticker)
                articles.extend(free_articles)
            
            if not articles:
                return {'success': False, 'message': 'No articles found'}
            
            # Process and store articles
            stored_count = self.process_and_store_news(company.id, articles)
            
            return {
                'success': True,
                'articles_found': len(articles),
                'articles_stored': stored_count,
                'company': company.name,
                'ticker': ticker
            }
            
        except Exception as e:
            logger.error(f"Error collecting news for {ticker}: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def bulk_collect_news(self, tickers: List[str], days_back: int = 7) -> Dict:
        """Collect news for multiple companies"""
        results = {'success': [], 'failed': []}
        
        for ticker in tickers:
            try:
                result = self.collect_company_news(ticker, days_back)
                if result['success']:
                    results['success'].append({
                        'ticker': ticker,
                        'articles_stored': result['articles_stored']
                    })
                else:
                    results['failed'].append({
                        'ticker': ticker,
                        'error': result['message']
                    })
                
                # Rate limiting between companies
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing news for {ticker}: {str(e)}")
                results['failed'].append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return results
    
    def get_sentiment_summary(self, ticker: str, days_back: int = 30) -> Dict:
        """Get sentiment summary for a company"""
        try:
            company = self.session.query(Company).filter(Company.ticker == ticker).first()
            if not company:
                return {}
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            news_items = self.session.query(NewsRaw).filter(
                NewsRaw.company_id == company.id,
                NewsRaw.published_at >= cutoff_date,
                NewsRaw.processed == True
            ).all()
            
            if not news_items:
                return {'message': 'No news data available'}
            
            # Calculate sentiment statistics
            scores = [item.sentiment_score for item in news_items if item.sentiment_score is not None]
            labels = [item.sentiment_label for item in news_items if item.sentiment_label]
            
            if not scores:
                return {'message': 'No sentiment scores available'}
            
            # Count sentiment labels
            label_counts = {}
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            return {
                'total_articles': len(news_items),
                'average_sentiment': sum(scores) / len(scores),
                'sentiment_distribution': label_counts,
                'latest_articles': [
                    {
                        'title': item.title,
                        'sentiment_score': item.sentiment_score,
                        'sentiment_label': item.sentiment_label,
                        'published_at': item.published_at.isoformat(),
                        'source': item.source
                    }
                    for item in sorted(news_items, key=lambda x: x.published_at, reverse=True)[:5]
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {ticker}: {str(e)}")
            return {'error': str(e)}
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    collector = NewsSentimentCollector()
    
    # Test with a single company
    ticker = 'AAPL'
    print(f"Collecting news for {ticker}...")
    
    result = collector.collect_company_news(ticker, days_back=3)
    print(f"Result: {result}")
    
    if result['success']:
        # Get sentiment summary
        summary = collector.get_sentiment_summary(ticker)
        print(f"Sentiment summary: {summary}")
    
    collector.close()

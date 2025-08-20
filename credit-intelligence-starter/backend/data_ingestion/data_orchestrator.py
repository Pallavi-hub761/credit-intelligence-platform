import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company
from .yahoo_finance_collector import YahooFinanceCollector
from .alpha_vantage_collector import AlphaVantageCollector
from .news_sentiment_collector import NewsSentimentCollector
import os
import schedule
import time
import threading

logger = logging.getLogger(__name__)

class DataOrchestrator:
    """Orchestrates data collection from multiple sources"""
    
    def __init__(self):
        self.yahoo_collector = YahooFinanceCollector()
        self.alpha_vantage_collector = AlphaVantageCollector()
        self.news_collector = NewsSentimentCollector()
        self.session = next(get_db())
        
    def initialize_default_companies(self) -> Dict:
        """Initialize database with default companies to track"""
        default_companies = [
            ('AAPL', 'Apple Inc.'),
            ('MSFT', 'Microsoft Corporation'),
            ('GOOGL', 'Alphabet Inc.'),
            ('AMZN', 'Amazon.com Inc.'),
            ('TSLA', 'Tesla Inc.'),
            ('META', 'Meta Platforms Inc.'),
            ('NVDA', 'NVIDIA Corporation'),
            ('JPM', 'JPMorgan Chase & Co.'),
            ('V', 'Visa Inc.'),
            ('JNJ', 'Johnson & Johnson'),
            ('WMT', 'Walmart Inc.'),
            ('PG', 'Procter & Gamble Co.'),
            ('UNH', 'UnitedHealth Group Inc.'),
            ('HD', 'Home Depot Inc.'),
            ('MA', 'Mastercard Inc.'),
            ('BAC', 'Bank of America Corp.'),
            ('XOM', 'Exxon Mobil Corporation'),
            ('KO', 'The Coca-Cola Company'),
            ('PFE', 'Pfizer Inc.'),
            ('INTC', 'Intel Corporation')
        ]
        
        added_companies = []
        for ticker, name in default_companies:
            try:
                company = self.yahoo_collector.add_company(ticker, name)
                added_companies.append({'ticker': ticker, 'name': name, 'id': company.id})
            except Exception as e:
                logger.error(f"Error adding company {ticker}: {str(e)}")
        
        return {
            'companies_added': len(added_companies),
            'companies': added_companies
        }
    
    def collect_all_data_for_company(self, ticker: str) -> Dict:
        """Collect all available data for a single company"""
        results = {
            'ticker': ticker,
            'yahoo_finance': {'success': False},
            'alpha_vantage': {'success': False},
            'news_sentiment': {'success': False},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Yahoo Finance data collection
            logger.info(f"Collecting Yahoo Finance data for {ticker}")
            yahoo_success = self.yahoo_collector.collect_price_data(ticker)
            results['yahoo_finance'] = {
                'success': yahoo_success,
                'data_type': 'price_data'
            }
            
            # Alpha Vantage data collection (with rate limiting)
            if os.getenv('ALPHA_VANTAGE_API_KEY'):
                logger.info(f"Collecting Alpha Vantage data for {ticker}")
                try:
                    av_data = self.alpha_vantage_collector.collect_comprehensive_data(ticker)
                    results['alpha_vantage'] = {
                        'success': bool(av_data),
                        'data_collected': list(av_data.keys()) if av_data else []
                    }
                except Exception as e:
                    logger.error(f"Alpha Vantage collection failed for {ticker}: {str(e)}")
                    results['alpha_vantage'] = {'success': False, 'error': str(e)}
            
            # News sentiment collection
            logger.info(f"Collecting news sentiment for {ticker}")
            try:
                news_result = self.news_collector.collect_company_news(ticker, days_back=7)
                results['news_sentiment'] = news_result
            except Exception as e:
                logger.error(f"News collection failed for {ticker}: {str(e)}")
                results['news_sentiment'] = {'success': False, 'error': str(e)}
            
        except Exception as e:
            logger.error(f"Error in comprehensive data collection for {ticker}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def bulk_data_collection(self, tickers: Optional[List[str]] = None) -> Dict:
        """Collect data for multiple companies"""
        if tickers is None:
            # Get all companies from database
            companies = self.session.query(Company).all()
            tickers = [company.ticker for company in companies]
        
        if not tickers:
            return {'error': 'No companies to process'}
        
        results = {
            'total_companies': len(tickers),
            'successful': [],
            'failed': [],
            'start_time': datetime.now().isoformat()
        }
        
        for ticker in tickers:
            try:
                logger.info(f"Processing {ticker}...")
                result = self.collect_all_data_for_company(ticker)
                
                # Determine if overall collection was successful
                success_count = sum([
                    result['yahoo_finance']['success'],
                    result['alpha_vantage']['success'],
                    result['news_sentiment']['success']
                ])
                
                if success_count > 0:
                    results['successful'].append(result)
                else:
                    results['failed'].append(result)
                
                # Rate limiting between companies
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {str(e)}")
                results['failed'].append({
                    'ticker': ticker,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        results['end_time'] = datetime.now().isoformat()
        results['success_rate'] = len(results['successful']) / len(tickers) * 100
        
        return results
    
    def daily_data_refresh(self) -> Dict:
        """Daily data refresh for all companies"""
        logger.info("Starting daily data refresh...")
        
        # Get companies that need updates (haven't been updated in 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # This would ideally check last update timestamps from the database
        # For now, we'll refresh all companies
        companies = self.session.query(Company).all()
        tickers = [company.ticker for company in companies]
        
        if not tickers:
            logger.warning("No companies found for daily refresh")
            return {'message': 'No companies to refresh'}
        
        # Focus on price data and news for daily refresh
        results = {
            'refresh_type': 'daily',
            'companies_processed': 0,
            'price_updates': 0,
            'news_updates': 0,
            'errors': []
        }
        
        for ticker in tickers:
            try:
                # Update price data (Yahoo Finance - more reliable for daily updates)
                price_success = self.yahoo_collector.collect_price_data(ticker, period="5d")
                if price_success:
                    results['price_updates'] += 1
                
                # Update news sentiment (last 2 days)
                news_result = self.news_collector.collect_company_news(ticker, days_back=2)
                if news_result['success']:
                    results['news_updates'] += 1
                
                results['companies_processed'] += 1
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Error refreshing {ticker}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        logger.info(f"Daily refresh completed: {results}")
        return results
    
    def weekly_comprehensive_refresh(self) -> Dict:
        """Weekly comprehensive data refresh including fundamental data"""
        logger.info("Starting weekly comprehensive refresh...")
        
        companies = self.session.query(Company).all()
        tickers = [company.ticker for company in companies]
        
        if not tickers:
            return {'message': 'No companies to refresh'}
        
        # Full data collection including Alpha Vantage fundamental data
        return self.bulk_data_collection(tickers)
    
    def setup_scheduled_tasks(self):
        """Setup scheduled data collection tasks"""
        # Daily refresh at 6 AM
        schedule.every().day.at("06:00").do(self.daily_data_refresh)
        
        # Weekly comprehensive refresh on Sundays at 2 AM
        schedule.every().sunday.at("02:00").do(self.weekly_comprehensive_refresh)
        
        # Start scheduler in a separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduled tasks setup completed")
    
    def get_data_status(self) -> Dict:
        """Get status of data collection for all companies"""
        companies = self.session.query(Company).all()
        
        status = {
            'total_companies': len(companies),
            'companies': [],
            'last_updated': datetime.now().isoformat()
        }
        
        for company in companies:
            # Get latest price data
            latest_price = self.session.query(Price).filter(
                Price.company_id == company.id
            ).order_by(Price.date.desc()).first()
            
            # Get latest news
            latest_news = self.session.query(NewsRaw).filter(
                NewsRaw.company_id == company.id
            ).order_by(NewsRaw.published_at.desc()).first()
            
            company_status = {
                'ticker': company.ticker,
                'name': company.name,
                'latest_price_date': latest_price.date.isoformat() if latest_price else None,
                'latest_news_date': latest_news.published_at.isoformat() if latest_news else None,
                'price_data_available': latest_price is not None,
                'news_data_available': latest_news is not None
            }
            
            status['companies'].append(company_status)
        
        return status
    
    def close(self):
        """Close all collectors and database sessions"""
        self.yahoo_collector.close()
        self.alpha_vantage_collector.close()
        self.news_collector.close()
        self.session.close()

# FastAPI endpoints integration
async def initialize_data_collection():
    """Initialize data collection system"""
    orchestrator = DataOrchestrator()
    
    try:
        # Initialize default companies
        init_result = orchestrator.initialize_default_companies()
        
        # Setup scheduled tasks
        orchestrator.setup_scheduled_tasks()
        
        return {
            'status': 'initialized',
            'companies_added': init_result['companies_added'],
            'scheduler_active': True
        }
    
    except Exception as e:
        logger.error(f"Error initializing data collection: {str(e)}")
        return {'status': 'error', 'message': str(e)}
    
    finally:
        orchestrator.close()

if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = DataOrchestrator()
    
    try:
        # Initialize companies
        print("Initializing companies...")
        init_result = orchestrator.initialize_default_companies()
        print(f"Initialized {init_result['companies_added']} companies")
        
        # Test data collection for a few companies
        test_tickers = ['AAPL', 'MSFT', 'GOOGL']
        print(f"Testing data collection for {test_tickers}...")
        
        results = orchestrator.bulk_data_collection(test_tickers)
        print(f"Collection results: {results['success_rate']:.1f}% success rate")
        
        # Get data status
        status = orchestrator.get_data_status()
        print(f"Data status: {status['total_companies']} companies tracked")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        orchestrator.close()

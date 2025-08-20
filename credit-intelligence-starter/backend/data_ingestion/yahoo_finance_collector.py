import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Price
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class YahooFinanceCollector:
    """Collects financial data from Yahoo Finance API"""
    
    def __init__(self):
        self.session = next(get_db())
    
    def add_company(self, ticker: str, name: str) -> Company:
        """Add a new company to track"""
        existing = self.session.query(Company).filter(Company.ticker == ticker).first()
        if existing:
            return existing
        
        company = Company(ticker=ticker, name=name)
        self.session.add(company)
        self.session.commit()
        logger.info(f"Added company: {name} ({ticker})")
        return company
    
    def collect_price_data(self, ticker: str, period: str = "1y") -> bool:
        """
        Collect historical price data for a ticker
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        """
        try:
            # Get company from database
            company = self.session.query(Company).filter(Company.ticker == ticker).first()
            if not company:
                logger.error(f"Company {ticker} not found in database")
                return False
            
            # Download data from Yahoo Finance
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data found for {ticker}")
                return False
            
            # Process and store data
            records_added = 0
            for date, row in hist.iterrows():
                # Check if record already exists
                existing = self.session.query(Price).filter(
                    Price.company_id == company.id,
                    Price.date == date.date()
                ).first()
                
                if not existing:
                    price_record = Price(
                        company_id=company.id,
                        date=date.date(),
                        open_price=float(row['Open']),
                        high_price=float(row['High']),
                        low_price=float(row['Low']),
                        close_price=float(row['Close']),
                        volume=int(row['Volume']),
                        adj_close=float(row['Close'])  # Yahoo Finance adjusted close
                    )
                    self.session.add(price_record)
                    records_added += 1
            
            self.session.commit()
            logger.info(f"Added {records_added} price records for {ticker}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data for {ticker}: {str(e)}")
            self.session.rollback()
            return False
    
    def get_company_info(self, ticker: str) -> dict:
        """Get detailed company information from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'employees': info.get('fullTimeEmployees', 0),
                'country': info.get('country', 'Unknown'),
                'website': info.get('website', ''),
                'business_summary': info.get('longBusinessSummary', ''),
                'pe_ratio': info.get('trailingPE', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'return_on_equity': info.get('returnOnEquity', None),
                'revenue_growth': info.get('revenueGrowth', None)
            }
        except Exception as e:
            logger.error(f"Error getting company info for {ticker}: {str(e)}")
            return {}
    
    def collect_financial_metrics(self, ticker: str) -> dict:
        """Collect key financial metrics for credit scoring"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Key financial metrics for credit scoring
            metrics = {
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'return_on_equity': info.get('returnOnEquity', None),
                'return_on_assets': info.get('returnOnAssets', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None),
                'revenue_growth': info.get('revenueGrowth', None),
                'earnings_growth': info.get('earningsGrowth', None),
                'free_cash_flow': info.get('freeCashflow', None),
                'total_cash': info.get('totalCash', None),
                'total_debt': info.get('totalDebt', None),
                'enterprise_value': info.get('enterpriseValue', None),
                'price_to_book': info.get('priceToBook', None),
                'price_to_earnings': info.get('trailingPE', None),
                'beta': info.get('beta', None),
                'market_cap': info.get('marketCap', None)
            }
            
            return {k: v for k, v in metrics.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error collecting financial metrics for {ticker}: {str(e)}")
            return {}
    
    def bulk_collect_data(self, tickers: List[str], period: str = "1y") -> dict:
        """Collect data for multiple tickers"""
        results = {'success': [], 'failed': []}
        
        for ticker in tickers:
            try:
                # Get company info and add to database
                info = self.get_company_info(ticker)
                company_name = info.get('name', ticker)
                
                self.add_company(ticker, company_name)
                
                # Collect price data
                if self.collect_price_data(ticker, period):
                    results['success'].append(ticker)
                else:
                    results['failed'].append(ticker)
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                results['failed'].append(ticker)
        
        return results
    
    def get_latest_prices(self, ticker: str, days: int = 30) -> pd.DataFrame:
        """Get latest price data for a ticker"""
        company = self.session.query(Company).filter(Company.ticker == ticker).first()
        if not company:
            return pd.DataFrame()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        prices = self.session.query(Price).filter(
            Price.company_id == company.id,
            Price.date >= cutoff_date.date()
        ).order_by(Price.date.desc()).all()
        
        if not prices:
            return pd.DataFrame()
        
        data = []
        for price in prices:
            data.append({
                'date': price.date,
                'open': price.open_price,
                'high': price.high_price,
                'low': price.low_price,
                'close': price.close_price,
                'volume': price.volume,
                'adj_close': price.adj_close
            })
        
        return pd.DataFrame(data)
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage and default companies to track
DEFAULT_COMPANIES = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'GOOGL', # Alphabet Inc.
    'AMZN',  # Amazon.com Inc.
    'TSLA',  # Tesla Inc.
    'META',  # Meta Platforms Inc.
    'NVDA',  # NVIDIA Corporation
    'JPM',   # JPMorgan Chase & Co.
    'V',     # Visa Inc.
    'JNJ',   # Johnson & Johnson
    'WMT',   # Walmart Inc.
    'PG',    # Procter & Gamble Co.
    'UNH',   # UnitedHealth Group Inc.
    'HD',    # Home Depot Inc.
    'MA',    # Mastercard Inc.
]

if __name__ == "__main__":
    collector = YahooFinanceCollector()
    
    # Collect data for default companies
    print("Starting data collection for default companies...")
    results = collector.bulk_collect_data(DEFAULT_COMPANIES)
    
    print(f"Successfully collected data for: {results['success']}")
    print(f"Failed to collect data for: {results['failed']}")
    
    collector.close()

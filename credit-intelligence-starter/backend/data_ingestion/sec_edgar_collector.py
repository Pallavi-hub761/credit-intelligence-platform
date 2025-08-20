import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Price, NewsRaw
from typing import List, Optional, Dict, Any
import logging
import time
import json
import re
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

class SECEdgarCollector:
    """Collects SEC EDGAR filings data for credit risk analysis"""
    
    def __init__(self, user_agent: str = "Credit Intelligence Platform contact@example.com"):
        self.session = next(get_db())
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.search_url = "https://efts.sec.gov/LATEST/search-index"
        self.user_agent = user_agent
        
        # SEC requires proper User-Agent header
        self.headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        # Rate limiting - SEC allows 10 requests per second
        self.request_delay = 0.1
        
        # Key filing types for credit analysis
        self.key_filings = {
            '10-K': 'Annual Report',
            '10-Q': 'Quarterly Report', 
            '8-K': 'Current Report',
            '10-K/A': 'Annual Report Amendment',
            '10-Q/A': 'Quarterly Report Amendment',
            'DEF 14A': 'Proxy Statement',
            'S-1': 'Registration Statement',
            '424B4': 'Prospectus'
        }
    
    def get_company_cik(self, ticker: str) -> Optional[str]:
        """Get CIK (Central Index Key) for a company ticker"""
        try:
            # Use SEC company tickers JSON endpoint
            url = "https://www.sec.gov/files/company_tickers.json"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            companies = response.json()
            
            for company_data in companies.values():
                if company_data['ticker'].upper() == ticker.upper():
                    # CIK needs to be padded to 10 digits
                    cik = str(company_data['cik_str']).zfill(10)
                    logger.info(f"Found CIK {cik} for ticker {ticker}")
                    return cik
            
            logger.warning(f"CIK not found for ticker {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting CIK for {ticker}: {str(e)}")
            return None
    
    def search_filings(self, cik: str, filing_types: List[str] = None, 
                      date_from: datetime = None, date_to: datetime = None,
                      count: int = 40) -> List[Dict]:
        """Search for company filings"""
        if filing_types is None:
            filing_types = list(self.key_filings.keys())
        
        if date_from is None:
            date_from = datetime.now() - timedelta(days=365)  # Last year
        
        if date_to is None:
            date_to = datetime.now()
        
        filings = []
        
        try:
            for filing_type in filing_types:
                # Build search parameters
                params = {
                    'q': f'cik:{cik} AND form:{filing_type}',
                    'dateRange': 'custom',
                    'startdt': date_from.strftime('%Y-%m-%d'),
                    'enddt': date_to.strftime('%Y-%m-%d'),
                    'count': count
                }
                
                response = requests.get(self.search_url, params=params, headers=self.headers)
                response.raise_for_status()
                
                data = response.json()
                
                if 'hits' in data and 'hits' in data['hits']:
                    for hit in data['hits']['hits']:
                        source = hit['_source']
                        filing = {
                            'cik': cik,
                            'form': source.get('form'),
                            'file_num': source.get('file_num'),
                            'filing_date': source.get('file_date'),
                            'period_of_report': source.get('period_of_report'),
                            'company_name': source.get('display_names', [''])[0],
                            'accession_number': source.get('accession_num'),
                            'primary_doc': source.get('primary_doc'),
                            'description': self.key_filings.get(source.get('form'), 'SEC Filing')
                        }
                        filings.append(filing)
                
                # Rate limiting
                time.sleep(self.request_delay)
        
        except Exception as e:
            logger.error(f"Error searching filings for CIK {cik}: {str(e)}")
        
        # Sort by filing date (newest first)
        filings.sort(key=lambda x: x.get('filing_date', ''), reverse=True)
        
        return filings
    
    def get_filing_content(self, cik: str, accession_number: str, primary_doc: str) -> Optional[str]:
        """Download and parse filing content"""
        try:
            # Clean accession number (remove dashes)
            acc_clean = accession_number.replace('-', '')
            
            # Build URL for the filing
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{primary_doc}"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Rate limiting
            time.sleep(self.request_delay)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error downloading filing {accession_number}: {str(e)}")
            return None
    
    def extract_financial_metrics(self, filing_content: str, form_type: str) -> Dict[str, Any]:
        """Extract key financial metrics from filing content"""
        metrics = {}
        
        try:
            # Parse HTML content
            soup = BeautifulSoup(filing_content, 'html.parser')
            text = soup.get_text().lower()
            
            # Common financial indicators to extract
            patterns = {
                'total_debt': [
                    r'total debt[^\d]*(\$?[\d,]+\.?\d*)',
                    r'total borrowings[^\d]*(\$?[\d,]+\.?\d*)',
                    r'long-term debt[^\d]*(\$?[\d,]+\.?\d*)'
                ],
                'cash_equivalents': [
                    r'cash and cash equivalents[^\d]*(\$?[\d,]+\.?\d*)',
                    r'cash and equivalents[^\d]*(\$?[\d,]+\.?\d*)'
                ],
                'total_revenue': [
                    r'total revenue[^\d]*(\$?[\d,]+\.?\d*)',
                    r'net revenue[^\d]*(\$?[\d,]+\.?\d*)',
                    r'total net sales[^\d]*(\$?[\d,]+\.?\d*)'
                ],
                'net_income': [
                    r'net income[^\d]*(\$?[\d,]+\.?\d*)',
                    r'net earnings[^\d]*(\$?[\d,]+\.?\d*)'
                ],
                'total_assets': [
                    r'total assets[^\d]*(\$?[\d,]+\.?\d*)'
                ],
                'shareholders_equity': [
                    r'shareholders.? equity[^\d]*(\$?[\d,]+\.?\d*)',
                    r'stockholders.? equity[^\d]*(\$?[\d,]+\.?\d*)'
                ]
            }
            
            for metric_name, regex_patterns in patterns.items():
                for pattern in regex_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        # Clean and convert the first match
                        value_str = matches[0].replace('$', '').replace(',', '')
                        try:
                            value = float(value_str)
                            metrics[metric_name] = value
                            break  # Use first successful match
                        except ValueError:
                            continue
            
            # Extract risk factors and business description
            risk_factors = self.extract_risk_factors(soup)
            if risk_factors:
                metrics['risk_factors_count'] = len(risk_factors)
                metrics['risk_factors'] = risk_factors[:10]  # Top 10 risk factors
            
            # Extract business segment information
            segments = self.extract_business_segments(soup)
            if segments:
                metrics['business_segments'] = segments
            
            # Calculate derived metrics
            if 'total_debt' in metrics and 'shareholders_equity' in metrics:
                if metrics['shareholders_equity'] > 0:
                    metrics['debt_to_equity'] = metrics['total_debt'] / metrics['shareholders_equity']
            
            if 'cash_equivalents' in metrics and 'total_debt' in metrics:
                metrics['net_debt'] = metrics['total_debt'] - metrics['cash_equivalents']
            
            if 'net_income' in metrics and 'total_revenue' in metrics:
                if metrics['total_revenue'] > 0:
                    metrics['profit_margin'] = metrics['net_income'] / metrics['total_revenue']
            
            if 'total_debt' in metrics and 'total_assets' in metrics:
                if metrics['total_assets'] > 0:
                    metrics['debt_to_assets'] = metrics['total_debt'] / metrics['total_assets']
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {str(e)}")
        
        return metrics
    
    def extract_risk_factors(self, soup: BeautifulSoup) -> List[str]:
        """Extract risk factors from SEC filing"""
        risk_factors = []
        
        try:
            # Look for risk factors section
            risk_section = None
            
            # Common patterns for risk factors sections
            risk_patterns = [
                'risk factors',
                'item 1a',
                'principal risks',
                'risk management'
            ]
            
            for pattern in risk_patterns:
                elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                if elements:
                    # Find the parent element and extract text
                    for element in elements:
                        parent = element.parent
                        if parent:
                            # Get text from this section
                            section_text = parent.get_text()
                            
                            # Split into sentences and filter for risk-related content
                            sentences = re.split(r'[.!?]+', section_text)
                            
                            for sentence in sentences:
                                sentence = sentence.strip()
                                if len(sentence) > 50 and any(risk_word in sentence.lower() 
                                    for risk_word in ['risk', 'may', 'could', 'uncertainty', 'adverse']):
                                    risk_factors.append(sentence[:200])  # Limit length
                            
                            if len(risk_factors) >= 20:  # Limit number of risk factors
                                break
                
                if risk_factors:
                    break
        
        except Exception as e:
            logger.error(f"Error extracting risk factors: {str(e)}")
        
        return risk_factors
    
    def extract_business_segments(self, soup: BeautifulSoup) -> List[str]:
        """Extract business segment information"""
        segments = []
        
        try:
            # Look for business segments section
            segment_patterns = [
                'business segments',
                'operating segments',
                'reportable segments',
                'business operations'
            ]
            
            for pattern in segment_patterns:
                elements = soup.find_all(text=re.compile(pattern, re.IGNORECASE))
                if elements:
                    for element in elements:
                        parent = element.parent
                        if parent:
                            text = parent.get_text()
                            
                            # Extract segment names (simple heuristic)
                            lines = text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if len(line) > 10 and len(line) < 100:
                                    # Check if it looks like a segment name
                                    if any(word in line.lower() for word in 
                                          ['division', 'segment', 'business', 'operations']):
                                        segments.append(line)
                            
                            if segments:
                                break
                
                if segments:
                    break
        
        except Exception as e:
            logger.error(f"Error extracting business segments: {str(e)}")
        
        return segments[:5]  # Limit to 5 segments
    
    def analyze_filing_sentiment(self, filing_content: str) -> Dict[str, float]:
        """Analyze sentiment of SEC filing content"""
        try:
            from textblob import TextBlob
            
            # Parse and clean content
            soup = BeautifulSoup(filing_content, 'html.parser')
            text = soup.get_text()
            
            # Focus on key sections for sentiment analysis
            key_sections = []
            
            # Extract management discussion and analysis
            mda_patterns = [
                'management.s discussion and analysis',
                'md&a',
                'liquidity and capital resources',
                'results of operations'
            ]
            
            for pattern in mda_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start = match.start()
                    # Extract ~2000 characters after the match
                    section = text[start:start+2000]
                    key_sections.append(section)
            
            if not key_sections:
                # If no specific sections found, use first 5000 characters
                key_sections = [text[:5000]]
            
            # Analyze sentiment
            sentiments = []
            for section in key_sections:
                blob = TextBlob(section)
                sentiments.append({
                    'polarity': blob.sentiment.polarity,
                    'subjectivity': blob.sentiment.subjectivity
                })
            
            # Calculate average sentiment
            avg_polarity = sum(s['polarity'] for s in sentiments) / len(sentiments)
            avg_subjectivity = sum(s['subjectivity'] for s in sentiments) / len(sentiments)
            
            return {
                'polarity': avg_polarity,
                'subjectivity': avg_subjectivity,
                'sentiment_label': 'positive' if avg_polarity > 0.1 else 'negative' if avg_polarity < -0.1 else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing filing sentiment: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.5, 'sentiment_label': 'neutral'}
    
    def collect_company_filings(self, ticker: str, months_back: int = 12) -> Dict[str, Any]:
        """Collect and analyze SEC filings for a company"""
        try:
            # Get company from database
            company = self.session.query(Company).filter(Company.ticker == ticker).first()
            if not company:
                logger.error(f"Company {ticker} not found in database")
                return {'success': False, 'message': f'Company {ticker} not found'}
            
            # Get CIK
            cik = self.get_company_cik(ticker)
            if not cik:
                return {'success': False, 'message': f'CIK not found for {ticker}'}
            
            # Search for filings
            date_from = datetime.now() - timedelta(days=months_back * 30)
            filings = self.search_filings(cik, date_from=date_from)
            
            if not filings:
                return {'success': False, 'message': 'No filings found'}
            
            # Process key filings
            processed_filings = []
            financial_metrics = {}
            
            for filing in filings[:10]:  # Process top 10 most recent filings
                try:
                    # Download filing content
                    content = self.get_filing_content(
                        cik, 
                        filing['accession_number'], 
                        filing['primary_doc']
                    )
                    
                    if content:
                        # Extract financial metrics
                        metrics = self.extract_financial_metrics(content, filing['form'])
                        
                        # Analyze sentiment
                        sentiment = self.analyze_filing_sentiment(content)
                        
                        filing_data = {
                            'form_type': filing['form'],
                            'filing_date': filing['filing_date'],
                            'description': filing['description'],
                            'metrics': metrics,
                            'sentiment': sentiment,
                            'accession_number': filing['accession_number']
                        }
                        
                        processed_filings.append(filing_data)
                        
                        # Store latest metrics for 10-K and 10-Q filings
                        if filing['form'] in ['10-K', '10-Q'] and metrics:
                            financial_metrics.update(metrics)
                
                except Exception as e:
                    logger.error(f"Error processing filing {filing.get('accession_number')}: {str(e)}")
                    continue
            
            # Store filing data as news-like entries for integration with existing system
            self.store_filing_data(company.id, processed_filings)
            
            return {
                'success': True,
                'company': company.name,
                'ticker': ticker,
                'cik': cik,
                'filings_processed': len(processed_filings),
                'financial_metrics': financial_metrics,
                'latest_filings': processed_filings[:5]
            }
            
        except Exception as e:
            logger.error(f"Error collecting filings for {ticker}: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def store_filing_data(self, company_id: int, filings: List[Dict]) -> int:
        """Store SEC filing data as structured news entries"""
        stored_count = 0
        
        for filing in filings:
            try:
                # Create a news-like entry for the filing
                title = f"SEC {filing['form_type']} Filing - {filing['description']}"
                
                # Create content summary
                content_parts = []
                if filing.get('metrics'):
                    metrics = filing['metrics']
                    if 'total_revenue' in metrics:
                        content_parts.append(f"Revenue: ${metrics['total_revenue']:,.0f}")
                    if 'net_income' in metrics:
                        content_parts.append(f"Net Income: ${metrics['net_income']:,.0f}")
                    if 'total_debt' in metrics:
                        content_parts.append(f"Total Debt: ${metrics['total_debt']:,.0f}")
                    if 'debt_to_equity' in metrics:
                        content_parts.append(f"Debt/Equity: {metrics['debt_to_equity']:.2f}")
                
                content = "; ".join(content_parts) if content_parts else f"SEC {filing['form_type']} filing"
                
                # Check if filing already exists
                existing = self.session.query(NewsRaw).filter(
                    NewsRaw.company_id == company_id,
                    NewsRaw.title.contains(filing['accession_number'])
                ).first()
                
                if existing:
                    continue
                
                # Create news record
                news_record = NewsRaw(
                    company_id=company_id,
                    title=title[:500],
                    content=content[:2000],
                    source='SEC EDGAR',
                    published_at=datetime.strptime(filing['filing_date'], '%Y-%m-%d'),
                    url=f"https://www.sec.gov/Archives/edgar/data/{filing.get('cik', '')}/{filing['accession_number'].replace('-', '')}/",
                    sentiment_score=filing.get('sentiment', {}).get('polarity', 0.0),
                    sentiment_label=filing.get('sentiment', {}).get('sentiment_label', 'neutral'),
                    processed=True
                )
                
                self.session.add(news_record)
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Error storing filing data: {str(e)}")
                continue
        
        try:
            self.session.commit()
            logger.info(f"Stored {stored_count} SEC filings")
        except Exception as e:
            logger.error(f"Error committing SEC filings: {str(e)}")
            self.session.rollback()
            stored_count = 0
        
        return stored_count
    
    def bulk_collect_filings(self, tickers: List[str], months_back: int = 12) -> Dict:
        """Collect SEC filings for multiple companies"""
        results = {'successful': [], 'failed': []}
        
        for ticker in tickers:
            try:
                result = self.collect_company_filings(ticker, months_back)
                if result['success']:
                    results['successful'].append({
                        'ticker': ticker,
                        'filings_processed': result['filings_processed']
                    })
                else:
                    results['failed'].append({
                        'ticker': ticker,
                        'error': result['message']
                    })
                
                # Rate limiting between companies
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing SEC filings for {ticker}: {str(e)}")
                results['failed'].append({
                    'ticker': ticker,
                    'error': str(e)
                })
        
        return results
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    collector = SECEdgarCollector()
    
    try:
        # Test with a single company
        ticker = 'AAPL'
        print(f"Collecting SEC filings for {ticker}...")
        
        result = collector.collect_company_filings(ticker, months_back=6)
        print(f"Result: {result}")
        
        if result['success']:
            print(f"Processed {result['filings_processed']} filings")
            if result.get('financial_metrics'):
                print("Financial metrics found:")
                for metric, value in result['financial_metrics'].items():
                    print(f"  {metric}: {value}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        collector.close()

import requests
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Price
from typing import List, Optional, Dict
import logging
import time
import os

logger = logging.getLogger(__name__)

class AlphaVantageCollector:
    """Collects financial data from Alpha Vantage API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHA_VANTAGE_API_KEY')
        self.base_url = 'https://www.alphavantage.co/query'
        self.session = next(get_db())
        
        if not self.api_key:
            logger.warning("Alpha Vantage API key not provided. Some features may not work.")
    
    def _make_request(self, params: dict) -> dict:
        """Make API request with rate limiting"""
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if 'Error Message' in data:
                raise ValueError(f"API Error: {data['Error Message']}")
            
            if 'Note' in data:
                logger.warning(f"API Note: {data['Note']}")
                time.sleep(60)  # Rate limit hit, wait 1 minute
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise
    
    def collect_daily_data(self, symbol: str, outputsize: str = 'compact') -> bool:
        """
        Collect daily price data from Alpha Vantage
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' (100 days) or 'full' (20+ years)
        """
        try:
            # Get company from database
            company = self.session.query(Company).filter(Company.ticker == symbol).first()
            if not company:
                logger.error(f"Company {symbol} not found in database")
                return False
            
            params = {
                'function': 'TIME_SERIES_DAILY_ADJUSTED',
                'symbol': symbol,
                'outputsize': outputsize
            }
            
            data = self._make_request(params)
            
            if 'Time Series (Daily)' not in data:
                logger.error(f"No time series data found for {symbol}")
                return False
            
            time_series = data['Time Series (Daily)']
            records_added = 0
            
            for date_str, values in time_series.items():
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Check if record already exists
                existing = self.session.query(Price).filter(
                    Price.company_id == company.id,
                    Price.date == date_obj
                ).first()
                
                if not existing:
                    price_record = Price(
                        company_id=company.id,
                        date=date_obj,
                        open_price=float(values['1. open']),
                        high_price=float(values['2. high']),
                        low_price=float(values['3. low']),
                        close_price=float(values['4. close']),
                        volume=int(values['6. volume']),
                        adj_close=float(values['5. adjusted close'])
                    )
                    self.session.add(price_record)
                    records_added += 1
            
            self.session.commit()
            logger.info(f"Added {records_added} price records for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting daily data for {symbol}: {str(e)}")
            self.session.rollback()
            return False
    
    def get_company_overview(self, symbol: str) -> dict:
        """Get company overview and fundamental data"""
        try:
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol
            }
            
            data = self._make_request(params)
            
            # Extract key financial metrics
            overview = {
                'name': data.get('Name', symbol),
                'description': data.get('Description', ''),
                'sector': data.get('Sector', 'Unknown'),
                'industry': data.get('Industry', 'Unknown'),
                'country': data.get('Country', 'Unknown'),
                'market_cap': self._safe_float(data.get('MarketCapitalization')),
                'pe_ratio': self._safe_float(data.get('PERatio')),
                'peg_ratio': self._safe_float(data.get('PEGRatio')),
                'book_value': self._safe_float(data.get('BookValue')),
                'dividend_per_share': self._safe_float(data.get('DividendPerShare')),
                'dividend_yield': self._safe_float(data.get('DividendYield')),
                'eps': self._safe_float(data.get('EPS')),
                'revenue_per_share': self._safe_float(data.get('RevenuePerShareTTM')),
                'profit_margin': self._safe_float(data.get('ProfitMargin')),
                'operating_margin': self._safe_float(data.get('OperatingMarginTTM')),
                'return_on_assets': self._safe_float(data.get('ReturnOnAssetsTTM')),
                'return_on_equity': self._safe_float(data.get('ReturnOnEquityTTM')),
                'revenue_ttm': self._safe_float(data.get('RevenueTTM')),
                'gross_profit_ttm': self._safe_float(data.get('GrossProfitTTM')),
                'diluted_eps_ttm': self._safe_float(data.get('DilutedEPSTTM')),
                'quarterly_earnings_growth': self._safe_float(data.get('QuarterlyEarningsGrowthYOY')),
                'quarterly_revenue_growth': self._safe_float(data.get('QuarterlyRevenueGrowthYOY')),
                'analyst_target_price': self._safe_float(data.get('AnalystTargetPrice')),
                'trailing_pe': self._safe_float(data.get('TrailingPE')),
                'forward_pe': self._safe_float(data.get('ForwardPE')),
                'price_to_sales_ratio': self._safe_float(data.get('PriceToSalesRatioTTM')),
                'price_to_book_ratio': self._safe_float(data.get('PriceToBookRatio')),
                'ev_to_revenue': self._safe_float(data.get('EVToRevenue')),
                'ev_to_ebitda': self._safe_float(data.get('EVToEBITDA')),
                'beta': self._safe_float(data.get('Beta')),
                '52_week_high': self._safe_float(data.get('52WeekHigh')),
                '52_week_low': self._safe_float(data.get('52WeekLow')),
                '50_day_ma': self._safe_float(data.get('50DayMovingAverage')),
                '200_day_ma': self._safe_float(data.get('200DayMovingAverage')),
                'shares_outstanding': self._safe_float(data.get('SharesOutstanding')),
                'shares_float': self._safe_float(data.get('SharesFloat')),
                'shares_short': self._safe_float(data.get('SharesShort')),
                'shares_short_prior_month': self._safe_float(data.get('SharesShortPriorMonth')),
                'short_ratio': self._safe_float(data.get('ShortRatio')),
                'short_percent_outstanding': self._safe_float(data.get('ShortPercentOutstanding')),
                'short_percent_float': self._safe_float(data.get('ShortPercentFloat')),
                'percent_insiders': self._safe_float(data.get('PercentInsiders')),
                'percent_institutions': self._safe_float(data.get('PercentInstitutions')),
                'forward_annual_dividend_rate': self._safe_float(data.get('ForwardAnnualDividendRate')),
                'forward_annual_dividend_yield': self._safe_float(data.get('ForwardAnnualDividendYield')),
                'payout_ratio': self._safe_float(data.get('PayoutRatio')),
                'dividend_date': data.get('DividendDate'),
                'ex_dividend_date': data.get('ExDividendDate'),
                'last_split_factor': data.get('LastSplitFactor'),
                'last_split_date': data.get('LastSplitDate')
            }
            
            return {k: v for k, v in overview.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error getting company overview for {symbol}: {str(e)}")
            return {}
    
    def get_income_statement(self, symbol: str) -> dict:
        """Get annual income statement data"""
        try:
            params = {
                'function': 'INCOME_STATEMENT',
                'symbol': symbol
            }
            
            data = self._make_request(params)
            
            if 'annualReports' not in data:
                return {}
            
            # Get the most recent annual report
            latest_report = data['annualReports'][0] if data['annualReports'] else {}
            
            income_metrics = {
                'fiscal_date_ending': latest_report.get('fiscalDateEnding'),
                'reported_currency': latest_report.get('reportedCurrency'),
                'gross_profit': self._safe_float(latest_report.get('grossProfit')),
                'total_revenue': self._safe_float(latest_report.get('totalRevenue')),
                'cost_of_revenue': self._safe_float(latest_report.get('costOfRevenue')),
                'cost_of_goods_and_services_sold': self._safe_float(latest_report.get('costofGoodsAndServicesSold')),
                'operating_income': self._safe_float(latest_report.get('operatingIncome')),
                'selling_general_administrative': self._safe_float(latest_report.get('sellingGeneralAndAdministrative')),
                'research_and_development': self._safe_float(latest_report.get('researchAndDevelopment')),
                'operating_expenses': self._safe_float(latest_report.get('operatingExpenses')),
                'investment_income_net': self._safe_float(latest_report.get('investmentIncomeNet')),
                'net_interest_income': self._safe_float(latest_report.get('netInterestIncome')),
                'interest_income': self._safe_float(latest_report.get('interestIncome')),
                'interest_expense': self._safe_float(latest_report.get('interestExpense')),
                'non_interest_income': self._safe_float(latest_report.get('nonInterestIncome')),
                'other_non_operating_income': self._safe_float(latest_report.get('otherNonOperatingIncome')),
                'depreciation': self._safe_float(latest_report.get('depreciation')),
                'depreciation_and_amortization': self._safe_float(latest_report.get('depreciationAndAmortization')),
                'income_before_tax': self._safe_float(latest_report.get('incomeBeforeTax')),
                'income_tax_expense': self._safe_float(latest_report.get('incomeTaxExpense')),
                'interest_and_debt_expense': self._safe_float(latest_report.get('interestAndDebtExpense')),
                'net_income_from_continuing_ops': self._safe_float(latest_report.get('netIncomeFromContinuingOps')),
                'comprehensive_income_net_of_tax': self._safe_float(latest_report.get('comprehensiveIncomeNetOfTax')),
                'ebit': self._safe_float(latest_report.get('ebit')),
                'ebitda': self._safe_float(latest_report.get('ebitda')),
                'net_income': self._safe_float(latest_report.get('netIncome'))
            }
            
            return {k: v for k, v in income_metrics.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error getting income statement for {symbol}: {str(e)}")
            return {}
    
    def get_balance_sheet(self, symbol: str) -> dict:
        """Get annual balance sheet data"""
        try:
            params = {
                'function': 'BALANCE_SHEET',
                'symbol': symbol
            }
            
            data = self._make_request(params)
            
            if 'annualReports' not in data:
                return {}
            
            # Get the most recent annual report
            latest_report = data['annualReports'][0] if data['annualReports'] else {}
            
            balance_metrics = {
                'fiscal_date_ending': latest_report.get('fiscalDateEnding'),
                'reported_currency': latest_report.get('reportedCurrency'),
                'total_assets': self._safe_float(latest_report.get('totalAssets')),
                'total_current_assets': self._safe_float(latest_report.get('totalCurrentAssets')),
                'cash_and_cash_equivalents': self._safe_float(latest_report.get('cashAndCashEquivalentsAtCarryingValue')),
                'cash_and_short_term_investments': self._safe_float(latest_report.get('cashAndShortTermInvestments')),
                'inventory': self._safe_float(latest_report.get('inventory')),
                'current_net_receivables': self._safe_float(latest_report.get('currentNetReceivables')),
                'total_non_current_assets': self._safe_float(latest_report.get('totalNonCurrentAssets')),
                'property_plant_equipment': self._safe_float(latest_report.get('propertyPlantEquipment')),
                'accumulated_depreciation_amortization': self._safe_float(latest_report.get('accumulatedDepreciationAmortizationPPE')),
                'intangible_assets': self._safe_float(latest_report.get('intangibleAssets')),
                'intangible_assets_excluding_goodwill': self._safe_float(latest_report.get('intangibleAssetsExcludingGoodwill')),
                'goodwill': self._safe_float(latest_report.get('goodwill')),
                'investments': self._safe_float(latest_report.get('investments')),
                'long_term_investments': self._safe_float(latest_report.get('longTermInvestments')),
                'short_term_investments': self._safe_float(latest_report.get('shortTermInvestments')),
                'other_current_assets': self._safe_float(latest_report.get('otherCurrentAssets')),
                'other_non_current_assets': self._safe_float(latest_report.get('otherNonCurrentAssets')),
                'total_liabilities': self._safe_float(latest_report.get('totalLiabilities')),
                'total_current_liabilities': self._safe_float(latest_report.get('totalCurrentLiabilities')),
                'current_accounts_payable': self._safe_float(latest_report.get('currentAccountsPayable')),
                'deferred_revenue': self._safe_float(latest_report.get('deferredRevenue')),
                'current_debt': self._safe_float(latest_report.get('currentDebt')),
                'short_term_debt': self._safe_float(latest_report.get('shortTermDebt')),
                'total_non_current_liabilities': self._safe_float(latest_report.get('totalNonCurrentLiabilities')),
                'capital_lease_obligations': self._safe_float(latest_report.get('capitalLeaseObligations')),
                'long_term_debt': self._safe_float(latest_report.get('longTermDebt')),
                'current_long_term_debt': self._safe_float(latest_report.get('currentLongTermDebt')),
                'long_term_debt_noncurrent': self._safe_float(latest_report.get('longTermDebtNoncurrent')),
                'short_long_term_debt_total': self._safe_float(latest_report.get('shortLongTermDebtTotal')),
                'other_current_liabilities': self._safe_float(latest_report.get('otherCurrentLiabilities')),
                'other_non_current_liabilities': self._safe_float(latest_report.get('otherNonCurrentLiabilities')),
                'total_shareholder_equity': self._safe_float(latest_report.get('totalShareholderEquity')),
                'treasury_stock': self._safe_float(latest_report.get('treasuryStock')),
                'retained_earnings': self._safe_float(latest_report.get('retainedEarnings')),
                'common_stock': self._safe_float(latest_report.get('commonStock')),
                'common_stock_shares_outstanding': self._safe_float(latest_report.get('commonStockSharesOutstanding'))
            }
            
            return {k: v for k, v in balance_metrics.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error getting balance sheet for {symbol}: {str(e)}")
            return {}
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert string to float"""
        if value is None or value == 'None' or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def collect_comprehensive_data(self, symbol: str) -> dict:
        """Collect all available data for a symbol"""
        try:
            # Add small delays between API calls to respect rate limits
            overview = self.get_company_overview(symbol)
            time.sleep(12)  # Alpha Vantage free tier: 5 calls per minute
            
            income = self.get_income_statement(symbol)
            time.sleep(12)
            
            balance = self.get_balance_sheet(symbol)
            time.sleep(12)
            
            # Collect daily price data
            price_success = self.collect_daily_data(symbol)
            
            return {
                'overview': overview,
                'income_statement': income,
                'balance_sheet': balance,
                'price_data_collected': price_success
            }
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive data for {symbol}: {str(e)}")
            return {}
    
    def close(self):
        """Close database session"""
        self.session.close()

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        print("Please set ALPHA_VANTAGE_API_KEY environment variable")
        exit(1)
    
    collector = AlphaVantageCollector(api_key)
    
    # Test with a single symbol
    symbol = 'AAPL'
    print(f"Collecting comprehensive data for {symbol}...")
    
    data = collector.collect_comprehensive_data(symbol)
    print(f"Data collected: {list(data.keys())}")
    
    collector.close()

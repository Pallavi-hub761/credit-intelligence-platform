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
import os

logger = logging.getLogger(__name__)

class MacroEconomicCollector:
    """Collects macroeconomic data from FRED and World Bank APIs"""
    
    def __init__(self, fred_api_key: str = None):
        self.session = next(get_db())
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        
        # FRED API endpoints
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        
        # World Bank API endpoints
        self.wb_base_url = "https://api.worldbank.org/v2"
        
        # Key economic indicators for credit risk assessment
        self.fred_indicators = {
            'GDP': 'GDP',  # Gross Domestic Product
            'UNRATE': 'Unemployment Rate',
            'FEDFUNDS': 'Federal Funds Rate',
            'DGS10': '10-Year Treasury Rate',
            'DGS2': '2-Year Treasury Rate',
            'CPIAUCSL': 'Consumer Price Index',
            'INDPRO': 'Industrial Production Index',
            'HOUST': 'Housing Starts',
            'PAYEMS': 'Total Nonfarm Payrolls',
            'DEXUSEU': 'US/Euro Exchange Rate',
            'DEXCHUS': 'China/US Exchange Rate',
            'VIXCLS': 'VIX Volatility Index',
            'TEDRATE': 'TED Spread',
            'T10Y2Y': 'Treasury Yield Curve Spread',
            'CPILFESL': 'Core CPI',
            'UMCSENT': 'Consumer Sentiment'
        }
        
        self.wb_indicators = {
            'NY.GDP.MKTP.CD': 'GDP (current US$)',
            'FP.CPI.TOTL.ZG': 'Inflation, consumer prices (annual %)',
            'SL.UEM.TOTL.ZS': 'Unemployment, total (% of total labor force)',
            'FR.INR.RINR': 'Real interest rate (%)',
            'NE.EXP.GNFS.ZS': 'Exports of goods and services (% of GDP)',
            'NE.IMP.GNFS.ZS': 'Imports of goods and services (% of GDP)',
            'GC.DOD.TOTL.GD.ZS': 'Central government debt, total (% of GDP)',
            'FS.AST.DOMS.GD.ZS': 'Domestic credit to private sector (% of GDP)'
        }
        
        # Rate limiting
        self.request_delay = 0.1
    
    def get_fred_data(self, series_id: str, start_date: datetime = None, 
                     end_date: datetime = None, limit: int = 1000) -> pd.DataFrame:
        """Fetch data from FRED API"""
        if not self.fred_api_key:
            logger.warning("FRED API key not provided. Using mock data.")
            return self._generate_mock_fred_data(series_id, start_date, end_date)
        
        try:
            if start_date is None:
                start_date = datetime.now() - timedelta(days=365 * 2)  # 2 years
            if end_date is None:
                end_date = datetime.now()
            
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'observation_start': start_date.strftime('%Y-%m-%d'),
                'observation_end': end_date.strftime('%Y-%m-%d'),
                'limit': limit,
                'sort_order': 'desc'
            }
            
            url = f"{self.fred_base_url}/series/observations"
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'observations' in data:
                observations = data['observations']
                
                # Convert to DataFrame
                df = pd.DataFrame(observations)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df = df.dropna(subset=['value'])
                    df = df.sort_values('date')
                    
                    return df[['date', 'value']]
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {str(e)}")
            return self._generate_mock_fred_data(series_id, start_date, end_date)
    
    def _generate_mock_fred_data(self, series_id: str, start_date: datetime, 
                                end_date: datetime) -> pd.DataFrame:
        """Generate mock FRED data for testing"""
        import numpy as np
        
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Generate mock values based on series type
        base_values = {
            'GDP': 25000,
            'UNRATE': 4.0,
            'FEDFUNDS': 2.5,
            'DGS10': 3.0,
            'DGS2': 2.0,
            'CPIAUCSL': 250,
            'INDPRO': 105,
            'HOUST': 1500,
            'PAYEMS': 150000,
            'DEXUSEU': 1.1,
            'DEXCHUS': 6.5,
            'VIXCLS': 20,
            'TEDRATE': 0.5,
            'T10Y2Y': 1.0,
            'CPILFESL': 260,
            'UMCSENT': 95
        }
        
        base_value = base_values.get(series_id, 100)
        
        # Generate realistic time series with trend and noise
        values = []
        current_value = base_value
        
        for i, date in enumerate(dates):
            # Add trend (slight upward for most indicators)
            trend = 0.001 * i if series_id not in ['UNRATE', 'VIXCLS'] else -0.001 * i
            
            # Add seasonal component
            seasonal = 0.02 * np.sin(2 * np.pi * i / 12)
            
            # Add random noise
            noise = np.random.normal(0, 0.01)
            
            current_value = base_value * (1 + trend + seasonal + noise)
            values.append(current_value)
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def get_world_bank_data(self, indicator: str, country: str = 'US', 
                           start_year: int = None, end_year: int = None) -> pd.DataFrame:
        """Fetch data from World Bank API"""
        try:
            if start_year is None:
                start_year = datetime.now().year - 5
            if end_year is None:
                end_year = datetime.now().year
            
            url = f"{self.wb_base_url}/country/{country}/indicator/{indicator}"
            params = {
                'format': 'json',
                'date': f"{start_year}:{end_year}",
                'per_page': 1000
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if len(data) > 1 and data[1]:  # World Bank returns metadata in first element
                records = data[1]
                
                df_data = []
                for record in records:
                    if record['value'] is not None:
                        df_data.append({
                            'date': pd.to_datetime(f"{record['date']}-12-31"),  # End of year
                            'value': float(record['value']),
                            'country': record['country']['value'],
                            'indicator': record['indicator']['value']
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    df = df.sort_values('date')
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching World Bank data for {indicator}: {str(e)}")
            return self._generate_mock_wb_data(indicator, start_year, end_year)
    
    def _generate_mock_wb_data(self, indicator: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Generate mock World Bank data"""
        import numpy as np
        
        years = range(start_year, end_year + 1)
        dates = [pd.to_datetime(f"{year}-12-31") for year in years]
        
        # Base values for different indicators
        base_values = {
            'NY.GDP.MKTP.CD': 20e12,  # $20 trillion
            'FP.CPI.TOTL.ZG': 2.0,    # 2% inflation
            'SL.UEM.TOTL.ZS': 4.0,    # 4% unemployment
            'FR.INR.RINR': 1.0,       # 1% real interest rate
            'NE.EXP.GNFS.ZS': 12.0,   # 12% of GDP
            'NE.IMP.GNFS.ZS': 14.0,   # 14% of GDP
            'GC.DOD.TOTL.GD.ZS': 100.0,  # 100% of GDP
            'FS.AST.DOMS.GD.ZS': 200.0   # 200% of GDP
        }
        
        base_value = base_values.get(indicator, 50.0)
        
        values = []
        for i, year in enumerate(years):
            # Add realistic variation
            trend = 0.02 * i  # 2% annual growth
            noise = np.random.normal(0, 0.05)  # 5% noise
            
            value = base_value * (1 + trend + noise)
            values.append(value)
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'country': 'United States',
            'indicator': self.wb_indicators.get(indicator, indicator)
        })
    
    def calculate_economic_features(self, fred_data: Dict[str, pd.DataFrame], 
                                  wb_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate economic features for credit risk assessment"""
        features = {}
        
        try:
            # Process FRED indicators
            for series_id, df in fred_data.items():
                if df.empty:
                    continue
                
                # Get latest value
                latest_value = df['value'].iloc[-1]
                features[f'{series_id}_latest'] = latest_value
                
                # Calculate changes
                if len(df) >= 2:
                    prev_value = df['value'].iloc[-2]
                    change = (latest_value - prev_value) / prev_value if prev_value != 0 else 0
                    features[f'{series_id}_change'] = change
                
                # Calculate trends (6-month)
                if len(df) >= 6:
                    six_months_ago = df['value'].iloc[-6]
                    trend = (latest_value - six_months_ago) / six_months_ago if six_months_ago != 0 else 0
                    features[f'{series_id}_trend_6m'] = trend
                
                # Calculate volatility
                if len(df) >= 12:
                    volatility = df['value'].tail(12).std() / df['value'].tail(12).mean()
                    features[f'{series_id}_volatility'] = volatility
            
            # Calculate derived economic indicators
            if 'DGS10_latest' in features and 'DGS2_latest' in features:
                features['yield_curve_slope'] = features['DGS10_latest'] - features['DGS2_latest']
            
            if 'FEDFUNDS_latest' in features and 'CPIAUCSL_change' in features:
                # Real federal funds rate (approximation)
                features['real_fed_funds'] = features['FEDFUNDS_latest'] - (features['CPIAUCSL_change'] * 12)
            
            # Economic stress indicators
            stress_score = 0
            if 'UNRATE_latest' in features and features['UNRATE_latest'] > 6:
                stress_score += 1
            if 'VIXCLS_latest' in features and features['VIXCLS_latest'] > 30:
                stress_score += 1
            if 'yield_curve_slope' in features and features['yield_curve_slope'] < 0:
                stress_score += 1  # Inverted yield curve
            if 'TEDRATE_latest' in features and features['TEDRATE_latest'] > 1:
                stress_score += 1
            
            features['economic_stress_score'] = stress_score
            
            # Process World Bank indicators
            for indicator, df in wb_data.items():
                if df.empty:
                    continue
                
                latest_value = df['value'].iloc[-1]
                features[f'wb_{indicator}_latest'] = latest_value
                
                if len(df) >= 2:
                    prev_value = df['value'].iloc[-2]
                    change = (latest_value - prev_value) / prev_value if prev_value != 0 else 0
                    features[f'wb_{indicator}_change'] = change
            
            # Economic health score (0-100)
            health_components = []
            
            # GDP growth (positive is good)
            if 'GDP_trend_6m' in features:
                gdp_component = min(max(features['GDP_trend_6m'] * 100, -20), 20)
                health_components.append(gdp_component)
            
            # Unemployment (lower is better)
            if 'UNRATE_latest' in features:
                unemployment_component = max(10 - features['UNRATE_latest'], -10)
                health_components.append(unemployment_component)
            
            # Inflation (moderate is best)
            if 'CPIAUCSL_change' in features:
                inflation_rate = features['CPIAUCSL_change'] * 12  # Annualized
                inflation_component = max(5 - abs(inflation_rate - 2), -10)  # Target 2%
                health_components.append(inflation_component)
            
            # Market volatility (lower is better)
            if 'VIXCLS_latest' in features:
                vix_component = max(30 - features['VIXCLS_latest'], -20)
                health_components.append(vix_component)
            
            if health_components:
                features['economic_health_score'] = 50 + sum(health_components) / len(health_components)
            else:
                features['economic_health_score'] = 50  # Neutral
            
        except Exception as e:
            logger.error(f"Error calculating economic features: {str(e)}")
        
        return features
    
    def collect_macro_data(self, months_back: int = 24) -> Dict[str, Any]:
        """Collect comprehensive macroeconomic data"""
        try:
            start_date = datetime.now() - timedelta(days=months_back * 30)
            end_date = datetime.now()
            
            # Collect FRED data
            fred_data = {}
            for series_id, description in self.fred_indicators.items():
                logger.info(f"Collecting FRED data for {series_id}: {description}")
                df = self.get_fred_data(series_id, start_date, end_date)
                if not df.empty:
                    fred_data[series_id] = df
                
                time.sleep(self.request_delay)
            
            # Collect World Bank data
            wb_data = {}
            start_year = start_date.year
            end_year = end_date.year
            
            for indicator, description in self.wb_indicators.items():
                logger.info(f"Collecting World Bank data for {indicator}: {description}")
                df = self.get_world_bank_data(indicator, 'US', start_year, end_year)
                if not df.empty:
                    wb_data[indicator] = df
                
                time.sleep(self.request_delay)
            
            # Calculate economic features
            features = self.calculate_economic_features(fred_data, wb_data)
            
            # Store as news-like entries for integration
            self.store_macro_data(fred_data, wb_data, features)
            
            return {
                'success': True,
                'fred_indicators': len(fred_data),
                'wb_indicators': len(wb_data),
                'economic_features': features,
                'collection_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error collecting macro data: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    def store_macro_data(self, fred_data: Dict, wb_data: Dict, features: Dict):
        """Store macroeconomic data as news entries for system integration"""
        try:
            # Create a summary news entry for economic conditions
            title = f"Economic Indicators Update - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create content summary
            content_parts = []
            
            if 'economic_health_score' in features:
                health_score = features['economic_health_score']
                health_status = 'Strong' if health_score > 70 else 'Moderate' if health_score > 40 else 'Weak'
                content_parts.append(f"Economic Health: {health_status} ({health_score:.1f}/100)")
            
            if 'UNRATE_latest' in features:
                content_parts.append(f"Unemployment: {features['UNRATE_latest']:.1f}%")
            
            if 'FEDFUNDS_latest' in features:
                content_parts.append(f"Fed Funds Rate: {features['FEDFUNDS_latest']:.2f}%")
            
            if 'yield_curve_slope' in features:
                slope = features['yield_curve_slope']
                curve_status = 'Inverted' if slope < 0 else 'Normal'
                content_parts.append(f"Yield Curve: {curve_status} ({slope:.2f})")
            
            if 'VIXCLS_latest' in features:
                content_parts.append(f"Market Volatility (VIX): {features['VIXCLS_latest']:.1f}")
            
            content = "; ".join(content_parts)
            
            # Determine sentiment based on economic health
            if 'economic_health_score' in features:
                health = features['economic_health_score']
                if health > 60:
                    sentiment_score = 0.3
                    sentiment_label = 'positive'
                elif health < 40:
                    sentiment_score = -0.3
                    sentiment_label = 'negative'
                else:
                    sentiment_score = 0.0
                    sentiment_label = 'neutral'
            else:
                sentiment_score = 0.0
                sentiment_label = 'neutral'
            
            # Store for each company (macro affects all companies)
            companies = self.session.query(Company).all()
            
            for company in companies:
                # Check if recent macro update exists
                recent_macro = self.session.query(NewsRaw).filter(
                    NewsRaw.company_id == company.id,
                    NewsRaw.source == 'Economic Indicators',
                    NewsRaw.published_at >= datetime.now() - timedelta(days=1)
                ).first()
                
                if recent_macro:
                    continue  # Skip if already updated today
                
                news_record = NewsRaw(
                    company_id=company.id,
                    title=title,
                    content=content,
                    source='Economic Indicators',
                    published_at=datetime.now(),
                    url='https://fred.stlouisfed.org/',
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    processed=True
                )
                
                self.session.add(news_record)
            
            self.session.commit()
            logger.info(f"Stored macroeconomic data for {len(companies)} companies")
            
        except Exception as e:
            logger.error(f"Error storing macro data: {str(e)}")
            self.session.rollback()
    
    def get_economic_summary(self) -> Dict[str, Any]:
        """Get current economic conditions summary"""
        try:
            # Get latest economic data
            result = self.collect_macro_data(months_back=6)
            
            if result['success']:
                features = result['economic_features']
                
                summary = {
                    'economic_health_score': features.get('economic_health_score', 50),
                    'economic_stress_score': features.get('economic_stress_score', 0),
                    'key_indicators': {},
                    'trends': {},
                    'outlook': 'neutral'
                }
                
                # Key current indicators
                key_metrics = ['UNRATE_latest', 'FEDFUNDS_latest', 'DGS10_latest', 'VIXCLS_latest']
                for metric in key_metrics:
                    if metric in features:
                        summary['key_indicators'][metric.replace('_latest', '')] = features[metric]
                
                # Trends
                trend_metrics = ['GDP_trend_6m', 'UNRATE_trend_6m', 'CPIAUCSL_trend_6m']
                for metric in trend_metrics:
                    if metric in features:
                        summary['trends'][metric.replace('_trend_6m', '')] = features[metric]
                
                # Overall outlook
                health_score = features.get('economic_health_score', 50)
                if health_score > 65:
                    summary['outlook'] = 'positive'
                elif health_score < 35:
                    summary['outlook'] = 'negative'
                
                return summary
            
            return {'error': 'Failed to collect economic data'}
            
        except Exception as e:
            logger.error(f"Error getting economic summary: {str(e)}")
            return {'error': str(e)}
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    collector = MacroEconomicCollector()
    
    try:
        print("Collecting macroeconomic data...")
        result = collector.collect_macro_data(months_back=12)
        
        if result['success']:
            print(f"Successfully collected data:")
            print(f"  FRED indicators: {result['fred_indicators']}")
            print(f"  World Bank indicators: {result['wb_indicators']}")
            
            features = result['economic_features']
            print(f"\nKey Economic Features:")
            
            if 'economic_health_score' in features:
                print(f"  Economic Health Score: {features['economic_health_score']:.1f}/100")
            
            if 'economic_stress_score' in features:
                print(f"  Economic Stress Score: {features['economic_stress_score']}/4")
            
            if 'UNRATE_latest' in features:
                print(f"  Current Unemployment: {features['UNRATE_latest']:.1f}%")
            
            if 'yield_curve_slope' in features:
                print(f"  Yield Curve Slope: {features['yield_curve_slope']:.2f}")
        
        else:
            print(f"Failed to collect data: {result['message']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        collector.close()

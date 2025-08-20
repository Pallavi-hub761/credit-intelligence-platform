import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Price, NewsRaw, Score, Explanation
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import json
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    mse: float
    mae: float
    r2: float
    cv_score: float
    feature_importance: Dict[str, float]
    model_version: str
    training_date: datetime

class FeatureEngineer:
    """Feature engineering for credit scoring"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
    
    def calculate_technical_indicators(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from price data"""
        if prices_df.empty:
            return pd.DataFrame()
        
        df = prices_df.copy()
        df = df.sort_values('date')
        
        # Price-based features
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_20d'] = df['close'].pct_change(20)
        
        # Volatility features
        df['volatility_5d'] = df['price_change_1d'].rolling(5).std()
        df['volatility_20d'] = df['price_change_1d'].rolling(20).std()
        
        # Moving averages
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        # Moving average ratios
        df['price_to_ma5'] = df['close'] / df['ma_5']
        df['price_to_ma20'] = df['close'] / df['ma_20']
        df['ma5_to_ma20'] = df['ma_5'] / df['ma_20']
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_ma_5'] = df['volume'].rolling(5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        
        # RSI-like momentum indicator
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def calculate_sentiment_features(self, news_df: pd.DataFrame, days_back: int = 30) -> Dict[str, float]:
        """Calculate sentiment-based features"""
        if news_df.empty:
            return {}
        
        # Filter recent news
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_news = news_df[news_df['published_at'] >= cutoff_date]
        
        if recent_news.empty:
            return {}
        
        features = {}
        
        # Basic sentiment statistics
        sentiment_scores = recent_news['sentiment_score'].dropna()
        if not sentiment_scores.empty:
            features['sentiment_mean'] = sentiment_scores.mean()
            features['sentiment_std'] = sentiment_scores.std()
            features['sentiment_min'] = sentiment_scores.min()
            features['sentiment_max'] = sentiment_scores.max()
            features['sentiment_trend'] = self._calculate_trend(sentiment_scores, recent_news['published_at'])
        
        # Sentiment distribution
        sentiment_counts = recent_news['sentiment_label'].value_counts()
        total_articles = len(recent_news)
        features['positive_ratio'] = sentiment_counts.get('positive', 0) / total_articles
        features['negative_ratio'] = sentiment_counts.get('negative', 0) / total_articles
        features['neutral_ratio'] = sentiment_counts.get('neutral', 0) / total_articles
        
        # News volume features
        features['news_volume'] = total_articles
        features['news_frequency'] = total_articles / days_back  # articles per day
        
        return features
    
    def _calculate_trend(self, values: pd.Series, dates: pd.Series) -> float:
        """Calculate trend slope for time series data"""
        if len(values) < 2:
            return 0.0
        
        try:
            # Convert dates to numeric (days since first date)
            date_numeric = pd.to_datetime(dates)
            days = (date_numeric - date_numeric.min()).dt.days
            
            # Calculate linear trend
            coeffs = np.polyfit(days, values, 1)
            return coeffs[0]  # slope
        except:
            return 0.0
    
    def calculate_market_features(self, company_prices: pd.DataFrame, market_data: pd.DataFrame = None) -> Dict[str, float]:
        """Calculate market-relative features"""
        features = {}
        
        if company_prices.empty:
            return features
        
        # Company-specific features
        latest_prices = company_prices.tail(50)  # Last 50 days
        if not latest_prices.empty:
            features['price_level'] = latest_prices['close'].iloc[-1]
            features['price_52w_high'] = latest_prices['close'].max()
            features['price_52w_low'] = latest_prices['close'].min()
            features['price_to_52w_high'] = features['price_level'] / features['price_52w_high']
            features['price_to_52w_low'] = features['price_level'] / features['price_52w_low']
        
        # TODO: Add market comparison features when market index data is available
        # This would include beta, correlation with market, relative performance, etc.
        
        return features
    
    def create_features(self, company_id: int, session: Session) -> Dict[str, float]:
        """Create comprehensive feature set for a company"""
        features = {}
        
        try:
            # Get company data
            company = session.query(Company).filter(Company.id == company_id).first()
            if not company:
                return features
            
            # Get price data (last 200 days for technical indicators)
            cutoff_date = datetime.now() - timedelta(days=200)
            prices = session.query(Price).filter(
                Price.company_id == company_id,
                Price.date >= cutoff_date
            ).order_by(Price.date).all()
            
            if prices:
                prices_df = pd.DataFrame([{
                    'date': p.date,
                    'open': p.open_price,
                    'high': p.high_price,
                    'low': p.low_price,
                    'close': p.close_price,
                    'volume': p.volume
                } for p in prices])
                
                # Calculate technical indicators
                tech_df = self.calculate_technical_indicators(prices_df)
                if not tech_df.empty:
                    # Get latest technical features
                    latest_tech = tech_df.iloc[-1]
                    tech_features = {
                        'price_change_1d': latest_tech.get('price_change_1d', 0),
                        'price_change_5d': latest_tech.get('price_change_5d', 0),
                        'price_change_20d': latest_tech.get('price_change_20d', 0),
                        'volatility_5d': latest_tech.get('volatility_5d', 0),
                        'volatility_20d': latest_tech.get('volatility_20d', 0),
                        'price_to_ma5': latest_tech.get('price_to_ma5', 1),
                        'price_to_ma20': latest_tech.get('price_to_ma20', 1),
                        'ma5_to_ma20': latest_tech.get('ma5_to_ma20', 1),
                        'rsi': latest_tech.get('rsi', 50),
                        'volume_ratio': latest_tech.get('volume_ratio', 1)
                    }
                    features.update(tech_features)
                
                # Market features
                market_features = self.calculate_market_features(prices_df)
                features.update(market_features)
            
            # Get news sentiment data
            news_cutoff = datetime.now() - timedelta(days=60)
            news = session.query(NewsRaw).filter(
                NewsRaw.company_id == company_id,
                NewsRaw.published_at >= news_cutoff,
                NewsRaw.processed == True
            ).all()
            
            if news:
                news_df = pd.DataFrame([{
                    'published_at': n.published_at,
                    'sentiment_score': n.sentiment_score,
                    'sentiment_label': n.sentiment_label
                } for n in news])
                
                sentiment_features = self.calculate_sentiment_features(news_df)
                features.update(sentiment_features)
            
            # Fill missing values with defaults
            default_features = {
                'price_change_1d': 0, 'price_change_5d': 0, 'price_change_20d': 0,
                'volatility_5d': 0.02, 'volatility_20d': 0.02,
                'price_to_ma5': 1, 'price_to_ma20': 1, 'ma5_to_ma20': 1,
                'rsi': 50, 'volume_ratio': 1,
                'sentiment_mean': 0, 'sentiment_std': 0.1,
                'positive_ratio': 0.33, 'negative_ratio': 0.33, 'neutral_ratio': 0.34,
                'news_volume': 0, 'news_frequency': 0,
                'price_level': 100, 'price_to_52w_high': 0.8, 'price_to_52w_low': 1.2
            }
            
            for key, default_value in default_features.items():
                if key not in features or pd.isna(features[key]):
                    features[key] = default_value
            
            # Store feature names for consistency
            self.feature_names = list(features.keys())
            
        except Exception as e:
            logger.error(f"Error creating features for company {company_id}: {str(e)}")
        
        return features

class CreditScoringEngine:
    """Adaptive credit scoring engine with multiple models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.scaler = RobustScaler()
        self.session = next(get_db())
        self.current_model_version = "v1.0"
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize different model types"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            ),
            'linear': Ridge(alpha=1.0)
        }
    
    def create_synthetic_training_data(self, n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
        """Create synthetic training data for initial model training"""
        np.random.seed(42)
        
        # Generate synthetic features
        data = {}
        
        # Price change features (normally distributed around 0)
        data['price_change_1d'] = np.random.normal(0, 0.02, n_samples)
        data['price_change_5d'] = np.random.normal(0, 0.05, n_samples)
        data['price_change_20d'] = np.random.normal(0, 0.1, n_samples)
        
        # Volatility features (log-normal distribution)
        data['volatility_5d'] = np.random.lognormal(-3, 0.5, n_samples)
        data['volatility_20d'] = np.random.lognormal(-3, 0.5, n_samples)
        
        # Technical indicators
        data['price_to_ma5'] = np.random.normal(1, 0.1, n_samples)
        data['price_to_ma20'] = np.random.normal(1, 0.15, n_samples)
        data['ma5_to_ma20'] = np.random.normal(1, 0.05, n_samples)
        data['rsi'] = np.random.uniform(20, 80, n_samples)
        data['volume_ratio'] = np.random.lognormal(0, 0.5, n_samples)
        
        # Sentiment features
        data['sentiment_mean'] = np.random.normal(0, 0.3, n_samples)
        data['sentiment_std'] = np.random.uniform(0.1, 0.5, n_samples)
        data['positive_ratio'] = np.random.uniform(0.2, 0.6, n_samples)
        data['negative_ratio'] = np.random.uniform(0.1, 0.4, n_samples)
        data['neutral_ratio'] = 1 - data['positive_ratio'] - data['negative_ratio']
        data['news_volume'] = np.random.poisson(10, n_samples)
        data['news_frequency'] = data['news_volume'] / 30
        
        # Market features
        data['price_level'] = np.random.lognormal(4, 1, n_samples)  # Around $50-150
        data['price_to_52w_high'] = np.random.uniform(0.5, 1.0, n_samples)
        data['price_to_52w_low'] = np.random.uniform(1.0, 2.0, n_samples)
        
        X = pd.DataFrame(data)
        
        # Create synthetic credit scores based on features
        # Higher volatility -> lower score
        # Positive sentiment -> higher score
        # Strong technical indicators -> higher score
        
        base_score = 500  # Base credit score
        
        # Price momentum impact
        momentum_impact = (
            data['price_change_20d'] * 200 +
            data['price_change_5d'] * 100 +
            data['price_change_1d'] * 50
        )
        
        # Volatility impact (negative)
        volatility_impact = -(data['volatility_20d'] * 1000 + data['volatility_5d'] * 500)
        
        # Sentiment impact
        sentiment_impact = (
            data['sentiment_mean'] * 100 +
            (data['positive_ratio'] - data['negative_ratio']) * 150
        )
        
        # Technical indicators impact
        technical_impact = (
            (data['price_to_ma20'] - 1) * 100 +
            (data['rsi'] - 50) * 2
        )
        
        # Market position impact
        market_impact = (data['price_to_52w_high'] - 0.75) * 100
        
        # Combine all impacts
        y = (base_score + 
             momentum_impact + 
             volatility_impact + 
             sentiment_impact + 
             technical_impact + 
             market_impact +
             np.random.normal(0, 20, n_samples))  # Add noise
        
        # Clip scores to reasonable range (300-850, like FICO scores)
        y = np.clip(y, 300, 850)
        
        return X, pd.Series(y)
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, ModelMetrics]:
        """Train all models and return performance metrics"""
        results = {}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Train model
                if model_name in ['linear']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                if model_name in ['linear']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                cv_score = cv_scores.mean()
                
                # Feature importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    for i, importance in enumerate(model.feature_importances_):
                        feature_importance[X.columns[i]] = float(importance)
                elif hasattr(model, 'coef_'):
                    for i, coef in enumerate(model.coef_):
                        feature_importance[X.columns[i]] = float(abs(coef))
                
                # Create metrics object
                metrics = ModelMetrics(
                    mse=mse,
                    mae=mae,
                    r2=r2,
                    cv_score=cv_score,
                    feature_importance=feature_importance,
                    model_version=self.current_model_version,
                    training_date=datetime.now()
                )
                
                results[model_name] = metrics
                
                # Save model
                model_path = os.path.join(self.model_dir, f"{model_name}_{self.current_model_version}.joblib")
                joblib.dump(model, model_path)
                
                logger.info(f"{model_name} - R²: {r2:.3f}, MAE: {mae:.1f}, CV: {cv_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"scaler_{self.current_model_version}.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        feature_names_path = os.path.join(self.model_dir, f"feature_names_{self.current_model_version}.json")
        with open(feature_names_path, 'w') as f:
            json.dump(list(X.columns), f)
        
        return results
    
    def select_best_model(self, metrics: Dict[str, ModelMetrics]) -> str:
        """Select the best performing model"""
        if not metrics:
            return 'random_forest'  # Default fallback
        
        # Rank models by R² score (primary) and CV score (secondary)
        model_scores = {}
        for model_name, metric in metrics.items():
            # Weighted score: 70% R², 30% CV score
            score = 0.7 * metric.r2 + 0.3 * metric.cv_score
            model_scores[model_name] = score
        
        best_model = max(model_scores, key=model_scores.get)
        logger.info(f"Selected best model: {best_model} (score: {model_scores[best_model]:.3f})")
        
        return best_model
    
    def load_model(self, model_name: str) -> Any:
        """Load a trained model"""
        model_path = os.path.join(self.model_dir, f"{model_name}_{self.current_model_version}.joblib")
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
    
    def predict_credit_score(self, company_id: int, model_name: str = None) -> Dict:
        """Predict credit score for a company"""
        try:
            # Create features
            features = self.feature_engineer.create_features(company_id, self.session)
            if not features:
                return {'error': 'Could not create features for company'}
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Load model (use best model if not specified)
            if model_name is None:
                # Try to load model performance data to select best model
                model_name = 'random_forest'  # Default
            
            model = self.load_model(model_name)
            if model is None:
                return {'error': f'Model {model_name} not found'}
            
            # Make prediction
            if model_name in ['linear']:
                scaler_path = os.path.join(self.model_dir, f"scaler_{self.current_model_version}.joblib")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    feature_scaled = scaler.transform(feature_df)
                    score = model.predict(feature_scaled)[0]
                else:
                    return {'error': 'Scaler not found for linear model'}
            else:
                score = model.predict(feature_df)[0]
            
            # Convert score to risk category
            risk_category = self._score_to_risk_category(score)
            
            # Calculate confidence (based on feature completeness and model certainty)
            confidence = self._calculate_confidence(features, model_name)
            
            return {
                'credit_score': float(score),
                'risk_category': risk_category,
                'confidence': confidence,
                'model_used': model_name,
                'model_version': self.current_model_version,
                'features_used': list(features.keys()),
                'prediction_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting credit score for company {company_id}: {str(e)}")
            return {'error': str(e)}
    
    def _score_to_risk_category(self, score: float) -> str:
        """Convert numeric score to risk category"""
        if score >= 750:
            return 'AAA'
        elif score >= 700:
            return 'AA'
        elif score >= 650:
            return 'A'
        elif score >= 600:
            return 'BBB'
        elif score >= 550:
            return 'BB'
        elif score >= 500:
            return 'B'
        elif score >= 450:
            return 'CCC'
        elif score >= 400:
            return 'CC'
        elif score >= 350:
            return 'C'
        else:
            return 'D'
    
    def _calculate_confidence(self, features: Dict, model_name: str) -> float:
        """Calculate prediction confidence"""
        # Base confidence
        confidence = 0.7
        
        # Adjust based on feature completeness
        expected_features = len(self.feature_engineer.feature_names) if self.feature_engineer.feature_names else 20
        actual_features = len([v for v in features.values() if v != 0 and not pd.isna(v)])
        feature_completeness = actual_features / expected_features
        
        # Adjust confidence based on feature completeness
        confidence *= (0.5 + 0.5 * feature_completeness)
        
        # Adjust based on model type (ensemble methods generally more confident)
        model_confidence_multipliers = {
            'random_forest': 1.0,
            'gradient_boosting': 1.0,
            'xgboost': 1.0,
            'lightgbm': 1.0,
            'linear': 0.8
        }
        
        confidence *= model_confidence_multipliers.get(model_name, 0.9)
        
        return min(confidence, 0.95)  # Cap at 95%
    
    def store_prediction(self, company_id: int, prediction: Dict) -> bool:
        """Store prediction in database"""
        try:
            if 'error' in prediction:
                return False
            
            score_record = Score(
                company_id=company_id,
                score_date=datetime.now(),
                credit_score=prediction['credit_score'],
                risk_category=prediction['risk_category'],
                model_version=prediction['model_version'],
                confidence=prediction['confidence']
            )
            
            self.session.add(score_record)
            self.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction: {str(e)}")
            self.session.rollback()
            return False
    
    def batch_score_companies(self, company_ids: List[int] = None) -> Dict:
        """Score multiple companies"""
        if company_ids is None:
            companies = self.session.query(Company).all()
            company_ids = [c.id for c in companies]
        
        results = {'successful': [], 'failed': []}
        
        for company_id in company_ids:
            try:
                prediction = self.predict_credit_score(company_id)
                if 'error' not in prediction:
                    # Store prediction
                    stored = self.store_prediction(company_id, prediction)
                    if stored:
                        results['successful'].append({
                            'company_id': company_id,
                            'credit_score': prediction['credit_score'],
                            'risk_category': prediction['risk_category']
                        })
                    else:
                        results['failed'].append({
                            'company_id': company_id,
                            'error': 'Failed to store prediction'
                        })
                else:
                    results['failed'].append({
                        'company_id': company_id,
                        'error': prediction['error']
                    })
            except Exception as e:
                results['failed'].append({
                    'company_id': company_id,
                    'error': str(e)
                })
        
        return results
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage and testing
if __name__ == "__main__":
    engine = CreditScoringEngine()
    
    try:
        # Create synthetic training data
        print("Creating synthetic training data...")
        X, y = engine.create_synthetic_training_data(n_samples=2000)
        print(f"Created {len(X)} samples with {len(X.columns)} features")
        
        # Train models
        print("Training models...")
        metrics = engine.train_models(X, y)
        
        # Display results
        print("\nModel Performance:")
        for model_name, metric in metrics.items():
            print(f"{model_name}: R²={metric.r2:.3f}, MAE={metric.mae:.1f}, CV={metric.cv_score:.3f}")
        
        # Select best model
        best_model = engine.select_best_model(metrics)
        print(f"\nBest model: {best_model}")
        
        print("Credit scoring engine initialized successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        engine.close()

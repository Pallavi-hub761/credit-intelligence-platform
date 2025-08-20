import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, Price, NewsRaw, Score, Explanation
import logging
import joblib
import json
import os
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)

@dataclass
class FeatureContribution:
    """Feature contribution to credit score"""
    feature_name: str
    value: float
    contribution: float
    importance: float
    description: str

@dataclass
class TrendAnalysis:
    """Trend analysis for a feature or score"""
    metric_name: str
    current_value: float
    trend_direction: str  # 'improving', 'declining', 'stable'
    trend_strength: float  # 0-1
    period_days: int
    historical_values: List[float]
    description: str

class ExplainabilityEngine:
    """Provides explanations for credit score predictions"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.session = next(get_db())
        self.feature_descriptions = self._get_feature_descriptions()
        
    def _get_feature_descriptions(self) -> Dict[str, str]:
        """Get human-readable descriptions for features"""
        return {
            'price_change_1d': 'Daily stock price change (%)',
            'price_change_5d': '5-day stock price change (%)',
            'price_change_20d': '20-day stock price change (%)',
            'volatility_5d': '5-day price volatility',
            'volatility_20d': '20-day price volatility',
            'price_to_ma5': 'Price relative to 5-day moving average',
            'price_to_ma20': 'Price relative to 20-day moving average',
            'ma5_to_ma20': '5-day to 20-day moving average ratio',
            'rsi': 'Relative Strength Index (momentum indicator)',
            'volume_ratio': 'Trading volume relative to average',
            'sentiment_mean': 'Average news sentiment score',
            'sentiment_std': 'News sentiment volatility',
            'positive_ratio': 'Proportion of positive news',
            'negative_ratio': 'Proportion of negative news',
            'neutral_ratio': 'Proportion of neutral news',
            'news_volume': 'Number of news articles',
            'news_frequency': 'Daily news frequency',
            'price_level': 'Current stock price level',
            'price_to_52w_high': 'Price relative to 52-week high',
            'price_to_52w_low': 'Price relative to 52-week low'
        }
    
    def calculate_feature_contributions(self, company_id: int, model_name: str = 'random_forest') -> List[FeatureContribution]:
        """Calculate individual feature contributions to the credit score"""
        contributions = []
        
        try:
            # Load model and get feature importance
            model_path = os.path.join(self.model_dir, f"{model_name}_v1.0.joblib")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return contributions
            
            model = joblib.load(model_path)
            
            # Get feature names
            feature_names_path = os.path.join(self.model_dir, "feature_names_v1.0.json")
            if os.path.exists(feature_names_path):
                with open(feature_names_path, 'r') as f:
                    feature_names = json.load(f)
            else:
                logger.error("Feature names file not found")
                return contributions
            
            # Get company features (this would come from FeatureEngineer)
            from .credit_scoring_engine import FeatureEngineer
            feature_engineer = FeatureEngineer()
            features = feature_engineer.create_features(company_id, self.session)
            
            if not features:
                return contributions
            
            # Create feature DataFrame
            feature_df = pd.DataFrame([features])
            
            # Get model prediction for baseline
            if hasattr(model, 'predict'):
                base_prediction = model.predict(feature_df)[0]
            else:
                base_prediction = 500  # Default baseline
            
            # Calculate feature importance from model
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                # Use uniform importance if not available
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            # Calculate contributions using SHAP-like approach
            # For tree-based models, we approximate SHAP values
            for i, feature_name in enumerate(feature_names):
                if feature_name in features:
                    feature_value = features[feature_name]
                    importance = importances[i] if i < len(importances) else 0
                    
                    # Estimate contribution based on feature value and importance
                    # This is a simplified approach - in production, use actual SHAP
                    contribution = self._estimate_feature_contribution(
                        feature_name, feature_value, importance, base_prediction
                    )
                    
                    contributions.append(FeatureContribution(
                        feature_name=feature_name,
                        value=feature_value,
                        contribution=contribution,
                        importance=importance,
                        description=self.feature_descriptions.get(feature_name, feature_name)
                    ))
            
            # Sort by absolute contribution
            contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
            
        except Exception as e:
            logger.error(f"Error calculating feature contributions: {str(e)}")
        
        return contributions
    
    def _estimate_feature_contribution(self, feature_name: str, value: float, importance: float, base_score: float) -> float:
        """Estimate feature contribution to credit score"""
        # This is a simplified estimation - in production, use SHAP or LIME
        
        # Define expected ranges and directions for different features
        feature_impacts = {
            'price_change_1d': {'direction': 1, 'scale': 100},
            'price_change_5d': {'direction': 1, 'scale': 80},
            'price_change_20d': {'direction': 1, 'scale': 60},
            'volatility_5d': {'direction': -1, 'scale': 200},
            'volatility_20d': {'direction': -1, 'scale': 150},
            'price_to_ma5': {'direction': 1, 'scale': 50, 'baseline': 1.0},
            'price_to_ma20': {'direction': 1, 'scale': 40, 'baseline': 1.0},
            'rsi': {'direction': 0, 'scale': 1, 'baseline': 50},  # Neutral around 50
            'sentiment_mean': {'direction': 1, 'scale': 100},
            'positive_ratio': {'direction': 1, 'scale': 80, 'baseline': 0.33},
            'negative_ratio': {'direction': -1, 'scale': 80, 'baseline': 0.33},
            'price_to_52w_high': {'direction': 1, 'scale': 60, 'baseline': 0.8},
            'news_volume': {'direction': 0.5, 'scale': 2}  # More news can be positive or negative
        }
        
        if feature_name not in feature_impacts:
            # Default neutral impact
            return 0.0
        
        impact_config = feature_impacts[feature_name]
        direction = impact_config['direction']
        scale = impact_config['scale']
        baseline = impact_config.get('baseline', 0.0)
        
        # Calculate deviation from baseline
        deviation = value - baseline
        
        # Calculate contribution
        if direction == 0:  # Neutral features (optimal around baseline)
            contribution = -abs(deviation) * scale * importance
        else:
            contribution = direction * deviation * scale * importance
        
        return contribution
    
    def analyze_trends(self, company_id: int, days_back: int = 90) -> List[TrendAnalysis]:
        """Analyze trends in key metrics"""
        trends = []
        
        try:
            # Get historical price data
            cutoff_date = datetime.now() - timedelta(days=days_back)
            prices = self.session.query(Price).filter(
                Price.company_id == company_id,
                Price.date >= cutoff_date
            ).order_by(Price.date).all()
            
            if prices:
                price_data = pd.DataFrame([{
                    'date': p.date,
                    'close': p.close_price,
                    'volume': p.volume
                } for p in prices])
                
                # Price trend
                price_trend = self._calculate_trend_analysis(
                    'Stock Price',
                    price_data['close'].values,
                    price_data['date'].values,
                    days_back
                )
                trends.append(price_trend)
                
                # Volatility trend
                price_changes = price_data['close'].pct_change().dropna()
                volatility_values = price_changes.rolling(5).std().dropna().values
                if len(volatility_values) > 5:
                    vol_trend = self._calculate_trend_analysis(
                        'Price Volatility',
                        volatility_values,
                        price_data['date'].iloc[-len(volatility_values):].values,
                        days_back
                    )
                    trends.append(vol_trend)
            
            # Get historical sentiment data
            news = self.session.query(NewsRaw).filter(
                NewsRaw.company_id == company_id,
                NewsRaw.published_at >= cutoff_date,
                NewsRaw.processed == True,
                NewsRaw.sentiment_score.isnot(None)
            ).order_by(NewsRaw.published_at).all()
            
            if news:
                # Group by day and calculate daily average sentiment
                news_data = pd.DataFrame([{
                    'date': n.published_at.date(),
                    'sentiment': n.sentiment_score
                } for n in news])
                
                daily_sentiment = news_data.groupby('date')['sentiment'].mean()
                
                if len(daily_sentiment) > 5:
                    sentiment_trend = self._calculate_trend_analysis(
                        'News Sentiment',
                        daily_sentiment.values,
                        daily_sentiment.index.values,
                        days_back
                    )
                    trends.append(sentiment_trend)
            
            # Get historical credit scores
            scores = self.session.query(Score).filter(
                Score.company_id == company_id,
                Score.score_date >= cutoff_date
            ).order_by(Score.score_date).all()
            
            if scores:
                score_values = [s.credit_score for s in scores]
                score_dates = [s.score_date for s in scores]
                
                if len(score_values) > 2:
                    score_trend = self._calculate_trend_analysis(
                        'Credit Score',
                        score_values,
                        score_dates,
                        days_back
                    )
                    trends.append(score_trend)
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
        
        return trends
    
    def _calculate_trend_analysis(self, metric_name: str, values: np.ndarray, dates: np.ndarray, period_days: int) -> TrendAnalysis:
        """Calculate trend analysis for a time series"""
        if len(values) < 2:
            return TrendAnalysis(
                metric_name=metric_name,
                current_value=values[-1] if len(values) > 0 else 0,
                trend_direction='stable',
                trend_strength=0.0,
                period_days=period_days,
                historical_values=values.tolist(),
                description=f"Insufficient data for {metric_name} trend analysis"
            )
        
        # Convert dates to numeric for trend calculation
        if hasattr(dates[0], 'timestamp'):
            date_numeric = np.array([d.timestamp() for d in dates])
        else:
            date_numeric = np.arange(len(dates))
        
        # Calculate linear trend
        try:
            coeffs = np.polyfit(date_numeric, values, 1)
            slope = coeffs[0]
            
            # Normalize slope by value range for comparison
            value_range = np.max(values) - np.min(values)
            if value_range > 0:
                normalized_slope = slope / value_range * len(values)
            else:
                normalized_slope = 0
            
            # Determine trend direction and strength
            if abs(normalized_slope) < 0.1:
                direction = 'stable'
                strength = abs(normalized_slope) / 0.1
            elif normalized_slope > 0:
                direction = 'improving'
                strength = min(abs(normalized_slope) / 2.0, 1.0)
            else:
                direction = 'declining'
                strength = min(abs(normalized_slope) / 2.0, 1.0)
            
            # Create description
            if direction == 'stable':
                description = f"{metric_name} has remained relatively stable over the past {period_days} days"
            elif direction == 'improving':
                description = f"{metric_name} has been improving over the past {period_days} days"
            else:
                description = f"{metric_name} has been declining over the past {period_days} days"
            
            return TrendAnalysis(
                metric_name=metric_name,
                current_value=float(values[-1]),
                trend_direction=direction,
                trend_strength=float(strength),
                period_days=period_days,
                historical_values=values.tolist(),
                description=description
            )
            
        except Exception as e:
            logger.error(f"Error calculating trend for {metric_name}: {str(e)}")
            return TrendAnalysis(
                metric_name=metric_name,
                current_value=float(values[-1]) if len(values) > 0 else 0,
                trend_direction='stable',
                trend_strength=0.0,
                period_days=period_days,
                historical_values=values.tolist(),
                description=f"Error calculating trend for {metric_name}"
            )
    
    def generate_explanation(self, company_id: int, score_id: int = None) -> Dict:
        """Generate comprehensive explanation for a credit score"""
        try:
            # Get company info
            company = self.session.query(Company).filter(Company.id == company_id).first()
            if not company:
                return {'error': 'Company not found'}
            
            # Get latest score if score_id not provided
            if score_id is None:
                latest_score = self.session.query(Score).filter(
                    Score.company_id == company_id
                ).order_by(Score.score_date.desc()).first()
                if not latest_score:
                    return {'error': 'No credit scores found for company'}
                score_id = latest_score.id
            
            score = self.session.query(Score).filter(Score.id == score_id).first()
            if not score:
                return {'error': 'Score not found'}
            
            # Calculate feature contributions
            contributions = self.calculate_feature_contributions(company_id)
            
            # Analyze trends
            trends = self.analyze_trends(company_id)
            
            # Get top positive and negative contributors
            positive_contributors = [c for c in contributions if c.contribution > 0][:5]
            negative_contributors = [c for c in contributions if c.contribution < 0][:5]
            
            # Generate summary
            summary = self._generate_summary(score, positive_contributors, negative_contributors, trends)
            
            explanation = {
                'company': {
                    'id': company.id,
                    'name': company.name,
                    'ticker': company.ticker
                },
                'score': {
                    'value': score.credit_score,
                    'risk_category': score.risk_category,
                    'confidence': score.confidence,
                    'date': score.score_date.isoformat(),
                    'model_version': score.model_version
                },
                'summary': summary,
                'feature_contributions': {
                    'positive': [asdict(c) for c in positive_contributors],
                    'negative': [asdict(c) for c in negative_contributors],
                    'total_features': len(contributions)
                },
                'trends': [asdict(t) for t in trends],
                'recommendations': self._generate_recommendations(score, contributions, trends),
                'generated_at': datetime.now().isoformat()
            }
            
            # Store explanation in database
            self._store_explanation(score_id, explanation)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {'error': str(e)}
    
    def _generate_summary(self, score: Score, positive_contributors: List[FeatureContribution], 
                         negative_contributors: List[FeatureContribution], trends: List[TrendAnalysis]) -> str:
        """Generate a human-readable summary of the credit score"""
        
        risk_descriptions = {
            'AAA': 'excellent credit quality with minimal risk',
            'AA': 'very strong credit quality with low risk',
            'A': 'strong credit quality with moderate risk',
            'BBB': 'adequate credit quality with moderate risk',
            'BB': 'speculative credit quality with elevated risk',
            'B': 'highly speculative with high risk',
            'CCC': 'substantial credit risk',
            'CC': 'very high credit risk',
            'C': 'extremely high credit risk',
            'D': 'default risk'
        }
        
        risk_desc = risk_descriptions.get(score.risk_category, 'unknown risk level')
        
        summary = f"Credit score of {score.credit_score:.0f} ({score.risk_category}) indicates {risk_desc}. "
        
        # Add key positive factors
        if positive_contributors:
            top_positive = positive_contributors[0]
            summary += f"Key strength: {top_positive.description.lower()} "
            summary += f"(contributing +{top_positive.contribution:.0f} points). "
        
        # Add key negative factors
        if negative_contributors:
            top_negative = negative_contributors[0]
            summary += f"Main concern: {top_negative.description.lower()} "
            summary += f"(reducing score by {abs(top_negative.contribution):.0f} points). "
        
        # Add trend information
        improving_trends = [t for t in trends if t.trend_direction == 'improving']
        declining_trends = [t for t in trends if t.trend_direction == 'declining']
        
        if improving_trends:
            summary += f"Positive trends observed in {', '.join([t.metric_name.lower() for t in improving_trends[:2]])}. "
        
        if declining_trends:
            summary += f"Concerning trends in {', '.join([t.metric_name.lower() for t in declining_trends[:2]])}. "
        
        return summary
    
    def _generate_recommendations(self, score: Score, contributions: List[FeatureContribution], 
                                trends: List[TrendAnalysis]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on score level
        if score.credit_score < 500:
            recommendations.append("Focus on fundamental business improvements to strengthen financial position")
        elif score.credit_score < 650:
            recommendations.append("Monitor key risk factors and implement risk mitigation strategies")
        
        # Based on negative contributors
        negative_contributors = [c for c in contributions if c.contribution < -10]
        for contributor in negative_contributors[:3]:
            if 'volatility' in contributor.feature_name:
                recommendations.append("Consider strategies to reduce stock price volatility")
            elif 'sentiment' in contributor.feature_name or 'negative' in contributor.feature_name:
                recommendations.append("Improve public relations and communication strategy")
            elif 'price_change' in contributor.feature_name:
                recommendations.append("Focus on sustainable business growth to support stock performance")
        
        # Based on declining trends
        declining_trends = [t for t in trends if t.trend_direction == 'declining' and t.trend_strength > 0.5]
        for trend in declining_trends:
            if 'Price' in trend.metric_name:
                recommendations.append("Address factors contributing to declining stock performance")
            elif 'Sentiment' in trend.metric_name:
                recommendations.append("Enhance stakeholder communication and address negative news coverage")
            elif 'Credit Score' in trend.metric_name:
                recommendations.append("Implement immediate risk mitigation measures")
        
        # General recommendations
        if len(recommendations) == 0:
            if score.credit_score >= 700:
                recommendations.append("Maintain current strong performance and monitor for emerging risks")
            else:
                recommendations.append("Continue monitoring key performance indicators and market conditions")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _store_explanation(self, score_id: int, explanation: Dict) -> bool:
        """Store explanation in database"""
        try:
            # Convert explanation to JSON string for storage
            explanation_json = json.dumps(explanation, default=str)
            
            explanation_record = Explanation(
                score_id=score_id,
                explanation_type='comprehensive',
                explanation_data=explanation_json,
                created_at=datetime.now()
            )
            
            self.session.add(explanation_record)
            self.session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing explanation: {str(e)}")
            self.session.rollback()
            return False
    
    def get_stored_explanation(self, score_id: int) -> Optional[Dict]:
        """Retrieve stored explanation from database"""
        try:
            explanation = self.session.query(Explanation).filter(
                Explanation.score_id == score_id,
                Explanation.explanation_type == 'comprehensive'
            ).order_by(Explanation.created_at.desc()).first()
            
            if explanation:
                return json.loads(explanation.explanation_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving explanation: {str(e)}")
            return None
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    explainer = ExplainabilityEngine()
    
    try:
        # Test with a company (assuming company_id 1 exists)
        company_id = 1
        
        print("Calculating feature contributions...")
        contributions = explainer.calculate_feature_contributions(company_id)
        
        print(f"Found {len(contributions)} feature contributions")
        for contrib in contributions[:5]:
            print(f"  {contrib.description}: {contrib.contribution:.1f}")
        
        print("\nAnalyzing trends...")
        trends = explainer.analyze_trends(company_id)
        
        print(f"Found {len(trends)} trends")
        for trend in trends:
            print(f"  {trend.metric_name}: {trend.trend_direction} ({trend.trend_strength:.2f})")
        
        print("\nGenerating comprehensive explanation...")
        explanation = explainer.generate_explanation(company_id)
        
        if 'error' not in explanation:
            print("Explanation generated successfully!")
            print(f"Summary: {explanation['summary']}")
        else:
            print(f"Error: {explanation['error']}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        explainer.close()

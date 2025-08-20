import re
import spacy
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Company, NewsRaw
from typing import List, Optional, Dict, Any, Tuple
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class CreditEventClassifier:
    """Advanced NLP system for classifying credit-relevant events and extracting entities"""
    
    def __init__(self):
        self.session = next(get_db())
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load spaCy model (fallback to basic if not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Using basic NLP features.")
            self.nlp = None
        
        # Credit risk event categories and their indicators
        self.event_categories = {
            'debt_restructuring': {
                'keywords': [
                    'debt restructuring', 'debt reorganization', 'debt modification',
                    'refinancing', 'debt renegotiation', 'payment terms modification',
                    'maturity extension', 'interest rate reduction', 'principal reduction'
                ],
                'risk_impact': -0.7,  # Negative impact on credit score
                'description': 'Debt restructuring or refinancing activities'
            },
            'financial_distress': {
                'keywords': [
                    'bankruptcy', 'chapter 11', 'chapter 7', 'insolvency', 'liquidation',
                    'financial distress', 'cash flow problems', 'covenant breach',
                    'default', 'missed payment', 'payment delay', 'credit facility drawn'
                ],
                'risk_impact': -0.9,
                'description': 'Signs of financial distress or bankruptcy'
            },
            'earnings_warning': {
                'keywords': [
                    'earnings warning', 'profit warning', 'guidance reduction',
                    'revenue decline', 'margin compression', 'cost overruns',
                    'disappointing results', 'below expectations', 'outlook lowered'
                ],
                'risk_impact': -0.5,
                'description': 'Negative earnings or guidance updates'
            },
            'management_changes': {
                'keywords': [
                    'ceo resignation', 'cfo departure', 'management turnover',
                    'executive changes', 'board changes', 'leadership transition',
                    'interim ceo', 'management shake-up'
                ],
                'risk_impact': -0.3,
                'description': 'Key management or leadership changes'
            },
            'regulatory_issues': {
                'keywords': [
                    'regulatory investigation', 'sec investigation', 'compliance issues',
                    'regulatory fine', 'penalty', 'sanctions', 'license revocation',
                    'regulatory action', 'consent decree', 'settlement'
                ],
                'risk_impact': -0.6,
                'description': 'Regulatory investigations or penalties'
            },
            'operational_issues': {
                'keywords': [
                    'plant closure', 'facility shutdown', 'production halt',
                    'supply chain disruption', 'operational challenges',
                    'safety incident', 'environmental incident', 'recall'
                ],
                'risk_impact': -0.4,
                'description': 'Operational disruptions or challenges'
            },
            'market_expansion': {
                'keywords': [
                    'market expansion', 'new product launch', 'acquisition completed',
                    'strategic partnership', 'joint venture', 'market entry',
                    'capacity expansion', 'investment in growth'
                ],
                'risk_impact': 0.3,
                'description': 'Positive business expansion activities'
            },
            'financial_strength': {
                'keywords': [
                    'strong earnings', 'revenue growth', 'profit increase',
                    'cash generation', 'debt reduction', 'credit facility repaid',
                    'dividend increase', 'share buyback', 'investment grade'
                ],
                'risk_impact': 0.5,
                'description': 'Signs of financial strength and stability'
            },
            'rating_changes': {
                'keywords': [
                    'credit rating', 'rating upgrade', 'rating downgrade',
                    'outlook positive', 'outlook negative', 'rating affirmed',
                    'moody\'s', 'standard & poor\'s', 's&p', 'fitch'
                ],
                'risk_impact': 0.0,  # Impact depends on direction
                'description': 'Credit rating agency actions'
            },
            'legal_issues': {
                'keywords': [
                    'lawsuit', 'litigation', 'legal settlement', 'court ruling',
                    'patent dispute', 'class action', 'legal liability',
                    'damages awarded', 'injunction'
                ],
                'risk_impact': -0.4,
                'description': 'Legal proceedings and disputes'
            }
        }
        
        # Entity types to extract
        self.entity_types = {
            'PERSON': 'People (executives, analysts, etc.)',
            'ORG': 'Organizations (companies, agencies, etc.)',
            'MONEY': 'Monetary amounts',
            'PERCENT': 'Percentages',
            'DATE': 'Dates and time periods',
            'GPE': 'Geopolitical entities (countries, cities)',
            'CARDINAL': 'Numbers and quantities'
        }
        
        # Financial metrics patterns
        self.financial_patterns = {
            'revenue': r'revenue[s]?\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'profit': r'(?:net\s+)?profit[s]?\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'debt': r'debt[s]?\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'loss': r'loss[es]?\s+(?:of\s+)?[\$]?([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'margin': r'margin[s]?\s+(?:of\s+)?([\d,]+\.?\d*)%?',
            'growth': r'growth[s]?\s+(?:of\s+)?([\d,]+\.?\d*)%?'
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """Extract named entities from text"""
        entities = defaultdict(list)
        
        try:
            if self.nlp:
                # Use spaCy for entity extraction
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ in self.entity_types:
                        entities[ent.label_].append({
                            'text': ent.text,
                            'start': ent.start_char,
                            'end': ent.end_char,
                            'confidence': 1.0  # spaCy doesn't provide confidence scores
                        })
            else:
                # Fallback: basic pattern matching
                entities = self._extract_entities_basic(text)
            
            # Extract financial metrics
            financial_entities = self._extract_financial_metrics(text)
            entities.update(financial_entities)
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
        
        return dict(entities)
    
    def _extract_entities_basic(self, text: str) -> Dict[str, List[Dict]]:
        """Basic entity extraction using regex patterns"""
        entities = defaultdict(list)
        
        # Money patterns
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|m|b))?'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities['MONEY'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.8
            })
        
        # Percentage patterns
        percent_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(percent_pattern, text):
            entities['PERCENT'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.9
            })
        
        # Date patterns (basic)
        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
        for match in re.finditer(date_pattern, text):
            entities['DATE'].append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7
            })
        
        return dict(entities)
    
    def _extract_financial_metrics(self, text: str) -> Dict[str, List[Dict]]:
        """Extract specific financial metrics"""
        financial_entities = defaultdict(list)
        
        for metric_name, pattern in self.financial_patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                financial_entities['FINANCIAL_METRIC'].append({
                    'text': match.group(),
                    'metric_type': metric_name,
                    'value': match.group(1) if match.groups() else None,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8
                })
        
        return dict(financial_entities)
    
    def classify_event(self, text: str, title: str = "") -> Dict[str, Any]:
        """Classify credit-relevant events in text"""
        combined_text = f"{title} {text}".lower()
        
        event_scores = {}
        detected_events = []
        
        # Score each event category
        for category, config in self.event_categories.items():
            score = 0
            matched_keywords = []
            
            for keyword in config['keywords']:
                if keyword.lower() in combined_text:
                    # Weight by keyword specificity (longer keywords = higher weight)
                    weight = len(keyword.split()) * 1.5
                    score += weight
                    matched_keywords.append(keyword)
            
            if score > 0:
                # Normalize score by text length
                normalized_score = min(score / (len(combined_text.split()) / 100), 1.0)
                
                event_scores[category] = {
                    'score': normalized_score,
                    'matched_keywords': matched_keywords,
                    'risk_impact': config['risk_impact'],
                    'description': config['description']
                }
        
        # Identify primary events (score > threshold)
        threshold = 0.1
        for category, data in event_scores.items():
            if data['score'] > threshold:
                detected_events.append({
                    'category': category,
                    'confidence': data['score'],
                    'risk_impact': data['risk_impact'],
                    'description': data['description'],
                    'keywords': data['matched_keywords']
                })
        
        # Sort by confidence
        detected_events.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate overall event sentiment
        if detected_events:
            avg_impact = np.mean([event['risk_impact'] for event in detected_events])
            event_sentiment = max(min(avg_impact, 1.0), -1.0)
        else:
            event_sentiment = 0.0
        
        return {
            'detected_events': detected_events,
            'event_sentiment': event_sentiment,
            'primary_category': detected_events[0]['category'] if detected_events else None,
            'confidence': detected_events[0]['confidence'] if detected_events else 0.0
        }
    
    def analyze_market_impact(self, events: List[Dict], entities: Dict) -> Dict[str, Any]:
        """Analyze potential market impact of detected events"""
        impact_analysis = {
            'overall_impact': 0.0,
            'impact_factors': [],
            'affected_areas': [],
            'urgency': 'low'
        }
        
        try:
            if not events:
                return impact_analysis
            
            # Calculate weighted impact
            total_impact = 0
            total_weight = 0
            
            for event in events:
                weight = event['confidence']
                impact = event['risk_impact'] * weight
                total_impact += impact
                total_weight += weight
                
                # Add impact factors
                impact_analysis['impact_factors'].append({
                    'category': event['category'],
                    'impact': event['risk_impact'],
                    'confidence': event['confidence'],
                    'description': event['description']
                })
            
            if total_weight > 0:
                impact_analysis['overall_impact'] = total_impact / total_weight
            
            # Determine affected areas
            financial_events = ['debt_restructuring', 'financial_distress', 'earnings_warning']
            operational_events = ['operational_issues', 'management_changes']
            regulatory_events = ['regulatory_issues', 'legal_issues']
            
            for event in events:
                category = event['category']
                if category in financial_events and 'Financial' not in impact_analysis['affected_areas']:
                    impact_analysis['affected_areas'].append('Financial')
                elif category in operational_events and 'Operational' not in impact_analysis['affected_areas']:
                    impact_analysis['affected_areas'].append('Operational')
                elif category in regulatory_events and 'Regulatory' not in impact_analysis['affected_areas']:
                    impact_analysis['affected_areas'].append('Regulatory')
            
            # Determine urgency
            max_impact = max([abs(event['risk_impact']) for event in events])
            max_confidence = max([event['confidence'] for event in events])
            
            urgency_score = max_impact * max_confidence
            
            if urgency_score > 0.6:
                impact_analysis['urgency'] = 'high'
            elif urgency_score > 0.3:
                impact_analysis['urgency'] = 'medium'
            else:
                impact_analysis['urgency'] = 'low'
            
            # Add monetary impact if available
            if 'MONEY' in entities:
                amounts = []
                for money_entity in entities['MONEY']:
                    # Extract numeric value
                    amount_text = money_entity['text'].replace('$', '').replace(',', '')
                    try:
                        if 'billion' in amount_text.lower() or 'b' in amount_text.lower():
                            amount = float(re.findall(r'[\d.]+', amount_text)[0]) * 1000000000
                        elif 'million' in amount_text.lower() or 'm' in amount_text.lower():
                            amount = float(re.findall(r'[\d.]+', amount_text)[0]) * 1000000
                        else:
                            amount = float(re.findall(r'[\d.]+', amount_text)[0])
                        amounts.append(amount)
                    except:
                        continue
                
                if amounts:
                    impact_analysis['monetary_impact'] = {
                        'total_amount': sum(amounts),
                        'max_amount': max(amounts),
                        'currency': 'USD'
                    }
        
        except Exception as e:
            logger.error(f"Error analyzing market impact: {str(e)}")
        
        return impact_analysis
    
    def process_news_article(self, article_id: int) -> Dict[str, Any]:
        """Process a single news article for event classification"""
        try:
            # Get article from database
            article = self.session.query(NewsRaw).filter(NewsRaw.id == article_id).first()
            if not article:
                return {'error': 'Article not found'}
            
            # Extract entities
            text = f"{article.title} {article.content or ''}"
            entities = self.extract_entities(text)
            
            # Classify events
            events = self.classify_event(text, article.title)
            
            # Analyze market impact
            impact = self.analyze_market_impact(events['detected_events'], entities)
            
            # Enhanced sentiment analysis
            enhanced_sentiment = self._analyze_enhanced_sentiment(text, events['detected_events'])
            
            # Create comprehensive analysis
            analysis = {
                'article_id': article_id,
                'company_id': article.company_id,
                'title': article.title,
                'source': article.source,
                'published_at': article.published_at.isoformat(),
                'entities': entities,
                'events': events,
                'market_impact': impact,
                'enhanced_sentiment': enhanced_sentiment,
                'processing_date': datetime.now().isoformat()
            }
            
            # Update article with enhanced analysis
            self._update_article_analysis(article, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing article {article_id}: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_enhanced_sentiment(self, text: str, events: List[Dict]) -> Dict[str, Any]:
        """Enhanced sentiment analysis considering event context"""
        # Base sentiment analysis
        blob = TextBlob(text)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # Adjust sentiment based on detected events
        event_adjustment = 0.0
        if events:
            event_impacts = [event['risk_impact'] for event in events]
            event_adjustment = np.mean(event_impacts) * 0.3  # 30% weight to events
        
        # Combined sentiment
        base_sentiment = (blob.sentiment.polarity + vader_scores['compound']) / 2
        adjusted_sentiment = base_sentiment + event_adjustment
        adjusted_sentiment = max(min(adjusted_sentiment, 1.0), -1.0)  # Clamp to [-1, 1]
        
        return {
            'base_sentiment': base_sentiment,
            'event_adjustment': event_adjustment,
            'final_sentiment': adjusted_sentiment,
            'confidence': max(blob.sentiment.subjectivity, vader_scores['neu']),
            'method': 'enhanced_nlp'
        }
    
    def _update_article_analysis(self, article: NewsRaw, analysis: Dict):
        """Update article with enhanced analysis results"""
        try:
            # Update sentiment with enhanced analysis
            enhanced_sentiment = analysis['enhanced_sentiment']
            article.sentiment_score = enhanced_sentiment['final_sentiment']
            
            # Update sentiment label
            sentiment_score = enhanced_sentiment['final_sentiment']
            if sentiment_score > 0.1:
                article.sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                article.sentiment_label = 'negative'
            else:
                article.sentiment_label = 'neutral'
            
            # Store analysis as JSON in content field extension or create new field
            # For now, we'll log the analysis
            logger.info(f"Enhanced analysis for article {article.id}: "
                       f"Events: {len(analysis['events']['detected_events'])}, "
                       f"Entities: {sum(len(v) for v in analysis['entities'].values())}, "
                       f"Impact: {analysis['market_impact']['overall_impact']:.2f}")
            
            self.session.commit()
            
        except Exception as e:
            logger.error(f"Error updating article analysis: {str(e)}")
            self.session.rollback()
    
    def bulk_process_articles(self, company_id: int = None, days_back: int = 30) -> Dict[str, Any]:
        """Process multiple articles for enhanced NLP analysis"""
        try:
            # Get articles to process
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            query = self.session.query(NewsRaw).filter(
                NewsRaw.published_at >= cutoff_date,
                NewsRaw.processed == True
            )
            
            if company_id:
                query = query.filter(NewsRaw.company_id == company_id)
            
            articles = query.order_by(NewsRaw.published_at.desc()).limit(100).all()
            
            results = {
                'processed_count': 0,
                'event_summary': defaultdict(int),
                'entity_summary': defaultdict(int),
                'impact_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'articles_analyzed': []
            }
            
            for article in articles:
                try:
                    analysis = self.process_news_article(article.id)
                    
                    if 'error' not in analysis:
                        results['processed_count'] += 1
                        
                        # Update summaries
                        for event in analysis['events']['detected_events']:
                            results['event_summary'][event['category']] += 1
                        
                        for entity_type, entities in analysis['entities'].items():
                            results['entity_summary'][entity_type] += len(entities)
                        
                        urgency = analysis['market_impact']['urgency']
                        results['impact_distribution'][urgency] += 1
                        
                        # Store key analysis info
                        results['articles_analyzed'].append({
                            'id': article.id,
                            'title': article.title[:100],
                            'events_count': len(analysis['events']['detected_events']),
                            'primary_event': analysis['events']['primary_category'],
                            'impact_score': analysis['market_impact']['overall_impact'],
                            'urgency': urgency
                        })
                
                except Exception as e:
                    logger.error(f"Error processing article {article.id}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk processing: {str(e)}")
            return {'error': str(e)}
    
    def get_event_trends(self, company_id: int, days_back: int = 90) -> Dict[str, Any]:
        """Analyze event trends for a company over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Get company articles
            articles = self.session.query(NewsRaw).filter(
                NewsRaw.company_id == company_id,
                NewsRaw.published_at >= cutoff_date,
                NewsRaw.processed == True
            ).order_by(NewsRaw.published_at.desc()).all()
            
            # Process articles and track trends
            weekly_events = defaultdict(lambda: defaultdict(int))
            overall_events = defaultdict(int)
            sentiment_trend = []
            
            for article in articles:
                # Quick event classification
                text = f"{article.title} {article.content or ''}"
                events = self.classify_event(text, article.title)
                
                # Get week of year
                week = article.published_at.strftime('%Y-W%U')
                
                for event in events['detected_events']:
                    category = event['category']
                    weekly_events[week][category] += 1
                    overall_events[category] += 1
                
                # Track sentiment
                sentiment_trend.append({
                    'date': article.published_at.strftime('%Y-%m-%d'),
                    'sentiment': article.sentiment_score or 0.0
                })
            
            # Calculate trends
            trend_analysis = {
                'total_articles': len(articles),
                'event_distribution': dict(overall_events),
                'weekly_trends': dict(weekly_events),
                'sentiment_trend': sentiment_trend[-30:],  # Last 30 data points
                'risk_indicators': self._identify_risk_indicators(overall_events),
                'trend_direction': self._calculate_trend_direction(weekly_events)
            }
            
            return trend_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing event trends: {str(e)}")
            return {'error': str(e)}
    
    def _identify_risk_indicators(self, events: Dict[str, int]) -> List[Dict]:
        """Identify key risk indicators from event patterns"""
        risk_indicators = []
        
        # High-risk event thresholds
        risk_thresholds = {
            'financial_distress': 1,  # Any occurrence is concerning
            'debt_restructuring': 2,
            'earnings_warning': 3,
            'regulatory_issues': 2,
            'management_changes': 4
        }
        
        for event_type, count in events.items():
            threshold = risk_thresholds.get(event_type, 5)
            if count >= threshold:
                risk_level = 'high' if count >= threshold * 2 else 'medium'
                
                risk_indicators.append({
                    'event_type': event_type,
                    'occurrence_count': count,
                    'risk_level': risk_level,
                    'description': self.event_categories[event_type]['description']
                })
        
        return risk_indicators
    
    def _calculate_trend_direction(self, weekly_events: Dict) -> str:
        """Calculate overall trend direction for events"""
        if len(weekly_events) < 2:
            return 'insufficient_data'
        
        # Get recent weeks vs earlier weeks
        weeks = sorted(weekly_events.keys())
        mid_point = len(weeks) // 2
        
        early_weeks = weeks[:mid_point]
        recent_weeks = weeks[mid_point:]
        
        early_total = sum(sum(weekly_events[week].values()) for week in early_weeks)
        recent_total = sum(sum(weekly_events[week].values()) for week in recent_weeks)
        
        if recent_total > early_total * 1.2:
            return 'increasing'
        elif recent_total < early_total * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def close(self):
        """Close database session"""
        self.session.close()

# Example usage
if __name__ == "__main__":
    classifier = CreditEventClassifier()
    
    try:
        # Test event classification
        test_text = """
        Company XYZ announced today that it has completed a debt restructuring agreement 
        with its lenders, reducing total debt by $500 million. The CEO stated that this 
        will improve the company's financial flexibility and reduce interest expenses by 
        15% annually. However, the company also warned that Q3 earnings may be below 
        analyst expectations due to supply chain disruptions.
        """
        
        print("Testing event classification...")
        entities = classifier.extract_entities(test_text)
        events = classifier.classify_event(test_text)
        impact = classifier.analyze_market_impact(events['detected_events'], entities)
        
        print(f"Entities found: {sum(len(v) for v in entities.values())}")
        print(f"Events detected: {len(events['detected_events'])}")
        print(f"Primary event: {events['primary_category']}")
        print(f"Overall impact: {impact['overall_impact']:.2f}")
        print(f"Urgency: {impact['urgency']}")
        
        if events['detected_events']:
            print("\nDetected events:")
            for event in events['detected_events'][:3]:
                print(f"  - {event['category']}: {event['confidence']:.2f} confidence")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        classifier.close()

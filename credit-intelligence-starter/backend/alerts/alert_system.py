import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from database.connection import get_db
from database.models import Company, Score, NewsRaw, Explanation
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import os
import redis
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    SCORE_DROP = "score_drop"
    SCORE_SPIKE = "score_spike"
    VOLATILITY_INCREASE = "volatility_increase"
    RATING_DOWNGRADE = "rating_downgrade"
    RATING_UPGRADE = "rating_upgrade"
    NEGATIVE_NEWS_CLUSTER = "negative_news_cluster"
    FINANCIAL_DISTRESS = "financial_distress"
    REGULATORY_ISSUE = "regulatory_issue"
    ECONOMIC_STRESS = "economic_stress"

@dataclass
class Alert:
    id: str
    company_id: int
    company_name: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    data: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    acknowledged: bool = False
    sent_notifications: List[str] = None

    def __post_init__(self):
        if self.sent_notifications is None:
            self.sent_notifications = []

class RealTimeAlertSystem:
    """Real-time alert system for credit score changes and risk events"""
    
    def __init__(self, redis_url: str = None, email_config: Dict = None):
        self.session = next(get_db())
        
        # Redis for real-time notifications (optional)
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
            except Exception as e:
                logger.warning(f"Redis connection failed: {str(e)}")
        
        # Email configuration
        self.email_config = email_config or {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('EMAIL_USERNAME'),
            'password': os.getenv('EMAIL_PASSWORD'),
            'from_email': os.getenv('FROM_EMAIL')
        }
        
        # Alert thresholds and configurations
        self.alert_config = {
            'score_drop_threshold': 50,  # Points
            'score_spike_threshold': 50,  # Points
            'volatility_threshold': 0.15,  # 15% volatility
            'time_window_hours': 24,  # Time window for analysis
            'negative_news_threshold': 3,  # Number of negative articles
            'sentiment_threshold': -0.5,  # Sentiment score threshold
            'max_alerts_per_hour': 5,  # Rate limiting
        }
        
        # Active alerts storage
        self.active_alerts: Dict[str, Alert] = {}
        
        # Load existing alerts from Redis if available
        self._load_active_alerts()
    
    def _load_active_alerts(self):
        """Load active alerts from Redis"""
        if not self.redis_client:
            return
        
        try:
            alert_keys = self.redis_client.keys("alert:*")
            for key in alert_keys:
                alert_data = self.redis_client.get(key)
                if alert_data:
                    alert_dict = json.loads(alert_data)
                    alert = Alert(**alert_dict)
                    self.active_alerts[alert.id] = alert
        except Exception as e:
            logger.error(f"Error loading alerts from Redis: {str(e)}")
    
    def _save_alert_to_redis(self, alert: Alert):
        """Save alert to Redis"""
        if not self.redis_client:
            return
        
        try:
            alert_dict = {
                'id': alert.id,
                'company_id': alert.company_id,
                'company_name': alert.company_name,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'data': alert.data,
                'created_at': alert.created_at.isoformat(),
                'expires_at': alert.expires_at.isoformat() if alert.expires_at else None,
                'acknowledged': alert.acknowledged,
                'sent_notifications': alert.sent_notifications
            }
            
            key = f"alert:{alert.id}"
            self.redis_client.setex(key, 86400 * 7, json.dumps(alert_dict))  # 7 days TTL
            
        except Exception as e:
            logger.error(f"Error saving alert to Redis: {str(e)}")
    
    def create_alert(self, company_id: int, alert_type: AlertType, 
                    severity: AlertSeverity, title: str, message: str, 
                    data: Dict[str, Any] = None) -> Alert:
        """Create a new alert"""
        
        # Get company info
        company = self.session.query(Company).filter(Company.id == company_id).first()
        if not company:
            raise ValueError(f"Company with ID {company_id} not found")
        
        # Generate unique alert ID
        alert_id = f"{company.ticker}_{alert_type.value}_{int(datetime.now().timestamp())}"
        
        # Create alert
        alert = Alert(
            id=alert_id,
            company_id=company_id,
            company_name=company.name,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            data=data or {},
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)  # Default 24h expiry
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self._save_alert_to_redis(alert)
        
        logger.info(f"Created {severity.value} alert for {company.ticker}: {title}")
        
        return alert
    
    def check_score_changes(self, company_id: int = None) -> List[Alert]:
        """Check for significant credit score changes"""
        alerts = []
        
        try:
            # Get companies to check
            if company_id:
                companies = [self.session.query(Company).filter(Company.id == company_id).first()]
            else:
                companies = self.session.query(Company).all()
            
            for company in companies:
                if not company:
                    continue
                
                # Get recent scores
                recent_scores = self.session.query(Score).filter(
                    Score.company_id == company.id
                ).order_by(desc(Score.score_date)).limit(10).all()
                
                if len(recent_scores) < 2:
                    continue
                
                current_score = recent_scores[0]
                previous_score = recent_scores[1]
                
                # Calculate change
                score_change = current_score.credit_score - previous_score.credit_score
                change_percent = abs(score_change) / previous_score.credit_score * 100
                
                # Check for significant drop
                if score_change <= -self.alert_config['score_drop_threshold']:
                    severity = AlertSeverity.CRITICAL if change_percent > 20 else AlertSeverity.HIGH
                    
                    alert = self.create_alert(
                        company_id=company.id,
                        alert_type=AlertType.SCORE_DROP,
                        severity=severity,
                        title=f"Significant Credit Score Drop - {company.ticker}",
                        message=f"Credit score dropped by {abs(score_change):.0f} points ({change_percent:.1f}%) "
                               f"from {previous_score.credit_score:.0f} to {current_score.credit_score:.0f}",
                        data={
                            'previous_score': previous_score.credit_score,
                            'current_score': current_score.credit_score,
                            'change': score_change,
                            'change_percent': change_percent,
                            'previous_date': previous_score.score_date.isoformat(),
                            'current_date': current_score.score_date.isoformat()
                        }
                    )
                    alerts.append(alert)
                
                # Check for significant spike (could indicate data issues)
                elif score_change >= self.alert_config['score_spike_threshold']:
                    severity = AlertSeverity.MEDIUM
                    
                    alert = self.create_alert(
                        company_id=company.id,
                        alert_type=AlertType.SCORE_SPIKE,
                        severity=severity,
                        title=f"Unusual Credit Score Increase - {company.ticker}",
                        message=f"Credit score increased by {score_change:.0f} points ({change_percent:.1f}%) "
                               f"from {previous_score.credit_score:.0f} to {current_score.credit_score:.0f}",
                        data={
                            'previous_score': previous_score.credit_score,
                            'current_score': current_score.credit_score,
                            'change': score_change,
                            'change_percent': change_percent,
                            'previous_date': previous_score.score_date.isoformat(),
                            'current_date': current_score.score_date.isoformat()
                        }
                    )
                    alerts.append(alert)
                
                # Check for high volatility
                if len(recent_scores) >= 5:
                    scores = [s.credit_score for s in recent_scores[:5]]
                    volatility = np.std(scores) / np.mean(scores)
                    
                    if volatility > self.alert_config['volatility_threshold']:
                        alert = self.create_alert(
                            company_id=company.id,
                            alert_type=AlertType.VOLATILITY_INCREASE,
                            severity=AlertSeverity.MEDIUM,
                            title=f"High Score Volatility - {company.ticker}",
                            message=f"Credit score showing high volatility ({volatility:.1%}) over recent periods",
                            data={
                                'volatility': volatility,
                                'recent_scores': scores,
                                'mean_score': np.mean(scores),
                                'std_dev': np.std(scores)
                            }
                        )
                        alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking score changes: {str(e)}")
        
        return alerts
    
    def check_news_sentiment_clusters(self, company_id: int = None) -> List[Alert]:
        """Check for clusters of negative news"""
        alerts = []
        
        try:
            # Time window for analysis
            cutoff_time = datetime.now() - timedelta(hours=self.alert_config['time_window_hours'])
            
            # Get companies to check
            if company_id:
                companies = [self.session.query(Company).filter(Company.id == company_id).first()]
            else:
                companies = self.session.query(Company).all()
            
            for company in companies:
                if not company:
                    continue
                
                # Get recent negative news
                negative_news = self.session.query(NewsRaw).filter(
                    and_(
                        NewsRaw.company_id == company.id,
                        NewsRaw.published_at >= cutoff_time,
                        NewsRaw.sentiment_score <= self.alert_config['sentiment_threshold']
                    )
                ).all()
                
                if len(negative_news) >= self.alert_config['negative_news_threshold']:
                    # Calculate average sentiment
                    avg_sentiment = np.mean([n.sentiment_score for n in negative_news if n.sentiment_score])
                    
                    severity = AlertSeverity.HIGH if avg_sentiment < -0.7 else AlertSeverity.MEDIUM
                    
                    alert = self.create_alert(
                        company_id=company.id,
                        alert_type=AlertType.NEGATIVE_NEWS_CLUSTER,
                        severity=severity,
                        title=f"Cluster of Negative News - {company.ticker}",
                        message=f"Detected {len(negative_news)} negative news articles in the last "
                               f"{self.alert_config['time_window_hours']} hours with average sentiment {avg_sentiment:.2f}",
                        data={
                            'article_count': len(negative_news),
                            'average_sentiment': avg_sentiment,
                            'time_window_hours': self.alert_config['time_window_hours'],
                            'articles': [
                                {
                                    'title': n.title,
                                    'sentiment': n.sentiment_score,
                                    'published_at': n.published_at.isoformat(),
                                    'source': n.source
                                }
                                for n in negative_news[:5]  # Include top 5
                            ]
                        }
                    )
                    alerts.append(alert)
        
        except Exception as e:
            logger.error(f"Error checking news sentiment: {str(e)}")
        
        return alerts
    
    def check_rating_changes(self, company_id: int = None) -> List[Alert]:
        """Check for credit rating changes"""
        alerts = []
        
        try:
            # Get companies to check
            if company_id:
                companies = [self.session.query(Company).filter(Company.id == company_id).first()]
            else:
                companies = self.session.query(Company).all()
            
            for company in companies:
                if not company:
                    continue
                
                # Get recent scores to check for rating changes
                recent_scores = self.session.query(Score).filter(
                    Score.company_id == company.id
                ).order_by(desc(Score.score_date)).limit(5).all()
                
                if len(recent_scores) < 2:
                    continue
                
                current_rating = recent_scores[0].risk_category
                previous_rating = recent_scores[1].risk_category
                
                if current_rating != previous_rating:
                    # Determine if upgrade or downgrade
                    rating_order = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
                    
                    try:
                        current_idx = rating_order.index(current_rating)
                        previous_idx = rating_order.index(previous_rating)
                        
                        if current_idx > previous_idx:  # Downgrade
                            alert_type = AlertType.RATING_DOWNGRADE
                            severity = AlertSeverity.HIGH if current_idx - previous_idx > 2 else AlertSeverity.MEDIUM
                            title = f"Credit Rating Downgrade - {company.ticker}"
                            message = f"Credit rating downgraded from {previous_rating} to {current_rating}"
                        else:  # Upgrade
                            alert_type = AlertType.RATING_UPGRADE
                            severity = AlertSeverity.LOW
                            title = f"Credit Rating Upgrade - {company.ticker}"
                            message = f"Credit rating upgraded from {previous_rating} to {current_rating}"
                        
                        alert = self.create_alert(
                            company_id=company.id,
                            alert_type=alert_type,
                            severity=severity,
                            title=title,
                            message=message,
                            data={
                                'previous_rating': previous_rating,
                                'current_rating': current_rating,
                                'change_date': recent_scores[0].score_date.isoformat(),
                                'score_change': recent_scores[0].credit_score - recent_scores[1].credit_score
                            }
                        )
                        alerts.append(alert)
                    
                    except ValueError:
                        # Rating not in standard list
                        continue
        
        except Exception as e:
            logger.error(f"Error checking rating changes: {str(e)}")
        
        return alerts
    
    def run_alert_checks(self, company_id: int = None) -> Dict[str, Any]:
        """Run all alert checks"""
        try:
            all_alerts = []
            
            # Run different types of checks
            score_alerts = self.check_score_changes(company_id)
            news_alerts = self.check_news_sentiment_clusters(company_id)
            rating_alerts = self.check_rating_changes(company_id)
            
            all_alerts.extend(score_alerts)
            all_alerts.extend(news_alerts)
            all_alerts.extend(rating_alerts)
            
            # Send notifications for new alerts
            for alert in all_alerts:
                self._send_notifications(alert)
            
            return {
                'success': True,
                'alerts_generated': len(all_alerts),
                'score_alerts': len(score_alerts),
                'news_alerts': len(news_alerts),
                'rating_alerts': len(rating_alerts),
                'alerts': [
                    {
                        'id': alert.id,
                        'company': alert.company_name,
                        'type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'created_at': alert.created_at.isoformat()
                    }
                    for alert in all_alerts
                ]
            }
        
        except Exception as e:
            logger.error(f"Error running alert checks: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        try:
            # Send email notification
            if self._should_send_email(alert):
                self._send_email_notification(alert)
            
            # Send to Redis/WebSocket for real-time updates
            if self.redis_client:
                self._send_realtime_notification(alert)
            
        except Exception as e:
            logger.error(f"Error sending notifications for alert {alert.id}: {str(e)}")
    
    def _should_send_email(self, alert: Alert) -> bool:
        """Determine if email should be sent for this alert"""
        # Check if email already sent
        if 'email' in alert.sent_notifications:
            return False
        
        # Only send email for medium severity and above
        if alert.severity in [AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            return True
        
        return False
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        if not all([self.email_config['username'], self.email_config['password'], 
                   self.email_config['from_email']]):
            logger.warning("Email configuration incomplete. Skipping email notification.")
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config.get('alert_recipients', self.email_config['from_email'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Create email body
            body = f"""
Credit Intelligence Alert

Company: {alert.company_name}
Alert Type: {alert.alert_type.value.replace('_', ' ').title()}
Severity: {alert.severity.value.upper()}
Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

{alert.message}

Additional Data:
{json.dumps(alert.data, indent=2)}

---
This is an automated alert from the Credit Intelligence Platform.
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            # Mark as sent
            alert.sent_notifications.append('email')
            self._save_alert_to_redis(alert)
            
            logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
    
    def _send_realtime_notification(self, alert: Alert):
        """Send real-time notification via Redis"""
        try:
            notification = {
                'type': 'alert',
                'alert': {
                    'id': alert.id,
                    'company_id': alert.company_id,
                    'company_name': alert.company_name,
                    'alert_type': alert.alert_type.value,
                    'severity': alert.severity.value,
                    'title': alert.title,
                    'message': alert.message,
                    'created_at': alert.created_at.isoformat(),
                    'data': alert.data
                }
            }
            
            # Publish to Redis channel
            self.redis_client.publish('credit_alerts', json.dumps(notification))
            
            # Mark as sent
            if 'realtime' not in alert.sent_notifications:
                alert.sent_notifications.append('realtime')
                self._save_alert_to_redis(alert)
            
        except Exception as e:
            logger.error(f"Error sending real-time notification: {str(e)}")
    
    def get_active_alerts(self, company_id: int = None, severity: AlertSeverity = None) -> List[Dict]:
        """Get active alerts"""
        alerts = []
        
        for alert in self.active_alerts.values():
            # Filter by company if specified
            if company_id and alert.company_id != company_id:
                continue
            
            # Filter by severity if specified
            if severity and alert.severity != severity:
                continue
            
            # Skip expired alerts
            if alert.expires_at and alert.expires_at < datetime.now():
                continue
            
            alerts.append({
                'id': alert.id,
                'company_id': alert.company_id,
                'company_name': alert.company_name,
                'alert_type': alert.alert_type.value,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'created_at': alert.created_at.isoformat(),
                'acknowledged': alert.acknowledged,
                'data': alert.data
            })
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda x: x['created_at'], reverse=True)
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            self._save_alert_to_redis(self.active_alerts[alert_id])
            logger.info(f"Alert {alert_id} acknowledged")
            return True
        return False
    
    def cleanup_expired_alerts(self):
        """Remove expired alerts"""
        current_time = datetime.now()
        expired_alerts = []
        
        for alert_id, alert in self.active_alerts.items():
            if alert.expires_at and alert.expires_at < current_time:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
            if self.redis_client:
                self.redis_client.delete(f"alert:{alert_id}")
        
        if expired_alerts:
            logger.info(f"Cleaned up {len(expired_alerts)} expired alerts")
    
    def close(self):
        """Close connections"""
        self.session.close()
        if self.redis_client:
            self.redis_client.close()

# Background task for continuous monitoring
async def continuous_alert_monitoring(alert_system: RealTimeAlertSystem, 
                                    check_interval: int = 300):  # 5 minutes
    """Continuous monitoring task"""
    logger.info("Starting continuous alert monitoring...")
    
    while True:
        try:
            # Run alert checks
            result = alert_system.run_alert_checks()
            
            if result['success']:
                logger.info(f"Alert check completed. Generated {result['alerts_generated']} alerts")
            else:
                logger.error(f"Alert check failed: {result.get('error')}")
            
            # Cleanup expired alerts
            alert_system.cleanup_expired_alerts()
            
            # Wait for next check
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error in continuous monitoring: {str(e)}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# Example usage
if __name__ == "__main__":
    # Initialize alert system
    alert_system = RealTimeAlertSystem()
    
    try:
        print("Running alert checks...")
        result = alert_system.run_alert_checks()
        
        if result['success']:
            print(f"Generated {result['alerts_generated']} alerts:")
            for alert in result['alerts']:
                print(f"  - {alert['severity'].upper()}: {alert['title']}")
        else:
            print(f"Error: {result['error']}")
        
        # Get active alerts
        active_alerts = alert_system.get_active_alerts()
        print(f"\nActive alerts: {len(active_alerts)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        alert_system.close()

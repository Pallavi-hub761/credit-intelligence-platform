from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from database.connection import get_db, init_db
from database.models import Company, Score, Price, NewsRaw, Explanation
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timedelta
import os
import logging
import asyncio

# Import our custom modules
from data_ingestion.data_orchestrator import DataOrchestrator
from data_ingestion.macro_economic_collector import MacroEconomicCollector
from scoring.credit_scoring_engine import CreditScoringEngine
from scoring.explainability import ExplainabilityEngine
from nlp.event_classifier import CreditEventClassifier
from alerts.alert_system import RealTimeAlertSystem
from ratings.agency_comparisons import AgencyRatingComparator

# Initialize components
orchestrator = DataOrchestrator()
macro_collector = MacroEconomicCollector()
scoring_engine = CreditScoringEngine()
explainability_engine = ExplainabilityEngine()
event_classifier = CreditEventClassifier()
alert_system = RealTimeAlertSystem()
rating_comparator = AgencyRatingComparator()

app = FastAPI(title="Credit Intelligence API", version="0.1.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API responses
class CompanyResponse(BaseModel):
    id: int
    name: str
    ticker: str
    
    class Config:
        from_attributes = True

class ScoreResponse(BaseModel):
    id: int
    company_id: int
    score_date: datetime
    credit_score: float
    risk_category: Optional[str]
    confidence: Optional[float]
    
    class Config:
        from_attributes = True

class ExplanationResponse(BaseModel):
    feature_name: str
    feature_value: Optional[float]
    contribution: Optional[float]
    importance: Optional[float]
    trend: Optional[str]
    explanation_text: Optional[str]
    
    class Config:
        from_attributes = True

class PredictionRequest(BaseModel):
    company_id: int
    model_name: Optional[str] = None

class DataCollectionRequest(BaseModel):
    tickers: Optional[List[str]] = None
    days_back: Optional[int] = 7

class TrainingRequest(BaseModel):
    n_samples: Optional[int] = 2000

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()

@app.get("/")
def root():
    return {"message": "Credit Intelligence API", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "ok", "database": "connected"}

@app.get("/companies", response_model=List[CompanyResponse])
def get_companies(db: Session = Depends(get_db)):
    """Get all companies"""
    companies = db.query(Company).all()
    return companies

@app.get("/companies/{company_id}/scores", response_model=List[ScoreResponse])
def get_company_scores(
    company_id: int, 
    limit: int = 30,
    db: Session = Depends(get_db)
):
    """Get credit scores for a company"""
    scores = db.query(Score).filter(
        Score.company_id == company_id
    ).order_by(Score.score_date.desc()).limit(limit).all()
    return scores

@app.get("/companies/{company_id}/scores/latest", response_model=ScoreResponse)
def get_latest_score(company_id: int, db: Session = Depends(get_db)):
    """Get latest credit score for a company"""
    score = db.query(Score).filter(
        Score.company_id == company_id
    ).order_by(Score.score_date.desc()).first()
    
    if not score:
        raise HTTPException(status_code=404, detail="No scores found for this company")
    return score

@app.get("/scores/{score_id}/explanations")
def get_score_explanations(score_id: int, db: Session = Depends(get_db)):
    """Get comprehensive explanations for a specific score"""
    try:
        explainer = ExplainabilityEngine()
        
        # Get the score to find company_id
        score = db.query(Score).filter(Score.id == score_id).first()
        if not score:
            raise HTTPException(status_code=404, detail="Score not found")
        
        # Generate comprehensive explanation
        explanation = explainer.generate_explanation(score.company_id, score_id)
        explainer.close()
        
        if 'error' in explanation:
            raise HTTPException(status_code=500, detail=explanation['error'])
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/explanations/{company_id}")
def get_explanations(company_id: int, db: Session = Depends(get_db)):
    """Get explanations for a company's credit score"""
    try:
        explanations = explainability_engine.get_explanations(company_id)
        return explanations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/analyze/{article_id}")
def analyze_article(article_id: int):
    """Analyze news article for events and entities"""
    try:
        analysis = event_classifier.process_news_article(article_id)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/nlp/bulk-analyze")
async def bulk_analyze_articles(background_tasks: BackgroundTasks, company_id: int = None, days_back: int = 30):
    """Bulk analyze articles for enhanced NLP"""
    try:
        background_tasks.add_task(event_classifier.bulk_process_articles, company_id, days_back)
        return {"message": "Bulk NLP analysis started", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/events/trends/{company_id}")
def get_event_trends(company_id: int, days_back: int = 90):
    """Get event trends for a company"""
    try:
        trends = event_classifier.get_event_trends(company_id, days_back)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/check")
async def check_alerts(background_tasks: BackgroundTasks, company_id: int = None):
    """Run alert checks for companies"""
    try:
        background_tasks.add_task(alert_system.run_alert_checks, company_id)
        return {"message": "Alert checks started", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/active")
def get_active_alerts(company_id: int = None, severity: str = None):
    """Get active alerts"""
    try:
        from alerts.alert_system import AlertSeverity
        severity_enum = AlertSeverity(severity) if severity else None
        alerts = alert_system.get_active_alerts(company_id, severity_enum)
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    try:
        success = alert_system.acknowledge_alert(alert_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ratings/compare/{company_id}")
def compare_ratings(company_id: int):
    """Compare internal rating with agency ratings"""
    try:
        comparison = rating_comparator.compare_with_internal_rating(company_id)
        if comparison:
            return {
                "company_id": comparison.company_id,
                "company_name": comparison.company_name,
                "ticker": comparison.ticker,
                "internal_rating": comparison.internal_rating,
                "internal_score": comparison.internal_score,
                "consensus_rating": comparison.consensus_rating,
                "rating_spread": comparison.rating_spread,
                "upgrade_probability": comparison.upgrade_probability,
                "downgrade_probability": comparison.downgrade_probability,
                "agency_ratings": [
                    {
                        "agency": r.agency,
                        "rating": r.rating,
                        "outlook": r.outlook,
                        "investment_grade": r.investment_grade,
                        "date": r.date.isoformat()
                    }
                    for r in comparison.agency_ratings
                ]
            }
        else:
            raise HTTPException(status_code=404, detail="Company not found or no ratings available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ratings/bulk-compare")
def bulk_compare_ratings():
    """Bulk compare ratings for all companies"""
    try:
        result = rating_comparator.bulk_compare_ratings()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ratings/distribution")
def get_rating_distribution():
    """Get rating distribution analysis"""
    try:
        distribution = rating_comparator.get_rating_distribution_analysis()
        return distribution
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Data Collection Endpoints
@app.post("/collect/news")
async def collect_news_data(background_tasks: BackgroundTasks, company_ticker: str = None):
    """Collect news sentiment data"""
    try:
        background_tasks.add_task(orchestrator.collect_news_sentiment, company_ticker)
        return {"message": "News collection started", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect/macro")
async def collect_macro_data(background_tasks: BackgroundTasks):
    """Collect macroeconomic data"""
    try:
        background_tasks.add_task(macro_collector.collect_macro_data)
        return {"message": "Macroeconomic data collection started", "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/economic/summary")
def get_economic_summary():
    """Get current economic conditions summary"""
    try:
        summary = macro_collector.get_economic_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/collect")
def collect_data(request: DataCollectionRequest, background_tasks: BackgroundTasks):
    """Collect data for specified companies"""
    def run_collection():
        orchestrator = DataOrchestrator()
        try:
            if request.tickers:
                result = orchestrator.bulk_data_collection(request.tickers)
            else:
                result = orchestrator.daily_data_refresh()
            return result
        finally:
            orchestrator.close()
    
    background_tasks.add_task(run_collection)
    return {"message": "Data collection started", "tickers": request.tickers}

@app.get("/data/status")
def get_data_status():
    """Get data collection status for all companies"""
    orchestrator = DataOrchestrator()
    try:
        status = orchestrator.get_data_status()
        return status
    finally:
        orchestrator.close()

# Scoring Engine Endpoints
@app.post("/scoring/train")
def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train credit scoring models"""
    def run_training():
        engine = CreditScoringEngine()
        try:
            # Create synthetic training data
            X, y = engine.create_synthetic_training_data(request.n_samples)
            
            # Train models
            metrics = engine.train_models(X, y)
            
            # Select best model
            best_model = engine.select_best_model(metrics)
            
            return {
                "status": "completed",
                "best_model": best_model,
                "metrics": {name: {
                    "r2": metric.r2,
                    "mae": metric.mae,
                    "cv_score": metric.cv_score
                } for name, metric in metrics.items()},
                "training_samples": request.n_samples
            }
        finally:
            engine.close()
    
    background_tasks.add_task(run_training)
    return {"message": "Model training started", "samples": request.n_samples}

@app.post("/scoring/predict")
def predict_credit_score(request: PredictionRequest):
    """Predict credit score for a company"""
    engine = CreditScoringEngine()
    try:
        prediction = engine.predict_credit_score(request.company_id, request.model_name)
        
        if 'error' not in prediction:
            # Store the prediction
            stored = engine.store_prediction(request.company_id, prediction)
            prediction['stored'] = stored
        
        return prediction
    finally:
        engine.close()

@app.post("/scoring/batch")
def batch_score_companies(background_tasks: BackgroundTasks, company_ids: Optional[List[int]] = None):
    """Score multiple companies"""
    def run_batch_scoring():
        engine = CreditScoringEngine()
        try:
            results = engine.batch_score_companies(company_ids)
            return results
        finally:
            engine.close()
    
    background_tasks.add_task(run_batch_scoring)
    return {"message": "Batch scoring started", "company_ids": company_ids}

# News and Sentiment Endpoints
@app.get("/companies/{company_id}/sentiment")
def get_sentiment_summary(company_id: int, days_back: int = 30):
    """Get sentiment summary for a company"""
    from data_ingestion.news_sentiment_collector import NewsSentimentCollector
    
    # Get company ticker
    db = next(get_db())
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    collector = NewsSentimentCollector()
    try:
        summary = collector.get_sentiment_summary(company.ticker, days_back)
        return summary
    finally:
        collector.close()

@app.post("/companies/{company_id}/collect-news")
def collect_company_news(company_id: int, days_back: int = 7, background_tasks: BackgroundTasks = BackgroundTasks()):
    """Collect news for a specific company"""
    # Get company ticker
    db = next(get_db())
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    def run_news_collection():
        from data_ingestion.news_sentiment_collector import NewsSentimentCollector
        collector = NewsSentimentCollector()
        try:
            result = collector.collect_company_news(company.ticker, days_back)
            return result
        finally:
            collector.close()
    
    background_tasks.add_task(run_news_collection)
    return {"message": f"News collection started for {company.ticker}", "days_back": days_back}

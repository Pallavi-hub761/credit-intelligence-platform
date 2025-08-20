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
import json

# Import our custom modules
from data_ingestion.data_orchestrator import DataOrchestrator
from data_ingestion.macro_economic_collector import MacroEconomicCollector
from scoring.credit_scoring_engine import CreditScoringEngine
from scoring.explainability import ExplainabilityEngine
from nlp.event_classifier import CreditEventClassifier

# In-memory cache for alerts (replaces Redis)
alert_cache = {}
notification_cache = []

# Initialize components
orchestrator = DataOrchestrator()
macro_collector = MacroEconomicCollector()
scoring_engine = CreditScoringEngine()
explainability_engine = ExplainabilityEngine()
event_classifier = CreditEventClassifier()

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
    risk_category: str
    model_version: str
    confidence: float
    
    class Config:
        from_attributes = True

class ExplanationResponse(BaseModel):
    id: int
    score_id: int
    feature_name: str
    importance: float
    contribution: float
    explanation: str
    
    class Config:
        from_attributes = True

class CompanyCreate(BaseModel):
    name: str
    ticker: str

class PredictionRequest(BaseModel):
    ticker: str
    features: Optional[Dict[str, Any]] = None

class DataCollectionRequest(BaseModel):
    tickers: List[str]
    days: Optional[int] = 30

class TrainingRequest(BaseModel):
    retrain: bool = False
    model_types: Optional[List[str]] = None

class AlertResponse(BaseModel):
    id: str
    type: str
    severity: str
    message: str
    timestamp: datetime
    data: Dict[str, Any]

# In-memory alert system (replaces Redis-based alerts)
class SimpleAlertSystem:
    def __init__(self):
        self.alerts = []
        self.max_alerts = 100
    
    def add_alert(self, alert_type: str, severity: str, message: str, data: Dict[str, Any]):
        alert = {
            "id": f"alert_{len(self.alerts)}_{int(datetime.now().timestamp())}",
            "type": alert_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now(),
            "data": data
        }
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        return alert
    
    def get_recent_alerts(self, limit: int = 50):
        return sorted(self.alerts, key=lambda x: x["timestamp"], reverse=True)[:limit]

simple_alert_system = SimpleAlertSystem()

# Mock agency rating comparator
class SimpleAgencyComparator:
    def get_agency_ratings(self, ticker: str):
        # Mock ratings data
        return {
            "ticker": ticker,
            "ratings": {
                "moodys": {"rating": "A2", "outlook": "Stable"},
                "sp": {"rating": "A", "outlook": "Positive"},
                "fitch": {"rating": "A+", "outlook": "Stable"}
            },
            "consensus": "A",
            "upgrade_probability": 0.25,
            "downgrade_probability": 0.15
        }

rating_comparator = SimpleAgencyComparator()

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    try:
        init_db()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")

@app.get("/")
async def root():
    return {"message": "Credit Intelligence Platform API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/companies", response_model=List[CompanyResponse])
async def get_companies(db: Session = Depends(get_db)):
    companies = db.query(Company).all()
    return companies

@app.post("/companies", response_model=CompanyResponse)
async def create_company(company: CompanyCreate, db: Session = Depends(get_db)):
    # Check if company already exists
    existing = db.query(Company).filter(Company.ticker == company.ticker).first()
    if existing:
        raise HTTPException(status_code=400, detail="Company already exists")
    
    db_company = Company(name=company.name, ticker=company.ticker)
    db.add(db_company)
    db.commit()
    db.refresh(db_company)
    return db_company

@app.get("/companies/{company_id}/scores", response_model=List[ScoreResponse])
async def get_company_scores(company_id: int, limit: int = 10, db: Session = Depends(get_db)):
    scores = db.query(Score).filter(Score.company_id == company_id).order_by(Score.score_date.desc()).limit(limit).all()
    return scores

@app.post("/predict")
async def predict_credit_score(request: PredictionRequest, db: Session = Depends(get_db)):
    try:
        # Get company
        company = db.query(Company).filter(Company.ticker == request.ticker).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        # Generate prediction using scoring engine
        prediction = scoring_engine.predict_credit_score(request.ticker, request.features or {})
        
        # Store score in database
        score = Score(
            company_id=company.id,
            credit_score=prediction["score"],
            risk_category=prediction["risk_category"],
            model_version=prediction["model_version"],
            confidence=prediction["confidence"]
        )
        db.add(score)
        db.commit()
        db.refresh(score)
        
        # Generate explanation
        explanation_data = explainability_engine.explain_prediction(
            request.ticker, prediction["score"], request.features or {}
        )
        
        # Store explanations
        for feature, data in explanation_data["feature_contributions"].items():
            explanation = Explanation(
                score_id=score.id,
                feature_name=feature,
                importance=data.get("importance", 0.0),
                contribution=data.get("contribution", 0.0),
                explanation=data.get("explanation", "")
            )
            db.add(explanation)
        
        db.commit()
        
        # Check for alerts
        if prediction["score"] < 300:  # High risk
            simple_alert_system.add_alert(
                "score_change", "high", 
                f"High risk score detected for {request.ticker}: {prediction['score']:.0f}",
                {"ticker": request.ticker, "score": prediction["score"]}
            )
        
        return {
            "score": prediction["score"],
            "risk_category": prediction["risk_category"],
            "confidence": prediction["confidence"],
            "explanation": explanation_data,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companies/{company_id}/explanations")
async def get_score_explanations(company_id: int, score_id: Optional[int] = None, db: Session = Depends(get_db)):
    query = db.query(Explanation).join(Score).filter(Score.company_id == company_id)
    
    if score_id:
        query = query.filter(Explanation.score_id == score_id)
    
    explanations = query.order_by(Explanation.id.desc()).limit(20).all()
    return explanations

@app.post("/data/collect")
async def collect_data(request: DataCollectionRequest, background_tasks: BackgroundTasks):
    """Trigger data collection for specified tickers"""
    background_tasks.add_task(orchestrator.collect_all_data, request.tickers, request.days)
    return {"message": f"Data collection started for {len(request.tickers)} companies", "tickers": request.tickers}

@app.post("/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train or retrain credit scoring models"""
    background_tasks.add_task(scoring_engine.train_models, request.retrain)
    return {"message": "Model training started", "retrain": request.retrain}

@app.get("/sentiment/{ticker}")
async def get_sentiment_analysis(ticker: str):
    """Get sentiment analysis for a ticker"""
    try:
        analysis = event_classifier.analyze_bulk_news([ticker])
        return analysis.get(ticker, {"sentiment": "neutral", "events": [], "confidence": 0.5})
    except Exception as e:
        return {"sentiment": "neutral", "events": [], "confidence": 0.5, "error": str(e)}

@app.post("/collect/macro")
async def collect_macro_data(background_tasks: BackgroundTasks):
    """Collect macroeconomic data"""
    background_tasks.add_task(macro_collector.collect_all_data)
    return {"message": "Macroeconomic data collection started"}

@app.post("/nlp/analyze")
async def analyze_events(tickers: List[str]):
    """Analyze events for given tickers"""
    try:
        results = event_classifier.analyze_bulk_news(tickers)
        return results
    except Exception as e:
        return {"error": str(e), "results": {}}

@app.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(limit: int = 50):
    """Get recent alerts"""
    alerts = simple_alert_system.get_recent_alerts(limit)
    return alerts

@app.get("/ratings/{ticker}")
async def get_agency_ratings(ticker: str):
    """Get agency rating comparisons"""
    try:
        ratings = rating_comparator.get_agency_ratings(ticker)
        return ratings
    except Exception as e:
        return {"error": str(e), "ticker": ticker}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

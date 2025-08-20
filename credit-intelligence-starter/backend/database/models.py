from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Company(Base):
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    ticker = Column(String(20), nullable=False, unique=True)
    
    # Relationships
    prices = relationship("Price", back_populates="company")
    news = relationship("NewsRaw", back_populates="company")
    scores = relationship("Score", back_populates="company")

class Price(Base):
    __tablename__ = 'prices'
    
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    
    # Relationship
    company = relationship("Company", back_populates="prices")

class NewsRaw(Base):
    __tablename__ = 'news_raw'
    
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False)
    title = Column(Text, nullable=False)
    content = Column(Text)
    source = Column(String(100))
    published_at = Column(DateTime, nullable=False)
    url = Column(Text)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # positive, negative, neutral
    processed = Column(Boolean, default=False)
    
    # Relationship
    company = relationship("Company", back_populates="news")

class Score(Base):
    __tablename__ = 'scores'
    
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=False)
    score_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    credit_score = Column(Float, nullable=False)  # 0-1000 scale
    risk_category = Column(String(20))  # AAA, AA, A, BBB, BB, B, CCC, CC, C, D
    model_version = Column(String(50))
    confidence = Column(Float)  # 0-1
    
    # Relationship
    company = relationship("Company", back_populates="scores")

class Explanation(Base):
    __tablename__ = 'explanations'
    
    id = Column(Integer, primary_key=True)
    score_id = Column(Integer, ForeignKey('scores.id'), nullable=False)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float)
    contribution = Column(Float)  # SHAP value or similar
    importance = Column(Float)  # 0-1
    trend = Column(String(20))  # improving, declining, stable
    explanation_text = Column(Text)
    
    # Relationship
    score = relationship("Score")

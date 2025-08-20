import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base
from database.connection import get_db
from main import app
import os

# Test database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_get_companies(client):
    response = client.get("/companies")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_predict_endpoint(client):
    test_data = {
        "company_id": 1,
        "features": {
            "revenue_growth": 0.15,
            "debt_to_equity": 0.3,
            "current_ratio": 2.1,
            "roa": 0.08,
            "sentiment_score": 0.2
        }
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    result = response.json()
    assert "credit_score" in result
    assert "risk_category" in result
    assert "confidence" in result

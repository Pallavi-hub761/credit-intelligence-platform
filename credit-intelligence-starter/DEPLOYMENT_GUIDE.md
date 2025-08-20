# Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.9+
- Node.js 18+
- PostgreSQL 15+
- Redis 7+

## Environment Setup

1. **Clone and Setup**
```bash
git clone <repository-url>
cd credit-intelligence-starter
```

2. **Environment Variables**
Create `.env` file:
```bash
# Database
DATABASE_URL=postgresql://credit_user:your_password@localhost:5432/credit_intelligence
REDIS_URL=redis://localhost:6379

# API Keys (obtain from respective providers)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
FRED_API_KEY=your_fred_api_key

# Email Configuration
EMAIL_USERNAME=your_email@domain.com
EMAIL_PASSWORD=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
FROM_EMAIL=alerts@yourdomain.com
```

## Quick Start (Development)

```bash
# Start all services
docker-compose up --build

# Access points:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - MLflow: http://localhost:5000
```

## Production Deployment

### Option 1: Docker Compose (Recommended for small-medium scale)

```bash
# Production deployment
docker-compose -f deployment/docker-compose.prod.yml up -d

# Monitor logs
docker-compose -f deployment/docker-compose.prod.yml logs -f
```

### Option 2: Kubernetes (Recommended for large scale)

```bash
# Create namespace
kubectl create namespace credit-intelligence

# Create secrets
kubectl create secret generic postgres-secret \
  --from-literal=password=your_secure_password \
  -n credit-intelligence

kubectl create secret generic api-keys \
  --from-literal=alpha-vantage=your_key \
  --from-literal=news-api=your_key \
  --from-literal=fred-api=your_key \
  -n credit-intelligence

# Deploy application
kubectl apply -f deployment/deploy.yml

# Check status
kubectl get pods -n credit-intelligence
```

## Initial Data Setup

1. **Create Sample Companies**
```bash
# Access backend container
docker exec -it credit-intelligence-backend bash

# Run setup script
python -c "
from database.connection import init_db
from database.models import Company
from database.connection import get_db

init_db()
session = next(get_db())

companies = [
    Company(name='Apple Inc.', ticker='AAPL'),
    Company(name='Microsoft Corporation', ticker='MSFT'),
    Company(name='Tesla Inc.', ticker='TSLA'),
    Company(name='Ford Motor Company', ticker='F'),
    Company(name='General Electric', ticker='GE')
]

for company in companies:
    session.add(company)
session.commit()
print('Sample companies created!')
"
```

2. **Collect Initial Data**
```bash
# Trigger data collection via API
curl -X POST "http://localhost:8000/data/collect" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "TSLA", "F", "GE"]}'

# Collect macroeconomic data
curl -X POST "http://localhost:8000/collect/macro"

# Collect news sentiment
curl -X POST "http://localhost:8000/collect/news"
```

3. **Train Initial Models**
```bash
# Train models via API
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"retrain": true}'
```

## Monitoring and Maintenance

### Health Checks
```bash
# Check service health
curl http://localhost:8000/health

# Check database connection
curl http://localhost:8000/companies

# Check model status
curl http://localhost:8000/model/status
```

### Log Monitoring
```bash
# View application logs
docker-compose logs -f backend frontend

# View specific service logs
docker-compose logs -f postgres redis
```

### Performance Monitoring
- **Application Metrics**: Available at backend `/metrics` endpoint
- **MLflow Tracking**: http://localhost:5000
- **Database Monitoring**: Use PostgreSQL monitoring tools
- **Redis Monitoring**: Use Redis monitoring commands

## Scaling Considerations

### Horizontal Scaling
- **Backend**: Scale backend replicas in Kubernetes
- **Frontend**: Use CDN for static assets
- **Database**: Consider read replicas for PostgreSQL
- **Cache**: Redis cluster for high availability

### Performance Optimization
- **Database Indexing**: Ensure proper indexes on frequently queried columns
- **Caching Strategy**: Implement Redis caching for expensive operations
- **API Rate Limiting**: Configure appropriate rate limits
- **Connection Pooling**: Use connection pooling for database connections

## Security Hardening

### Production Security Checklist
- [ ] Change default passwords
- [ ] Enable SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up API authentication
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup strategy implementation

### SSL Certificate Setup
```bash
# Using Let's Encrypt with Certbot
certbot --nginx -d yourdomain.com -d api.yourdomain.com

# Update nginx.conf with certificate paths
# Restart nginx
docker-compose restart nginx
```

## Backup and Recovery

### Database Backup
```bash
# Create backup
docker exec postgres pg_dump -U credit_user credit_intelligence > backup.sql

# Restore backup
docker exec -i postgres psql -U credit_user credit_intelligence < backup.sql
```

### Model Artifacts Backup
```bash
# Backup MLflow artifacts
docker exec backend tar -czf /tmp/mlruns-backup.tar.gz /app/mlruns
docker cp backend:/tmp/mlruns-backup.tar.gz ./mlruns-backup.tar.gz
```

## Troubleshooting

### Common Issues

1. **Database Connection Issues**
```bash
# Check PostgreSQL status
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up postgres
```

2. **API Key Issues**
```bash
# Verify API keys are set
docker exec backend printenv | grep API_KEY

# Test API connectivity
curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=YOUR_KEY"
```

3. **Model Training Issues**
```bash
# Check MLflow logs
docker-compose logs mlflow

# Verify training data
curl http://localhost:8000/companies/1/scores
```

4. **Frontend Issues**
```bash
# Check frontend logs
docker-compose logs frontend

# Verify API connectivity
curl http://localhost:8000/companies
```

### Performance Issues

1. **Slow API Responses**
- Check database query performance
- Verify Redis cache is working
- Monitor CPU and memory usage

2. **High Memory Usage**
- Adjust Docker memory limits
- Optimize ML model memory usage
- Implement data pagination

3. **Database Performance**
- Add indexes on frequently queried columns
- Optimize SQL queries
- Consider connection pooling

## Maintenance Tasks

### Regular Maintenance (Weekly)
- [ ] Check system logs for errors
- [ ] Verify backup integrity
- [ ] Monitor disk space usage
- [ ] Review performance metrics
- [ ] Update security patches

### Model Maintenance (Monthly)
- [ ] Review model performance metrics
- [ ] Check for data drift
- [ ] Retrain models if necessary
- [ ] Update feature engineering
- [ ] Validate model explanations

### Infrastructure Maintenance (Quarterly)
- [ ] Update dependencies
- [ ] Review security configurations
- [ ] Optimize resource allocation
- [ ] Plan capacity upgrades
- [ ] Disaster recovery testing

## Support and Documentation

- **API Documentation**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **System Logs**: `docker-compose logs`
- **Health Checks**: http://localhost:8000/health

For additional support, refer to the main README.md and create issues in the repository.

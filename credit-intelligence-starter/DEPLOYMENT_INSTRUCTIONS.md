# Quick Deployment Instructions

## Option 1: Heroku Deployment (Recommended - 15 minutes)

### Prerequisites
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Create Heroku account: https://signup.heroku.com/

### Steps:

```bash
# 1. Navigate to project directory
cd C:\Users\Admin\Downloads\credit-intelligence-starter

# 2. Initialize git repository
git init
git add .
git commit -m "Initial commit - Credit Intelligence Platform"

# 3. Login to Heroku
heroku login

# 4. Create Heroku app
heroku create your-credit-intelligence-app

# 5. Add PostgreSQL and Redis addons
heroku addons:create heroku-postgresql:mini
heroku addons:create heroku-redis:mini

# 6. Set environment variables (optional - app works without API keys)
heroku config:set ALPHA_VANTAGE_API_KEY=your_key_here
heroku config:set NEWS_API_KEY=your_key_here
heroku config:set FRED_API_KEY=your_key_here

# 7. Deploy to Heroku
git push heroku main

# 8. Initialize database
heroku run python -c "from database.connection import init_db; init_db()"

# 9. Open your app
heroku open
```

Your app will be available at: `https://your-credit-intelligence-app.herokuapp.com`

## Option 2: Railway Deployment (Alternative - 10 minutes)

### Steps:

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login to Railway
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up

# 5. Add PostgreSQL
railway add postgresql

# 6. Add Redis
railway add redis

# 7. Set environment variables
railway variables set ALPHA_VANTAGE_API_KEY=your_key
railway variables set NEWS_API_KEY=your_key
railway variables set FRED_API_KEY=your_key
```

## Option 3: Render Deployment (Free tier available)

### Steps:

1. Go to https://render.com and create account
2. Connect your GitHub repository
3. Create new Web Service
4. Use these settings:
   - **Build Command**: `pip install -r backend/requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.9
5. Add PostgreSQL database from Render dashboard
6. Add Redis instance from Render dashboard
7. Set environment variables in Render dashboard

## Option 4: AWS/GCP (Production - 1-2 hours)

### AWS ECS Deployment:

```bash
# 1. Install AWS CLI and configure
aws configure

# 2. Build and push Docker images
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t credit-intelligence-backend ./backend
docker tag credit-intelligence-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-intelligence-backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/credit-intelligence-backend:latest

# 3. Deploy using provided ECS task definition
aws ecs create-service --cluster credit-intelligence --service-name backend --task-definition credit-intelligence-backend
```

### GCP Cloud Run Deployment:

```bash
# 1. Install gcloud CLI and authenticate
gcloud auth login
gcloud config set project your-project-id

# 2. Build and deploy
gcloud builds submit --tag gcr.io/your-project-id/credit-intelligence-backend ./backend
gcloud run deploy credit-intelligence --image gcr.io/your-project-id/credit-intelligence-backend --platform managed --region us-central1 --allow-unauthenticated
```

## Post-Deployment Setup

### 1. Initialize Sample Data
```bash
# Create sample companies via API
curl -X POST "https://your-app-url.com/companies" \
  -H "Content-Type: application/json" \
  -d '{"name": "Apple Inc.", "ticker": "AAPL"}'

curl -X POST "https://your-app-url.com/companies" \
  -H "Content-Type: application/json" \
  -d '{"name": "Microsoft Corporation", "ticker": "MSFT"}'
```

### 2. Trigger Initial Data Collection
```bash
# Collect data for sample companies
curl -X POST "https://your-app-url.com/data/collect" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT"]}'

# Collect macroeconomic data
curl -X POST "https://your-app-url.com/collect/macro"
```

### 3. Train Initial Models
```bash
# Train models
curl -X POST "https://your-app-url.com/train" \
  -H "Content-Type: application/json" \
  -d '{"retrain": true}'
```

## Custom Domain Setup (Optional)

### For Heroku:
```bash
# Add custom domain
heroku domains:add www.your-domain.com

# Configure DNS (point to Heroku's DNS target)
# Add CNAME record: www -> your-app-name.herokuapp.com

# Add SSL certificate
heroku certs:auto:enable
```

### For other providers:
- Follow provider-specific domain configuration
- Most providers offer automatic SSL certificates

## Monitoring & Logs

### Heroku:
```bash
# View logs
heroku logs --tail

# Check app status
heroku ps

# Scale dynos
heroku ps:scale web=2
```

### Health Check:
Visit: `https://your-app-url.com/health`

## Troubleshooting

### Common Issues:

1. **Database Connection Error**
   - Ensure PostgreSQL addon is properly configured
   - Check DATABASE_URL environment variable

2. **API Key Issues**
   - App works without API keys (uses mock data)
   - Set keys as environment variables if needed

3. **Memory Issues**
   - Upgrade to higher tier dyno/instance
   - Optimize model loading in production

4. **Slow Performance**
   - Add Redis caching
   - Enable database connection pooling
   - Scale to multiple instances

## Success Verification

1. **Backend API**: `https://your-app-url.com/docs` (Swagger UI)
2. **Health Check**: `https://your-app-url.com/health`
3. **Companies List**: `https://your-app-url.com/companies`
4. **Frontend**: `https://your-app-url.com` (if deployed separately)

Your Credit Intelligence Platform is now live and accessible to the public!

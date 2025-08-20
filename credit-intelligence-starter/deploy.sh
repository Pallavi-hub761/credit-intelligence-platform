#!/bin/bash

# Quick Deployment Script for Credit Intelligence Platform
# Choose your deployment option by uncommenting the relevant section

echo "🚀 Credit Intelligence Platform Deployment Script"
echo "=================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit - Credit Intelligence Platform"
fi

echo "Choose deployment option:"
echo "1. Heroku (Recommended - Free tier available)"
echo "2. Railway (Fast deployment)"
echo "3. Render (Free tier available)"
echo "4. Manual setup instructions"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "🔧 Deploying to Heroku..."
        
        # Check if Heroku CLI is installed
        if ! command -v heroku &> /dev/null; then
            echo "❌ Heroku CLI not found. Please install it from: https://devcenter.heroku.com/articles/heroku-cli"
            exit 1
        fi
        
        # Login to Heroku
        echo "🔐 Please login to Heroku..."
        heroku login
        
        # Create app
        read -p "Enter your app name (or press Enter for auto-generated): " app_name
        if [ -z "$app_name" ]; then
            heroku create
        else
            heroku create $app_name
        fi
        
        # Add addons
        echo "📦 Adding PostgreSQL and Redis..."
        heroku addons:create heroku-postgresql:mini
        heroku addons:create heroku-redis:mini
        
        # Set environment variables (optional)
        read -p "Do you have API keys to set? (y/n): " has_keys
        if [ "$has_keys" = "y" ]; then
            read -p "Alpha Vantage API Key (optional): " av_key
            read -p "News API Key (optional): " news_key
            read -p "FRED API Key (optional): " fred_key
            
            [ ! -z "$av_key" ] && heroku config:set ALPHA_VANTAGE_API_KEY=$av_key
            [ ! -z "$news_key" ] && heroku config:set NEWS_API_KEY=$news_key
            [ ! -z "$fred_key" ] && heroku config:set FRED_API_KEY=$fred_key
        fi
        
        # Deploy
        echo "🚀 Deploying to Heroku..."
        git push heroku main
        
        # Initialize database
        echo "🗄️ Initializing database..."
        heroku run python -c "from database.connection import init_db; init_db()"
        
        # Get app URL
        app_url=$(heroku info -s | grep web_url | cut -d= -f2)
        echo "✅ Deployment complete!"
        echo "🌐 Your app is available at: $app_url"
        echo "📚 API Documentation: ${app_url}docs"
        echo "🏥 Health Check: ${app_url}health"
        
        # Open app
        heroku open
        ;;
        
    2)
        echo "🚂 Deploying to Railway..."
        
        # Check if Railway CLI is installed
        if ! command -v railway &> /dev/null; then
            echo "📦 Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        
        # Login and deploy
        railway login
        railway init
        railway up
        
        # Add databases
        railway add postgresql
        railway add redis
        
        echo "✅ Railway deployment initiated!"
        echo "🌐 Check your Railway dashboard for the app URL"
        ;;
        
    3)
        echo "🎨 Render deployment instructions:"
        echo "1. Go to https://render.com and create an account"
        echo "2. Connect your GitHub repository"
        echo "3. Create a new Web Service"
        echo "4. Use these settings:"
        echo "   - Build Command: pip install -r backend/requirements.txt"
        echo "   - Start Command: uvicorn main:app --host 0.0.0.0 --port \$PORT"
        echo "   - Environment: Python 3.9"
        echo "5. Add PostgreSQL database from Render dashboard"
        echo "6. Add Redis instance from Render dashboard"
        echo "7. Set environment variables in Render dashboard"
        ;;
        
    4)
        echo "📖 Manual deployment instructions:"
        echo "Please refer to DEPLOYMENT_INSTRUCTIONS.md for detailed setup guides"
        echo "Available options:"
        echo "- Heroku (easiest)"
        echo "- Railway (fast)"
        echo "- Render (free tier)"
        echo "- AWS ECS (production)"
        echo "- GCP Cloud Run (production)"
        ;;
        
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
echo "🎉 Next Steps:"
echo "1. Visit your app URL to see the dashboard"
echo "2. Check API documentation at /docs endpoint"
echo "3. Initialize sample data using the API endpoints"
echo "4. Monitor logs and performance"
echo ""
echo "📞 Need help? Check DEPLOYMENT_INSTRUCTIONS.md for troubleshooting"

# 🚀 Deployment Guide

This guide covers how to deploy the Stock Prediction Dashboard locally and to the cloud.

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)

## 🏠 Local Deployment

### Quick Start

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Trading-Tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Option 1: Use the launcher script
   ./run_app.sh
   
   # Option 2: Run directly
   python -m streamlit run app.py
   
   # Option 3: Use streamlit command
   streamlit run app.py
   ```

4. **Access the dashboard**
   - Open your browser
   - Navigate to `http://localhost:8501`
   - The dashboard will load automatically

### Configuration

1. **API Keys (Optional)**
   - Edit `config.py` to add your API keys
   - Get free API keys from:
     - [News API](https://newsapi.org/) - for news sentiment
     - [Twitter Developer Portal](https://developer.twitter.com/) - for social sentiment

2. **Customization**
   - Modify `config.py` to change default settings
   - Adjust technical indicator parameters
   - Change prediction horizons

## ☁️ Cloud Deployment

### Streamlit Cloud (Recommended)

1. **Prepare your repository**
   - Push your code to GitHub
   - Ensure all files are committed

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set the main file path to `app.py`
   - Click "Deploy"

3. **Configure environment variables**
   - In Streamlit Cloud, go to your app settings
   - Add environment variables for API keys:
     ```
     NEWS_API_KEY=your_news_api_key
     TWITTER_API_KEY=your_twitter_api_key
     ```

### Heroku Deployment

1. **Create Heroku app**
   ```bash
   heroku create your-app-name
   ```

2. **Add buildpacks**
   ```bash
   heroku buildpacks:add heroku/python
   ```

3. **Create Procfile**
   ```bash
   echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
   ```

4. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and run**
   ```bash
   docker build -t stock-dashboard .
   docker run -p 8501:8501 stock-dashboard
   ```

## 🔧 Production Considerations

### Security

1. **API Keys**
   - Never commit API keys to version control
   - Use environment variables
   - Rotate keys regularly

2. **Data Privacy**
   - Portfolio data is stored locally by default
   - Consider database storage for production

### Performance

1. **Caching**
   - Data is cached for 5 minutes
   - Adjust cache duration in `config.py`

2. **Resource Limits**
   - Monitor memory usage
   - Consider limiting concurrent users

### Monitoring

1. **Logs**
   - Streamlit provides built-in logging
   - Monitor for errors and performance issues

2. **Health Checks**
   - Add health check endpoints
   - Monitor API response times

## 🛠️ Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Kill existing process
   lsof -ti:8501 | xargs kill -9
   
   # Or use different port
   streamlit run app.py --server.port 8502
   ```

2. **Dependencies not found**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   ```

3. **API rate limits**
   - Implement rate limiting
   - Use caching to reduce API calls
   - Consider paid API plans

### Performance Issues

1. **Slow loading**
   - Reduce data period
   - Disable unused features
   - Optimize chart rendering

2. **Memory usage**
   - Limit concurrent users
   - Implement data cleanup
   - Use streaming for large datasets

## 📊 Scaling

### Horizontal Scaling

1. **Load Balancer**
   - Use nginx or similar
   - Distribute traffic across instances

2. **Multiple Instances**
   - Deploy multiple app instances
   - Use shared storage for data

### Vertical Scaling

1. **Resource Allocation**
   - Increase CPU/memory
   - Use dedicated servers

2. **Database**
   - Use PostgreSQL for portfolio data
   - Implement proper indexing

## 🔄 Updates and Maintenance

### Regular Updates

1. **Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Code Updates**
   - Pull latest changes
   - Test thoroughly
   - Deploy during low traffic

### Backup

1. **Portfolio Data**
   - Backup `portfolio.json` regularly
   - Consider cloud storage

2. **Configuration**
   - Version control configuration
   - Document changes

## 📞 Support

For deployment issues:
1. Check Streamlit documentation
2. Review error logs
3. Test locally first
4. Consider community forums

---

**Note**: This dashboard is for educational purposes. Ensure compliance with data provider terms of service and local regulations. 
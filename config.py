# Configuration file for Stock Prediction Dashboard
# Add your API keys here for enhanced functionality

# News API Configuration
# Get your free API key from: https://newsapi.org/
NEWS_API_KEY = None  # Replace with your actual API key

# Twitter API Configuration  
# Get your API keys from: https://developer.twitter.com/
TWITTER_API_KEY = None
TWITTER_API_SECRET = None
TWITTER_ACCESS_TOKEN = None
TWITTER_ACCESS_TOKEN_SECRET = None

# Alpha Vantage API (Alternative to Yahoo Finance)
# Get your free API key from: https://www.alphavantage.co/
ALPHA_VANTAGE_API_KEY = None

# Model Configuration
DEFAULT_TRAINING_PERIOD = "2y"  # Default period for training models
PREDICTION_HORIZONS = [1, 5, 20]  # Days for predictions
CACHE_DURATION = 300  # Cache data for 5 minutes

# Technical Indicators Configuration
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Sentiment Analysis Configuration
SENTIMENT_ANALYSIS_ENABLED = True
NEWS_LOOKBACK_DAYS = 7
SOCIAL_MENTIONS_LIMIT = 100

# UI Configuration
DEFAULT_TICKER = "AAPL"
DEFAULT_PERIOD = "2y"
SHOW_TECHNICAL_INDICATORS = True
SHOW_SENTIMENT_ANALYSIS = True
SHOW_PREDICTIONS = True

# Risk Management
MAX_PREDICTION_CONFIDENCE = 0.9
MIN_DATA_POINTS_FOR_TRAINING = 100 
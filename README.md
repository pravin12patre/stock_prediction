# Stock Prediction Dashboard

A comprehensive stock analysis and prediction tool built with Python and Streamlit. This dashboard combines advanced technical analysis, enhanced machine learning predictions, and sentiment analysis to provide insights into stock price movements. The dashboard now features the enhanced AI models by default, providing superior prediction accuracy and confidence estimation.

## Features

### Technical Analysis
- **Interactive Price Charts**: Candlestick charts with volume
- **Technical Indicators**: 
  - Moving Averages (SMA 20, SMA 50, EMA 12, EMA 26)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Stochastic Oscillator
  - Rate of Change (ROC)

### Enhanced Machine Learning Predictions
- **Multi-horizon Predictions**: 1 day, 3 days, 5 days, 14 days, 20 days, and 30 days forecasts
- **Advanced Ensemble Models**: Random Forest, Gradient Boosting, Extra Trees, Ridge Regression, SVR, and CatBoost
- **Confidence Metrics**: Advanced prediction confidence estimation with model agreement analysis
- **Feature Engineering**: 60+ advanced technical and price-based features
- **Model Selection**: Automatic feature selection and hyperparameter optimization
- **Robust Error Handling**: Handles stocks with limited historical data

### Sentiment Analysis
- **News Sentiment**: Analysis of financial news articles
- **Social Media Sentiment**: Twitter and social media sentiment
- **Sentiment Visualization**: Interactive sentiment charts

### User Interface
- **Responsive Design**: Wide layout with sidebar controls
- **Real-time Data**: Live stock data from Yahoo Finance
- **Interactive Charts**: Plotly-powered interactive visualizations
- **Customizable Analysis**: Toggle different analysis components

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## How to Use

### Basic Usage
1. **Enter a Stock Ticker**: Type a valid stock symbol (e.g., AAPL, MSFT, GOOGL) in the sidebar
2. **Select Data Period**: Choose from 1 year, 2 years, 5 years, or maximum available data
3. **Configure Analysis**: Toggle technical indicators, sentiment analysis, and predictions in the sidebar

### Understanding the Dashboard

#### Price Chart Tab
- **Candlestick Chart**: Shows open, high, low, and close prices
- **Technical Indicators**: Overlay moving averages, Bollinger Bands, RSI, and MACD
- **Volume Analysis**: Volume bars with color coding

#### Predictions Tab
- **Price Forecasts**: Predicted prices for 1 day, 1 week, and 1 month
- **Confidence Gauges**: Visual representation of prediction confidence
- **Trend Analysis**: Bullish/bearish signals based on predictions

#### Technical Analysis Tab
- **Key Indicators**: RSI, MACD, and moving average analysis
- **Signal Interpretation**: Buy/sell signals based on technical indicators
- **Bollinger Bands**: Price position relative to volatility bands

#### Sentiment Tab
- **News Sentiment**: Sentiment analysis of financial news
- **Social Sentiment**: Social media sentiment analysis
- **Overall Sentiment**: Combined sentiment score and interpretation

## Configuration

### API Keys (Optional)
For enhanced sentiment analysis, you can add API keys:

1. **News API**: Get a free key from [newsapi.org](https://newsapi.org/)
2. **Twitter API**: Get keys from [Twitter Developer Portal](https://developer.twitter.com/)

Add your API keys to the `data_fetcher.py` file:
```python
def __init__(self):
    self.news_api_key = "your_news_api_key_here"
    self.twitter_api_key = "your_twitter_api_key_here"
```

### Enhanced Model Training
- Models are automatically trained when you first enable predictions
- Training uses the last 2 years of data by default with robust data quality checks
- Advanced ensemble models with automatic hyperparameter optimization
- Models are cached for faster subsequent predictions
- Handles stocks with limited historical data (minimum 100 days)
- Automatic feature selection based on importance scores

## Project Structure

```
stock_prediction/
├── app.py                 # Main Streamlit application (Enhanced Dashboard)
├── enhanced_app.py        # Enhanced dashboard (alternative)
├── enhanced_ml_predictor.py # Enhanced ML models and predictions
├── data_fetcher.py        # Data fetching and technical indicators
├── ml_predictor.py        # Original ML models and predictions
├── visualizer.py          # Chart creation and visualization
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Important Disclaimers

### Educational Purpose Only
This tool is designed for **educational and research purposes only**. It should not be used as the sole basis for investment decisions.

### No Financial Advice
- The predictions and analysis provided are not financial advice
- Always conduct your own research before making investment decisions
- Consult with qualified financial advisors for personalized investment advice

### Prediction Limitations
- **No 100% Accuracy**: Stock predictions are inherently uncertain
- **Market Volatility**: Unpredictable events can significantly impact stock prices
- **Model Limitations**: Machine learning models have limitations and may not capture all market factors
- **Historical Data**: Past performance does not guarantee future results

### Risk Warning
- **Investment Risk**: All investments carry risk of loss
- **Market Risk**: Stock prices can go down as well as up
- **Liquidity Risk**: Some stocks may be difficult to buy or sell quickly

## Data Sources

- **Stock Data**: Yahoo Finance (via yfinance)
- **Technical Indicators**: Calculated using pandas-ta
- **Sentiment Data**: Placeholder implementation (requires API keys for real data)

## Updates and Maintenance

### Regular Updates
- Stock data is cached for 5 minutes to reduce API calls
- Models are retrained when new data is available
- Technical indicators are calculated in real-time

### Performance Optimization
- Data caching to improve load times
- Efficient chart rendering with Plotly
- Optimized ML model training

## Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Improving documentation
- Adding new technical indicators
- Enhancing ML models

## License

This project is for educational purposes. Please ensure compliance with any applicable terms of service for data sources used.

## Support

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify that the stock ticker symbol is valid
3. Ensure you have a stable internet connection for data fetching
4. Check the console for any error messages

---

**Remember**: This tool is for educational purposes only. Always do your own research and consult with financial professionals before making investment decisions. 
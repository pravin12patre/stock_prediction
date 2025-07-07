import yfinance as yf
import pandas as pd
import numpy as np
try:
    import pandas_ta as ta
except ImportError:
    # Fallback for compatibility issues
    print("Warning: pandas_ta not available, using basic technical indicators")
    ta = None
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import time
from config import *

class StockDataFetcher:
    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.twitter_api_key = TWITTER_API_KEY
        self.twitter_api_secret = TWITTER_API_SECRET
        self.twitter_access_token = TWITTER_ACCESS_TOKEN
        self.twitter_access_token_secret = TWITTER_ACCESS_TOKEN_SECRET
        
    def fetch_stock_data(self, ticker, period="2y"):
        """Fetch historical stock data"""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            # Check if data is empty or None
            if data is None or data.empty:
                print(f"No data found for ticker {ticker}")
                return None
            
            # Adjust minimum data requirements based on period
            if period in ['5d', '1mo']:
                min_data_points = 3  # Very short periods need minimal data
            elif period in ['2mo', '3mo']:
                min_data_points = 5  # Short periods need some data
            elif period in ['6mo']:
                min_data_points = 10  # Medium periods need more data
            else:
                min_data_points = 10  # Longer periods need substantial data
                
            # Check if we have enough data points
            if len(data) < min_data_points:
                print(f"Insufficient data for {ticker} ({period}): only {len(data)} data points, need at least {min_data_points}")
                return None
                
            print(f"Successfully fetched {len(data)} data points for {ticker} ({period})")
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def calculate_technical_indicators(self, data):
        """Calculate various technical indicators"""
        if data is None or data.empty:
            return None
            
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate indicators based on available data
        data_length = len(df)
        
        # Moving averages - adjust length based on available data
        if ta is not None:
            if data_length >= 20:
                df['SMA_20'] = ta.sma(df['Close'], length=20)
            else:
                df['SMA_20'] = ta.sma(df['Close'], length=min(data_length-1, 5))
                
            if data_length >= 50:
                df['SMA_50'] = ta.sma(df['Close'], length=50)
            else:
                df['SMA_50'] = ta.sma(df['Close'], length=min(data_length-1, 10))
                
            if data_length >= 12:
                df['EMA_12'] = ta.ema(df['Close'], length=12)
            else:
                df['EMA_12'] = ta.ema(df['Close'], length=min(data_length-1, 3))
                
            if data_length >= 26:
                df['EMA_26'] = ta.ema(df['Close'], length=26)
            else:
                df['EMA_26'] = ta.ema(df['Close'], length=min(data_length-1, 5))
        else:
            # Fallback calculations without pandas_ta
            if data_length >= 20:
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
            else:
                df['SMA_20'] = df['Close'].rolling(window=min(data_length-1, 5)).mean()
                
            if data_length >= 50:
                df['SMA_50'] = df['Close'].rolling(window=50).mean()
            else:
                df['SMA_50'] = df['Close'].rolling(window=min(data_length-1, 10)).mean()
                
            if data_length >= 12:
                df['EMA_12'] = df['Close'].ewm(span=12).mean()
            else:
                df['EMA_12'] = df['Close'].ewm(span=min(data_length-1, 3)).mean()
                
            if data_length >= 26:
                df['EMA_26'] = df['Close'].ewm(span=26).mean()
            else:
                df['EMA_26'] = df['Close'].ewm(span=min(data_length-1, 5)).mean()
        
        # MACD - only if we have enough data
        if data_length >= 26:
            if ta is not None:
                try:
                    macd = ta.macd(df['Close'])
                    if macd is not None:
                        df = pd.concat([df, macd], axis=1)
                    else:
                        # Create placeholder MACD columns if calculation fails
                        df['MACD_12_26_9'] = np.nan
                        df['MACDh_12_26_9'] = np.nan
                        df['MACDs_12_26_9'] = np.nan
                except Exception as e:
                    print(f"Error calculating MACD for {data_length} data points: {e}")
                    # Create placeholder MACD columns
                    df['MACD_12_26_9'] = np.nan
                    df['MACDh_12_26_9'] = np.nan
                    df['MACDs_12_26_9'] = np.nan
            else:
                # Fallback MACD calculation
                ema12 = df['Close'].ewm(span=12).mean()
                ema26 = df['Close'].ewm(span=26).mean()
                df['MACD_12_26_9'] = ema12 - ema26
                df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9).mean()
                df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        else:
            # Create placeholder MACD columns
            df['MACD_12_26_9'] = np.nan
            df['MACDh_12_26_9'] = np.nan
            df['MACDs_12_26_9'] = np.nan
        
        # RSI - adjust length based on available data
        if data_length >= 14:
            if ta is not None:
                df['RSI'] = ta.rsi(df['Close'], length=14)
            else:
                # Fallback RSI calculation
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
        else:
            if ta is not None:
                df['RSI'] = ta.rsi(df['Close'], length=min(data_length-1, 3))
            else:
                # Fallback RSI calculation for short periods
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=min(data_length-1, 3)).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=min(data_length-1, 3)).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands - adjust length based on available data
        if data_length >= 20:
            try:
                bb = ta.bbands(df['Close'])
                if bb is not None:
                    df = pd.concat([df, bb], axis=1)
                else:
                    # Create placeholder BB columns if calculation fails
                    df['BBL_5_2.0'] = np.nan
                    df['BBM_5_2.0'] = np.nan
                    df['BBU_5_2.0'] = np.nan
                    df['BBB_5_2.0'] = np.nan
                    df['BBP_5_2.0'] = np.nan
            except Exception as e:
                print(f"Error calculating Bollinger Bands for {data_length} data points: {e}")
                # Create placeholder BB columns
                df['BBL_5_2.0'] = np.nan
                df['BBM_5_2.0'] = np.nan
                df['BBU_5_2.0'] = np.nan
                df['BBB_5_2.0'] = np.nan
                df['BBP_5_2.0'] = np.nan
        else:
            # Create placeholder BB columns for short periods
            df['BBL_5_2.0'] = np.nan
            df['BBM_5_2.0'] = np.nan
            df['BBU_5_2.0'] = np.nan
            df['BBB_5_2.0'] = np.nan
            df['BBP_5_2.0'] = np.nan
        
        # Stochastic - adjust length based on available data
        if data_length >= 14:
            try:
                stoch = ta.stoch(df['High'], df['Low'], df['Close'])
                if stoch is not None:
                    df = pd.concat([df, stoch], axis=1)
                else:
                    # Create placeholder Stochastic columns if calculation fails
                    df['STOCHk_14_3_3'] = np.nan
                    df['STOCHd_14_3_3'] = np.nan
            except Exception as e:
                print(f"Error calculating Stochastic for {data_length} data points: {e}")
                # Create placeholder Stochastic columns
                df['STOCHk_14_3_3'] = np.nan
                df['STOCHd_14_3_3'] = np.nan
        else:
            # Create placeholder Stochastic columns for short periods
            df['STOCHk_14_3_3'] = np.nan
            df['STOCHd_14_3_3'] = np.nan
        
        # Volume indicators - adjust length based on available data
        if data_length >= 20:
            if ta is not None:
                df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
            else:
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        else:
            if ta is not None:
                df['Volume_SMA'] = ta.sma(df['Volume'], length=min(data_length-1, 3))
            else:
                df['Volume_SMA'] = df['Volume'].rolling(window=min(data_length-1, 3)).mean()
        
        # Price momentum - adjust length based on available data
        if data_length >= 10:
            if ta is not None:
                df['ROC_10'] = ta.roc(df['Close'], length=10)
            else:
                df['ROC_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        else:
            if ta is not None:
                df['ROC_10'] = ta.roc(df['Close'], length=min(data_length-1, 2))
            else:
                df['ROC_10'] = ((df['Close'] - df['Close'].shift(min(data_length-1, 2))) / df['Close'].shift(min(data_length-1, 2))) * 100
        
        return df
    
    def get_news_sentiment(self, ticker, days=7):
        """Get news sentiment for a stock"""
        if self.news_api_key:
            try:
                # Use real News API
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{ticker}" stock',
                    'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'pageSize': 50
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    
                    if articles:
                        sentiments = []
                        for article in articles:
                            title = article.get('title', '')
                            description = article.get('description', '')
                            content = f"{title} {description}"
                            
                            # Analyze sentiment
                            blob = TextBlob(content)
                            sentiments.append(blob.sentiment.polarity)
                        
                        avg_sentiment = np.mean(sentiments)
                        return {
                            'sentiment_score': avg_sentiment,
                            'sentiment_label': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral',
                            'articles_count': len(articles),
                            'articles': articles[:5]  # Return first 5 articles for display
                        }
                
                # Fallback to placeholder if API fails
                return self._get_placeholder_sentiment()
                
            except Exception as e:
                print(f"Error fetching news sentiment: {e}")
                return self._get_placeholder_sentiment()
        else:
            # Use placeholder if no API key
            return self._get_placeholder_sentiment()
    
    def _get_placeholder_sentiment(self):
        """Generate placeholder sentiment data"""
        sentiment_score = np.random.uniform(-1, 1)
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral',
            'articles_count': np.random.randint(5, 20),
            'articles': []
        }
    
    def get_social_sentiment(self, ticker):
        """Get social media sentiment (placeholder - requires API keys)"""
        # This is a placeholder. You'll need to add your Twitter API keys
        # and implement the actual social media fetching logic
        try:
            # Placeholder sentiment score (random for demo)
            sentiment_score = np.random.uniform(-1, 1)
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': 'Positive' if sentiment_score > 0 else 'Negative' if sentiment_score < 0 else 'Neutral',
                'mentions_count': np.random.randint(10, 100)
            }
        except Exception as e:
            print(f"Error fetching social sentiment: {e}")
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'mentions_count': 0
            }
    
    def prepare_features(self, data):
        """Prepare features for ML model"""
        if data is None or data.empty:
            return None
            
        df = data.copy()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_20d'] = df['Close'].pct_change(20)
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change()
        
        if 'Volume_SMA' in df.columns:
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        else:
            df['Volume_Ratio'] = np.nan
        
        # Technical indicator features - check if columns exist before using them
        if 'SMA_20' in df.columns:
            df['SMA_Ratio'] = df['Close'] / df['SMA_20']
        else:
            df['SMA_Ratio'] = np.nan
            
        if 'EMA_12' in df.columns:
            df['EMA_Ratio'] = df['Close'] / df['EMA_12']
        else:
            df['EMA_Ratio'] = np.nan
        
        # Bollinger Bands position - check if columns exist
        if 'BBL_5_2.0' in df.columns and 'BBU_5_2.0' in df.columns:
            bb_range = df['BBU_5_2.0'] - df['BBL_5_2.0']
            # Avoid division by zero
            df['BB_Position'] = np.where(bb_range != 0, 
                                        (df['Close'] - df['BBL_5_2.0']) / bb_range, 
                                        0.5)  # Default to middle if range is zero
        else:
            df['BB_Position'] = np.nan
        
        # Remove NaN values but be more lenient for short periods
        # For very short periods, fill NaN with forward fill and then drop remaining NaN
        if len(df) < 20:
            # Fill NaN values with forward fill for short periods
            df = df.fillna(method='ffill').fillna(method='bfill')
            # If still have NaN, fill with 0 for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(0)
        else:
            # For longer periods, drop NaN rows
            df = df.dropna()
        
        # Final check - if still no data, return None
        if df.empty:
            print("Warning: All data was dropped due to NaN values")
            return None
            
        print(f"Features data shape after processing: {df.shape}")
        return df 
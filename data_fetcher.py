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
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from config import *
import re

class StockDataFetcher:
    def __init__(self):
        self.news_api_key = NEWS_API_KEY
        self.twitter_api_key = TWITTER_API_KEY
        self.twitter_api_secret = TWITTER_API_SECRET
        self.twitter_access_token = TWITTER_ACCESS_TOKEN
        self.twitter_access_token_secret = TWITTER_ACCESS_TOKEN_SECRET
        
    def fetch_yahoo_finance_news(self, ticker, days=7):
        """Fetch news from Yahoo Finance for a specific stock"""
        try:
            # Get stock object
            stock = yf.Ticker(ticker)
            
            # Fetch news from Yahoo Finance
            news = stock.news
            
            if not news:
                print(f"No Yahoo Finance news found for {ticker}")
                return []
            
            # Filter news by date (last N days)
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered_news = []
            
            for article in news:
                # Convert timestamp to datetime
                if 'providerPublishTime' in article:
                    article_date = datetime.fromtimestamp(article['providerPublishTime'])
                elif 'published' in article:
                    # Handle different timestamp formats
                    try:
                        article_date = datetime.fromtimestamp(article['published'])
                    except:
                        article_date = datetime.now()
                else:
                    article_date = datetime.now()
                
                # Only include recent articles
                if article_date >= cutoff_date:
                    # Get title and summary, handle empty/placeholder values
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    
                    # If title is empty or just "...", try to get it from other fields
                    if not title or title == '...':
                        title = article.get('headline', '') or article.get('name', '') or 'No Title Available'
                    
                    # If summary is empty or just "...", try to get it from other fields
                    if not summary or summary == '...':
                        summary = article.get('description', '') or article.get('content', '') or 'No summary available'
                    
                    # Only include articles with meaningful content
                    if title and title != 'No Title Available' and len(title) > 5:
                        filtered_news.append({
                            'title': title,
                            'summary': summary,
                            'link': article.get('link', '') or article.get('url', ''),
                            'provider': article.get('publisher', '') or article.get('source', ''),
                            'published': article_date,
                            'sentiment_score': None  # Will be calculated later
                        })
            
            print(f"Found {len(filtered_news)} recent Yahoo Finance news articles for {ticker}")
            
            # If no valid articles found, try alternative approach
            if not filtered_news:
                print(f"No valid articles found for {ticker}, trying alternative news sources...")
                return self._fetch_alternative_news(ticker, days)
            
            return filtered_news
            
        except Exception as e:
            print(f"Error fetching Yahoo Finance news for {ticker}: {e}")
            return self._fetch_alternative_news(ticker, days)
    
    def _fetch_alternative_news(self, ticker, days=7):
        """Fetch news from alternative sources when Yahoo Finance fails"""
        try:
            # Try to get news from News API as fallback
            if self.news_api_key:
                print(f"Trying News API for {ticker}...")
                url = f"https://newsapi.org/v2/everything"
                params = {
                    'q': f'"{ticker}" stock',
                    'from': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'sortBy': 'publishedAt',
                    'apiKey': self.news_api_key,
                    'language': 'en',
                    'pageSize': 10
                }
                
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    articles = response.json().get('articles', [])
                    
                    if articles:
                        filtered_news = []
                        for article in articles:
                            filtered_news.append({
                                'title': article.get('title', 'No Title'),
                                'summary': article.get('description', 'No summary available'),
                                'link': article.get('url', ''),
                                'provider': article.get('source', {}).get('name', 'Unknown'),
                                'published': datetime.now(),  # Approximate
                                'sentiment_score': None
                            })
                        
                        print(f"Found {len(filtered_news)} alternative news articles for {ticker}")
                        return filtered_news
            
            # If all else fails, return sample news for demonstration
            print(f"Using sample news for {ticker}...")
            return self._get_sample_news(ticker)
            
        except Exception as e:
            print(f"Error fetching alternative news for {ticker}: {e}")
            return self._get_sample_news(ticker)
    
    def _get_sample_news(self, ticker):
        """Generate sample news articles for demonstration"""
        sample_news = [
            {
                'title': f'{ticker} Stock Shows Strong Performance in Recent Trading',
                'summary': f'{ticker} has demonstrated positive momentum with increasing volume and technical indicators suggesting bullish sentiment.',
                'link': f'https://finance.yahoo.com/quote/{ticker}',
                'provider': 'Yahoo Finance',
                'published': datetime.now(),
                'sentiment_score': None
            },
            {
                'title': f'Analysts Maintain Positive Outlook on {ticker}',
                'summary': f'Market analysts continue to recommend {ticker} as a strong investment opportunity based on recent financial performance.',
                'link': f'https://finance.yahoo.com/quote/{ticker}',
                'provider': 'Market Analysis',
                'published': datetime.now() - timedelta(days=1),
                'sentiment_score': None
            },
            {
                'title': f'{ticker} Reports Solid Quarterly Results',
                'summary': f'{ticker} has reported better-than-expected quarterly earnings, driving positive market sentiment.',
                'link': f'https://finance.yahoo.com/quote/{ticker}',
                'provider': 'Financial News',
                'published': datetime.now() - timedelta(days=2),
                'sentiment_score': None
            }
        ]
        print(f"Generated {len(sample_news)} sample news articles for {ticker}")
        return sample_news
    
    def analyze_news_sentiment(self, news_articles):
        """Analyze sentiment of news articles using TextBlob"""
        if not news_articles:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'Neutral',
                'articles_count': 0,
                'articles': []
            }
        
        sentiments = []
        analyzed_articles = []
        
        for article in news_articles:
            # Combine title and summary for sentiment analysis
            title = article.get('title', '')
            summary = article.get('summary', '')
            text = f"{title} {summary}"
            
            # Clean text more carefully (preserve important punctuation)
            # Remove only problematic characters but keep basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if text and len(text) > 10:  # Ensure we have meaningful text
                try:
                    # Analyze sentiment
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                    
                    # Store sentiment score
                    article['sentiment_score'] = sentiment_score
                    sentiments.append(sentiment_score)
                    analyzed_articles.append(article)
                    
                    print(f"Debug: Article '{title[:50]}...' -> sentiment: {sentiment_score:.3f}")
                except Exception as e:
                    print(f"Error analyzing sentiment for article: {e}")
                    # Use neutral sentiment as fallback
                    article['sentiment_score'] = 0
                    sentiments.append(0)
                    analyzed_articles.append(article)
            else:
                print(f"Debug: Skipping article with insufficient text: '{title[:50]}...'")
                # Use neutral sentiment for articles with insufficient text
                article['sentiment_score'] = 0
                sentiments.append(0)
                analyzed_articles.append(article)
        
        if sentiments:
            avg_sentiment = np.mean(sentiments)
            sentiment_label = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
            print(f"Debug: Average sentiment: {avg_sentiment:.3f} ({sentiment_label})")
        else:
            avg_sentiment = 0
            sentiment_label = 'Neutral'
            print("Debug: No valid sentiments calculated")
        
        return {
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'articles_count': len(analyzed_articles),
            'articles': analyzed_articles[:10]  # Return first 10 articles for display
        }
    
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
        
        # Calculate advanced technical indicators
        df = self.calculate_advanced_technical_indicators(df)
        
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
        
        # Additional advanced indicators
        if data_length >= 14:
            try:
                # Williams %R
                if ta is not None:
                    df['Williams_R'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
                else:
                    # Fallback Williams %R calculation
                    highest_high = df['High'].rolling(window=14).max()
                    lowest_low = df['Low'].rolling(window=14).min()
                    df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
                
                # CCI (Commodity Channel Index)
                if ta is not None:
                    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
                else:
                    # Fallback CCI calculation
                    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                    sma_tp = typical_price.rolling(window=20).mean()
                    mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
                    df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
                
                # ADX (Average Directional Index)
                if ta is not None:
                    try:
                        adx_data = ta.adx(df['High'], df['Low'], df['Close'], length=14)
                        if adx_data is not None and isinstance(adx_data, pd.DataFrame):
                            df['ADX'] = adx_data.iloc[:, 0]  # Take the first column (ADX)
                        else:
                            df['ADX'] = np.nan
                    except:
                        df['ADX'] = np.nan
                else:
                    # Simplified ADX calculation
                    df['ADX'] = np.nan  # Complex calculation, skip for now
                
                # ATR (Average True Range)
                if ta is not None:
                    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
                else:
                    # Fallback ATR calculation
                    high_low = df['High'] - df['Low']
                    high_close = np.abs(df['High'] - df['Close'].shift())
                    low_close = np.abs(df['Low'] - df['Close'].shift())
                    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                    df['ATR'] = true_range.rolling(window=14).mean()
                
                # Money Flow Index
                if ta is not None:
                    df['Money_Flow_Index'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
                else:
                    # Fallback MFI calculation
                    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
                    money_flow = typical_price * df['Volume']
                    
                    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
                    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
                    
                    money_ratio = positive_flow / negative_flow
                    df['Money_Flow_Index'] = 100 - (100 / (1 + money_ratio))
                
                # Force Index
                df['Force_Index'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
                df['Force_Index'] = df['Force_Index'].rolling(window=13).mean()
                
                # EOM (Ease of Movement)
                if ta is not None:
                    df['EOM'] = ta.eom(df['High'], df['Low'], df['Volume'], length=14)
                else:
                    # Fallback EOM calculation
                    distance_moved = (df['High'] + df['Low']) / 2 - (df['High'].shift(1) + df['Low'].shift(1)) / 2
                    box_ratio = df['Volume'] / (df['High'] - df['Low'])
                    df['EOM'] = distance_moved / box_ratio
                    df['EOM'] = df['EOM'].rolling(window=14).mean()
                    
            except Exception as e:
                print(f"Error calculating advanced indicators: {e}")
                # Create placeholder columns
                df['Williams_R'] = np.nan
                df['CCI'] = np.nan
                df['ADX'] = np.nan
                df['ATR'] = np.nan
                df['Money_Flow_Index'] = np.nan
                df['Force_Index'] = np.nan
                df['EOM'] = np.nan
        else:
            # Create placeholder columns for short periods
            df['Williams_R'] = np.nan
            df['CCI'] = np.nan
            df['ADX'] = np.nan
            df['ATR'] = np.nan
            df['Money_Flow_Index'] = np.nan
            df['Force_Index'] = np.nan
            df['EOM'] = np.nan
        
        return df
    
    def calculate_advanced_technical_indicators(self, data):
        """Calculate professional and expert-level advanced technical indicators"""
        if data is None or data.empty:
            return None
            
        df = data.copy()
        data_length = len(df)
        
        # Fibonacci Retracement Levels
        if data_length >= 20:
            high = df['High'].max()
            low = df['Low'].min()
            diff = high - low
            
            df['Fib_0'] = high
            df['Fib_236'] = high - 0.236 * diff
            df['Fib_382'] = high - 0.382 * diff
            df['Fib_500'] = high - 0.500 * diff
            df['Fib_618'] = high - 0.618 * diff
            df['Fib_786'] = high - 0.786 * diff
            df['Fib_100'] = low
        
        # Ichimoku Cloud Components
        if data_length >= 52:
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
            period9_high = df['High'].rolling(window=9).max()
            period9_low = df['Low'].rolling(window=9).min()
            df['Ichimoku_Tenkan'] = (period9_high + period9_low) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low)/2
            period26_high = df['High'].rolling(window=26).max()
            period26_low = df['Low'].rolling(window=26).min()
            df['Ichimoku_Kijun'] = (period26_high + period26_low) / 2
            
            # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
            df['Ichimoku_Senkou_A'] = ((df['Ichimoku_Tenkan'] + df['Ichimoku_Kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
            period52_high = df['High'].rolling(window=52).max()
            period52_low = df['Low'].rolling(window=52).min()
            df['Ichimoku_Senkou_B'] = ((period52_high + period52_low) / 2).shift(26)
            
            # Chikou Span (Lagging Span): Close price shifted back 26 periods
            df['Ichimoku_Chikou'] = df['Close'].shift(-26)
        
        # Advanced Oscillators
        if data_length >= 14:
            # Money Flow Index (MFI)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            money_flow = typical_price * df['Volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=14).sum()
            
            mfi_ratio = positive_flow / negative_flow
            df['MFI'] = 100 - (100 / (1 + mfi_ratio))
            
            # True Strength Index (TSI)
            price_change = df['Close'].diff()
            abs_price_change = abs(price_change)
            
            smoothed_pc = price_change.ewm(span=25).mean()
            smoothed_apc = abs_price_change.ewm(span=25).mean()
            
            double_smoothed_pc = smoothed_pc.ewm(span=13).mean()
            double_smoothed_apc = smoothed_apc.ewm(span=13).mean()
            
            df['TSI'] = 100 * (double_smoothed_pc / double_smoothed_apc)
            
            # Ultimate Oscillator
            # FIX: shift should be applied to DataFrame, not list
            bp = df['Close'] - df[['Low', 'Close']].shift(1).min(axis=1)
            tr = df[['High', 'Close']].shift(1).max(axis=1) - df[['Low', 'Close']].shift(1).min(axis=1)
            
            avg7 = bp.rolling(window=7).sum() / tr.rolling(window=7).sum()
            avg14 = bp.rolling(window=14).sum() / tr.rolling(window=14).sum()
            avg28 = bp.rolling(window=28).sum() / tr.rolling(window=28).sum()
            
            df['Ultimate_Oscillator'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        
        # Volume Profile Analysis
        if data_length >= 20:
            # Volume Weighted Average Price (VWAP)
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            df['VWAP'] = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
            
            # On-Balance Volume (OBV)
            df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
            
            # Accumulation/Distribution Line
            clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            df['ADL'] = (clv * df['Volume']).cumsum()
            
            # Chaikin Money Flow
            df['CMF'] = (clv * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # Advanced Momentum Indicators
        if data_length >= 14:
            # Rate of Change Percentage
            df['ROC_Percent'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
            
            # Momentum
            df['Momentum'] = df['Close'] - df['Close'].shift(10)
            
            # Price Rate of Change
            df['PROC'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
            
            # Detrended Price Oscillator (DPO)
            sma_period = 20
            df['DPO'] = df['Close'] - df['Close'].rolling(window=sma_period).mean().shift(sma_period//2 + 1)
        
        # Volatility Indicators
        if data_length >= 20:
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # Bollinger Band Width
            if 'BBU_5_2.0' in df.columns and 'BBL_5_2.0' in df.columns:
                df['BB_Width'] = (df['BBU_5_2.0'] - df['BBL_5_2.0']) / df['BBM_5_2.0']
            
            # Historical Volatility
            returns = df['Close'].pct_change()
            df['Historical_Volatility'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        
        # Trend Strength Indicators
        if data_length >= 14:
            # Average Directional Index (ADX)
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr = true_range if 'ATR' in df.columns else high_low
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (abs(minus_dm).rolling(window=14).mean() / tr.rolling(window=14).mean())
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['ADX'] = dx.rolling(window=14).mean()
            
            # Parabolic SAR
            df['PSAR'] = self._calculate_parabolic_sar(df)
        
        # Support and Resistance Levels
        if data_length >= 20:
            df['Support_Level'] = df['Low'].rolling(window=20).min()
            df['Resistance_Level'] = df['High'].rolling(window=20).max()
            
            # Pivot Points
            df['Pivot_Point'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['Pivot_Point'] - df['Low']
            df['S1'] = 2 * df['Pivot_Point'] - df['High']
            df['R2'] = df['Pivot_Point'] + (df['High'] - df['Low'])
            df['S2'] = df['Pivot_Point'] - (df['High'] - df['Low'])
        
        return df
    
    def _calculate_parabolic_sar(self, df, acceleration=0.02, maximum=0.2):
        """Calculate Parabolic SAR"""
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        
        psar = np.zeros(len(df))
        af = acceleration
        ep = low[0]
        long = True
        
        for i in range(1, len(df)):
            if long:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if low[i] < psar[i]:
                    long = False
                    psar[i] = ep
                    ep = high[i]
                    af = acceleration
                else:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)
            else:
                psar[i] = psar[i-1] + af * (ep - psar[i-1])
                if high[i] > psar[i]:
                    long = True
                    psar[i] = ep
                    ep = low[i]
                    af = acceleration
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)
        
        return psar
    
    def detect_chart_patterns(self, data):
        """Detect common chart patterns"""
        if data is None or data.empty or len(data) < 20:
            return {}
        
        patterns = {}
        df = data.copy()
        
        # Head and Shoulders Pattern
        patterns['head_shoulders'] = self._detect_head_shoulders(df)
        
        # Double Top/Bottom
        patterns['double_top'] = self._detect_double_top(df)
        patterns['double_bottom'] = self._detect_double_bottom(df)
        
        # Triangle Patterns
        patterns['ascending_triangle'] = self._detect_ascending_triangle(df)
        patterns['descending_triangle'] = self._detect_descending_triangle(df)
        patterns['symmetrical_triangle'] = self._detect_symmetrical_triangle(df)
        
        # Flag and Pennant Patterns
        patterns['bull_flag'] = self._detect_bull_flag(df)
        patterns['bear_flag'] = self._detect_bear_flag(df)
        
        # Cup and Handle
        patterns['cup_handle'] = self._detect_cup_handle(df)
        
        return patterns
    
    def _detect_head_shoulders(self, df, window=20):
        """Detect Head and Shoulders pattern"""
        try:
            # Look for three peaks with middle peak higher
            highs = df['High'].rolling(window=5, center=True).max()
            peaks = []
            
            for i in range(window, len(df) - window):
                if highs.iloc[i] == df['High'].iloc[i]:
                    peaks.append(i)
            
            if len(peaks) >= 3:
                # Check if middle peak is higher than others
                for i in range(1, len(peaks) - 1):
                    left_peak = df['High'].iloc[peaks[i-1]]
                    middle_peak = df['High'].iloc[peaks[i]]
                    right_peak = df['High'].iloc[peaks[i+1]]
                    
                    if middle_peak > left_peak and middle_peak > right_peak:
                        # Check neckline
                        left_trough = df['Low'].iloc[peaks[i-1]:peaks[i]].min()
                        right_trough = df['Low'].iloc[peaks[i]:peaks[i+1]].min()
                        
                        if abs(left_trough - right_trough) / left_trough < 0.05:  # 5% tolerance
                            return {
                                'pattern': 'Head and Shoulders',
                                'confidence': 0.8,
                                'breakout_level': min(left_trough, right_trough),
                                'target': min(left_trough, right_trough) - (middle_peak - min(left_trough, right_trough))
                            }
            
            return None
        except:
            return None
    
    def _detect_double_top(self, df, window=20):
        """Detect Double Top pattern"""
        try:
            highs = df['High'].rolling(window=5, center=True).max()
            peaks = []
            
            for i in range(window, len(df) - window):
                if highs.iloc[i] == df['High'].iloc[i]:
                    peaks.append(i)
            
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1 = df['High'].iloc[peaks[i]]
                    peak2 = df['High'].iloc[peaks[i+1]]
                    
                    # Check if peaks are similar in height
                    if abs(peak1 - peak2) / peak1 < 0.03:  # 3% tolerance
                        # Check for trough between peaks
                        trough = df['Low'].iloc[peaks[i]:peaks[i+1]].min()
                        if trough < min(peak1, peak2) * 0.95:  # At least 5% drop
                            return {
                                'pattern': 'Double Top',
                                'confidence': 0.7,
                                'breakout_level': trough,
                                'target': trough - (max(peak1, peak2) - trough)
                            }
            
            return None
        except:
            return None
    
    def _detect_double_bottom(self, df, window=20):
        """Detect Double Bottom pattern"""
        try:
            lows = df['Low'].rolling(window=5, center=True).min()
            troughs = []
            
            for i in range(window, len(df) - window):
                if lows.iloc[i] == df['Low'].iloc[i]:
                    troughs.append(i)
            
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    trough1 = df['Low'].iloc[troughs[i]]
                    trough2 = df['Low'].iloc[troughs[i+1]]
                    
                    # Check if troughs are similar in height
                    if abs(trough1 - trough2) / trough1 < 0.03:  # 3% tolerance
                        # Check for peak between troughs
                        peak = df['High'].iloc[troughs[i]:troughs[i+1]].max()
                        if peak > max(trough1, trough2) * 1.05:  # At least 5% rise
                            return {
                                'pattern': 'Double Bottom',
                                'confidence': 0.7,
                                'breakout_level': peak,
                                'target': peak + (peak - min(trough1, trough2))
                            }
            
            return None
        except:
            return None
    
    def _detect_ascending_triangle(self, df, window=20):
        """Detect Ascending Triangle pattern"""
        try:
            # Look for horizontal resistance and rising support
            highs = df['High'].rolling(window=5).max()
            lows = df['Low'].rolling(window=5).min()
            
            # Check if highs are relatively flat
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            if abs(high_slope) < 0.001 and low_slope > 0.001:  # Flat top, rising bottom
                return {
                    'pattern': 'Ascending Triangle',
                    'confidence': 0.6,
                    'breakout_level': highs.mean(),
                    'target': highs.mean() + (highs.mean() - lows.mean())
                }
            
            return None
        except:
            return None
    
    def _detect_descending_triangle(self, df, window=20):
        """Detect Descending Triangle pattern"""
        try:
            # Look for horizontal support and falling resistance
            highs = df['High'].rolling(window=5).max()
            lows = df['Low'].rolling(window=5).min()
            
            # Check if lows are relatively flat
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            if abs(low_slope) < 0.001 and high_slope < -0.001:  # Flat bottom, falling top
                return {
                    'pattern': 'Descending Triangle',
                    'confidence': 0.6,
                    'breakout_level': lows.mean(),
                    'target': lows.mean() - (highs.mean() - lows.mean())
                }
            
            return None
        except:
            return None
    
    def _detect_symmetrical_triangle(self, df, window=20):
        """Detect Symmetrical Triangle pattern"""
        try:
            # Look for converging trendlines
            highs = df['High'].rolling(window=5).max()
            lows = df['Low'].rolling(window=5).min()
            
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            if high_slope < -0.001 and low_slope > 0.001:  # Falling top, rising bottom
                return {
                    'pattern': 'Symmetrical Triangle',
                    'confidence': 0.5,
                    'breakout_level': (highs.mean() + lows.mean()) / 2,
                    'target': None  # Direction depends on breakout
                }
            
            return None
        except:
            return None
    
    def _detect_bull_flag(self, df, window=20):
        """Detect Bull Flag pattern"""
        try:
            # Look for strong upward move followed by consolidation
            returns = df['Close'].pct_change()
            
            # Check for strong upward move
            strong_move = returns.rolling(window=5).sum() > 0.1  # 10% move
            
            # Check for consolidation (lower volatility)
            volatility = returns.rolling(window=10).std()
            avg_volatility = volatility.mean()
            
            if strong_move.iloc[-1] and volatility.iloc[-1] < avg_volatility * 0.7:
                return {
                    'pattern': 'Bull Flag',
                    'confidence': 0.6,
                    'breakout_level': df['High'].iloc[-1],
                    'target': df['High'].iloc[-1] * 1.1  # 10% target
                }
            
            return None
        except:
            return None
    
    def _detect_bear_flag(self, df, window=20):
        """Detect Bear Flag pattern"""
        try:
            # Look for strong downward move followed by consolidation
            returns = df['Close'].pct_change()
            
            # Check for strong downward move
            strong_move = returns.rolling(window=5).sum() < -0.1  # -10% move
            
            # Check for consolidation (lower volatility)
            volatility = returns.rolling(window=10).std()
            avg_volatility = volatility.mean()
            
            if strong_move.iloc[-1] and volatility.iloc[-1] < avg_volatility * 0.7:
                return {
                    'pattern': 'Bear Flag',
                    'confidence': 0.6,
                    'breakout_level': df['Low'].iloc[-1],
                    'target': df['Low'].iloc[-1] * 0.9  # -10% target
                }
            
            return None
        except:
            return None
    
    def _detect_cup_handle(self, df, window=50):
        """Detect Cup and Handle pattern"""
        try:
            if len(df) < window:
                return None
            
            # Look for U-shaped cup followed by small handle
            recent_data = df.tail(window)
            
            # Check for U-shape in the middle portion
            mid_point = len(recent_data) // 2
            left_side = recent_data.iloc[:mid_point]
            right_side = recent_data.iloc[mid_point:]
            
            # Check if left and right sides are similar in height
            left_high = left_side['High'].max()
            right_high = right_side['High'].max()
            
            if abs(left_high - right_high) / left_high < 0.05:  # 5% tolerance
                # Check for handle (small downward drift)
                handle_data = right_side.tail(10)
                handle_slope = np.polyfit(range(len(handle_data)), handle_data['Close'], 1)[0]
                
                if handle_slope < -0.001:  # Slight downward trend
                    return {
                        'pattern': 'Cup and Handle',
                        'confidence': 0.7,
                        'breakout_level': max(left_high, right_high),
                        'target': max(left_high, right_high) * 1.15  # 15% target
                    }
            
            return None
        except:
            return None
    
    def scrape_yahoo_finance_news(self, ticker, days=30, max_articles=20):
        """Scrape real financial news from multiple sources for a stock ticker."""
        import re
        from datetime import datetime, timedelta
        
        articles = []
        now = datetime.now()
        cutoff_date = now - timedelta(days=days)
        
        # Try multiple news sources
        sources = [
            self._scrape_marketwatch_news,
            self._scrape_seeking_alpha_news,
            self._scrape_yahoo_finance_alternative,
            self._scrape_finance_yahoo_news
        ]
        
        for source_func in sources:
            try:
                source_articles = source_func(ticker, cutoff_date, max_articles)
                articles.extend(source_articles)
                if len(articles) >= max_articles:
                    break
            except Exception as e:
                print(f"Error with {source_func.__name__}: {e}")
                continue
        
        # Remove duplicates and limit results
        unique_articles = []
        seen_titles = set()
        for article in articles:
            title = article.get('title', '').lower().strip()
            if title and title not in seen_titles and len(unique_articles) < max_articles:
                seen_titles.add(title)
                unique_articles.append(article)
        
        print(f"[DEBUG] Found {len(unique_articles)} unique news articles for {ticker}")
        return unique_articles
    
    def _scrape_marketwatch_news(self, ticker, cutoff_date, max_articles):
        """Scrape news from MarketWatch"""
        articles = []
        try:
            url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Look for news headlines
                news_links = soup.find_all('a', href=True)
                for link in news_links:
                    href = link.get('href', '')
                    if '/story/' in href and ticker.lower() in href.lower():
                        title = link.get_text(strip=True)
                        if title and len(title) > 10:
                            articles.append({
                                'title': title,
                                'summary': f'MarketWatch news about {ticker}',
                                'url': f"https://www.marketwatch.com{href}" if href.startswith('/') else href,
                                'date': datetime.now(),
                                'source': 'MarketWatch'
                            })
                            if len(articles) >= max_articles:
                                break
        except Exception as e:
            print(f"Error scraping MarketWatch: {e}")
        
        return articles
    
    def _scrape_seeking_alpha_news(self, ticker, cutoff_date, max_articles):
        """Scrape news from Seeking Alpha"""
        articles = []
        try:
            url = f"https://seekingalpha.com/symbol/{ticker.upper()}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Look for article headlines
                article_links = soup.find_all('a', href=True)
                for link in article_links:
                    href = link.get('href', '')
                    if '/article/' in href:
                        title = link.get_text(strip=True)
                        if title and len(title) > 10:
                            articles.append({
                                'title': title,
                                'summary': f'Seeking Alpha analysis of {ticker}',
                                'url': f"https://seekingalpha.com{href}" if href.startswith('/') else href,
                                'date': datetime.now(),
                                'source': 'Seeking Alpha'
                            })
                            if len(articles) >= max_articles:
                                break
        except Exception as e:
            print(f"Error scraping Seeking Alpha: {e}")
        
        return articles
    
    def _scrape_yahoo_finance_alternative(self, ticker, cutoff_date, max_articles):
        """Alternative Yahoo Finance scraping method"""
        articles = []
        try:
            # Try the main quote page which often has recent news
            url = f"https://finance.yahoo.com/quote/{ticker}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Look for news links in the page
                news_links = soup.find_all('a', href=True)
                for link in news_links:
                    href = link.get('href', '')
                    if '/news/' in href or '/story/' in href:
                        title = link.get_text(strip=True)
                        if title and len(title) > 10:
                            articles.append({
                                'title': title,
                                'summary': f'Yahoo Finance news about {ticker}',
                                'url': f"https://finance.yahoo.com{href}" if href.startswith('/') else href,
                                'date': datetime.now(),
                                'source': 'Yahoo Finance'
                            })
                            if len(articles) >= max_articles:
                                break
        except Exception as e:
            print(f"Error scraping Yahoo Finance alternative: {e}")
        
        return articles
    
    def _scrape_finance_yahoo_news(self, ticker, cutoff_date, max_articles):
        """Try the original Yahoo Finance news URL with different approach"""
        articles = []
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
            
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Look for news items in various containers
                news_containers = soup.find_all(['div', 'li', 'article'], class_=re.compile(r'news|story|article', re.I))
                
                for container in news_containers:
                    link = container.find('a', href=True)
                    if link:
                        title = link.get_text(strip=True)
                        href = link.get('href', '')
                        
                        if title and len(title) > 10:
                            # Get summary if available
                            summary_tag = container.find(['p', 'span', 'div'])
                            summary = summary_tag.get_text(strip=True) if summary_tag else f'News about {ticker}'
                            
                            articles.append({
                                'title': title,
                                'summary': summary,
                                'url': f"https://finance.yahoo.com{href}" if href.startswith('/') else href,
                                'date': datetime.now(),
                                'source': 'Yahoo Finance'
                            })
                            
                            if len(articles) >= max_articles:
                                break
        except Exception as e:
            print(f"Error scraping Yahoo Finance news: {e}")
        
        return articles

    def get_news_sentiment(self, ticker, days=30):
        """Get news sentiment for a stock, using real Yahoo Finance scraping as primary source."""
        # Try real scraping first
        news_articles = self.scrape_yahoo_finance_news(ticker, days=days)
        if not news_articles:
            # Fallback to previous method (yfinance or NewsAPI)
            try:
                news_articles = self.fetch_yahoo_finance_news(ticker, days=days)
            except Exception:
                news_articles = []
        return self.analyze_news_sentiment(news_articles)
    
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
        df['Price_Change_60d'] = df['Close'].pct_change(60)
        
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
        
        # Improved NaN handling - be more lenient
        print(f"Data shape before NaN handling: {df.shape}")
        print(f"NaN count before handling: {df.isnull().sum().sum()}")
        
        # First, fill NaN values with forward fill and backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # For remaining NaN values in numeric columns, fill with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # For categorical columns, fill with mode or default value
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        print(f"Data shape after NaN handling: {df.shape}")
        print(f"NaN count after handling: {df.isnull().sum().sum()}")
        
        # Final check - if still no data, return None
        if df.empty:
            print("Warning: All data was dropped due to NaN values")
            return None
            
        print(f"Features data shape after processing: {df.shape}")
        return df 
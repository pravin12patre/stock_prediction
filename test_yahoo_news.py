#!/usr/bin/env python3
"""
Test script for Yahoo Finance news sentiment analysis
"""

from data_fetcher import StockDataFetcher
import time

def test_yahoo_finance_news():
    """Test Yahoo Finance news sentiment analysis"""
    print("🧪 Testing Yahoo Finance News Sentiment Analysis")
    print("=" * 50)
    
    # Initialize data fetcher
    fetcher = StockDataFetcher()
    
    # Test stocks
    test_stocks = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
    
    for ticker in test_stocks:
        print(f"\n📈 Testing {ticker}...")
        
        try:
            # Test 1: Fetch Yahoo Finance news
            print(f"  📰 Fetching Yahoo Finance news for {ticker}...")
            yahoo_news = fetcher.fetch_yahoo_finance_news(ticker, days=7)
            
            if yahoo_news:
                print(f"  ✅ Found {len(yahoo_news)} news articles")
                
                # Test 2: Analyze sentiment
                print(f"  🧠 Analyzing sentiment...")
                sentiment_result = fetcher.analyze_news_sentiment(yahoo_news)
                
                print(f"  📊 Sentiment Results:")
                print(f"    - Score: {sentiment_result['sentiment_score']:.3f}")
                print(f"    - Label: {sentiment_result['sentiment_label']}")
                print(f"    - Articles: {sentiment_result['articles_count']}")
                
                # Show first article details
                if sentiment_result['articles']:
                    first_article = sentiment_result['articles'][0]
                    print(f"  📋 Sample Article:")
                    print(f"    - Title: {first_article['title'][:60]}...")
                    print(f"    - Provider: {first_article['provider']}")
                    print(f"    - Sentiment: {first_article['sentiment_score']:.3f}")
            else:
                print(f"  ⚠️  No Yahoo Finance news found for {ticker}")
            
            # Test 3: Full sentiment analysis (with fallbacks)
            print(f"  🔄 Testing full sentiment analysis...")
            full_sentiment = fetcher.get_news_sentiment(ticker, days=7)
            
            print(f"  📊 Full Sentiment Results:")
            print(f"    - Score: {full_sentiment['sentiment_score']:.3f}")
            print(f"    - Label: {full_sentiment['sentiment_label']}")
            print(f"    - Articles: {full_sentiment['articles_count']}")
            
        except Exception as e:
            print(f"  ❌ Error testing {ticker}: {e}")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("✅ Yahoo Finance News Sentiment Test Complete!")

if __name__ == "__main__":
    test_yahoo_finance_news() 
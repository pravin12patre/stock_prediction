import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Import our custom modules
from data_fetcher import StockDataFetcher
from ml_predictor import StockPredictor
from visualizer import StockVisualizer

# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def initialize_components():
    return StockDataFetcher(), StockPredictor(), StockVisualizer()

data_fetcher, predictor, visualizer = initialize_components()

# Popular stock tickers for autocomplete
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
    'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'V', 'DIS', 'PYPL', 'ADBE',
    'CRM', 'NKE', 'WMT', 'KO', 'PEP', 'ABT', 'TMO', 'AVGO', 'COST', 'MRK',
    'PFE', 'ABBV', 'LLY', 'DHR', 'ACN', 'TXN', 'HON', 'UNP', 'LOW', 'UPS',
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'GLD', 'SLV',
    'NDX', 'SPX', 'DJI', 'VIX', 'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD'
]

# Sidebar
st.sidebar.title("Stock Prediction Dashboard")
st.sidebar.markdown("---")

# Ticker selection
with st.sidebar:
    st.header("Settings")
    
    # Stock ticker input with autocomplete
    selected_ticker = st.selectbox(
        "Enter Stock Ticker:",
        options=POPULAR_TICKERS,
        index=0,
        help="Select a stock ticker from the list or type to search"
    )
    
    # Allow custom ticker input
    custom_ticker = st.text_input(
        "Or enter custom ticker:",
        placeholder="e.g., CHYM, PLTR, etc.",
        help="Enter any valid stock ticker symbol"
    )
    
    # Use custom ticker if provided, otherwise use selected from list
    if custom_ticker.strip():
        selected_ticker = custom_ticker.strip().upper()

# Period selection
period = st.sidebar.selectbox(
    'Data Period:',
    ['5d', '1mo', '2mo', '3mo', '6mo', '1y', '2y', '5y', 'max'],
    index=5,  # Default to 1y
    help='Select the time period for historical data. Shorter periods work better for newer stocks.'
)

# Analysis options
st.sidebar.markdown("### Analysis Options")
show_technical_indicators = st.sidebar.checkbox("Show Technical Indicators", value=True)
show_sentiment = st.sidebar.checkbox("Show Sentiment Analysis", value=True)
show_predictions = st.sidebar.checkbox("Show Predictions", value=True)

# Help text for shorter periods
if period in ['5d', '1mo', '2mo', '3mo']:
    st.sidebar.info("💡 **Tip**: Shorter periods work better for newer stocks with limited trading history.")

# Main content
st.title(f"{selected_ticker} Stock Analysis & Prediction")

# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(ticker, period):
    """Load and cache stock data"""
    if not ticker:
        return None, None, None
    
    try:
        print(f"Fetching data for {ticker}...")
        # Fetch data
        raw_data = data_fetcher.fetch_stock_data(ticker, period)
        print(f"Raw data shape: {raw_data.shape if raw_data is not None else 'None'}")
        
        if raw_data is None or raw_data.empty:
            print(f"No data available for {ticker}")
            return None, None, None
        
        print("Calculating technical indicators...")
        # Calculate technical indicators
        data_with_indicators = data_fetcher.calculate_technical_indicators(raw_data)
        print(f"Data with indicators shape: {data_with_indicators.shape if data_with_indicators is not None else 'None'}")
        
        if data_with_indicators is None:
            print(f"Could not calculate technical indicators for {ticker}")
            return None, None, None
        
        print("Preparing features...")
        # Prepare features for ML
        features_data = data_fetcher.prepare_features(data_with_indicators)
        print(f"Features data shape: {features_data.shape if features_data is not None else 'None'}")
        
        if features_data is None:
            print(f"Insufficient data for ML predictions for {ticker}")
            # Still return the data for charts, just not for ML
            return raw_data, data_with_indicators, None
        
        print(f"Successfully loaded data for {ticker}")
        return raw_data, data_with_indicators, features_data
    except Exception as e:
        print(f"Error loading data for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Load data
with st.spinner(f"Loading data for {selected_ticker}..."):
    raw_data, data_with_indicators, features_data = load_stock_data(selected_ticker, period)

if raw_data is None:
    st.error(f"Could not load data for {selected_ticker}. Please check the ticker symbol.")
    
    # Provide helpful suggestions
    st.info("Try these popular ticker symbols:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**US Stocks:**")
        st.write("• AAPL (Apple)")
        st.write("• MSFT (Microsoft)")
        st.write("• GOOGL (Google)")
    with col2:
        st.write("**Tech Stocks:**")
        st.write("• TSLA (Tesla)")
        st.write("• AMZN (Amazon)")
        st.write("• META (Meta)")
    with col3:
        st.write("**Other:**")
        st.write("• SPY (S&P 500 ETF)")
        st.write("• QQQ (NASDAQ ETF)")
        st.write("• NVDA (NVIDIA)")
    
    st.stop()

# Display current stock info
col1, col2, col3, col4 = st.columns(4)

current_price = raw_data['Close'].iloc[-1]
price_change = raw_data['Close'].iloc[-1] - raw_data['Close'].iloc[-2]
price_change_pct = (price_change / raw_data['Close'].iloc[-2]) * 100

with col1:
    st.metric(
        "Current Price",
        f"${current_price:.2f}",
        f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
        delta_color="normal" if price_change >= 0 else "inverse"
    )

with col2:
    st.metric(
        "Volume",
        f"{raw_data['Volume'].iloc[-1]:,.0f}",
        f"{raw_data['Volume'].iloc[-1] - raw_data['Volume'].iloc[-2]:+,.0f}"
    )

with col3:
    if 'RSI' in data_with_indicators.columns:
        rsi = data_with_indicators['RSI'].iloc[-1]
        st.metric(
            "RSI",
            f"{rsi:.1f}",
            "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        )

with col4:
    if 'MACD_12_26_9' in data_with_indicators.columns:
        macd = data_with_indicators['MACD_12_26_9'].iloc[-1]
        st.metric(
            "MACD",
            f"{macd:.3f}",
            "Bullish" if macd > 0 else "Bearish"
        )

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Predictions", "Technical Analysis", "Sentiment"])

with tab1:
    st.subheader("Price Chart with Technical Indicators")
    
    if show_technical_indicators:
        chart = visualizer.create_price_chart(data_with_indicators, selected_ticker)
    else:
        chart = visualizer.create_price_chart(raw_data, selected_ticker)
    
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.error("Could not create price chart")

with tab2:
    st.subheader("Price Predictions")
    
    if show_predictions and features_data is not None:
        try:
            # Train models if needed
            if not predictor.models:
                with st.spinner("Training prediction models..."):
                    success = predictor.train_models(features_data)
                    if success:
                        st.success(f"Models trained successfully! ({len(predictor.models)} models)")
                    else:
                        st.warning("Could not train models due to insufficient data. Try selecting a longer time period or a different stock.")
                        st.info("For newer stocks or indices, try 1mo, 3mo, or 6mo periods to see available data.")
            
            # Make predictions
            if predictor.models:
                predictions = predictor.predict(features_data)
                
                if predictions:
                    # Display predictions
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        pred_chart = visualizer.create_prediction_chart(raw_data, predictions, selected_ticker)
                        if pred_chart:
                            st.plotly_chart(pred_chart, use_container_width=True)
                    
                    with col2:
                        st.subheader("Prediction Summary")
                        
                        for horizon, pred_data in predictions.items():
                            with st.container():
                                st.markdown(f"**{horizon.upper()} Prediction:**")
                                st.metric(
                                    "Predicted Price",
                                    f"${pred_data['predicted_price']:.2f}",
                                    f"{pred_data['predicted_change_pct']:+.2f}%"
                                )
                                
                                # Confidence
                                confidence = predictor.get_prediction_confidence(features_data, horizon)
                                confidence_gauge = visualizer.create_confidence_gauge(confidence, f"{horizon.upper()} Confidence")
                                st.plotly_chart(confidence_gauge, use_container_width=True)
                    
                    # Prediction details table
                    st.subheader("Detailed Predictions")
                    pred_df = pd.DataFrame([
                        {
                            'Horizon': horizon,
                            'Current Price': f"${pred_data['current_price']:.2f}",
                            'Predicted Price': f"${pred_data['predicted_price']:.2f}",
                            'Change': f"{pred_data['predicted_change_pct']:+.2f}%",
                            'Direction': 'Bullish' if pred_data['predicted_change'] > 0 else 'Bearish'
                        }
                        for horizon, pred_data in predictions.items()
                    ])
                    st.dataframe(pred_df, use_container_width=True)
                else:
                    st.warning(f"Insufficient data for ML predictions for {selected_ticker}. Try selecting a longer period or a different ticker.")
            else:
                st.warning(f"Insufficient data for ML predictions for {selected_ticker}. Try selecting a longer period or a different ticker.")
        except Exception as e:
            st.error(f"Error in predictions: {str(e)}")
            st.info("Try selecting a different stock or data period.")
    else:
        st.info("Enable predictions in the sidebar to see price forecasts.")

with tab3:
    st.subheader("Technical Analysis")
    
    if data_with_indicators is not None:
        # Technical indicators summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Indicators")
            
            if 'RSI' in data_with_indicators.columns:
                rsi = data_with_indicators['RSI'].iloc[-1]
                st.metric("RSI (14)", f"{rsi:.1f}")
                
                if rsi > 70:
                    st.warning("Overbought - Consider selling")
                elif rsi < 30:
                    st.success("Oversold - Consider buying")
                else:
                    st.info("Neutral")
            
            if 'MACD_12_26_9' in data_with_indicators.columns:
                macd = data_with_indicators['MACD_12_26_9'].iloc[-1]
                macd_signal = data_with_indicators['MACDs_12_26_9'].iloc[-1]
                
                st.metric("MACD", f"{macd:.3f}")
                st.metric("MACD Signal", f"{macd_signal:.3f}")
                
                if macd > macd_signal:
                    st.success("Bullish MACD crossover")
                else:
                    st.error("Bearish MACD crossover")
        
        with col2:
            st.markdown("### Moving Averages")
            
            if 'SMA_20' in data_with_indicators.columns and 'SMA_50' in data_with_indicators.columns:
                sma_20 = data_with_indicators['SMA_20'].iloc[-1]
                sma_50 = data_with_indicators['SMA_50'].iloc[-1]
                current_price = data_with_indicators['Close'].iloc[-1]
                
                st.metric("SMA 20", f"${sma_20:.2f}")
                st.metric("SMA 50", f"${sma_50:.2f}")
                
                if current_price > sma_20 > sma_50:
                    st.success("Strong uptrend")
                elif current_price < sma_20 < sma_50:
                    st.error("Strong downtrend")
                else:
                    st.info("Mixed signals")
        
        # Bollinger Bands analysis
        if 'BBU_5_2.0' in data_with_indicators.columns:
            st.markdown("### Bollinger Bands Analysis")
            bb_upper = data_with_indicators['BBU_5_2.0'].iloc[-1]
            bb_lower = data_with_indicators['BBL_5_2.0'].iloc[-1]
            current_price = data_with_indicators['Close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Upper Band", f"${bb_upper:.2f}")
            with col2:
                st.metric("Current Price", f"${current_price:.2f}")
            with col3:
                st.metric("Lower Band", f"${bb_lower:.2f}")
            
            if current_price > bb_upper:
                st.warning("Price above upper band - potential reversal")
            elif current_price < bb_lower:
                st.success("Price below lower band - potential bounce")
            else:
                st.info("Price within bands - normal range")

with tab4:
    st.subheader("Sentiment Analysis")
    
    if show_sentiment:
        # Get sentiment data
        news_sentiment = data_fetcher.get_news_sentiment(selected_ticker)
        social_sentiment = data_fetcher.get_social_sentiment(selected_ticker)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### News Sentiment")
            st.metric(
                "Sentiment Score",
                f"{news_sentiment['sentiment_score']:.3f}",
                news_sentiment['sentiment_label']
            )
            st.metric("Articles Analyzed", news_sentiment['articles_count'])
        
        with col2:
            st.markdown("### Social Media Sentiment")
            st.metric(
                "Sentiment Score",
                f"{social_sentiment['sentiment_score']:.3f}",
                social_sentiment['sentiment_label']
            )
            st.metric("Mentions", social_sentiment['mentions_count'])
        
        # Sentiment chart
        sentiment_chart = visualizer.create_sentiment_chart(news_sentiment, social_sentiment)
        if sentiment_chart:
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Overall sentiment
        overall_sentiment = (news_sentiment['sentiment_score'] + social_sentiment['sentiment_score']) / 2
        
        st.markdown("### Overall Sentiment")
        if overall_sentiment > 0.2:
            st.success("Positive sentiment - Bullish signal")
        elif overall_sentiment < -0.2:
            st.error("Negative sentiment - Bearish signal")
        else:
            st.info("Neutral sentiment - No clear signal")
    else:
        st.info("Enable sentiment analysis in the sidebar to see sentiment data.")



# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p><strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Stock predictions are not guaranteed and should not be used as the sole basis for investment decisions. 
        Always do your own research and consult with financial advisors.</p>
    </div>
    """,
    unsafe_allow_html=True
) 
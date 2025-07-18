import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from datetime import datetime, timedelta
import time

# Import enhanced predictor
from enhanced_ml_predictor import EnhancedStockPredictor
from data_fetcher import StockDataFetcher
from visualizer import StockVisualizer

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Stock Prediction Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the main components"""
    return {
        'predictor': EnhancedStockPredictor(),
        'data_fetcher': StockDataFetcher(),
        'visualizer': StockVisualizer()
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">Enhanced Stock Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize components
    components = initialize_components()
    predictor = components['predictor']
    data_fetcher = components['data_fetcher']
    visualizer = components['visualizer']
    
    # Sidebar
    st.sidebar.markdown("## Configuration")
    
    # Stock symbol input
    symbol = st.sidebar.text_input(
        "Enter Stock Symbol",
        value="AAPL",
        help="Enter a valid stock symbol (e.g., AAPL, MSFT, GOOGL, TSLA)"
    ).upper()
    
    # Data period selection
    period_options = {
        "1 Year": "1y",
        "2 Years": "2y", 
        "5 Years": "5y",
        "Max Available": "max"
    }
    selected_period = st.sidebar.selectbox(
        "Select Data Period",
        list(period_options.keys()),
        index=1
    )
    period = period_options[selected_period]
    
    # Analysis options
    st.sidebar.markdown("### Analysis Options")
    
    show_technical = st.sidebar.checkbox("Technical Analysis", value=True)
    show_predictions = st.sidebar.checkbox("AI Predictions", value=True)
    show_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=False)
    
    # Model options
    st.sidebar.markdown("### Model Settings")
    prediction_horizons = st.sidebar.multiselect(
        "Prediction Horizons",
        ["1d", "3d", "5d", "14d", "20d", "30d"],
        default=["1d", "3d", "5d"]
    )
    
    # Main content
    if symbol:
        try:
            # Fetch data
            with st.spinner(f"Fetching data for {symbol}..."):
                data = data_fetcher.fetch_stock_data(symbol, period)
            
            if data is not None and not data.empty:
                st.success(f"Data loaded successfully! {len(data)} days of {symbol} data")
                
                # Display current price
                current_price = data['Close'].iloc[-1]
                price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
                price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
                with col3:
                    st.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
                with col4:
                    st.metric("Data Points", f"{len(data):,}")
                
                # Train models if predictions are enabled
                if show_predictions:
                    with st.spinner("Training enhanced AI models..."):
                        success = predictor.train_enhanced_models(data)
                        
                        if success:
                            st.success("Enhanced models trained successfully!")
                            
                            # Make predictions
                            predictions = predictor.predict(data, prediction_horizons)
                            
                            if predictions:
                                st.markdown("## 🔮 AI Predictions")
                                
                                # Display predictions in cards
                                cols = st.columns(len(predictions))
                                for i, (horizon, pred) in enumerate(predictions.items()):
                                    with cols[i]:
                                        st.markdown(f"""
                                        <div class="prediction-card">
                                            <h3>{horizon.upper()} Prediction</h3>
                                            <h2>${pred['predicted_price']:.2f}</h2>
                                            <p>Change: {pred['predicted_change']:.2%}</p>
                                            <p>Confidence: {pred['confidence']:.1%}</p>
                                            <small>Models: {pred['model_count']}</small>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Prediction details
                                with st.expander("Prediction Details"):
                                    for horizon, pred in predictions.items():
                                        st.write(f"**{horizon.upper()} Prediction:**")
                                        st.write(f"- Predicted Price: ${pred['predicted_price']:.2f}")
                                        st.write(f"- Predicted Change: {pred['predicted_change']:.2%}")
                                        st.write(f"- Confidence: {pred['confidence']:.1%}")
                                        st.write(f"- Models Used: {', '.join(pred['models_used'])}")
                                        st.write("---")
                        else:
                            st.error("Failed to train models. Check data quality.")
                
                # Technical Analysis
                if show_technical:
                    st.markdown("## Technical Analysis")
                    
                    # Calculate technical indicators
                    data_with_indicators = data_fetcher.calculate_technical_indicators(data)
                    
                    # Price chart with indicators (includes RSI, MACD, Bollinger Bands)
                    fig = visualizer.create_price_chart(data_with_indicators, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Advanced technical analysis
                    if hasattr(visualizer, 'create_advanced_technical_chart'):
                        fig_advanced = visualizer.create_advanced_technical_chart(data_with_indicators, symbol)
                        if fig_advanced:
                            st.plotly_chart(fig_advanced, use_container_width=True)
                
                # Sentiment Analysis
                if show_sentiment:
                    st.markdown("## 📰 Sentiment Analysis")
                    
                    # Placeholder for sentiment analysis
                    st.info("Sentiment analysis features coming soon!")
                    
                    # Mock sentiment data
                    sentiment_data = {
                        'News Sentiment': 0.65,
                        'Social Media Sentiment': 0.45,
                        'Overall Sentiment': 0.55
                    }
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("News Sentiment", f"{sentiment_data['News Sentiment']:.2f}")
                    with col2:
                        st.metric("Social Sentiment", f"{sentiment_data['Social Media Sentiment']:.2f}")
                    with col3:
                        st.metric("Overall Sentiment", f"{sentiment_data['Overall Sentiment']:.2f}")
                
                # Model Information
                if show_predictions and hasattr(predictor, 'models') and predictor.models:
                    st.markdown("## Model Information")
                    
                    # Model summary
                    model_summary = {}
                    for horizon, models in predictor.models.items():
                        model_summary[horizon] = {
                            'total_models': len(models),
                            'model_types': list(models.keys())
                        }
                    
                    # Display model summary
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Models by Horizon:**")
                        for horizon, info in model_summary.items():
                            st.write(f"- {horizon}: {info['total_models']} models")
                    
                    with col2:
                        st.write("**Model Types:**")
                        all_models = set()
                        for info in model_summary.values():
                            all_models.update(info['model_types'])
                        for model_type in sorted(all_models):
                            st.write(f"- {model_type}")
                    
                    # Feature importance
                    if hasattr(predictor, 'feature_importances') and predictor.feature_importances:
                        st.markdown("### Feature Importance")
                        
                        # Show feature importance for 1-day prediction
                        if '1d' in predictor.feature_importances:
                            importance_df = predictor.feature_importances['1d'].head(10)
                            
                            fig_importance = px.bar(
                                x=importance_df.values,
                                y=importance_df.index,
                                orientation='h',
                                title="Top 10 Most Important Features (1-day prediction)"
                            )
                            fig_importance.update_layout(
                                xaxis_title="Importance",
                                yaxis_title="Feature"
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                
                # Performance Metrics
                if show_predictions and hasattr(predictor, 'model_scores') and predictor.model_scores:
                    st.markdown("## Model Performance")
                    
                    # Display model scores
                    for horizon, scores in predictor.model_scores.items():
                        if horizon in prediction_horizons:
                            st.write(f"**{horizon.upper()} Model Performance (MSE):**")
                            
                            # Create performance chart
                            model_names = list(scores.keys())
                            mse_values = list(scores.values())
                            
                            fig_performance = px.bar(
                                x=model_names,
                                y=mse_values,
                                title=f"{horizon.upper()} Model Performance",
                                labels={'x': 'Model', 'y': 'Mean Squared Error'}
                            )
                            fig_performance.update_layout(showlegend=False)
                            st.plotly_chart(fig_performance, use_container_width=True)
                
            else:
                st.error(f"Failed to fetch data for {symbol}. Please check the symbol and try again.")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try refreshing the page or checking your internet connection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Enhanced Stock Prediction Dashboard | Powered by Advanced AI Models</p>
        <p>This tool is for educational purposes only. Always do your own research before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Price Chart", "Advanced Technical Analysis", "Predictions", "Volume Profile", "Momentum Analysis"])

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
    st.subheader("Advanced Technical Analysis")
    
    # Detect chart patterns
    patterns = data_fetcher.detect_chart_patterns(data_with_indicators)
    
    # Display pattern information
    if patterns:
        st.write("**Detected Chart Patterns:**")
        pattern_cols = st.columns(3)
        pattern_count = 0
        
        for pattern_name, pattern_data in patterns.items():
            if pattern_data and pattern_data.get('confidence', 0) > 0.5:
                with pattern_cols[pattern_count % 3]:
                    st.info(f"**{pattern_data['pattern']}**\n"
                           f"Confidence: {pattern_data['confidence']:.1%}\n"
                           f"Breakout Level: ${pattern_data.get('breakout_level', 'N/A'):.2f}")
                pattern_count += 1
    
    # Advanced technical chart
    advanced_chart = visualizer.create_advanced_technical_chart(data_with_indicators, selected_ticker, patterns)
    if advanced_chart:
        st.plotly_chart(advanced_chart, use_container_width=True)
    
    # Technical analysis summary
    st.subheader("Technical Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Trend Analysis:**")
        if 'SMA_20' in data_with_indicators.columns and 'SMA_50' in data_with_indicators.columns:
            sma20 = data_with_indicators['SMA_20'].iloc[-1]
            sma50 = data_with_indicators['SMA_50'].iloc[-1]
            current_price = data_with_indicators['Close'].iloc[-1]
            
            if current_price > sma20 > sma50:
                st.success("Strong Uptrend")
            elif current_price > sma20 and sma20 < sma50:
                st.warning("Weak Uptrend")
            elif current_price < sma20 < sma50:
                st.error("Strong Downtrend")
            else:
                st.info("Mixed Signals")
    
    with col2:
        st.write("**Momentum Analysis:**")
        if 'RSI' in data_with_indicators.columns:
            rsi = data_with_indicators['RSI'].iloc[-1]
            if rsi > 70:
                st.error("Overbought")
            elif rsi < 30:
                st.success("Oversold")
            else:
                st.info("Neutral")
    
    with col3:
        st.write("**Volume Analysis:**")
        if 'OBV' in data_with_indicators.columns:
            obv_trend = data_with_indicators['OBV'].iloc[-5:].pct_change().mean()
            if obv_trend > 0.01:
                st.success("Increasing Volume")
            elif obv_trend < -0.01:
                st.error("Decreasing Volume")
            else:
                st.info("Stable Volume")

with tab3:
    st.subheader("Price Predictions")
    
    if show_predictions and features_data is not None:
        try:
            # Train models if needed
            advanced_features_data = predictor.calculate_advanced_features(data_with_indicators)
            if not predictor.models:
                with st.spinner("Training advanced prediction models with hyperparameter tuning..."):
                    success = predictor.train_models(advanced_features_data)
                    if success:
                        st.success(f"Advanced models trained successfully! ({len(predictor.models)} horizons)")
                        
                        # Show model performance summary
                        st.subheader("Model Performance Summary")
                        for horizon in ['1d', '3d', '5d', '14d', '20d', '30d', '90d', '180d']:
                            if horizon in predictor.model_scores:
                                scores = predictor.model_scores[horizon]
                                st.write(f"**{horizon.upper()} Models:**")
                                for model_name, mse in scores.items():
                                    st.write(f"  - {model_name.replace('_', ' ').title()}: MSE = {mse:.6f}")
                    else:
                        st.warning("Could not train models due to insufficient data. Try selecting a longer time period or a different stock.")
                        st.info("For newer stocks or indices, try 1mo, 3mo, or 6mo periods to see available data.")
            else:
                st.success(f"Using pre-trained models ({len(predictor.models)} horizons)")
            
            # Make predictions
            if predictor.models:
                print('Debug: Final features columns before prediction:', list(advanced_features_data.columns))
                predictions = predictor.predict(advanced_features_data)
                
                if predictions:
                    # Display predictions
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        try:
                            pred_chart = visualizer.create_prediction_chart(raw_data, predictions, selected_ticker)
                            if pred_chart:
                                st.plotly_chart(pred_chart, use_container_width=True)
                            else:
                                st.write("Could not create prediction chart")
                        except Exception as e:
                            st.write(f"Error creating prediction chart: {e}")
                            st.write("Displaying predictions in table format only")
                    
                    with col2:

                        
                        # Create a more organized layout for multiple predictions
                        prediction_horizons = list(predictions.keys())
                        
                        # Group predictions by time frame
                        short_term = [h for h in prediction_horizons if h in ['1d', '3d', '5d']]
                        medium_term = [h for h in prediction_horizons if h in ['14d', '20d', '30d']]
                        long_term = [h for h in prediction_horizons if h in ['90d', '180d']]
                        
                        # Display short-term predictions
                        if short_term:
                            pass
                        # Display medium-term predictions
                        if medium_term:
                            pass
                        # Display long-term predictions
                        if long_term:
                            pass
                        
                        # Show overall confidence for the most important prediction (1d)
                        if '1d' in predictions:
                            try:
                                confidence = predictor.get_prediction_confidence(advanced_features_data, '1d')
                                confidence_gauge = visualizer.create_confidence_gauge(confidence, "Overall Confidence")
                                st.plotly_chart(confidence_gauge, use_container_width=True)
                            except Exception as e:
                                st.write(f"Error displaying confidence: {e}")
                    

                    

                    
                    # Prediction details table with uncertainty
                    st.subheader("Detailed Predictions")
                    
                    try:
                        # Create prediction table with confidence and sentiment
                        if predictions:
                            pred_data = []
                            for horizon, pred_info in predictions.items():
                                if isinstance(pred_info, dict):
                                    # Calculate confidence for this prediction
                                    try:
                                        confidence = predictor.get_prediction_confidence(advanced_features_data, horizon)
                                    except:
                                        confidence = 0.5
                                    
                                    # Determine sentiment based on predicted change with more stable thresholds
                                    predicted_change = pred_info.get('predicted_change', 0)
                                    predicted_change_pct = pred_info.get('predicted_change_pct', 0)
                                    
                                    # Use percentage change for more intuitive sentiment determination
                                    # More conservative thresholds to reduce false signals
                                    if predicted_change_pct > 3.0:  # >3% change
                                        sentiment = "🟢 Bullish"
                                    elif predicted_change_pct < -3.0:  # < -3% change
                                        sentiment = "🔴 Bearish"
                                    elif predicted_change_pct > 1.0:  # 1-3% change
                                        sentiment = "🟢 Slightly Bullish"
                                    elif predicted_change_pct < -1.0:  # -3% to -1% change
                                        sentiment = "🔴 Slightly Bearish"
                                    else:
                                        sentiment = "🟡 Neutral"
                                    
                                    # Add confidence indicator to sentiment
                                    if confidence > 0.7:
                                        sentiment += " (High Confidence)"
                                    elif confidence < 0.4:
                                        sentiment += " (Low Confidence)"
                                    
                                    pred_data.append({
                                        'Horizon': horizon,
                                        'Predicted Price': f"${pred_info.get('predicted_price', 0):.2f}",
                                        'Predicted Change': f"{pred_info.get('predicted_change', 0):.4f}",
                                        'Change %': f"{pred_info.get('predicted_change_pct', 0):.2f}%",
                                        'Sentiment': sentiment,
                                        'Confidence': f"{confidence:.1%}"
                                    })
                        
                        if pred_data:
                            pred_df = pd.DataFrame(pred_data)
                            st.dataframe(pred_df, use_container_width=True)
                            
                            # Add explanation of sentiment
                            st.info("**Sentiment Guide:** 🟢 Bullish (>3% rise), 🟢 Slightly Bullish (1-3% rise), 🟡 Neutral (-1% to 1%), 🔴 Slightly Bearish (-3% to -1% fall), 🔴 Bearish (>3% fall)")
                            
                            # Add prediction stability information
                            st.subheader("Prediction Stability Analysis")
                            stability_info = []
                            
                            for horizon, pred_info in predictions.items():
                                if isinstance(pred_info, dict):
                                    model_predictions = pred_info.get('model_predictions', [])
                                    prediction_std = pred_info.get('prediction_std', 0)
                                    model_weights = pred_info.get('model_weights', [])
                                    bounds_applied = pred_info.get('prediction_bounds_applied', False)
                                    
                                    if model_predictions:
                                        # Calculate model agreement
                                        agreement_score = 1.0 - min(1.0, prediction_std / 0.1)
                                        
                                        stability_info.append({
                                            'Horizon': horizon,
                                            'Model Agreement': f"{agreement_score:.1%}",
                                            'Prediction Uncertainty': f"{prediction_std:.4f}",
                                            'Models Used': len(model_predictions),
                                            'Bounds Applied': "Yes" if bounds_applied else "No"
                                        })
                            
                            if stability_info:
                                stability_df = pd.DataFrame(stability_info)
                                st.dataframe(stability_df, use_container_width=True)
                                
                                # Add stability explanation
                                st.info("""
                                **Stability Metrics:**
                                - **Model Agreement**: Higher percentage means models agree more
                                - **Prediction Uncertainty**: Lower values indicate more confident predictions
                                - **Bounds Applied**: Shows if extreme predictions were capped for stability
                                """)
                        else:
                            st.warning("No valid predictions to display in table")
                    except Exception as e:
                        st.error(f"Error displaying predictions: {e}")
                    
                    # NEW: Model Validation Section
                    st.subheader("Model Validation & Accuracy Testing")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Run Model Validation", type="primary"):
                            with st.spinner("Running comprehensive model validation..."):
                                try:
                                    # Run validation for different horizons
                                    validation_results = {}
                                    for horizon in [1, 5, 20]:
                                        horizon_str = f'{horizon}d'
                                        results = predictor.validate_model(raw_data, horizon_str)
                                        if results:
                                            validation_results[horizon_str] = results
                                    
                                    if validation_results:
                                        st.success("Model validation completed!")
                                        
                                        # Display validation summary
                                        st.write("### Validation Summary")
                                        for horizon, results in validation_results.items():
                                            if results:
                                                best_model = max(results.keys(), key=lambda x: results[x]['R2'])
                                                best_r2 = results[best_model]['R2']
                                                best_direction = results[best_model]['Directional_Accuracy']
                                                
                                                st.write(f"**{horizon} Horizon:**")
                                                st.write(f"- Best Model: {best_model}")
                                                st.write(f"- R² Score: {best_r2:.4f}")
                                                st.write(f"- Directional Accuracy: {best_direction:.4f}")
                                    else:
                                        st.warning("No validation results generated")
                                        
                                except Exception as e:
                                    st.error(f"Error during validation: {e}")
                    
                    with col2:
                        if st.button("Run Backtesting"):
                            with st.spinner("Running backtesting simulation..."):
                                try:
                                    # Run backtesting for different horizons
                                    backtest_results = {}
                                    for horizon in [1, 5, 20]:
                                        horizon_str = f'{horizon}d'
                                        results = predictor.backtest_model(raw_data, horizon_str)
                                        if results:
                                            backtest_results[horizon_str] = results
                                    
                                    if backtest_results:
                                        st.success("Backtesting completed!")
                                        
                                        # Display backtest summary
                                        st.write("### Backtest Summary")
                                        for horizon, results in backtest_results.items():
                                            st.write(f"**{horizon} Horizon:**")
                                            st.write(f"- Total Return: {results['Total_Return']:.4f}")
                                            st.write(f"- Sharpe Ratio: {results['Sharpe_Ratio']:.4f}")
                                            st.write(f"- Win Rate: {results['Win_Rate']:.4f}")
                                            st.write(f"- Max Drawdown: {results['Max_Drawdown']:.4f}")
                                    else:
                                        st.warning("No backtest results generated")
                                        
                                except Exception as e:
                                    st.error(f"Error during backtesting: {e}")
                    
                    with col3:
                        if st.button("Generate Full Report"):
                            with st.spinner("Generating comprehensive accuracy report..."):
                                try:
                                    # Generate full accuracy report
                                    accuracy_report = predictor.generate_accuracy_report(raw_data, [1, 5, 20])
                                    st.success("Accuracy report generated!")
                                    
                                    # Display key insights
                                    st.write("### Key Insights")
                                    if accuracy_report:
                                        for horizon, results in accuracy_report.items():
                                            validation = results.get('validation', {})
                                            backtest = results.get('backtest', {})
                                            
                                            if validation:
                                                best_model = max(validation.keys(), key=lambda x: validation[x]['R2'])
                                                st.write(f"**{horizon}:** Best model is {best_model}")
                                            
                                            if backtest:
                                                if backtest['Excess_Return'] > 0:
                                                    st.write(f"**{horizon}:** Strategy outperforms buy & hold")
                                                else:
                                                    st.write(f"**{horizon}:** Buy & hold outperforms strategy")
                                
                                except Exception as e:
                                    st.error(f"Error generating report: {e}")
                    
                    # Display detailed validation results if available
                    if hasattr(predictor, 'validation_results') and predictor.validation_results:
                        st.write("### Detailed Validation Results")
                        
                        # Create tabs for different horizons
                        horizon_tabs = st.tabs(list(predictor.validation_results.keys()))
                        
                        for i, (horizon, results) in enumerate(predictor.validation_results.items()):
                            with horizon_tabs[i]:
                                if results:
                                    # Create metrics display
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Model Performance Metrics**")
                                        for model_name, metrics in results.items():
                                            st.write(f"**{model_name}:**")
                                            st.write(f"- R²: {metrics['R2']:.4f}")
                                            st.write(f"- Directional Accuracy: {metrics['Directional_Accuracy']:.4f}")
                                            st.write(f"- Strategy Sharpe: {metrics['Strategy_Sharpe']:.4f}")
                                    
                                    with col2:
                                        st.write("**Trading Performance**")
                                        for model_name, metrics in results.items():
                                            st.write(f"**{model_name}:**")
                                            st.write(f"- Strategy Return: {metrics['Strategy_Total_Return']:.4f}")
                                            st.write(f"- Buy & Hold Return: {metrics['Buy_Hold_Total_Return']:.4f}")
                                            st.write(f"- Outperformance: {metrics['Outperformance']:.4f}")
                    
                    # Display backtest results if available
                    if hasattr(predictor, 'backtest_results') and predictor.backtest_results:
                        st.write("### Backtest Results")
                        
                        # Create tabs for different horizons
                        backtest_tabs = st.tabs(list(predictor.backtest_results.keys()))
                        
                        for i, (horizon, results) in enumerate(predictor.backtest_results.items()):
                            with backtest_tabs[i]:
                                if results:
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Performance Metrics**")
                                        st.metric("Total Return", f"{results['Total_Return']:.4f}")
                                        st.metric("Buy & Hold Return", f"{results['Buy_Hold_Return']:.4f}")
                                        st.metric("Excess Return", f"{results['Excess_Return']:.4f}")
                                    
                                    with col2:
                                        st.write("**Risk Metrics**")
                                        st.metric("Sharpe Ratio", f"{results['Sharpe_Ratio']:.4f}")
                                        st.metric("Max Drawdown", f"{results['Max_Drawdown']:.4f}")
                                        st.metric("Win Rate", f"{results['Win_Rate']:.4f}")
                                    
                                    st.write(f"**Trading Summary:** {results['Number_of_Trades']} trades executed")
                                    st.write(f"**Final Portfolio Value:** ${results['Final_Portfolio_Value']:.2f}")
                    
                else:
                    st.warning(f"Could not generate predictions for {selected_ticker}. This might be due to insufficient recent data.")
                    st.info("Try selecting a different stock or data period.")
            else:
                st.warning(f"Could not train models for {selected_ticker}. Try selecting a longer time period or a different ticker.")
        except Exception as e:
            st.error(f"Error in predictions: {str(e)}")
            st.info("Try selecting a different stock or data period.")
    elif show_predictions and features_data is None:
        st.write(f"Debug: features_data is None")
        st.warning(f"Insufficient data for ML predictions for {selected_ticker}. Try selecting a longer period or a different ticker.")
        st.info("For newer stocks or indices, try 1mo, 3mo, or 6mo periods to see available data.")
    else:
        st.write(f"Debug: show_predictions={show_predictions}, features_data is None={features_data is None}")
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
    st.subheader("Volume Profile Analysis")
    
    # Volume profile chart
    volume_profile_chart = visualizer.create_volume_profile_chart(data_with_indicators, selected_ticker)
    if volume_profile_chart:
        st.plotly_chart(volume_profile_chart, use_container_width=True)
    
    # Volume analysis metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Volume Metrics:**")
        if 'Volume' in data_with_indicators.columns:
            avg_volume = data_with_indicators['Volume'].mean()
            current_volume = data_with_indicators['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            st.metric("Current Volume", f"{current_volume:,.0f}")
            st.metric("Avg Volume", f"{avg_volume:,.0f}")
            st.metric("Volume Ratio", f"{volume_ratio:.2f}x")
    
    with col2:
        st.write("**Volume Indicators:**")
        if 'OBV' in data_with_indicators.columns:
            obv = data_with_indicators['OBV'].iloc[-1]
            obv_change = data_with_indicators['OBV'].iloc[-1] - data_with_indicators['OBV'].iloc[-5]
            st.metric("OBV", f"{obv:,.0f}")
            st.metric("OBV Change (5d)", f"{obv_change:+,.0f}")
        
        if 'CMF' in data_with_indicators.columns:
            cmf = data_with_indicators['CMF'].iloc[-1]
            st.metric("CMF", f"{cmf:.3f}")
    
    with col3:
        st.write("**Volume Analysis:**")
        if 'Volume' in data_with_indicators.columns:
            # Volume trend analysis
            recent_volume = data_with_indicators['Volume'].tail(10).mean()
            older_volume = data_with_indicators['Volume'].tail(30).head(20).mean()
            
            if recent_volume > older_volume * 1.2:
                st.success("Increasing Volume Trend")
            elif recent_volume < older_volume * 0.8:
                st.error("Decreasing Volume Trend")
            else:
                st.info("Stable Volume Trend")

with tab5:
    st.subheader("Momentum Analysis")
    
    # Momentum analysis chart
    momentum_chart = visualizer.create_momentum_analysis_chart(data_with_indicators, selected_ticker)
    if momentum_chart:
        st.plotly_chart(momentum_chart, use_container_width=True)
    
    # Advanced momentum indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Momentum Oscillators:**")
        if 'RSI' in data_with_indicators.columns:
            rsi = data_with_indicators['RSI'].iloc[-1]
            st.metric("RSI", f"{rsi:.1f}")
            
        if 'MFI' in data_with_indicators.columns:
            mfi = data_with_indicators['MFI'].iloc[-1]
            st.metric("MFI", f"{mfi:.1f}")
            
        if 'Ultimate_Oscillator' in data_with_indicators.columns:
            uo = data_with_indicators['Ultimate_Oscillator'].iloc[-1]
            st.metric("Ultimate Osc", f"{uo:.1f}")
    
    with col2:
        st.write("**Trend Strength:**")
        if 'ADX' in data_with_indicators.columns:
            adx = data_with_indicators['ADX'].iloc[-1]
            st.metric("ADX", f"{adx:.1f}")
            
            if adx > 25:
                st.success("Strong Trend")
            elif adx > 20:
                st.warning("Moderate Trend")
            else:
                st.info("Weak Trend")
        
        if 'TSI' in data_with_indicators.columns:
            tsi = data_with_indicators['TSI'].iloc[-1]
            st.metric("TSI", f"{tsi:.3f}")
    
    with col3:
        st.write("**Price Momentum:**")
        if 'ROC_10' in data_with_indicators.columns:
            roc = data_with_indicators['ROC_10'].iloc[-1]
            st.metric("ROC (10)", f"{roc:.2f}%")
            
        if 'Momentum' in data_with_indicators.columns:
            momentum = data_with_indicators['Momentum'].iloc[-1]
            st.metric("Momentum", f"{momentum:.2f}")
            
        if 'PROC' in data_with_indicators.columns:
            proc = data_with_indicators['PROC'].iloc[-1]
            st.metric("PROC", f"{proc:.2f}%")

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
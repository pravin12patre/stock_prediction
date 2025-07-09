import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class StockVisualizer:
    def __init__(self):
        pass
    
    def create_price_chart(self, data, ticker):
        """Create interactive price chart with technical indicators"""
        if data is None or data.empty:
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker} Stock Price', 'Volume', 'RSI', 'MACD'),
            row_width=[0.5, 0.1, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BBU_5_2.0' in data.columns and 'BBL_5_2.0' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BBU_5_2.0'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BBL_5_2.0'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty'
                ),
                row=1, col=1
            )
        
        # Volume
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # MACD
        if 'MACD_12_26_9' in data.columns and 'MACDh_12_26_9' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_12_26_9'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACDh_12_26_9'],
                    mode='lines',
                    name='MACD Histogram',
                    line=dict(color='orange', width=1)
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Stock Analysis',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_prediction_chart(self, data, predictions, ticker):
        """Create chart showing predictions"""
        if data is None or data.empty or not predictions:
            return None
            
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add predictions
        current_date = data.index[-1]
        colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink']
        
        # Sort predictions by horizon for consistent ordering
        sorted_predictions = sorted(predictions.items(), key=lambda x: self._extract_days(x[0]))
        
        for i, (horizon, pred_data) in enumerate(sorted_predictions):
            if not isinstance(pred_data, dict):
                continue
                
            # Calculate future date based on horizon
            days = self._extract_days(horizon)
            if days is None:
                continue
                
            future_date = current_date + pd.Timedelta(days=days)
            predicted_price = pred_data.get('predicted_price', 0)
            
            # Use color cycling to handle any number of predictions
            color = colors[i % len(colors)]
            
            # Add prediction point
            fig.add_trace(
                go.Scatter(
                    x=[future_date],
                    y=[predicted_price],
                    mode='markers+text',
                    name=f'{horizon} Prediction',
                    text=[f"${predicted_price:.2f}"],
                    textposition="top center",
                    marker=dict(size=12, color=color),
                    line=dict(color=color, width=2)
                )
            )
            
            # Add line from current to prediction
            fig.add_trace(
                go.Scatter(
                    x=[current_date, future_date],
                    y=[data['Close'].iloc[-1], predicted_price],
                    mode='lines',
                    name=f'{horizon} Trend',
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=False
                )
            )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} Price Predictions',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def _extract_days(self, horizon):
        """Extract number of days from horizon string"""
        try:
            # Handle formats like '1d', '5d', '20d', etc.
            if isinstance(horizon, str) and horizon.endswith('d'):
                return int(horizon[:-1])
            # Handle integer horizons
            elif isinstance(horizon, int):
                return horizon
            else:
                return None
        except (ValueError, AttributeError):
            return None
    
    def create_sentiment_chart(self, news_sentiment, social_sentiment):
        """Create sentiment visualization"""
        fig = go.Figure()
        
        # News sentiment
        fig.add_trace(
            go.Bar(
                x=['News Sentiment'],
                y=[news_sentiment['sentiment_score']],
                name='News',
                marker_color='blue',
                text=[f"{news_sentiment['sentiment_score']:.3f}"],
                textposition='auto'
            )
        )
        
        # Social sentiment
        fig.add_trace(
            go.Bar(
                x=['Social Sentiment'],
                y=[social_sentiment['sentiment_score']],
                name='Social',
                marker_color='green',
                text=[f"{social_sentiment['sentiment_score']:.3f}"],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title='Sentiment Analysis',
            yaxis_title='Sentiment Score (-1 to 1)',
            height=400,
            yaxis=dict(range=[-1, 1])
        )
        
        return fig
    
    def create_confidence_gauge(self, confidence, title="Prediction Confidence"):
        """Create a gauge chart for prediction confidence"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig 

    def create_advanced_technical_chart(self, data, ticker, patterns=None):
        """Create advanced technical analysis chart with professional indicators"""
        if data is None or data.empty:
            return None
            
        # Create subplots for advanced analysis
        fig = make_subplots(
            rows=6, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(
                f'{ticker} Advanced Technical Analysis', 
                'Volume Profile', 
                'Advanced Oscillators',
                'Ichimoku Cloud',
                'Fibonacci & Support/Resistance',
                'Pattern Analysis'
            ),
            row_width=[0.4, 0.1, 0.15, 0.15, 0.1, 0.1]
        )
        
        # Main price chart with candlesticks
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Add VWAP if available
        if 'VWAP' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['VWAP'],
                    mode='lines',
                    name='VWAP',
                    line=dict(color='purple', width=2, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add Fibonacci retracement levels
        if 'Fib_0' in data.columns:
            fib_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
            fib_levels = ['Fib_0', 'Fib_236', 'Fib_382', 'Fib_500', 'Fib_618', 'Fib_786', 'Fib_100']
            fib_labels = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']
            
            for i, (level, label) in enumerate(zip(fib_levels, fib_labels)):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[level],
                        mode='lines',
                        name=f'Fib {label}',
                        line=dict(color=fib_colors[i], width=1, dash='dot'),
                        showlegend=False
                    ),
                    row=5, col=1
                )
        
        # Add Ichimoku Cloud components
        if 'Ichimoku_Tenkan' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Ichimoku_Tenkan'],
                    mode='lines',
                    name='Tenkan-sen',
                    line=dict(color='blue', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Ichimoku_Kijun'],
                    mode='lines',
                    name='Kijun-sen',
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )
            
            # Add Senkou Span A and B with fill
            if 'Ichimoku_Senkou_A' in data.columns and 'Ichimoku_Senkou_B' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Ichimoku_Senkou_A'],
                        mode='lines',
                        name='Senkou Span A',
                        line=dict(color='green', width=1),
                        fill='tonexty'
                    ),
                    row=4, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['Ichimoku_Senkou_B'],
                        mode='lines',
                        name='Senkou Span B',
                        line=dict(color='red', width=1),
                        fill='tonexty'
                    ),
                    row=4, col=1
                )
        
        # Volume with color coding
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add OBV if available
        if 'OBV' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['OBV'],
                    mode='lines',
                    name='OBV',
                    line=dict(color='purple', width=1),
                    yaxis='y2'
                ),
                row=2, col=1
            )
        
        # Advanced Oscillators
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # Add MFI if available
        if 'MFI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MFI'],
                    mode='lines',
                    name='MFI',
                    line=dict(color='orange', width=1),
                    yaxis='y3'
                ),
                row=3, col=1
            )
            
            # Add MFI overbought/oversold lines
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=3, col=1)
        
        # Add Ultimate Oscillator if available
        if 'Ultimate_Oscillator' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Ultimate_Oscillator'],
                    mode='lines',
                    name='Ultimate Osc',
                    line=dict(color='brown', width=1),
                    yaxis='y4'
                ),
                row=3, col=1
            )
            
            # Add Ultimate Oscillator levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Support and Resistance levels
        if 'Support_Level' in data.columns and 'Resistance_Level' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Resistance_Level'],
                    mode='lines',
                    name='Resistance',
                    line=dict(color='red', width=2, dash='dash')
                ),
                row=5, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Support_Level'],
                    mode='lines',
                    name='Support',
                    line=dict(color='green', width=2, dash='dash')
                ),
                row=5, col=1
            )
        
        # Pivot Points
        if 'Pivot_Point' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Pivot_Point'],
                    mode='lines',
                    name='Pivot Point',
                    line=dict(color='black', width=1)
                ),
                row=5, col=1
            )
        
        # Pattern Analysis
        if patterns:
            self._add_pattern_annotations(fig, data, patterns, row=6)
        
        # Update layout with professional styling
        fig.update_layout(
            title=f'{ticker} Professional Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=1200,
            showlegend=True,
            template='plotly_white',
            font=dict(size=10)
        )
        
        # Update y-axes for multiple indicators
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="Oscillators", row=3, col=1)
        fig.update_yaxes(title_text="Ichimoku", row=4, col=1)
        fig.update_yaxes(title_text="Levels", row=5, col=1)
        fig.update_yaxes(title_text="Patterns", row=6, col=1)
        
        return fig
    
    def _add_pattern_annotations(self, fig, data, patterns, row=6):
        """Add pattern detection annotations to the chart"""
        for pattern_name, pattern_data in patterns.items():
            if pattern_data and pattern_data.get('confidence', 0) > 0.5:
                # Add pattern annotation
                fig.add_annotation(
                    x=data.index[-1],
                    y=data['Close'].iloc[-1],
                    text=f"{pattern_data['pattern']}<br>Confidence: {pattern_data['confidence']:.1%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red",
                    ax=40,
                    ay=-40,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1,
                    row=row,
                    col=1
                )
    
    def create_volume_profile_chart(self, data, ticker):
        """Create volume profile analysis chart"""
        if data is None or data.empty:
            return None
            
        # Calculate volume profile
        price_bins = pd.cut(data['Close'], bins=20)
        volume_profile = data.groupby(price_bins)['Volume'].sum()
        
        fig = go.Figure()
        
        # Volume profile bars
        fig.add_trace(
            go.Bar(
                x=volume_profile.values,
                y=[f"{interval.left:.2f}-{interval.right:.2f}" for interval in volume_profile.index],
                orientation='h',
                name='Volume Profile',
                marker_color='lightblue'
            )
        )
        
        # Current price line
        current_price = data['Close'].iloc[-1]
        fig.add_vline(
            x=volume_profile.max() * 0.8,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current Price: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{ticker} Volume Profile Analysis',
            xaxis_title='Volume',
            yaxis_title='Price Range',
            height=600,
            template='plotly_white'
        )
        
        return fig
    
    def create_momentum_analysis_chart(self, data, ticker):
        """Create comprehensive momentum analysis chart"""
        if data is None or data.empty:
            return None
            
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Action', 'RSI & MFI', 'MACD', 'Volume Momentum'),
            row_width=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Price action
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MFI
        if 'MFI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MFI'],
                    mode='lines',
                    name='MFI',
                    line=dict(color='orange', width=1)
                ),
                row=2, col=1
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'MACD_12_26_9' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_12_26_9'],
                    mode='lines',
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                row=3, col=1
            )
            
            if 'MACDs_12_26_9' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['MACDs_12_26_9'],
                        mode='lines',
                        name='MACD Signal',
                        line=dict(color='red', width=1)
                    ),
                    row=3, col=1
                )
        
        # Volume momentum
        if 'OBV' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['OBV'],
                    mode='lines',
                    name='OBV',
                    line=dict(color='green', width=1)
                ),
                row=4, col=1
            )
        
        fig.update_layout(
            title=f'{ticker} Momentum Analysis',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        return fig 
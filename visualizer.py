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
        if data is None or data.empty:
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
        colors = ['red', 'orange', 'green']
        
        for i, (horizon, pred_data) in enumerate(predictions.items()):
            # Calculate future date
            if horizon == '1d':
                future_date = current_date + pd.Timedelta(days=1)
            elif horizon == '5d':
                future_date = current_date + pd.Timedelta(days=5)
            elif horizon == '20d':
                future_date = current_date + pd.Timedelta(days=20)
            else:
                continue
            
            # Add prediction point
            fig.add_trace(
                go.Scatter(
                    x=[future_date],
                    y=[pred_data['predicted_price']],
                    mode='markers+text',
                    name=f'{horizon} Prediction',
                    text=[f"${pred_data['predicted_price']:.2f}"],
                    textposition="top center",
                    marker=dict(size=12, color=colors[i]),
                    line=dict(color=colors[i], width=2)
                )
            )
            
            # Add line from current to prediction
            fig.add_trace(
                go.Scatter(
                    x=[current_date, future_date],
                    y=[pred_data['current_price'], pred_data['predicted_price']],
                    mode='lines',
                    name=f'{horizon} Trend',
                    line=dict(color=colors[i], width=2, dash='dash'),
                    showlegend=False
                )
            )
        
        fig.update_layout(
            title=f'{ticker} Price Predictions',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500
        )
        
        return fig
    
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
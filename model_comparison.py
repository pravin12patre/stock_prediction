import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import both predictors
from ml_predictor import StockPredictor
from advanced_ml_predictor import AdvancedStockPredictor

class ModelComparison:
    def __init__(self):
        self.basic_predictor = StockPredictor()
        self.advanced_predictor = AdvancedStockPredictor()
        self.results = {}
        
    def fetch_sample_data(self, symbol='AAPL', period='2y'):
        """Fetch sample data for testing"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def compare_models(self, symbol='AAPL', period='2y'):
        """Compare different AI models for stock prediction"""
        print(f"Comparing AI Models for {symbol}")
        print("=" * 60)
        
        # Fetch data
        data = self.fetch_sample_data(symbol, period)
        if data is None:
            print("Failed to fetch data")
            return
        
        print(f"Data loaded: {len(data)} days of {symbol} data")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print()
        
        # Test basic models
        print("Testing Basic Models (Current Implementation)")
        print("-" * 40)
        basic_success = self.basic_predictor.train_models(data)
        if basic_success:
            basic_predictions = self.basic_predictor.predict(data, horizons=[1, 3, 5])
            print("Basic models trained successfully")
            for horizon, pred in basic_predictions.items():
                print(f"   {horizon}: ${pred['predicted_price']:.2f} ({pred['predicted_change']:.2%})")
        else:
            print("Basic models failed to train")
        print()
        
        # Test advanced models
        print("Testing Advanced AI Models")
        print("-" * 40)
        advanced_success = self.advanced_predictor.train_advanced_models(data)
        if advanced_success:
            advanced_predictions = self.advanced_predictor.predict(data, horizons=[1, 3, 5])
            print("Advanced models trained successfully")
            
            # Get model summary
            model_summary = self.advanced_predictor.get_model_summary()
            for horizon, models in model_summary.items():
                print(f"   {horizon} horizon:")
                print(f"     - Traditional ML: {len(models['traditional_ml'])} models")
                print(f"     - Deep Learning: {len(models['deep_learning'])} models")
                print(f"     - Total: {models['total_models']} models")
            
            print("\nAdvanced Model Predictions:")
            for horizon, pred in advanced_predictions.items():
                print(f"   {horizon}: ${pred['predicted_price']:.2f} ({pred['predicted_change']:.2%}) - Confidence: {pred['confidence']:.1%}")
        else:
            print("Advanced models failed to train")
        print()
        
        # Compare results
        if basic_success and advanced_success:
            print("Model Comparison Summary")
            print("-" * 40)
            current_price = data['Close'].iloc[-1]
            print(f"Current {symbol} price: ${current_price:.2f}")
            print()
            
            print("1-day predictions:")
            if '1d' in basic_predictions and '1d' in advanced_predictions:
                basic_1d = basic_predictions['1d']['predicted_price']
                advanced_1d = advanced_predictions['1d']['predicted_price']
                print(f"   Basic: ${basic_1d:.2f} (diff: ${basic_1d - current_price:.2f})")
                print(f"   Advanced: ${advanced_1d:.2f} (diff: ${advanced_1d - current_price:.2f})")
                print(f"   Difference: ${abs(advanced_1d - basic_1d):.2f}")
        
        return {
            'basic_success': basic_success,
            'advanced_success': advanced_success,
            'basic_predictions': basic_predictions if basic_success else {},
            'advanced_predictions': advanced_predictions if advanced_success else {},
            'current_price': data['Close'].iloc[-1] if data is not None else None
        }
    
    def explain_model_advantages(self):
        """Explain the advantages of different AI models"""
        print("\nAI Model Advantages for Stock Prediction")
        print("=" * 60)
        
        advantages = {
            "Traditional ML (Current)": {
                "Models": ["Random Forest", "Gradient Boosting", "Ridge Regression", "SVR"],
                "Pros": [
                    "Fast training and prediction",
                    "Good for small datasets",
                    "Easy to interpret",
                    "Stable performance"
                ],
                "Cons": [
                    "Limited to linear/non-linear patterns",
                    "May miss complex temporal relationships",
                    "Feature engineering dependent"
                ],
                "Best For": "Short-term predictions, small datasets, quick analysis"
            },
            "Gradient Boosting (XGBoost/LightGBM/CatBoost)": {
                "Models": ["XGBoost", "LightGBM", "CatBoost"],
                "Pros": [
                    "Excellent performance on structured data",
                    "Built-in regularization",
                    "Handles missing values well",
                    "Feature importance ranking"
                ],
                "Cons": [
                    "Can overfit with small datasets",
                    "Black box nature",
                    "Sensitive to hyperparameters"
                ],
                "Best For": "Medium-term predictions, feature-rich datasets"
            },
            "Deep Learning (LSTM/CNN-LSTM)": {
                "Models": ["LSTM", "CNN-LSTM", "Transformer"],
                "Pros": [
                    "Captures complex temporal patterns",
                    "Learns from sequential data",
                    "Can model non-linear relationships",
                    "State-of-the-art performance"
                ],
                "Cons": [
                    "Requires large datasets",
                    "Long training time",
                    "Computationally expensive",
                    "Black box predictions"
                ],
                "Best For": "Long-term predictions, complex market patterns"
            },
            "Ensemble Methods": {
                "Models": ["Voting", "Stacking", "Blending"],
                "Pros": [
                    "Reduces overfitting",
                    "Improves prediction stability",
                    "Combines strengths of different models",
                    "Better generalization"
                ],
                "Cons": [
                    "More complex to implement",
                    "Harder to interpret",
                    "Computational overhead"
                ],
                "Best For": "Production systems, high-accuracy requirements"
            }
        }
        
        for model_type, info in advantages.items():
            print(f"\n{model_type}")
            print(f"   Models: {', '.join(info['Models'])}")
            print(f"   Pros: {', '.join(info['Pros'])}")
            print(f"   Cons: {', '.join(info['Cons'])}")
            print(f"   Best For: {info['Best For']}")
    
    def recommend_improvements(self):
        """Recommend specific improvements for better predictions"""
        print("\nRecommendations for Better Stock Predictions")
        print("=" * 60)
        
        recommendations = [
            {
                "Category": "Data Quality",
                "Improvements": [
                    "Use higher frequency data (intraday)",
                    "Include alternative data (news, social media)",
                    "Add macroeconomic indicators",
                    "Include sector/industry data"
                ]
            },
            {
                "Category": "Feature Engineering",
                "Improvements": [
                    "Add market regime indicators",
                    "Include volatility clustering features",
                    "Create cross-asset features",
                    "Add sentiment analysis features"
                ]
            },
            {
                "Category": "Model Architecture",
                "Improvements": [
                    "Use attention mechanisms for time series",
                    "Implement multi-task learning",
                    "Add uncertainty quantification",
                    "Use ensemble of diverse models"
                ]
            },
            {
                "Category": "Training Strategy",
                "Improvements": [
                    "Use walk-forward optimization",
                    "Implement online learning",
                    "Add regularization techniques",
                    "Use cross-validation with time series split"
                ]
            },
            {
                "Category": "Risk Management",
                "Improvements": [
                    "Add position sizing based on confidence",
                    "Implement stop-loss mechanisms",
                    "Use portfolio optimization",
                    "Add drawdown protection"
                ]
            }
        ]
        
        for rec in recommendations:
            print(f"\n📋 {rec['Category']}:")
            for improvement in rec['Improvements']:
                print(f"   • {improvement}")

def main():
    """Main function to run the comparison"""
    print("AI Model Comparison for Stock Prediction")
    print("=" * 60)
    
    comparator = ModelComparison()
    
    # Run comparison
    results = comparator.compare_models('AAPL', '2y')
    
    # Explain advantages
    comparator.explain_model_advantages()
    
    # Provide recommendations
    comparator.recommend_improvements()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("• Basic models: Good for quick analysis and small datasets")
    print("• Advanced models: Better for complex patterns and long-term predictions")
    print("• Ensemble methods: Most robust for production use")
    print("• Deep learning: Best for capturing complex market dynamics")
    print("\nFor best results, combine multiple approaches!")

if __name__ == "__main__":
    main() 
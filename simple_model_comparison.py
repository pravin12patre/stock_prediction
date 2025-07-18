import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import the current predictor
from ml_predictor import StockPredictor

class SimpleModelComparison:
    def __init__(self):
        self.current_predictor = StockPredictor()
        
    def fetch_sample_data(self, symbol='AAPL', period='2y'):
        """Fetch sample data for testing"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def test_current_models(self, symbol='AAPL', period='2y'):
        """Test the current implementation"""
        print(f"Testing Current AI Models for {symbol}")
        print("=" * 60)
        
        # Fetch data
        data = self.fetch_sample_data(symbol, period)
        if data is None:
            print("Failed to fetch data")
            return
        
        print(f"Data loaded: {len(data)} days of {symbol} data")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print()
        
        # Test current models
        print("Current Models Analysis")
        print("-" * 40)
        success = self.current_predictor.train_models(data)
        if success:
            predictions = self.current_predictor.predict(data, horizons=[1, 3, 5])
            print("Models trained successfully")
            current_price = data['Close'].iloc[-1]
            print(f"Current {symbol} price: ${current_price:.2f}")
            print()
            
            for horizon, pred in predictions.items():
                price_diff = pred['predicted_price'] - current_price
                print(f"   {horizon}: ${pred['predicted_price']:.2f} ({pred['predicted_change']:.2%}) - Diff: ${price_diff:.2f}")
        else:
            print("Models failed to train")
        
        return success, predictions if success else {}, data['Close'].iloc[-1] if data is not None else None
    
    def explain_ai_models(self):
        """Explain different AI models for stock prediction"""
        print("\nAI Models for Stock Prediction - Complete Guide")
        print("=" * 70)
        
        models_info = {
            "Current Models (Your Implementation)": {
                "Models": ["Random Forest", "Gradient Boosting", "Ridge Regression", "SVR"],
                "Type": "Traditional Machine Learning",
                "Strengths": [
                    "Fast training and prediction",
                    "Good for small to medium datasets",
                    "Easy to interpret and debug",
                    "Stable performance",
                    "No special hardware required"
                ],
                "Limitations": [
                    "Limited to linear/non-linear patterns",
                    "May miss complex temporal relationships",
                    "Feature engineering dependent",
                    "Struggles with long-term dependencies"
                ],
                "Best Use Cases": [
                    "Short-term predictions (1-5 days)",
                    "Quick analysis and prototyping",
                    "Small datasets (< 10,000 samples)",
                    "Real-time trading systems"
                ]
            },
            "Advanced Gradient Boosting": {
                "Models": ["XGBoost", "LightGBM", "CatBoost"],
                "Type": "Ensemble Learning",
                "Strengths": [
                    "Excellent performance on structured data",
                    "Built-in regularization prevents overfitting",
                    "Handles missing values automatically",
                    "Feature importance ranking",
                    "Fast prediction speed"
                ],
                "Limitations": [
                    "Can overfit with small datasets",
                    "Black box nature (harder to interpret)",
                    "Sensitive to hyperparameters",
                    "Requires careful tuning"
                ],
                "Best Use Cases": [
                    "Medium-term predictions (1-30 days)",
                    "Feature-rich datasets",
                    "Competition-level performance",
                    "Production trading systems"
                ]
            },
            "Deep Learning Models": {
                "Models": ["LSTM", "GRU", "CNN-LSTM", "Transformer"],
                "Type": "Neural Networks",
                "Strengths": [
                    "Captures complex temporal patterns",
                    "Learns from sequential data automatically",
                    "Can model highly non-linear relationships",
                    "State-of-the-art performance",
                    "Can handle multiple time scales"
                ],
                "Limitations": [
                    "Requires large datasets (> 10,000 samples)",
                    "Long training time",
                    "Computationally expensive",
                    "Black box predictions",
                    "Needs GPU for optimal performance"
                ],
                "Best Use Cases": [
                    "Long-term predictions (30+ days)",
                    "Complex market patterns",
                    "Multi-timeframe analysis",
                    "Research and advanced strategies"
                ]
            },
            "Hybrid Models": {
                "Models": ["CNN-LSTM", "Attention-based", "Multi-task Learning"],
                "Type": "Combined Approaches",
                "Strengths": [
                    "Combines strengths of multiple approaches",
                    "Better feature extraction",
                    "Improved generalization",
                    "Can handle multiple objectives",
                    "More robust predictions"
                ],
                "Limitations": [
                    "Complex to implement and debug",
                    "Higher computational cost",
                    "More hyperparameters to tune",
                    "Requires domain expertise"
                ],
                "Best Use Cases": [
                    "Production trading systems",
                    "Multi-asset portfolios",
                    "Risk management systems",
                    "Advanced research projects"
                ]
            },
            "Ensemble Methods": {
                "Models": ["Voting", "Stacking", "Blending", "Bagging"],
                "Type": "Model Combination",
                "Strengths": [
                    "Reduces overfitting",
                    "Improves prediction stability",
                    "Combines strengths of different models",
                    "Better generalization",
                    "More reliable confidence estimates"
                ],
                "Limitations": [
                    "More complex to implement",
                    "Harder to interpret",
                    "Computational overhead",
                    "Requires diverse base models"
                ],
                "Best Use Cases": [
                    "Production systems",
                    "High-accuracy requirements",
                    "Risk-sensitive applications",
                    "Competition submissions"
                ]
            }
        }
        
        for model_type, info in models_info.items():
            print(f"\n{model_type}")
            print(f"   Type: {info['Type']}")
            print(f"   Models: {', '.join(info['Models'])}")
            print(f"   Strengths:")
            for strength in info['Strengths']:
                print(f"      • {strength}")
            print(f"   Limitations:")
            for limitation in info['Limitations']:
                print(f"      • {limitation}")
            print(f"   Best Use Cases:")
            for use_case in info['Best Use Cases']:
                print(f"      • {use_case}")
    
    def recommend_improvements(self):
        """Recommend specific improvements for better predictions"""
        print("\nRecommendations for Better Stock Predictions")
        print("=" * 60)
        
        recommendations = [
            {
                "Category": "Data Quality & Sources",
                "Improvements": [
                    "Use higher frequency data (1-minute, 5-minute bars)",
                    "Include alternative data (news sentiment, social media)",
                    "Add macroeconomic indicators (GDP, inflation, interest rates)",
                    "Include sector/industry data and correlations",
                    "Add options data and implied volatility",
                    "Include foreign exchange and commodity data"
                ]
            },
            {
                "Category": "Feature Engineering",
                "Improvements": [
                    "Add market regime indicators (bull/bear/sideways)",
                    "Include volatility clustering features (GARCH models)",
                    "Create cross-asset features and correlations",
                    "Add sentiment analysis features from news/social media",
                    "Include options-based features (put-call ratios)",
                    "Add technical pattern recognition features"
                ]
            },
            {
                "Category": "Model Architecture",
                "Improvements": [
                    "Use attention mechanisms for time series",
                    "Implement multi-task learning (price + volatility)",
                    "Add uncertainty quantification (confidence intervals)",
                    "Use ensemble of diverse models (different algorithms)",
                    "Implement online learning for real-time updates",
                    "Add reinforcement learning for trading strategies"
                ]
            },
            {
                "Category": "Training Strategy",
                "Improvements": [
                    "Use walk-forward optimization (time series CV)",
                    "Implement online learning with concept drift detection",
                    "Add regularization techniques (L1/L2, dropout)",
                    "Use cross-validation with time series split",
                    "Implement early stopping and learning rate scheduling",
                    "Add data augmentation techniques"
                ]
            },
            {
                "Category": "Risk Management",
                "Improvements": [
                    "Add position sizing based on prediction confidence",
                    "Implement stop-loss mechanisms",
                    "Use portfolio optimization (Markowitz, Black-Litterman)",
                    "Add drawdown protection and risk limits",
                    "Include scenario analysis and stress testing",
                    "Add correlation-based risk management"
                ]
            },
            {
                "Category": "Performance Evaluation",
                "Improvements": [
                    "Use multiple metrics (Sharpe ratio, Sortino ratio)",
                    "Implement backtesting with transaction costs",
                    "Add out-of-sample testing on different time periods",
                    "Include regime-specific performance analysis",
                    "Add statistical significance testing",
                    "Implement cross-validation across different market conditions"
                ]
            }
        ]
        
        for rec in recommendations:
            print(f"\n📋 {rec['Category']}:")
            for improvement in rec['Improvements']:
                print(f"   • {improvement}")
    
    def show_implementation_guide(self):
        """Show how to implement advanced models"""
        print("\nImplementation Guide for Advanced Models")
        print("=" * 60)
        
        print("\n1. Gradient Boosting Implementation:")
        print("   pip install xgboost lightgbm catboost")
        print("   - XGBoost: Best for structured data")
        print("   - LightGBM: Fastest training")
        print("   - CatBoost: Best for categorical features")
        
        print("\n2. Deep Learning Implementation:")
        print("   pip install tensorflow keras")
        print("   - LSTM: For sequential data")
        print("   - CNN-LSTM: For pattern + sequence")
        print("   - Transformer: For attention-based modeling")
        
        print("\n3. Ensemble Methods:")
        print("   - Voting: Simple average of predictions")
        print("   - Stacking: Meta-learner on base predictions")
        print("   - Blending: Weighted combination")
        
        print("\n4. Advanced Features:")
        print("   - Hyperparameter optimization (Optuna)")
        print("   - Model interpretability (SHAP)")
        print("   - Model tracking (MLflow)")
        print("   - Uncertainty quantification")

def main():
    """Main function to run the comparison"""
    print("AI Model Comparison for Stock Prediction")
    print("=" * 60)
    
    comparator = SimpleModelComparison()
    
    # Test current models
    success, predictions, current_price = comparator.test_current_models('AAPL', '2y')
    
    # Explain AI models
    comparator.explain_ai_models()
    
    # Provide recommendations
    comparator.recommend_improvements()
    
    # Show implementation guide
    comparator.show_implementation_guide()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("• Your current models: Good foundation for basic predictions")
    print("• Gradient Boosting: Next step for better performance")
    print("• Deep Learning: For complex patterns and long-term predictions")
    print("• Ensemble Methods: Most robust for production use")
    print("\nStart with gradient boosting, then move to deep learning!")
    print("Check the advanced_requirements.txt for all dependencies")

if __name__ == "__main__":
    main() 
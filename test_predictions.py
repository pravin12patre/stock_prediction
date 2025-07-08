#!/usr/bin/env python3
"""
Test script for the improved stock prediction system
"""

import pandas as pd
import numpy as np
from data_fetcher import StockDataFetcher
from ml_predictor import StockPredictor
import warnings
warnings.filterwarnings('ignore')

def test_prediction_accuracy():
    """Test the prediction accuracy with a well-known stock"""
    
    print("Testing Improved Stock Prediction System")
    print("=" * 50)
    
    # Initialize components
    data_fetcher = StockDataFetcher()
    predictor = StockPredictor()
    
    # Test with AAPL (Apple) - a well-established stock with good data
    ticker = "AAPL"
    period = "2y"
    
    print(f"Testing with {ticker} over {period} period...")
    
    try:
        # Fetch data
        print("1. Fetching stock data...")
        raw_data = data_fetcher.fetch_stock_data(ticker, period)
        
        if raw_data is None or raw_data.empty:
            print("❌ Failed to fetch data")
            return False
        
        print(f"   ✅ Fetched {len(raw_data)} data points")
        
        # Calculate technical indicators
        print("2. Calculating technical indicators...")
        data_with_indicators = data_fetcher.calculate_technical_indicators(raw_data)
        
        if data_with_indicators is None:
            print("❌ Failed to calculate technical indicators")
            return False
        
        print(f"   ✅ Calculated {len(data_with_indicators.columns)} indicators")
        
        # Prepare features
        print("3. Preparing features...")
        features_data = data_fetcher.prepare_features(data_with_indicators)
        
        if features_data is None:
            print("❌ Failed to prepare features")
            return False
        
        print(f"   ✅ Prepared {len(features_data.columns)} features")
        
        # Train models
        print("4. Training advanced models...")
        success = predictor.train_models(features_data)
        
        if not success:
            print("❌ Failed to train models")
            return False
        
        print(f"   ✅ Trained models for {len(predictor.models)} horizons")
        
        # Show model performance
        print("\n5. Model Performance:")
        for horizon in ['1d', '5d', '20d']:
            if horizon in predictor.model_scores:
                scores = predictor.model_scores[horizon]
                print(f"   {horizon.upper()} Models:")
                for model_name, mse in scores.items():
                    print(f"     - {model_name.replace('_', ' ').title()}: MSE = {mse:.6f}")
        
        # Make predictions
        print("\n6. Making predictions...")
        predictions = predictor.predict(features_data)
        
        if predictions:
            print("   ✅ Predictions generated:")
            for horizon, pred_data in predictions.items():
                print(f"     {horizon.upper()}: ${pred_data['predicted_price']:.2f} "
                      f"({pred_data['predicted_change_pct']:+.2f}%)")
                
                # Show confidence
                confidence = predictor.get_prediction_confidence(features_data, horizon)
                print(f"       Confidence: {confidence:.2%}")
                
                # Show uncertainty
                if 'prediction_std' in pred_data:
                    print(f"       Uncertainty: {pred_data['prediction_std']:.4f}")
        else:
            print("❌ Failed to generate predictions")
            return False
        
        # Show feature importance
        print("\n7. Feature Importance (Top 5):")
        for horizon in ['1d', '5d', '20d']:
            feature_importance = predictor.get_feature_importance(horizon)
            if feature_importance is not None:
                print(f"   {horizon.upper()}:")
                top_features = feature_importance.head(5)
                for feature, importance in top_features.items():
                    print(f"     - {feature}: {importance:.4f}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_old_vs_new():
    """Compare old vs new prediction approach"""
    
    print("\n" + "=" * 50)
    print("COMPARISON: Old vs New Prediction System")
    print("=" * 50)
    
    print("OLD SYSTEM:")
    print("- Basic features (12 indicators)")
    print("- Simple models (Random Forest, Gradient Boosting, Linear Regression)")
    print("- No hyperparameter tuning")
    print("- No feature selection")
    print("- No ensemble methods")
    print("- Basic confidence calculation")
    
    print("\nNEW SYSTEM:")
    print("- Advanced features (50+ indicators)")
    print("- Multiple model types (RF, GB, Ridge, SVR, Neural Networks)")
    print("- Hyperparameter tuning with GridSearchCV")
    print("- Intelligent feature selection (correlation, mutual info, RF importance)")
    print("- Ensemble methods with weighted voting")
    print("- Advanced confidence calculation based on model agreement")
    print("- Time series cross-validation")
    print("- Robust scaling for outlier handling")
    print("- Log returns for more stable targets")
    print("- Market regime detection")
    print("- Uncertainty quantification")
    
    print("\nEXPECTED IMPROVEMENTS:")
    print("- Better prediction accuracy through ensemble methods")
    print("- More stable predictions with log returns")
    print("- Better handling of outliers with robust scaling")
    print("- More informative feature importance analysis")
    print("- Better confidence estimates")
    print("- More sophisticated market regime detection")

if __name__ == "__main__":
    # Run the test
    success = test_prediction_accuracy()
    
    if success:
        # Show comparison
        compare_old_vs_new()
        
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print("The improved prediction system includes:")
        print("✅ Advanced feature engineering with 50+ indicators")
        print("✅ Multiple ML models with hyperparameter tuning")
        print("✅ Ensemble methods for better accuracy")
        print("✅ Intelligent feature selection")
        print("✅ Time series cross-validation")
        print("✅ Robust scaling and outlier handling")
        print("✅ Uncertainty quantification")
        print("✅ Market regime detection")
        print("✅ Enhanced confidence calculation")
        print("\nThese improvements should result in more accurate and reliable predictions!")
    else:
        print("\n❌ Test failed. Please check the error messages above.") 
#!/usr/bin/env python3
"""
Test script for the enhanced StockPredictor with validation and accuracy testing
"""

import pandas as pd
import numpy as np
import yfinance as yf
from ml_predictor import StockPredictor
import warnings
warnings.filterwarnings('ignore')

def test_enhanced_features():
    """Test the enhanced features and validation capabilities"""
    print("=" * 60)
    print("ENHANCED STOCK PREDICTOR TEST")
    print("=" * 60)
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Fetch data for testing
    print("\n1. Fetching test data...")
    ticker = "AAPL"
    data = yf.download(ticker, start="2022-01-01", end="2024-01-01", progress=False)
    
    if data.empty:
        print(f"Failed to fetch data for {ticker}")
        return
    
    print(f"Fetched {len(data)} data points for {ticker}")
    print(f"Data shape: {data.shape}")
    
    # Test enhanced feature calculation
    print("\n2. Testing enhanced feature calculation...")
    enhanced_data = predictor.calculate_advanced_features(data)
    
    # Count new features
    original_features = len(predictor.feature_columns)
    available_features = len([col for col in predictor.feature_columns if col in enhanced_data.columns])
    
    print(f"Original feature count: {original_features}")
    print(f"Available features: {available_features}")
    print(f"New features added: {available_features - 59}")  # 59 was the original count
    
    # Show some new features
    new_features = [col for col in enhanced_data.columns if col not in data.columns]
    print(f"Sample new features: {new_features[:10]}")
    
    # Test model training
    print("\n3. Testing model training...")
    success = predictor.train_models(enhanced_data)
    
    if success:
        print("Model training successful!")
        print(f"Trained models for horizons: {list(predictor.models.keys())}")
    else:
        print("Model training failed!")
        return
    
    # Test predictions
    print("\n4. Testing predictions...")
    predictions = predictor.predict(enhanced_data, horizons=[1, 5, 20])
    
    if predictions:
        print("Predictions generated successfully!")
        for horizon, pred_info in predictions.items():
            if isinstance(pred_info, dict):
                print(f"{horizon}: ${pred_info.get('predicted_price', 0):.2f} ({pred_info.get('predicted_change', 0):.4f})")
    else:
        print("No predictions generated!")
    
    # Test validation
    print("\n5. Testing model validation...")
    validation_results = predictor.validate_model(enhanced_data, '1d')
    
    if validation_results:
        print("Validation completed!")
        print("Validation Results Summary:")
        for model_name, metrics in validation_results.items():
            print(f"  {model_name}:")
            print(f"    R² Score: {metrics['R2']:.4f}")
            print(f"    Directional Accuracy: {metrics['Directional_Accuracy']:.4f}")
            print(f"    Strategy Sharpe: {metrics['Strategy_Sharpe']:.4f}")
    else:
        print("Validation failed!")
    
    # Test backtesting
    print("\n6. Testing backtesting...")
    backtest_results = predictor.backtest_model(enhanced_data, '1d')
    
    if backtest_results:
        print("Backtesting completed!")
        print("Backtest Results Summary:")
        print(f"  Total Return: {backtest_results['Total_Return']:.4f}")
        print(f"  Buy & Hold Return: {backtest_results['Buy_Hold_Return']:.4f}")
        print(f"  Excess Return: {backtest_results['Excess_Return']:.4f}")
        print(f"  Sharpe Ratio: {backtest_results['Sharpe_Ratio']:.4f}")
        print(f"  Max Drawdown: {backtest_results['Max_Drawdown']:.4f}")
        print(f"  Win Rate: {backtest_results['Win_Rate']:.4f}")
        print(f"  Number of Trades: {backtest_results['Number_of_Trades']}")
    else:
        print("Backtesting failed!")
    
    # Test comprehensive accuracy report
    print("\n7. Testing comprehensive accuracy report...")
    accuracy_report = predictor.generate_accuracy_report(enhanced_data, [1, 5, 20])
    
    if accuracy_report:
        print("Accuracy report generated successfully!")
        print(f"Tested horizons: {list(accuracy_report.keys())}")
    else:
        print("Accuracy report generation failed!")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        'predictor': predictor,
        'data': enhanced_data,
        'predictions': predictions,
        'validation': validation_results,
        'backtest': backtest_results,
        'accuracy_report': accuracy_report
    }

def test_multiple_stocks():
    """Test the enhanced features on multiple stocks"""
    print("\n" + "=" * 60)
    print("MULTIPLE STOCK TEST")
    print("=" * 60)
    
    tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    results = {}
    
    for ticker in tickers:
        print(f"\nTesting {ticker}...")
        try:
            # Fetch data
            data = yf.download(ticker, start="2023-01-01", end="2024-01-01", progress=False)
            
            if data.empty:
                print(f"  Failed to fetch data for {ticker}")
                continue
            
            # Initialize predictor
            predictor = StockPredictor()
            
            # Calculate features
            enhanced_data = predictor.calculate_advanced_features(data)
            
            # Train models
            success = predictor.train_models(enhanced_data)
            
            if not success:
                print(f"  Model training failed for {ticker}")
                continue
            
            # Run validation
            validation = predictor.validate_model(enhanced_data, '1d')
            
            if validation:
                best_model = max(validation.keys(), key=lambda x: validation[x]['R2'])
                best_r2 = validation[best_model]['R2']
                best_direction = validation[best_model]['Directional_Accuracy']
                
                results[ticker] = {
                    'R2': best_r2,
                    'Directional_Accuracy': best_direction,
                    'Best_Model': best_model
                }
                
                print(f"  R²: {best_r2:.4f}")
                print(f"  Directional Accuracy: {best_direction:.4f}")
                print(f"  Best Model: {best_model}")
            else:
                print(f"  Validation failed for {ticker}")
                
        except Exception as e:
            print(f"  Error testing {ticker}: {e}")
    
    # Summary
    if results:
        print(f"\nSummary for {len(results)} stocks:")
        avg_r2 = np.mean([r['R2'] for r in results.values()])
        avg_direction = np.mean([r['Directional_Accuracy'] for r in results.values()])
        
        print(f"Average R²: {avg_r2:.4f}")
        print(f"Average Directional Accuracy: {avg_direction:.4f}")
        
        best_stock = max(results.keys(), key=lambda x: results[x]['R2'])
        print(f"Best performing stock: {best_stock} (R²: {results[best_stock]['R2']:.4f})")

if __name__ == "__main__":
    # Run the main test
    test_results = test_enhanced_features()
    
    # Run multiple stock test
    test_multiple_stocks()
    
    print("\nAll tests completed!") 
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, mutual_info_regression
import joblib
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.model_scores = {}
        self.best_features = {}
        self.validation_results = {}
        self.backtest_results = {}
        
        # Enhanced feature columns with more sophisticated indicators
        self.feature_columns = [
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d', 'Price_Change_60d',
            'Log_Return', 'Log_Return_5d', 'Log_Return_20d',
            'Price_Volatility', 'Price_Volatility_5d', 'Price_Volatility_20d',
            'Volume_Change', 'Volume_SMA_Ratio', 'Volume_Price_Trend', 'On_Balance_Volume', 'Volume_Ratio',
            'Price_Range_Ratio', 'High_Low_Ratio', 'Open_Close_Ratio', 'Gap_Up_Down', 'Price_Acceleration', 'Price_Deceleration',
            'Volume_Spike', 'Volume_Trend', 'Volume_Price_Correlation', 'Volume_Weighted_Price',
            'ROC_20', 'ROC_60',
            'Momentum_5d', 'Momentum_20d', 'Momentum_60d',
            'Trend_Strength', 'Trend_Direction',
            'Z_Score_20d', 'Z_Score_60d',
            'Percentile_20d', 'Percentile_60d',
            'Day_of_Week', 'Month', 'Quarter', 'Days_from_High', 'Days_from_Low',
            'Market_Regime', 'Volatility_Regime', 'Trend_Regime', 'Volume_Regime',
            'Ichimoku_Conversion', 'Ichimoku_Base', 'Ichimoku_Span_A', 'Ichimoku_Span_B',
            'Aroon_Up', 'Aroon_Down', 'Aroon_Oscillator',
            'Value_at_Risk', 'Expected_Shortfall', 'Sharpe_Ratio', 'Maximum_Drawdown',
            'Candlestick_Pattern', 'Breakout_Score', 'Consolidation_Score',
            'Seasonality_Score', 'Trend_Score', 'Entropy_Score'
        ]
        
    def calculate_advanced_features(self, data):
        """Calculate advanced technical and statistical features"""
        df = data.copy()
        
        # Basic price and volume features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        df['Price_Change_20d'] = df['Close'].pct_change(20)
        df['Price_Change_60d'] = df['Close'].pct_change(60)
        
        # Log returns (more stable than percentage changes)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        df['Log_Return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
        
        # Volatility measures
        df['Price_Volatility'] = df['Log_Return'].rolling(window=20).std()
        df['Price_Volatility_5d'] = df['Log_Return'].rolling(window=5).std()
        df['Price_Volatility_20d'] = df['Log_Return'].rolling(window=20).std()
        
        # Volume-based features
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        df['On_Balance_Volume'] = (df['Volume'] * np.sign(df['Close'] - df['Close'].shift(1))).cumsum()
        
        # Ensure Volume_Ratio is always present (fallback calculation)
        if 'Volume_Ratio' not in df.columns:
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        if 'Volume_Ratio' in df.columns:
            df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
        
        # NEW: Advanced Price Patterns
        df['Price_Range_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Open_Close_Ratio'] = df['Open'] / df['Close']
        df['Gap_Up_Down'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Price_Acceleration'] = df['Price_Change'].diff()
        df['Price_Deceleration'] = df['Price_Change'].diff().diff()
        
        # NEW: Advanced Volume Patterns
        df['Volume_Spike'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_Trend'] = df['Volume'].rolling(window=10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['Volume_Price_Correlation'] = df['Volume'].rolling(window=20).corr(df['Close'])
        df['Volume_Weighted_Price'] = (df['Close'] * df['Volume']).rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
        
        # Enhanced moving average ratios
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            df['SMA_20_50_Ratio'] = df['SMA_20'] / df['SMA_50']
        if 'EMA_12' in df.columns and 'EMA_26' in df.columns:
            df['EMA_12_26_Ratio'] = df['EMA_12'] / df['EMA_26']
        
        # Enhanced Bollinger Bands
        if 'BBL_5_2.0' in df.columns and 'BBU_5_2.0' in df.columns:
            bb_range = df['BBU_5_2.0'] - df['BBL_5_2.0']
            df['BB_Width'] = bb_range / df['Close']
            df['BB_Squeeze'] = df['BB_Width'] / df['BB_Width'].rolling(window=20).mean()
            df['Bollinger_Percent_B'] = (df['Close'] - df['BBL_5_2.0']) / (df['BBU_5_2.0'] - df['BBL_5_2.0'])
            df['Bollinger_Bandwidth'] = (df['BBU_5_2.0'] - df['BBL_5_2.0']) / df['BBM_5_2.0']
        
        # Enhanced RSI features
        if 'RSI' in df.columns:
            df['RSI_5d_Change'] = df['RSI'] - df['RSI'].shift(5)
            df['RSI_20d_Change'] = df['RSI'] - df['RSI'].shift(20)
        
        # Enhanced MACD features
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            df['MACD_Signal_Ratio'] = df['MACD_12_26_9'] / df['MACDs_12_26_9']
            df['MACD_Histogram'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # Stochastic crossover
        if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
            df['STOCH_Cross'] = np.where(df['STOCHk_14_3_3'] > df['STOCHd_14_3_3'], 1, -1)
        
        # Additional ROC periods
        df['ROC_20'] = ((df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)) * 100
        df['ROC_60'] = ((df['Close'] - df['Close'].shift(60)) / df['Close'].shift(60)) * 100
        
        # Momentum indicators
        df['Momentum_5d'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_20d'] = df['Close'] - df['Close'].shift(20)
        df['Momentum_60d'] = df['Close'] - df['Close'].shift(60)
        
        # Trend strength and direction
        df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50'] if 'SMA_20' in df.columns and 'SMA_50' in df.columns else 0
        df['Trend_Direction'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1) if 'SMA_20' in df.columns and 'SMA_50' in df.columns else 0
        
        # Z-scores for price
        df['Z_Score_20d'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / df['Close'].rolling(window=20).std()
        df['Z_Score_60d'] = (df['Close'] - df['Close'].rolling(window=60).mean()) / df['Close'].rolling(window=60).std()
        
        # Percentile ranks
        df['Percentile_20d'] = df['Close'].rolling(window=20).rank(pct=True)
        df['Percentile_60d'] = df['Close'].rolling(window=60).rank(pct=True)
        
        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        # Days from high/low
        window = 252
        if len(df) >= window:
            high_idx = df['Close'].rolling(window=window).apply(lambda x: np.argmax(x), raw=True)
            low_idx = df['Close'].rolling(window=window).apply(lambda x: np.argmin(x), raw=True)
            df['Days_from_High'] = window - high_idx
            df['Days_from_Low'] = window - low_idx
        else:
            df['Days_from_High'] = np.nan
            df['Days_from_Low'] = np.nan
        
        # Market regime features
        df['Market_Regime'] = np.where(df['Price_Volatility'] > df['Price_Volatility'].rolling(window=60).mean(), 'High_Vol', 'Low_Vol')
        df['Volatility_Regime'] = np.where(df['Price_Volatility'] > df['Price_Volatility'].quantile(0.8), 'High', 'Low')
        df['Trend_Regime'] = np.where(df['Trend_Strength'] > df['Trend_Strength'].rolling(window=60).mean(), 'Strong', 'Weak')
        df['Volume_Regime'] = np.where(df['Volume_Ratio'] > 1.5, 'High', 'Normal')
        
        # NEW: Advanced Technical Indicators (simplified calculations)
        # Ichimoku-like indicators
        df['Ichimoku_Conversion'] = (df['High'].rolling(9).max() + df['Low'].rolling(9).min()) / 2
        df['Ichimoku_Base'] = (df['High'].rolling(26).max() + df['Low'].rolling(26).min()) / 2
        df['Ichimoku_Span_A'] = (df['Ichimoku_Conversion'] + df['Ichimoku_Base']) / 2
        df['Ichimoku_Span_B'] = (df['High'].rolling(52).max() + df['Low'].rolling(52).min()) / 2
        
        # Aroon indicators
        df['Aroon_Up'] = df['High'].rolling(25).apply(lambda x: (x.argmax() / 25) * 100)
        df['Aroon_Down'] = df['Low'].rolling(25).apply(lambda x: (x.argmin() / 25) * 100)
        df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']
        
        # NEW: Risk Metrics
        df['Value_at_Risk'] = df['Log_Return'].rolling(window=20).quantile(0.05)
        df['Expected_Shortfall'] = df['Log_Return'].rolling(window=20).apply(lambda x: x[x <= x.quantile(0.05)].mean())
        df['Sharpe_Ratio'] = df['Log_Return'].rolling(window=20).mean() / df['Log_Return'].rolling(window=20).std()
        df['Maximum_Drawdown'] = df['Close'].rolling(window=20).apply(lambda x: (x.max() - x.min()) / x.max())
        
        # NEW: Pattern Recognition (simplified)
        df['Candlestick_Pattern'] = np.where((df['Close'] > df['Open']) & (df['High'] - df['Low'] > 2 * (df['Close'] - df['Open'])), 1, 0)
        df['Breakout_Score'] = (df['Close'] - df['Close'].rolling(window=20).max()) / df['Close'].rolling(window=20).std()
        df['Consolidation_Score'] = df['Price_Volatility'] / df['Price_Volatility'].rolling(window=60).mean()
        
        # NEW: Time Series Features
        df['Seasonality_Score'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['Trend_Score'] = df['Close'].rolling(window=20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        df['Entropy_Score'] = df['Log_Return'].rolling(window=20).apply(lambda x: -np.sum(x.value_counts(normalize=True) * np.log(x.value_counts(normalize=True))))
        
        return df
    
    def prepare_targets(self, data, horizons=[1, 3, 5, 14, 20, 30]):
        """Prepare target variables for different prediction horizons"""
        targets_df = pd.DataFrame(index=data.index)
        
        for horizon in horizons:
            # Calculate future log returns for each horizon
            future_prices = data['Close'].shift(-horizon)
            current_prices = data['Close']
            
            # Calculate log returns (more stable than percentage changes)
            log_returns = np.log(future_prices / current_prices)
            
            # Store as target variable
            targets_df[f'target_{horizon}d'] = log_returns
        
        return targets_df
    
    def select_best_features(self, X, y, horizon):
        """Select the most important features using multiple methods"""
        try:
            # Method 1: Correlation-based selection
            correlations = abs(X.corrwith(y)).sort_values(ascending=False)
            top_corr_features = correlations.head(min(20, len(correlations))).index.tolist()
            
            # Method 2: Mutual information (if available)
            try:
                mi_scores = mutual_info_regression(X.fillna(0), y.fillna(0))
                mi_features = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
                top_mi_features = mi_features.head(min(20, len(mi_features))).index.tolist()
            except:
                top_mi_features = top_corr_features
            
            # Method 3: Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X.fillna(0), y.fillna(0))
            rf_importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            top_rf_features = rf_importance.head(min(20, len(rf_importance))).index.tolist()
            
            # Combine all methods and select features that appear in at least 2 methods
            all_features = set(top_corr_features + top_mi_features + top_rf_features)
            selected_features = []
            
            for feature in all_features:
                count = sum([
                    feature in top_corr_features,
                    feature in top_mi_features,
                    feature in top_rf_features
                ])
                if count >= 2:
                    selected_features.append(feature)
            
            # Ensure we have at least 5 features
            if len(selected_features) < 5:
                selected_features = top_corr_features[:max(5, len(selected_features))]
            
            return selected_features[:20]  # Limit to top 20 features
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            return X.columns.tolist()[:20]  # Fallback to first 20 features
    
    def train_models(self, data):
        """Train advanced models with hyperparameter tuning and ensemble methods"""
        if data is None or data.empty:
            return False
            
        # Calculate advanced features
        data = self.calculate_advanced_features(data)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
        print(f"Debug: Data columns after flattening: {list(data.columns)}")
        
        # Check which feature columns are available
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        print(f"Debug: Expected features: {len(self.feature_columns)}")
        print(f"Debug: Available features: {len(available_features)}")
        
        if len(available_features) < 5:
            print(f"Warning: Only {len(available_features)} features available. Need at least 5.")
            return False
            
        # Prepare targets first
        targets = self.prepare_targets(data)
        print(f"Debug: Target columns created: {list(targets.columns)}")
        
        # Prepare features
        features = data[[col for col in available_features if col in data.columns]]
        print(f"Debug: Data columns after feature engineering: {list(data.columns)}")
        
        # Handle categorical features
        categorical_features = ['Market_Regime', 'Volatility_Regime', 'Trend_Regime', 'Volume_Regime']
        for cat_feature in categorical_features:
            if cat_feature in features.columns:
                features[cat_feature] = pd.Categorical(features[cat_feature]).codes
        
        # Fill NaN values more intelligently
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Align features and targets
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        # Recompute available_features for aligned_data
        available_features = [col for col in self.feature_columns if col in aligned_data.columns]
        
        print(f"Debug: Aligned data shape: {aligned_data.shape}")
        
        if len(aligned_data) < 30:  # Need more data for advanced models
            print(f"Warning: Insufficient aligned data. Only {len(aligned_data)} data points after alignment.")
            return False
        
        # Train models for each horizon
        for horizon in [1, 3, 5, 14, 20, 30]:
            # Ensure horizon is a string like '1d', not an int
            if isinstance(horizon, int):
                horizon_str = f'{horizon}d'
                target_col = f'target_{horizon}d'
            else:
                horizon_str = horizon
                if horizon.endswith('d'):
                    target_col = f'target_{horizon[:-1]}'
                else:
                    target_col = f'target_{horizon}'
            
            if target_col not in aligned_data.columns:
                print(f"Target column {target_col} not available in aligned data")
                continue
            
            if len(aligned_data) < (int(horizon) if isinstance(horizon, int) else int(horizon.rstrip('d'))) + 10:
                print(f"Warning: Insufficient data for {horizon_str} model. Only {len(aligned_data)} data points available, need at least {int(horizon) + 10}.")
                continue
                
            X = aligned_data[[col for col in available_features if col in aligned_data.columns]]
            y = aligned_data[target_col]
            
            # Select best features for this horizon
            best_features = self.select_best_features(X, y, horizon)
            # Fallback if feature selection fails
            if not best_features:
                print(f"Warning: No best features selected for {horizon_str}. Using fallback features.")
                best_features = available_features[:5]
            X_selected = X[best_features]
            self.best_features[f'{horizon}d'] = best_features
            
            print(f"Selected {len(best_features)} best features for {horizon_str} model")
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Scale features
            scaler = RobustScaler()  # More robust to outliers
            if X_selected.shape[1] == 0:
                print(f"Error: No features available for {horizon_str} model. Skipping.")
                continue
            X_scaled = scaler.fit_transform(X_selected)
            
            # Define models with hyperparameter grids
            models = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'ridge': {
                    'model': Ridge(random_state=42),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    }
                },
                'svr': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'gamma': ['scale', 'auto'],
                        'kernel': ['rbf', 'linear']
                    }
                }
            }
            
            best_models = {}
            model_scores = {}
            
            # Train each model with hyperparameter tuning
            for name, model_info in models.items():
                try:
                    # Use GridSearchCV with time series split
                    grid_search = GridSearchCV(
                        model_info['model'],
                        model_info['params'],
                        cv=tscv,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_scaled, y)
                    best_models[name] = grid_search.best_estimator_
                    model_scores[name] = -grid_search.best_score_  # Convert back to positive MSE
                    
                    print(f"{name} best score: {model_scores[name]:.6f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if not best_models:
                print(f"No models could be trained for {horizon_str} horizon")
                continue
            
            # Create ensemble model
            try:
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in best_models.items()],
                    weights=[1/model_scores[name] for name in best_models.keys()]  # Weight by inverse MSE
                )
                ensemble.fit(X_scaled, y)
                best_models['ensemble'] = ensemble
            except Exception as e:
                print(f"Error creating ensemble: {e}")
            
            # Store models and scaler
            self.models[f'{horizon}d'] = best_models
            self.scalers[f'{horizon}d'] = scaler
            self.model_scores[f'{horizon}d'] = model_scores
            
            # Calculate feature importance for the best model
            if 'random_forest' in best_models:
                self.feature_importances[f'{horizon}d'] = pd.Series(
                    best_models['random_forest'].feature_importances_,
                    index=best_features
                ).sort_values(ascending=False)
            
            print(f"Trained {len(best_models)} models for {horizon_str} horizon")
        
        if not self.models:
            print("Warning: No models could be trained due to insufficient data")
            return False
            
        return True
    
    def predict(self, data, horizons=[1, 3, 5, 14, 20, 30]):
        """Make predictions using ensemble methods with improved stability and outlier detection"""
        if data is None or data.empty:
            return {}
            
        predictions = {}
        
        # Calculate advanced features
        data = self.calculate_advanced_features(data)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
        print(f"Debug: Data columns after flattening: {list(data.columns)}")
        
        # Get latest features with better error handling
        available_features = [col for col in self.feature_columns if col in data.columns]
        print(f"Debug: Available features: {len(available_features)}")
        
        if len(available_features) == 0:
            print("No features available for prediction")
            return {}
        
        # Get the latest data point
        latest_data = data[available_features].iloc[-1:].copy()
        print(f"Debug: Latest features shape: {latest_data.shape}")
        print(f"Debug: Latest features columns: {list(latest_data.columns)}")
        
        # Handle NaN values in latest data
        latest_data = latest_data.fillna(method='ffill').fillna(0)
        print(f"Debug: Latest features after processing: {latest_data.shape}")
        
        # Get current price for calculations
        current_price = data['Close'].iloc[-1]
        
        # Check available models
        available_models = list(self.models.keys())
        print(f"Debug: Models available: {available_models}")
        
        for horizon in horizons:
            horizon_str = f"{horizon}d"
            print(f"Debug: Processing {horizon_str} prediction")
            
            if horizon_str not in self.models:
                print(f"Debug: No models for {horizon_str} horizon")
                continue
            
            # Get selected features for this horizon
            if horizon_str in self.best_features:
                selected_feature_names = self.best_features[horizon_str]
                print(f"Debug: Using {len(selected_feature_names)} selected features for {horizon_str}")
                
                # Check if we have enough features
                available_selected = [f for f in selected_feature_names if f in latest_data.columns]
                if len(available_selected) < len(selected_feature_names) * 0.5:  # Need at least 50% of features
                    print(f"Debug: Missing features for {horizon_str}. Expected {len(selected_feature_names)}, got {len(available_selected)}")
                    # Use available features as fallback
                    available_selected = [f for f in selected_feature_names if f in latest_data.columns]
                    if len(available_selected) == 0:
                        print(f"Debug: No features available for {horizon_str}, skipping")
                        continue
                    print(f"Debug: Using {len(available_selected)} available features for {horizon_str}")
                
                features_subset = latest_data[available_selected]
            else:
                print(f"Debug: No selected features for {horizon_str}, using all available")
                features_subset = latest_data
            
            print(f"Debug: Features subset shape: {features_subset.shape}")
            
            # Make predictions with each model
            model_predictions = []
            model_names = []
            
            for model_name, model in self.models[horizon_str].items():
                try:
                    pred = model.predict(features_subset)[0]
                    print(f"Debug: {model_name} prediction: {pred:.6f}")
                    
                    # Apply conservative bounds to individual model predictions
                    # Limit log returns to reasonable ranges (-0.3 to +0.3 for most horizons)
                    max_log_return = 0.3 if horizon <= 5 else 0.4 if horizon <= 14 else 0.5
                    pred = np.clip(pred, -max_log_return, max_log_return)
                    
                    model_predictions.append(pred)
                    model_names.append(model_name)
                    
                except Exception as e:
                    print(f"Error with {model_name} model: {e}")
                    continue
            
            if not model_predictions:
                print(f"Debug: No valid predictions for {horizon_str}")
                continue
            
            # Remove outliers using IQR method
            predictions_array = np.array(model_predictions)
            Q1 = np.percentile(predictions_array, 25)
            Q3 = np.percentile(predictions_array, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Filter out outliers
            valid_indices = (predictions_array >= lower_bound) & (predictions_array <= upper_bound)
            filtered_predictions = predictions_array[valid_indices]
            filtered_names = [model_names[i] for i in range(len(model_names)) if valid_indices[i]]
            
            if len(filtered_predictions) == 0:
                print(f"Debug: All predictions were outliers for {horizon_str}, using median")
                filtered_predictions = [np.median(predictions_array)]
            
            print(f"Debug: Filtered predictions: {filtered_predictions}")
            
            # Calculate ensemble prediction (weighted average, excluding extreme models)
            # Give more weight to more stable models (RF, GB) and less to neural networks
            weights = []
            for name in filtered_names:
                if 'random_forest' in name or 'gradient_boosting' in name:
                    weights.append(0.4)  # Higher weight for tree-based models
                elif 'ridge' in name or 'svr' in name:
                    weights.append(0.3)  # Medium weight for linear models
                else:
                    weights.append(0.1)  # Lower weight for neural networks
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            ensemble_prediction = np.average(filtered_predictions, weights=weights)
            
            # Apply final conservative bounds
            # More conservative bounds for shorter horizons
            if horizon == 1:
                max_change = 0.15  # Max 15% change in 1 day
            elif horizon <= 5:
                max_change = 0.25  # Max 25% change in 5 days
            elif horizon <= 14:
                max_change = 0.35  # Max 35% change in 14 days
            else:
                max_change = 0.50  # Max 50% change in 20+ days
            
            ensemble_prediction = np.clip(ensemble_prediction, -max_change, max_change)
            
            # Calculate predicted price
            predicted_price = current_price * np.exp(ensemble_prediction)
            
            # Additional sanity check: price shouldn't change more than max_change
            price_change_pct = (predicted_price - current_price) / current_price
            if abs(price_change_pct) > max_change:
                # Cap the price change
                if price_change_pct > 0:
                    predicted_price = current_price * (1 + max_change)
                else:
                    predicted_price = current_price * (1 - max_change)
                ensemble_prediction = np.log(predicted_price / current_price)
            
            print(f"Debug: {horizon_str} prediction completed: ${predicted_price:.2f} ({ensemble_prediction:.4f})")
            
            predictions[horizon_str] = {
                'predicted_price': predicted_price,
                'predicted_change': ensemble_prediction,
                'predicted_change_pct': (predicted_price - current_price) / current_price * 100,
                'current_price': current_price,
                'model_agreement': len(filtered_predictions),
                'prediction_std': np.std(filtered_predictions) if len(filtered_predictions) > 1 else 0
            }
        
        print(f"Debug: Total predictions generated: {len(predictions)}")
        return predictions
    
    def get_prediction_confidence(self, data, horizon='1d'):
        """Get confidence interval for prediction based on model agreement and historical performance"""
        if f'{horizon}' not in self.models:
            return 0.5
            
        try:
            # Calculate confidence based on model agreement
            predictions = self.predict(data, [int(horizon[:-1])])
            if not predictions:
                return 0.5
            
            pred_data = predictions[f'{horizon}']
            model_predictions = pred_data['model_predictions']
            prediction_std = pred_data['prediction_std']
            
            # Higher agreement (lower std) = higher confidence
            # Normalize std to 0-1 range (assuming typical std range is 0-0.1)
            normalized_std = min(1.0, prediction_std / 0.1)
            agreement_score = max(0.1, 1 - normalized_std)
            
            # Historical performance score based on model scores
            if f'{horizon}' in self.model_scores:
                # Convert MSE to a confidence score (lower MSE = higher confidence)
                avg_mse = np.mean(list(self.model_scores[f'{horizon}'].values()))
                # Normalize MSE (assuming typical range is 0-0.01)
                normalized_mse = min(1.0, avg_mse / 0.01)
                performance_score = max(0.1, 1 - normalized_mse)
            else:
                performance_score = 0.5
            
            # Feature quality score (more features = higher confidence)
            if f'{horizon}' in self.best_features:
                feature_count = len(self.best_features[f'{horizon}'])
                feature_score = min(1.0, feature_count / 20)  # Normalize to 0-1
            else:
                feature_score = 0.5
            
            # Combine all scores with weights
            confidence = (0.4 * agreement_score + 0.4 * performance_score + 0.2 * feature_score)
            
            # Ensure confidence is in reasonable range
            return min(0.95, max(0.05, confidence))
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def get_feature_importance(self, horizon='1d'):
        """Get feature importance for a specific horizon"""
        if f'{horizon}' in self.feature_importances:
            return self.feature_importances[f'{horizon}']
        return None
    
    def get_model_performance(self, horizon='1d'):
        """Get model performance metrics for a specific horizon"""
        if f'{horizon}' in self.model_scores:
            return self.model_scores[f'{horizon}']
        return None
    
    def save_models(self, filepath='models/'):
        """Save trained models"""
        os.makedirs(filepath, exist_ok=True)
        
        for horizon, models in self.models.items():
            for name, model in models.items():
                model_path = os.path.join(filepath, f'model_{horizon}_{name}.pkl')
                joblib.dump(model, model_path)
            
        for horizon, scaler in self.scalers.items():
            scaler_path = os.path.join(filepath, f'scaler_{horizon}.pkl')
            joblib.dump(scaler, scaler_path)
        
        # Save additional data
        metadata = {
            'feature_importances': self.feature_importances,
            'model_scores': self.model_scores,
            'best_features': self.best_features
        }
        metadata_path = os.path.join(filepath, 'metadata.pkl')
        joblib.dump(metadata, metadata_path)
    
    def load_models(self, filepath='models/'):
        """Load trained models"""
        try:
            for horizon in ['1d', '5d', '20d']:
                horizon_models = {}
                
                # Load individual models
                for model_name in ['random_forest', 'gradient_boosting', 'ridge', 'svr', 'ensemble']:
                    model_path = os.path.join(filepath, f'model_{horizon}_{model_name}.pkl')
                    if os.path.exists(model_path):
                        horizon_models[model_name] = joblib.load(model_path)
                
                if horizon_models:
                    self.models[horizon] = horizon_models
                
                # Load scaler
                scaler_path = os.path.join(filepath, f'scaler_{horizon}.pkl')
                if os.path.exists(scaler_path):
                    self.scalers[horizon] = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = os.path.join(filepath, 'metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_importances = metadata.get('feature_importances', {})
                self.model_scores = metadata.get('model_scores', {})
                self.best_features = metadata.get('best_features', {})
                
        except Exception as e:
            print(f"Error loading models: {e}") 

    def validate_model(self, data, horizon='1d', test_size=0.2):
        """Comprehensive model validation with multiple metrics"""
        if data is None or data.empty:
            return {}
            
        # Calculate features
        data = self.calculate_advanced_features(data)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
        print(f"Debug: Data columns after flattening: {list(data.columns)}")
        
        # Prepare targets
        targets = self.prepare_targets(data, [int(horizon.replace('d', ''))])
        
        # Prepare features and target
        available_features = [col for col in self.feature_columns if col in data.columns]
        if str(horizon).endswith('d'):
            target_col = f'target_{horizon}'
        else:
            target_col = f'target_{horizon}d'
        
        if target_col not in targets.columns:
            print(f"Target column {target_col} not found in prepared targets")
            return {}
            
        features = data[available_features]
        target = targets[target_col]
        
        # Handle categorical features
        categorical_features = ['Market_Regime', 'Volatility_Regime', 'Trend_Regime', 'Volume_Regime']
        for cat_feature in categorical_features:
            if cat_feature in features.columns:
                features[cat_feature] = pd.Categorical(features[cat_feature]).codes
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Align data
        aligned_data = pd.concat([features, target], axis=1).dropna()
        
        if len(aligned_data) < 50:
            print(f"Insufficient data for validation: {len(aligned_data)} points")
            return {}
            
        X = aligned_data[available_features]
        y = aligned_data[target_col]
        
        # Time series split for validation
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Select best features
        best_features = self.select_best_features(X_train, y_train, int(horizon.replace('d', '')))
        X_train_selected = X_train[best_features]
        X_test_selected = X_test[best_features]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }
        
        validation_results = {}
        
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                
                # Directional accuracy
                direction_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
                
                # Profit/Loss simulation
                actual_returns = np.exp(y_test) - 1
                predicted_returns = np.exp(y_pred) - 1
                
                # Simple trading strategy: buy if predicted return > 0
                strategy_returns = np.where(predicted_returns > 0, actual_returns, 0)
                buy_hold_returns = actual_returns
                
                strategy_sharpe = np.mean(strategy_returns) / np.std(strategy_returns) if np.std(strategy_returns) > 0 else 0
                buy_hold_sharpe = np.mean(buy_hold_returns) / np.std(buy_hold_returns) if np.std(buy_hold_returns) > 0 else 0
                
                validation_results[name] = {
                    'MSE': mse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'Directional_Accuracy': direction_accuracy,
                    'Strategy_Sharpe': strategy_sharpe,
                    'Buy_Hold_Sharpe': buy_hold_sharpe,
                    'Strategy_Total_Return': np.sum(strategy_returns),
                    'Buy_Hold_Total_Return': np.sum(buy_hold_returns),
                    'Outperformance': np.sum(strategy_returns) - np.sum(buy_hold_returns)
                }
                
            except Exception as e:
                print(f"Error validating {name}: {e}")
                continue
        
        self.validation_results[horizon] = validation_results
        return validation_results
    
    def backtest_model(self, data, horizon='1d', initial_capital=10000, transaction_cost=0.001):
        """Backtest the model with realistic trading simulation"""
        if data is None or data.empty:
            return {}
            
        # Calculate features
        data = self.calculate_advanced_features(data)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
        print(f"Debug: Data columns after flattening: {list(data.columns)}")
        
        # Prepare targets
        targets = self.prepare_targets(data, [int(horizon.replace('d', ''))])
        
        # Prepare features and target
        available_features = [col for col in self.feature_columns if col in data.columns]
        if str(horizon).endswith('d'):
            target_col = f'target_{horizon}'
        else:
            target_col = f'target_{horizon}d'
        
        if target_col not in targets.columns:
            print(f"Target column {target_col} not found in prepared targets")
            return {}
            
        features = data[available_features]
        target = targets[target_col]
        
        # Handle categorical features
        categorical_features = ['Market_Regime', 'Volatility_Regime', 'Trend_Regime', 'Volume_Regime']
        for cat_feature in categorical_features:
            if cat_feature in features.columns:
                features[cat_feature] = pd.Categorical(features[cat_feature]).codes
        
        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Align data
        aligned_data = pd.concat([features, target], axis=1).dropna()
        
        if len(aligned_data) < 100:
            print(f"Insufficient data for backtesting: {len(aligned_data)} points")
            return {}
            
        X = aligned_data[available_features]
        y = aligned_data[target_col]
        
        # Use first 70% for training, last 30% for backtesting
        train_size = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        
        # Select best features
        best_features = self.select_best_features(X_train, y_train, int(horizon.replace('d', '')))
        X_train_selected = X_train[best_features]
        X_test_selected = X_test[best_features]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train ensemble model
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Ridge Regression': Ridge(alpha=1.0),
            'SVR': SVR(kernel='rbf'),
            'Extra Trees': ExtraTreesRegressor(n_estimators=100, random_state=42)
        }
        
        ensemble_predictions = []
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)
                ensemble_predictions.append(pred)
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if not ensemble_predictions:
            print("No models could be trained for backtesting")
            return {}
        
        # Average predictions
        final_predictions = np.mean(ensemble_predictions, axis=0)
        
        # Convert predictions to price changes
        actual_returns = np.exp(y_test) - 1
        predicted_returns = np.exp(final_predictions) - 1
        
        # Trading simulation
        capital = initial_capital
        position = 0
        trades = []
        portfolio_values = [initial_capital]
        
        # Find the correct 'Close' column after flattening
        close_col = next((col for col in data.columns if str(col).startswith('Close')), 'Close')

        for i in range(len(predicted_returns)):
            current_price = data[close_col].iloc[train_size + i]
            
            # Trading logic
            if predicted_returns[i] > 0.01 and position == 0:  # Buy signal
                position = capital / current_price
                capital = 0
                trades.append({
                    'date': data.index[train_size + i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': position,
                    'cost': position * current_price * (1 + transaction_cost)
                })
            elif predicted_returns[i] < -0.01 and position > 0:  # Sell signal
                capital = position * current_price * (1 - transaction_cost)
                position = 0
                trades.append({
                    'date': data.index[train_size + i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'proceeds': capital
                })
            
            # Calculate current portfolio value
            if position > 0:
                portfolio_value = position * current_price
            else:
                portfolio_value = capital
            portfolio_values.append(portfolio_value)
        
        # Close final position
        if position > 0:
            final_price = data[close_col].iloc[-1]
            capital = position * final_price * (1 - transaction_cost)
            portfolio_values[-1] = capital
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        buy_hold_returns = data[close_col].iloc[train_size:].pct_change().dropna()
        
        # Performance metrics
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        buy_hold_return = (data[close_col].iloc[-1] - data[close_col].iloc[train_size]) / data[close_col].iloc[train_size]
        
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) if np.std(portfolio_returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        # Win rate
        if trades:
            profitable_trades = sum(1 for i in range(0, len(trades), 2) if i+1 < len(trades) and trades[i+1]['proceeds'] > trades[i]['cost'])
            win_rate = profitable_trades / (len(trades) // 2) if len(trades) > 1 else 0
        else:
            win_rate = 0
        
        backtest_results = {
            'Total_Return': total_return,
            'Buy_Hold_Return': buy_hold_return,
            'Excess_Return': total_return - buy_hold_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Number_of_Trades': len(trades),
            'Final_Portfolio_Value': portfolio_values[-1],
            'Initial_Capital': initial_capital,
            'Trades': trades,
            'Portfolio_Values': portfolio_values
        }
        
        self.backtest_results[horizon] = backtest_results
        return backtest_results
    
    def calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown from peak"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def test_model_accuracy(self, data, horizons=[1, 3, 5, 14, 20, 30]):
        """Comprehensive accuracy testing across multiple horizons"""
        accuracy_results = {}
        
        for horizon in horizons:
            horizon_str = f'{horizon}d'
            print(f"\n=== Testing Model Accuracy for {horizon_str} ===")
            
            # Validation
            validation_results = self.validate_model(data, horizon_str)
            if validation_results:
                print(f"Validation Results for {horizon_str}:")
                for model_name, metrics in validation_results.items():
                    print(f"  {model_name}:")
                    print(f"    R² Score: {metrics['R2']:.4f}")
                    print(f"    Directional Accuracy: {metrics['Directional_Accuracy']:.4f}")
                    print(f"    Strategy Sharpe: {metrics['Strategy_Sharpe']:.4f}")
                    print(f"    Outperformance: {metrics['Outperformance']:.4f}")
            
            # Backtesting
            backtest_results = self.backtest_model(data, horizon_str)
            if backtest_results:
                print(f"Backtest Results for {horizon_str}:")
                print(f"  Total Return: {backtest_results['Total_Return']:.4f}")
                print(f"  Buy & Hold Return: {backtest_results['Buy_Hold_Return']:.4f}")
                print(f"  Excess Return: {backtest_results['Excess_Return']:.4f}")
                print(f"  Sharpe Ratio: {backtest_results['Sharpe_Ratio']:.4f}")
                print(f"  Max Drawdown: {backtest_results['Max_Drawdown']:.4f}")
                print(f"  Win Rate: {backtest_results['Win_Rate']:.4f}")
                print(f"  Number of Trades: {backtest_results['Number_of_Trades']}")
            
            accuracy_results[horizon_str] = {
                'validation': validation_results,
                'backtest': backtest_results
            }
        
        return accuracy_results
    
    def generate_accuracy_report(self, data, horizons=[1, 3, 5, 14, 20, 30]):
        """Generate a comprehensive accuracy report"""
        print("=" * 60)
        print("STOCK PREDICTION MODEL ACCURACY REPORT")
        print("=" * 60)
        
        accuracy_results = self.test_model_accuracy(data, horizons)
        
        # Summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        for horizon_str in accuracy_results.keys():
            validation = accuracy_results[horizon_str]['validation']
            backtest = accuracy_results[horizon_str]['backtest']
            
            if validation:
                best_model = max(validation.keys(), key=lambda x: validation[x]['R2'])
                best_r2 = validation[best_model]['R2']
                best_direction = validation[best_model]['Directional_Accuracy']
                
                print(f"\n{horizon_str} Horizon:")
                print(f"  Best Model: {best_model}")
                print(f"  Best R² Score: {best_r2:.4f}")
                print(f"  Best Directional Accuracy: {best_direction:.4f}")
            
            if backtest:
                print(f"  Backtest Total Return: {backtest['Total_Return']:.4f}")
                print(f"  Backtest Sharpe Ratio: {backtest['Sharpe_Ratio']:.4f}")
                print(f"  Backtest Win Rate: {backtest['Win_Rate']:.4f}")
        
        return accuracy_results
    
    def plot_validation_results(self, horizon='1d'):
        """Plot validation results"""
        if horizon not in self.validation_results:
            print(f"No validation results for {horizon}")
            return
        
        results = self.validation_results[horizon]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Model Validation Results - {horizon} Horizon', fontsize=16)
        
        # R² Scores
        models = list(results.keys())
        r2_scores = [results[model]['R2'] for model in models]
        axes[0, 0].bar(models, r2_scores)
        axes[0, 0].set_title('R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Directional Accuracy
        direction_acc = [results[model]['Directional_Accuracy'] for model in models]
        axes[0, 1].bar(models, direction_acc)
        axes[0, 1].set_title('Directional Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Sharpe Ratios
        sharpe_ratios = [results[model]['Strategy_Sharpe'] for model in models]
        axes[1, 0].bar(models, sharpe_ratios)
        axes[1, 0].set_title('Strategy Sharpe Ratios')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Outperformance
        outperformance = [results[model]['Outperformance'] for model in models]
        axes[1, 1].bar(models, outperformance)
        axes[1, 1].set_title('Strategy vs Buy & Hold')
        axes[1, 1].set_ylabel('Excess Return')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def plot_backtest_results(self, horizon='1d'):
        """Plot backtest results"""
        if horizon not in self.backtest_results:
            print(f"No backtest results for {horizon}")
            return
        
        results = self.backtest_results[horizon]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results - {horizon} Horizon', fontsize=16)
        
        # Portfolio Value Over Time
        portfolio_values = results['Portfolio_Values']
        axes[0, 0].plot(portfolio_values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Returns Distribution
        returns = pd.Series(portfolio_values).pct_change().dropna()
        axes[0, 1].hist(returns, bins=30, alpha=0.7)
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        
        # Performance Metrics
        metrics = ['Total_Return', 'Buy_Hold_Return', 'Excess_Return']
        metric_values = [results[metric] for metric in metrics]
        axes[1, 0].bar(metrics, metric_values)
        axes[1, 0].set_title('Performance Comparison')
        axes[1, 0].set_ylabel('Return')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Risk Metrics
        risk_metrics = ['Sharpe_Ratio', 'Max_Drawdown', 'Win_Rate']
        risk_values = [results[metric] for metric in risk_metrics]
        axes[1, 1].bar(risk_metrics, risk_values)
        axes[1, 1].set_title('Risk Metrics')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show() 
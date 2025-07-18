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

# Advanced ML - CatBoost works without OpenMP issues
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Will use alternative models.")

warnings.filterwarnings('ignore')

class EnhancedStockPredictor:
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
            if isinstance(horizon, int):
                target_col = f'target_{horizon}d'
                targets_df[target_col] = data['Close'].shift(-horizon) / data['Close'] - 1
            else:
                horizon_int = int(horizon.rstrip('d'))
                target_col = f'target_{horizon}'
                targets_df[target_col] = data['Close'].shift(-horizon_int) / data['Close'] - 1
        
        return targets_df
    
    def select_best_features(self, X, y, horizon):
        """Select best features using multiple methods"""
        try:
            # Use mutual information for feature selection
            selector = SelectKBest(score_func=mutual_info_regression, k=min(20, X.shape[1]))
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            if len(selected_features) < 5:
                # Fallback to correlation-based selection
                correlations = X.corrwith(y).abs().sort_values(ascending=False)
                selected_features = correlations.head(min(15, len(correlations))).index.tolist()
            
            return selected_features
        except Exception as e:
            print(f"Error in feature selection: {e}")
            return X.columns.tolist()[:15]  # Fallback
    
    def train_enhanced_models(self, data):
        """Train enhanced models with CatBoost and improved ensemble methods"""
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
            
        # Prepare targets first - adjust horizons based on available data
        data_length = len(data)
        if data_length < 40:
            # For small datasets, use shorter horizons
            available_horizons = [1, 3, 5]
            print(f"Warning: Limited data ({data_length} points). Using shorter horizons: {available_horizons}")
        elif data_length < 60:
            available_horizons = [1, 3, 5, 14]
            print(f"Warning: Moderate data ({data_length} points). Using medium horizons: {available_horizons}")
        else:
            available_horizons = [1, 3, 5, 14, 20, 30]
            print(f"Sufficient data ({data_length} points). Using all horizons: {available_horizons}")
        
        targets = self.prepare_targets(data, available_horizons)
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
        
        # Remove any infinite values
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Align features and targets
        aligned_data = pd.concat([features, targets], axis=1).dropna()
        
        # Additional data quality check
        if aligned_data.empty:
            print("Error: No data remaining after alignment and cleaning.")
            return False
        # Recompute available_features for aligned_data
        available_features = [col for col in self.feature_columns if col in aligned_data.columns]
        
        print(f"Debug: Aligned data shape: {aligned_data.shape}")
        
        if len(aligned_data) < 20:  # Reduced minimum requirement for better compatibility
            print(f"Warning: Insufficient aligned data. Only {len(aligned_data)} data points after alignment.")
            return False
        
        # Train models for each horizon
        for horizon in available_horizons:
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
            
            # Define enhanced models with hyperparameter grids
            models = {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'extra_trees': {
                    'model': ExtraTreesRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [10, 15, 20, None],
                        'min_samples_split': [2, 5, 10]
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
            
            # Add CatBoost if available
            if CATBOOST_AVAILABLE:
                models['catboost'] = {
                    'model': CatBoostRegressor(random_state=42, verbose=False),
                    'params': {
                        'iterations': [100, 200, 300],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'depth': [4, 6, 8],
                        'l2_leaf_reg': [1, 3, 5]
                    }
                }
            
            best_models = {}
            model_scores = {}
            
            # Train each model with hyperparameter tuning
            successful_models = 0
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
                    successful_models += 1
                    
                    print(f"{name} best score: {model_scores[name]:.6f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if successful_models < 2:  # Need at least 2 models for ensemble
                print(f"Warning: Only {successful_models} models trained successfully for {horizon_str}. Need at least 2.")
                continue
            
            if not best_models:
                print(f"No models could be trained for {horizon_str} horizon")
                continue
            
            # Create enhanced ensemble model with weighted voting
            try:
                # Calculate weights based on inverse MSE (better models get higher weights)
                weights = [1/model_scores[name] for name in best_models.keys()]
                # Normalize weights
                total_weight = sum(weights)
                normalized_weights = [w/total_weight for w in weights]
                
                ensemble = VotingRegressor(
                    estimators=[(name, model) for name, model in best_models.items()],
                    weights=normalized_weights
                )
                ensemble.fit(X_scaled, y)
                best_models['ensemble'] = ensemble
                
                print(f"Ensemble weights: {dict(zip(best_models.keys(), normalized_weights))}")
                
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
        """Make predictions using enhanced ensemble methods with improved stability and outlier detection"""
        if data is None or data.empty:
            return {}
            
        predictions = {}
        
        # Calculate features
        data = self.calculate_advanced_features(data)
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
        
        for horizon in horizons:
            horizon_str = f'{horizon}d' if isinstance(horizon, int) else horizon
            
            if horizon_str not in self.models:
                continue
            
            # Get latest data point
            available_features = [col for col in self.best_features.get(horizon_str, []) if col in data.columns]
            if not available_features:
                continue
            
            # Ensure we only use numeric features that were used during training
            numeric_features = []
            for feature in available_features:
                if feature in data.columns and pd.api.types.is_numeric_dtype(data[feature]):
                    numeric_features.append(feature)
            
            if not numeric_features:
                print(f"Warning: No numeric features available for {horizon_str}")
                continue
            
            # Ensure we have the exact same features that were used during training
            if horizon_str in self.best_features:
                training_features = self.best_features[horizon_str]
                # Only use features that were actually used during training
                final_features = [f for f in numeric_features if f in training_features]
            else:
                final_features = numeric_features
            
            if not final_features:
                print(f"Warning: No matching features found for {horizon_str}")
                continue
            
            latest_features = data[final_features].iloc[-1:].fillna(0)
            
            # Scale features
            if horizon_str in self.scalers:
                latest_scaled = self.scalers[horizon_str].transform(latest_features)
            else:
                latest_scaled = latest_features.values
            
            model_predictions = []
            model_names = []
            
            # Get predictions from all models
            for model_name, model in self.models[horizon_str].items():
                try:
                    pred = model.predict(latest_scaled)[0]
                    model_predictions.append(pred)
                    model_names.append(model_name)
                    print(f"Debug: {model_name} prediction: {pred:.6f}")
                except Exception as e:
                    print(f"Error getting prediction from {model_name}: {e}")
                    continue
            
            if model_predictions:
                # Enhanced outlier detection and filtering
                predictions_array = np.array(model_predictions)
                
                # Remove extreme outliers (beyond 3 standard deviations)
                mean_pred = np.mean(predictions_array)
                std_pred = np.std(predictions_array)
                lower_bound = mean_pred - 3 * std_pred
                upper_bound = mean_pred + 3 * std_pred
                
                # Filter predictions within reasonable bounds
                filtered_predictions = []
                filtered_names = []
                for pred, name in zip(model_predictions, model_names):
                    if lower_bound <= pred <= upper_bound and -0.5 <= pred <= 0.5:  # Additional sanity check
                        filtered_predictions.append(pred)
                        filtered_names.append(name)
                
                if filtered_predictions:
                    # Use weighted ensemble prediction
                    ensemble_prediction = np.mean(filtered_predictions)
                    current_price = data['Close'].iloc[-1]
                    predicted_price = current_price * (1 + ensemble_prediction)
                    
                    # Calculate confidence based on model agreement
                    confidence = len(filtered_predictions) / len(model_predictions)
                    
                    predictions[horizon_str] = {
                        'predicted_price': predicted_price,
                        'predicted_change': ensemble_prediction,
                        'confidence': confidence,
                        'model_count': len(filtered_predictions),
                        'models_used': filtered_names
                    }
                    
                    print(f"Debug: {horizon_str} prediction completed: ${predicted_price:.2f} ({ensemble_prediction:.4f})")
                else:
                    print(f"Warning: All predictions filtered out for {horizon_str}")
        
        print(f"Debug: Total predictions generated: {len(predictions)}")
        return predictions
    
    def get_prediction_confidence(self, data, horizon='1d'):
        """Get prediction confidence based on model agreement"""
        predictions = self.predict(data, [horizon])
        if horizon in predictions:
            return predictions[horizon]['confidence']
        return 0.0
    
    def get_feature_importance(self, horizon='1d'):
        """Get feature importance for the specified horizon"""
        return self.feature_importances.get(horizon, pd.Series())
    
    def get_model_performance(self, horizon='1d'):
        """Get model performance metrics for the specified horizon"""
        return self.model_scores.get(horizon, {})
    
    def save_models(self, filepath='models/'):
        """Save trained models to disk"""
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        
        for horizon, models in self.models.items():
            for name, model in models.items():
                model_path = os.path.join(filepath, f'{horizon}_{name}_model.pkl')
                joblib.dump(model, model_path)
        
        # Save scalers
        for horizon, scaler in self.scalers.items():
            scaler_path = os.path.join(filepath, f'{horizon}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
        
        # Save feature importances and other metadata
        metadata = {
            'feature_importances': self.feature_importances,
            'model_scores': self.model_scores,
            'best_features': self.best_features
        }
        metadata_path = os.path.join(filepath, 'metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models from disk"""
        if not os.path.exists(filepath):
            print(f"Model directory {filepath} not found")
            return False
        
        try:
            # Load models
            model_files = [f for f in os.listdir(filepath) if f.endswith('_model.pkl')]
            for model_file in model_files:
                parts = model_file.replace('_model.pkl', '').split('_')
                horizon = parts[0]
                model_name = '_'.join(parts[1:])
                
                if horizon not in self.models:
                    self.models[horizon] = {}
                
                model_path = os.path.join(filepath, model_file)
                self.models[horizon][model_name] = joblib.load(model_path)
            
            # Load scalers
            scaler_files = [f for f in os.listdir(filepath) if f.endswith('_scaler.pkl')]
            for scaler_file in scaler_files:
                horizon = scaler_file.replace('_scaler.pkl', '')
                scaler_path = os.path.join(filepath, scaler_file)
                self.scalers[horizon] = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = os.path.join(filepath, 'metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_importances = metadata.get('feature_importances', {})
                self.model_scores = metadata.get('model_scores', {})
                self.best_features = metadata.get('best_features', {})
            
            print(f"Models loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Example usage
if __name__ == "__main__":
    predictor = EnhancedStockPredictor()
    print("Enhanced Stock Predictor initialized with:")
    print("- Traditional ML: Random Forest, Gradient Boosting, Extra Trees")
    print("- Linear Models: Ridge, SVR")
    if CATBOOST_AVAILABLE:
        print("- Advanced ML: CatBoost")
    else:
        print("- Advanced ML: CatBoost not available")
    print("- Enhanced Feature Engineering: 60+ technical indicators")
    print("- Weighted Ensemble Methods: Better prediction stability") 
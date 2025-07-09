import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.model_scores = {}
        self.best_features = {}
        
        # Enhanced feature columns with more sophisticated indicators
        self.feature_columns = [
            # Price-based features
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d', 'Price_Change_60d',
            'Log_Return', 'Log_Return_5d', 'Log_Return_20d',
            'Price_Volatility', 'Price_Volatility_5d', 'Price_Volatility_20d',
            
            # Volume features
            'Volume_Change', 'Volume_Ratio', 'Volume_SMA_Ratio',
            'Volume_Price_Trend', 'On_Balance_Volume',
            
            # Technical indicators
            'SMA_Ratio', 'EMA_Ratio', 'SMA_20_50_Ratio', 'EMA_12_26_Ratio',
            'BB_Position', 'BB_Width', 'BB_Squeeze',
            'RSI', 'RSI_5d_Change', 'RSI_20d_Change',
            'MACD_12_26_9', 'MACD_Signal_Ratio', 'MACD_Histogram',
            'STOCHk_14_3_3', 'STOCHd_14_3_3', 'STOCH_Cross',
            'ROC_10', 'ROC_20', 'ROC_60',
            
            # Advanced indicators
            'Williams_R', 'CCI', 'ADX', 'ATR',
            'Money_Flow_Index', 'Force_Index', 'EOM',
            
            # Momentum and trend features
            'Momentum_5d', 'Momentum_20d', 'Momentum_60d',
            'Trend_Strength', 'Trend_Direction',
            
            # Statistical features
            'Z_Score_20d', 'Z_Score_60d',
            'Percentile_20d', 'Percentile_60d',
            
            # Time-based features
            'Day_of_Week', 'Month', 'Quarter',
            'Days_from_High', 'Days_from_Low',
            
            # Market regime features
            'Market_Regime', 'Volatility_Regime',
            'Trend_Regime', 'Volume_Regime'
        ]
        
    def calculate_advanced_features(self, data):
        """Calculate advanced technical and statistical features"""
        df = data.copy()
        
        # Log returns (more stable than percentage changes)
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Return_5d'] = np.log(df['Close'] / df['Close'].shift(5))
        df['Log_Return_20d'] = np.log(df['Close'] / df['Close'].shift(20))
        
        # Volatility measures
        df['Price_Volatility'] = df['Log_Return'].rolling(window=20).std()
        df['Price_Volatility_5d'] = df['Log_Return'].rolling(window=5).std()
        df['Price_Volatility_20d'] = df['Log_Return'].rolling(window=20).std()
        
        # Volume-based features
        df['Volume_SMA_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['Volume_Price_Trend'] = (df['Close'] - df['Close'].shift(1)) * df['Volume']
        df['On_Balance_Volume'] = (df['Volume'] * np.sign(df['Close'] - df['Close'].shift(1))).cumsum()
        
        # Ensure Volume_Ratio is always present (fallback calculation)
        if 'Volume_Ratio' not in df.columns:
            # Calculate Volume_Ratio as current volume / average volume
            df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        # If Volume_Ratio exists but has NaN values, fill them
        if 'Volume_Ratio' in df.columns:
            df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)  # Default to 1.0 if no volume data
        
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
        
        # Days from high/low (use index difference, not Timestamp subtraction)
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
        
        return df
    
    def prepare_targets(self, data, horizons=[1, 5, 20]):
        """Prepare target variables for different prediction horizons"""
        targets = {}
        data_length = len(data)
        
        for horizon in horizons:
            if data_length > horizon:
                target_col = f'target_{horizon}d'
                # Use log returns for more stable targets
                targets[target_col] = np.log(data['Close'].shift(-horizon) / data['Close'])
            else:
                print(f"Skipping {horizon}d target - insufficient data (need {horizon+1}, have {data_length})")
        
        return targets
    
    def select_best_features(self, X, y, horizon):
        """Select the most important features using multiple methods"""
        try:
            # Method 1: Correlation-based selection
            correlations = abs(X.corrwith(y)).sort_values(ascending=False)
            top_corr_features = correlations.head(min(20, len(correlations))).index.tolist()
            
            # Method 2: Mutual information (if available)
            try:
                from sklearn.feature_selection import mutual_info_regression
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
        
        # Check which feature columns are available
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        print(f"Debug: Expected features: {len(self.feature_columns)}")
        print(f"Debug: Available features: {len(available_features)}")
        
        if len(available_features) < 5:
            print(f"Warning: Only {len(available_features)} features available. Need at least 5.")
            return False
            
        # Prepare targets first
        targets = self.prepare_targets(data)
        print(f"Debug: Target columns created: {list(targets.keys())}")
        
        # Prepare features
        features = data[available_features]
        
        # Handle categorical features
        categorical_features = ['Market_Regime', 'Volatility_Regime', 'Trend_Regime', 'Volume_Regime']
        for cat_feature in categorical_features:
            if cat_feature in features.columns:
                features[cat_feature] = pd.Categorical(features[cat_feature]).codes
        
        # Fill NaN values more intelligently
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Align features and targets
        aligned_data = pd.concat([features, pd.DataFrame(targets)], axis=1).dropna()
        
        print(f"Debug: Aligned data shape: {aligned_data.shape}")
        
        if len(aligned_data) < 30:  # Need more data for advanced models
            print(f"Warning: Insufficient aligned data. Only {len(aligned_data)} data points after alignment.")
            return False
        
        # Train models for each horizon
        for horizon in [1, 5, 20]:
            target_col = f'target_{horizon}d'
            
            if target_col not in aligned_data.columns:
                print(f"Target column {target_col} not available in aligned data")
                continue
            
            if len(aligned_data) < horizon + 10:
                print(f"Warning: Insufficient data for {horizon}d model. Only {len(aligned_data)} data points available, need at least {horizon + 10}.")
                continue
                
            X = aligned_data[available_features]
            y = aligned_data[target_col]
            
            # Select best features for this horizon
            best_features = self.select_best_features(X, y, horizon)
            X_selected = X[best_features]
            self.best_features[f'{horizon}d'] = best_features
            
            print(f"Selected {len(best_features)} best features for {horizon}d model")
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Scale features
            scaler = RobustScaler()  # More robust to outliers
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
                print(f"No models could be trained for {horizon}d horizon")
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
            
            print(f"Trained {len(best_models)} models for {horizon}d horizon")
        
        if not self.models:
            print("Warning: No models could be trained due to insufficient data")
            return False
            
        return True
    
    def predict(self, data, horizons=[1, 5, 20]):
        """Make predictions using ensemble methods"""
        if data is None or data.empty:
            return {}
            
        predictions = {}
        
        # Calculate advanced features
        data = self.calculate_advanced_features(data)
        
        # Get latest features with better error handling
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        # Use the last row without dropping NaN first
        latest_features = data[available_features].iloc[-1:].copy()
        
        # Fill NaN values before any operations
        latest_features = latest_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Handle categorical features
        categorical_features = ['Market_Regime', 'Volatility_Regime', 'Trend_Regime', 'Volume_Regime']
        for cat_feature in categorical_features:
            if cat_feature in latest_features.columns:
                # Convert to numeric safely
                try:
                    latest_features[cat_feature] = pd.Categorical(latest_features[cat_feature]).codes
                except:
                    latest_features[cat_feature] = 0
        
        # Ensure we have at least some features
        if latest_features.empty or latest_features.isnull().all().all():
            print("Warning: No valid features available for prediction")
            return {}
        
        for horizon in horizons:
            if f'{horizon}d' not in self.models:
                continue
            
            # Use best features for this horizon
            if f'{horizon}d' in self.best_features:
                selected_features = self.best_features[f'{horizon}d']
                # Only use features that are actually available
                available_selected = [f for f in selected_features if f in latest_features.columns]
                if available_selected:
                    features_subset = latest_features[available_selected]
                else:
                    print(f"Warning: No selected features available for {horizon}d prediction")
                    continue
            else:
                features_subset = latest_features
            
            # Ensure we have the right number of features for scaling
            if features_subset.shape[1] != self.scalers[f'{horizon}d'].n_features_in_:
                print(f"Warning: Feature mismatch for {horizon}d. Expected {self.scalers[f'{horizon}d'].n_features_in_}, got {features_subset.shape[1]}")
                # Try to match the expected features
                expected_features = self.scalers[f'{horizon}d'].feature_names_in_ if hasattr(self.scalers[f'{horizon}d'], 'feature_names_in_') else None
                if expected_features is not None:
                    missing_features = set(expected_features) - set(features_subset.columns)
                    for feature in missing_features:
                        features_subset[feature] = 0
                    features_subset = features_subset[expected_features]
            
            # Scale features
            try:
                scaled_features = self.scalers[f'{horizon}d'].transform(features_subset)
            except Exception as e:
                print(f"Error scaling features for {horizon}d: {e}")
                continue
            
            # Make predictions with all models
            model_predictions = []
            for name, model in self.models[f'{horizon}d'].items():
                try:
                    pred = model.predict(scaled_features)[0]
                    model_predictions.append(pred)
                except Exception as e:
                    print(f"Error with {name} model: {e}")
                    continue
            
            if not model_predictions:
                continue
            
            # Use ensemble prediction (average of all models)
            pred_change = np.mean(model_predictions)
            
            # Calculate predicted price (convert from log return)
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * np.exp(pred_change)
            
            predictions[f'{horizon}d'] = {
                'predicted_price': predicted_price,
                'predicted_change': pred_change,
                'predicted_change_pct': (np.exp(pred_change) - 1) * 100,
                'current_price': current_price,
                'model_predictions': model_predictions,
                'prediction_std': np.std(model_predictions)  # Uncertainty measure
            }
        
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
            agreement_score = max(0.1, 1 - prediction_std)
            
            # Historical performance score
            if f'{horizon}' in self.model_scores:
                avg_mse = np.mean(list(self.model_scores[f'{horizon}'].values()))
                performance_score = max(0.1, 1 - avg_mse)
            else:
                performance_score = 0.5
            
            # Combine scores
            confidence = (agreement_score + performance_score) / 2
            
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
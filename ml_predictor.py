import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class StockPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            'Price_Change', 'Price_Change_5d', 'Price_Change_20d',
            'Volume_Change', 'Volume_Ratio', 'SMA_Ratio', 'EMA_Ratio',
            'BB_Position', 'RSI', 'MACD_12_26_9', 'STOCHk_14_3_3', 'ROC_10'
        ]
        
    def prepare_targets(self, data, horizons=[1, 5, 20]):
        """Prepare target variables for different prediction horizons"""
        targets = {}
        data_length = len(data)
        
        for horizon in horizons:
            # Only create targets if we have enough data
            if data_length > horizon:
                target_col = f'target_{horizon}d'
                targets[target_col] = data['Close'].shift(-horizon) / data['Close'] - 1
            else:
                print(f"Skipping {horizon}d target - insufficient data (need {horizon+1}, have {data_length})")
        
        return targets
    
    def train_models(self, data):
        """Train models for different prediction horizons"""
        if data is None or data.empty:
            return False
            
        # Prepare features and targets
        # Check which feature columns are available
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        if len(available_features) < 2:  # Need at least 2 features for basic prediction
            print(f"Warning: Only {len(available_features)} features available. Need at least 2.")
            return False
            
        # Prepare targets first
        targets = self.prepare_targets(data)
        
        # Then prepare features and align them
        features = data[available_features]
        
        # Align features and targets before dropping NaN
        aligned_data = pd.concat([features, pd.DataFrame(targets)], axis=1).dropna()
        
        print(f"Debug: Aligned data shape: {aligned_data.shape}")
        print(f"Debug: Available features: {available_features}")
        print(f"Debug: Target columns in aligned data: {[col for col in aligned_data.columns if 'target' in col]}")
        
        if len(aligned_data) < 5:  # Need at least 5 data points
            print(f"Warning: Insufficient aligned data. Only {len(aligned_data)} data points after alignment.")
            return False
        
        # Train models for each horizon
        for horizon in [1, 5, 20]:
            target_col = f'target_{horizon}d'
            
            # Check if target column exists in aligned data
            if target_col not in aligned_data.columns:
                print(f"Target column {target_col} not available in aligned data")
                continue
            
            # Adjust minimum data requirements based on horizon
            min_data_points = max(3, horizon + 1)  # Need at least horizon+1 points
            
            if len(aligned_data) < min_data_points:
                print(f"Warning: Insufficient data for {horizon}d model. Only {len(aligned_data)} data points available, need at least {min_data_points}.")
                continue
                
            X = aligned_data[available_features]
            y = aligned_data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            
            best_model = None
            best_score = -np.inf
            
            for name, model in models.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            # Store best model and scaler
            self.models[f'{horizon}d'] = best_model
            self.scalers[f'{horizon}d'] = scaler
            
            print(f"Trained {horizon}d model with R² score: {best_score:.4f}")
        
        if not self.models:
            print("Warning: No models could be trained due to insufficient data")
            return False
            
        return True
    
    def predict(self, data, horizons=[1, 5, 20]):
        """Make predictions for different horizons"""
        if data is None or data.empty:
            return {}
            
        predictions = {}
        
        # Get latest features
        available_features = [col for col in self.feature_columns if col in data.columns]
        latest_features = data[available_features].iloc[-1:].dropna()
        
        if latest_features.empty:
            return {}
        
        for horizon in horizons:
            if f'{horizon}d' not in self.models:
                continue
                
            # Scale features
            scaled_features = self.scalers[f'{horizon}d'].transform(latest_features)
            
            # Make prediction
            pred_change = self.models[f'{horizon}d'].predict(scaled_features)[0]
            
            # Calculate predicted price
            current_price = data['Close'].iloc[-1]
            predicted_price = current_price * (1 + pred_change)
            
            predictions[f'{horizon}d'] = {
                'predicted_price': predicted_price,
                'predicted_change': pred_change,
                'predicted_change_pct': pred_change * 100,
                'current_price': current_price
            }
        
        return predictions
    
    def get_prediction_confidence(self, data, horizon='1d'):
        """Get confidence interval for prediction (simplified)"""
        if f'{horizon}' not in self.models:
            return 0.5  # Default confidence
            
        # This is a simplified confidence calculation
        # In a real implementation, you'd use techniques like:
        # - Bootstrap sampling
        # - Quantile regression
        # - Ensemble variance
        
        # For now, return a confidence based on recent model performance
        recent_data = data.tail(30)
        if len(recent_data) < 10:
            return 0.5
            
        available_features = [col for col in self.feature_columns if col in recent_data.columns]
        features = recent_data[available_features].dropna()
        if features.empty:
            return 0.5
            
        # Calculate prediction variance as a proxy for confidence
        predictions = []
        for i in range(len(features)):
            if i == 0:
                continue
            scaled_features = self.scalers[f'{horizon}'].transform(features.iloc[i:i+1])
            pred = self.models[f'{horizon}'].predict(scaled_features)[0]
            predictions.append(pred)
        
        if len(predictions) < 2:
            return 0.5
            
        # Higher variance = lower confidence
        variance = np.var(predictions)
        confidence = max(0.1, min(0.9, 1 - variance))
        
        return confidence
    
    def save_models(self, filepath='models/'):
        """Save trained models"""
        os.makedirs(filepath, exist_ok=True)
        
        for horizon, model in self.models.items():
            model_path = os.path.join(filepath, f'model_{horizon}.pkl')
            joblib.dump(model, model_path)
            
        for horizon, scaler in self.scalers.items():
            scaler_path = os.path.join(filepath, f'scaler_{horizon}.pkl')
            joblib.dump(scaler, scaler_path)
    
    def load_models(self, filepath='models/'):
        """Load trained models"""
        for horizon in ['1d', '5d', '20d']:
            model_path = os.path.join(filepath, f'model_{horizon}.pkl')
            scaler_path = os.path.join(filepath, f'scaler_{horizon}.pkl')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[horizon] = joblib.load(model_path)
                self.scalers[horizon] = joblib.load(scaler_path) 
"""
Energy Consumption Predictor - Enhanced Version
A modern time series forecasting system for energy consumption prediction
using advanced ML techniques and proper software engineering practices.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Modern ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Time series specific libraries
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnergyConsumptionPredictor:
    """
    Advanced Energy Consumption Predictor with multiple ML models
    and comprehensive feature engineering.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.best_model = None
        self.best_score = float('inf')
        
    def generate_enhanced_data(self, start_date='2023-01-01', periods=8760, freq='H'):
        """
        Generate realistic energy consumption data with multiple patterns
        """
        np.random.seed(self.random_state)
        
        # Create datetime index
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        
        # Base consumption with seasonal patterns
        base_consumption = 50
        
        # Daily pattern (higher during day, lower at night)
        daily_pattern = 15 * np.sin(2 * np.pi * np.arange(periods) / 24 + np.pi/2)
        
        # Weekly pattern (higher on weekdays)
        weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(periods) / (24*7))
        
        # Seasonal pattern (higher in summer/winter for AC/heating)
        seasonal_pattern = 20 * np.sin(2 * np.pi * np.arange(periods) / (24*365.25) + np.pi)
        
        # Weather effect (temperature correlation)
        temperature = 20 + 15 * np.sin(2 * np.pi * np.arange(periods) / (24*365.25)) + np.random.normal(0, 5, periods)
        weather_effect = 0.5 * np.abs(temperature - 20)  # More energy when temp deviates from 20°C
        
        # Random noise
        noise = np.random.normal(0, 3, periods)
        
        # Combine all patterns
        energy_consumption = (base_consumption + daily_pattern + weekly_pattern + 
                            seasonal_pattern + weather_effect + noise)
        
        # Ensure positive values
        energy_consumption = np.maximum(energy_consumption, 10)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'energy_consumption': energy_consumption,
            'temperature': temperature,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek,
            'month': dates.month,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'is_holiday': np.random.choice([0, 1], size=periods, p=[0.97, 0.03])
        })
        
        return df.set_index('timestamp')
    
    def create_advanced_features(self, df):
        """
        Create comprehensive feature set for energy prediction
        """
        df_features = df.copy()
        
        # Lag features (multiple horizons)
        for lag in [1, 2, 3, 6, 12, 24, 48, 168]:  # 1h to 1 week
            df_features[f'lag_{lag}h'] = df_features['energy_consumption'].shift(lag)
        
        # Rolling statistics
        for window in [6, 12, 24, 168]:
            df_features[f'rolling_mean_{window}h'] = df_features['energy_consumption'].rolling(window).mean()
            df_features[f'rolling_std_{window}h'] = df_features['energy_consumption'].rolling(window).std()
            df_features[f'rolling_min_{window}h'] = df_features['energy_consumption'].rolling(window).min()
            df_features[f'rolling_max_{window}h'] = df_features['energy_consumption'].rolling(window).max()
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df_features[f'ema_{alpha}'] = df_features['energy_consumption'].ewm(alpha=alpha).mean()
        
        # Time-based features
        df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
        df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        # Interaction features
        df_features['temp_hour'] = df_features['temperature'] * df_features['hour']
        df_features['temp_weekend'] = df_features['temperature'] * df_features['is_weekend']
        
        # Difference features
        df_features['energy_diff_1h'] = df_features['energy_consumption'].diff(1)
        df_features['energy_diff_24h'] = df_features['energy_consumption'].diff(24)
        
        return df_features
    
    def prepare_data(self, df, target_col='energy_consumption', test_size=0.2):
        """
        Prepare data for training with proper time series split
        """
        # Remove target from features
        feature_cols = [col for col in df.columns if col != target_col]
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        # Time series split (no shuffling)
        split_idx = int(len(df_clean) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models and compare performance
        """
        # Initialize models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=self.random_state),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=self.random_state),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        }
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['standard'] = scaler
        
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Use scaled data for linear models
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_train_model = X_train_scaled
            else:
                X_train_model = X_train
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_model, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            
            # Train model
            model.fit(X_train_model, y_train)
            
            # Store results
            results[name] = {
                'model': model,
                'cv_score': -cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.models[name] = model
            
            # Track best model (lower CV score is better for MSE)
            cv_score = -cv_scores.mean()
            if cv_score < self.best_score:
                self.best_score = cv_score
                self.best_model = name
        
        return results
    
    def evaluate_models(self, X_test, y_test, results):
        """
        Evaluate all trained models on test set
        """
        evaluation_results = {}
        
        for name, result in results.items():
            model = result['model']
            
            # Use appropriate data scaling
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_test_model = self.scalers['standard'].transform(X_test)
            else:
                X_test_model = X_test
            
            # Predictions
            y_pred = model.predict(X_test_model)
            
            # Metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            evaluation_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2,
                'CV_Score': result['cv_score']
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X_test.columns, model.feature_importances_))
        
        return evaluation_results
    
    def plot_results(self, X_test, y_test, evaluation_results):
        """
        Create comprehensive visualization of results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        models = list(evaluation_results.keys())
        r2_scores = [evaluation_results[model]['R²'] for model in models]
        rmse_scores = [evaluation_results[model]['RMSE'] for model in models]
        
        axes[0, 0].bar(models, r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Model Comparison - R² Score')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. RMSE comparison
        axes[0, 1].bar(models, rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Model Comparison - RMSE')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Best model predictions
        if self.best_model is not None:
            best_model = self.models[self.best_model]
            if self.best_model in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                X_test_scaled = self.scalers['standard'].transform(X_test)
                y_pred_best = best_model.predict(X_test_scaled)
            else:
                y_pred_best = best_model.predict(X_test)
        else:
            # Fallback to first available model
            first_model_name = list(self.models.keys())[0]
            best_model = self.models[first_model_name]
            y_pred_best = best_model.predict(X_test)
            self.best_model = first_model_name
        
        axes[1, 0].plot(y_test.index[-100:], y_test.iloc[-100:], label='Actual', linewidth=2)
        axes[1, 0].plot(y_test.index[-100:], y_pred_best[-100:], label='Predicted', linewidth=2, alpha=0.8)
        axes[1, 0].set_title(f'Predictions - {self.best_model} (Last 100 hours)')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Energy Consumption (kWh)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature importance (if available)
        if self.best_model in self.feature_importance:
            importance = self.feature_importance[self.best_model]
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*top_features)
            
            axes[1, 1].barh(features, importances, color='lightgreen', alpha=0.7)
            axes[1, 1].set_title(f'Top 10 Feature Importance - {self.best_model}')
            axes[1, 1].set_xlabel('Importance')
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance not available\nfor this model type', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def generate_forecast(self, df, steps=24):
        """
        Generate future predictions
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Train models first.")
        
        model = self.models[self.best_model]
        last_data = df.tail(200).copy()  # Use last 200 hours for context
        
        forecasts = []
        
        for step in range(steps):
            # Create features for the next time step
            features_df = self.create_advanced_features(last_data)
            features_df = features_df.dropna()
            
            # Get the latest feature vector
            latest_features = features_df.iloc[-1:].drop('energy_consumption', axis=1)
            
            # Make prediction
            if self.best_model in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                latest_features_scaled = self.scalers['standard'].transform(latest_features)
                prediction = model.predict(latest_features_scaled)[0]
            else:
                prediction = model.predict(latest_features)[0]
            
            forecasts.append(prediction)
            
            # Add prediction to data for next iteration
            next_time = last_data.index[-1] + pd.Timedelta(hours=1)
            new_row = last_data.iloc[-1].copy()
            new_row.name = next_time
            new_row['energy_consumption'] = prediction
            
            # Update time-based features
            new_row['hour'] = next_time.hour
            new_row['day_of_week'] = next_time.dayofweek
            new_row['month'] = next_time.month
            new_row['is_weekend'] = int(next_time.dayofweek >= 5)
            
            last_data = pd.concat([last_data, new_row.to_frame().T])
        
        return forecasts

def main():
    """
    Main execution function
    """
    print("🔋 Energy Consumption Predictor - Enhanced Version")
    print("=" * 50)
    
    # Initialize predictor
    predictor = EnergyConsumptionPredictor()
    
    # Generate enhanced dataset
    print("📊 Generating enhanced dataset...")
    df = predictor.generate_enhanced_data(periods=8760)  # 1 year of hourly data
    
    # Create advanced features
    print("🔧 Creating advanced features...")
    df_features = predictor.create_advanced_features(df)
    
    # Prepare data
    print("📋 Preparing data for training...")
    X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {len(X_train.columns)}")
    
    # Train models
    print("\n🤖 Training multiple ML models...")
    results = predictor.train_models(X_train, y_train)
    
    # Evaluate models
    print("\n📈 Evaluating models...")
    evaluation_results = predictor.evaluate_models(X_test, y_test, results)
    
    # Display results
    print("\n🏆 Model Performance Summary:")
    print("-" * 70)
    print(f"{'Model':<20} {'R²':<8} {'RMSE':<8} {'MAE':<8} {'CV Score':<10}")
    print("-" * 70)
    
    for model, metrics in evaluation_results.items():
        print(f"{model:<20} {metrics['R²']:<8.3f} {metrics['RMSE']:<8.2f} "
              f"{metrics['MAE']:<8.2f} {metrics['CV_Score']:<10.2f}")
    
    print(f"\n🥇 Best Model: {predictor.best_model}")
    
    # Plot results
    print("\n📊 Generating visualizations...")
    predictor.plot_results(X_test, y_test, evaluation_results)
    
    # Generate forecast
    print("\n🔮 Generating 24-hour forecast...")
    forecast = predictor.generate_forecast(df_features, steps=24)
    
    print("Next 24 hours forecast:")
    for i, pred in enumerate(forecast):
        print(f"Hour {i+1}: {pred:.2f} kWh")
    
    return predictor, df_features, evaluation_results

if __name__ == "__main__":
    predictor, data, results = main()

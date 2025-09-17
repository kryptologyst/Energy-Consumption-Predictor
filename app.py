"""
Flask Web Application for Energy Consumption Predictor
Modern web interface with real-time predictions and interactive dashboards
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from energy_predictor import EnergyConsumptionPredictor
from database import EnergyDatabase
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['SECRET_KEY'] = 'energy-predictor-2024'

# Global variables
predictor = None
database = None
trained_models = None

def initialize_system():
    """Initialize the prediction system and database"""
    global predictor, database, trained_models
    
    # Initialize database
    database = EnergyDatabase()
    
    # Check if we have data, if not populate sample data
    stats = database.get_database_stats()
    if stats['energy_records'] == 0:
        database.populate_sample_data(days=90)  # 3 months of data
    
    # Initialize predictor
    predictor = EnergyConsumptionPredictor()
    
    # Get data and train models
    df = database.get_energy_data()
    if not df.empty:
        # Rename columns to match predictor expectations
        df = df.rename(columns={'energy_kwh': 'energy_consumption'})
        
        # Add required columns
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Convert categorical weather condition to numeric
        weather_mapping = {'clear': 0, 'partly_cloudy': 1, 'cloudy': 2, 'rainy': 3, 'stormy': 4}
        df['weather_condition'] = df['weather_condition'].map(weather_mapping).fillna(0)
        
        # Create features and train
        df_features = predictor.create_advanced_features(df)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
        
        trained_models = predictor.train_models(X_train, y_train)
        evaluation_results = predictor.evaluate_models(X_test, y_test, trained_models)
        
        # Store model performance in database
        for model_name, metrics in evaluation_results.items():
            database.insert_model_performance(model_name, metrics)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/dashboard_data')
def get_dashboard_data():
    """Get data for the main dashboard"""
    try:
        # Get recent energy data
        recent_data = database.get_energy_data(
            start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        )
        
        if recent_data.empty:
            return jsonify({'error': 'No data available'})
        
        # Prepare data for charts
        timestamps = recent_data.index.strftime('%Y-%m-%d %H:%M').tolist()
        energy_values = recent_data['energy_kwh'].tolist()
        temperature_values = recent_data['temperature'].tolist()
        
        # Calculate statistics
        stats = {
            'current_consumption': round(energy_values[-1], 2),
            'avg_consumption': round(np.mean(energy_values), 2),
            'max_consumption': round(np.max(energy_values), 2),
            'min_consumption': round(np.min(energy_values), 2),
            'total_records': len(energy_values)
        }
        
        # Get best model info
        best_model = database.get_best_model()
        
        return jsonify({
            'timestamps': timestamps,
            'energy_values': energy_values,
            'temperature_values': temperature_values,
            'stats': stats,
            'best_model': best_model.to_dict() if best_model is not None else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict_energy():
    """Generate energy consumption predictions"""
    try:
        data = request.get_json()
        hours = data.get('hours', 24)
        
        if predictor is None or predictor.best_model is None:
            return jsonify({'error': 'Model not trained yet'})
        
        # Get recent data for context
        df = database.get_energy_data()
        df = df.rename(columns={'energy_kwh': 'energy_consumption'})
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Convert categorical weather condition to numeric
        weather_mapping = {'clear': 0, 'partly_cloudy': 1, 'cloudy': 2, 'rainy': 3, 'stormy': 4}
        df['weather_condition'] = df['weather_condition'].map(weather_mapping).fillna(0)
        
        # Create features
        df_features = predictor.create_advanced_features(df)
        
        # Generate predictions
        predictions = predictor.generate_forecast(df_features, steps=hours)
        
        # Create future timestamps
        last_timestamp = df.index[-1]
        future_timestamps = [
            (last_timestamp + timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M')
            for i in range(hours)
        ]
        
        return jsonify({
            'timestamps': future_timestamps,
            'predictions': [round(p, 2) for p in predictions],
            'model_used': predictor.best_model
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/model_performance')
def get_model_performance():
    """Get model performance comparison"""
    try:
        # Get recent data and retrain if needed
        df = database.get_energy_data()
        if df.empty:
            return jsonify({'error': 'No data available'})
        
        df = df.rename(columns={'energy_kwh': 'energy_consumption'})
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        
        # Convert categorical weather condition to numeric
        weather_mapping = {'clear': 0, 'partly_cloudy': 1, 'cloudy': 2, 'rainy': 3, 'stormy': 4}
        df['weather_condition'] = df['weather_condition'].map(weather_mapping).fillna(0)
        
        df_features = predictor.create_advanced_features(df)
        X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
        
        if trained_models is None:
            models = predictor.train_models(X_train, y_train)
        else:
            models = trained_models
            
        evaluation_results = predictor.evaluate_models(X_test, y_test, models)
        
        # Format for frontend
        model_data = []
        for model_name, metrics in evaluation_results.items():
            model_data.append({
                'name': model_name,
                'r2': round(metrics['R²'], 3),
                'rmse': round(metrics['RMSE'], 2),
                'mae': round(metrics['MAE'], 2)
            })
        
        return jsonify({'models': model_data})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/feature_importance')
def get_feature_importance():
    """Get feature importance for the best model"""
    try:
        if predictor is None or predictor.best_model is None:
            return jsonify({'error': 'Model not trained yet'})
        
        if predictor.best_model in predictor.feature_importance:
            importance = predictor.feature_importance[predictor.best_model]
            
            # Get top 10 features
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features = [f[0] for f in top_features]
            importances = [round(f[1], 4) for f in top_features]
            
            return jsonify({
                'features': features,
                'importances': importances,
                'model': predictor.best_model
            })
        else:
            return jsonify({'error': 'Feature importance not available for this model type'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/weather_correlation')
def get_weather_correlation():
    """Get correlation between weather and energy consumption"""
    try:
        df = database.get_energy_data()
        if df.empty:
            return jsonify({'error': 'No data available'})
        
        # Calculate correlations
        correlations = {}
        if 'temperature' in df.columns:
            correlations['temperature'] = round(df['energy_kwh'].corr(df['temperature']), 3)
        if 'humidity' in df.columns:
            correlations['humidity'] = round(df['energy_kwh'].corr(df['humidity']), 3)
        
        # Get hourly patterns
        df['hour'] = df.index.hour
        hourly_avg = df.groupby('hour')['energy_kwh'].mean().round(2)
        
        return jsonify({
            'correlations': correlations,
            'hourly_pattern': {
                'hours': hourly_avg.index.tolist(),
                'consumption': hourly_avg.values.tolist()
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/export_data')
def export_data():
    """Export energy data as CSV"""
    try:
        filename = database.export_data('energy_consumption')
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🚀 Starting Energy Consumption Predictor Web App...")
    initialize_system()
    print("✅ System initialized successfully!")
    print("🌐 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

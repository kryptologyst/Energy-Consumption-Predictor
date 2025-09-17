"""
Mock Database System for Energy Consumption Data
Provides SQLite-based storage and retrieval for energy consumption records
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

class EnergyDatabase:
    """
    Mock database for storing energy consumption data and predictions
    """
    
    def __init__(self, db_path="energy_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Energy consumption table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS energy_consumption (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                energy_kwh REAL NOT NULL,
                temperature REAL,
                humidity REAL,
                weather_condition TEXT,
                is_weekend INTEGER,
                is_holiday INTEGER,
                building_id TEXT DEFAULT 'main',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                predicted_energy_kwh REAL NOT NULL,
                actual_energy_kwh REAL,
                model_name TEXT NOT NULL,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                r2_score REAL,
                rmse REAL,
                mae REAL,
                training_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT
            )
        ''')
        
        # Weather data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                solar_radiation REAL,
                weather_condition TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_energy_data(self, df):
        """Insert energy consumption data from DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        # Prepare data for insertion
        data_to_insert = []
        for idx, row in df.iterrows():
            data_to_insert.append((
                idx.strftime('%Y-%m-%d %H:%M:%S'),
                row['energy_consumption'],
                row.get('temperature', None),
                row.get('humidity', None),
                row.get('weather_condition', 'clear'),
                row.get('is_weekend', 0),
                row.get('is_holiday', 0),
                'main'
            ))
        
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT INTO energy_consumption 
            (timestamp, energy_kwh, temperature, humidity, weather_condition, is_weekend, is_holiday, building_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert)
        
        conn.commit()
        conn.close()
        
        return len(data_to_insert)
    
    def get_energy_data(self, start_date=None, end_date=None, building_id='main'):
        """Retrieve energy consumption data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT timestamp, energy_kwh, temperature, humidity, weather_condition, 
                   is_weekend, is_holiday
            FROM energy_consumption 
            WHERE building_id = ?
        '''
        params = [building_id]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp'
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        return df
    
    def insert_predictions(self, predictions_data):
        """Insert prediction results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executemany('''
            INSERT INTO predictions 
            (timestamp, predicted_energy_kwh, actual_energy_kwh, model_name, confidence_interval_lower, confidence_interval_upper)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', predictions_data)
        
        conn.commit()
        conn.close()
    
    def insert_model_performance(self, model_name, metrics, parameters=None):
        """Store model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        params_json = json.dumps(parameters) if parameters else None
        
        cursor.execute('''
            INSERT INTO model_performance 
            (model_name, r2_score, rmse, mae, parameters)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, metrics.get('R²'), metrics.get('RMSE'), metrics.get('MAE'), params_json))
        
        conn.commit()
        conn.close()
    
    def get_best_model(self):
        """Get the best performing model"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT model_name, r2_score, rmse, mae, training_date
            FROM model_performance 
            ORDER BY r2_score DESC, rmse ASC
            LIMIT 1
        '''
        
        result = pd.read_sql_query(query, conn)
        conn.close()
        
        return result.iloc[0] if not result.empty else None
    
    def generate_mock_weather_data(self, start_date, end_date):
        """Generate realistic weather data for the given period"""
        date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        weather_data = []
        np.random.seed(42)
        
        for timestamp in date_range:
            # Seasonal temperature variation
            day_of_year = timestamp.timetuple().tm_yday
            base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25)
            
            # Daily temperature variation
            hour_temp_variation = 5 * np.sin(2 * np.pi * timestamp.hour / 24)
            
            # Random variation
            temp = base_temp + hour_temp_variation + np.random.normal(0, 3)
            
            # Humidity (inversely related to temperature)
            humidity = max(30, min(90, 70 - (temp - 20) * 1.5 + np.random.normal(0, 10)))
            
            # Wind speed
            wind_speed = max(0, np.random.exponential(5))
            
            # Solar radiation (depends on hour and season)
            if 6 <= timestamp.hour <= 18:
                solar_base = 800 * np.sin(np.pi * (timestamp.hour - 6) / 12)
                seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)
                solar_radiation = solar_base * seasonal_factor * (1 + np.random.normal(0, 0.2))
            else:
                solar_radiation = 0
            
            # Weather condition
            conditions = ['clear', 'partly_cloudy', 'cloudy', 'rainy', 'stormy']
            weights = [0.4, 0.3, 0.2, 0.08, 0.02]
            weather_condition = np.random.choice(conditions, p=weights)
            
            weather_data.append({
                'timestamp': timestamp,
                'temperature': round(temp, 2),
                'humidity': round(humidity, 2),
                'wind_speed': round(wind_speed, 2),
                'solar_radiation': round(max(0, solar_radiation), 2),
                'weather_condition': weather_condition
            })
        
        return pd.DataFrame(weather_data).set_index('timestamp')
    
    def populate_sample_data(self, days=365):
        """Populate database with sample data"""
        print("🗄️ Populating database with sample data...")
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Generate weather data
        weather_df = self.generate_mock_weather_data(start_date, end_date)
        
        # Insert weather data
        conn = sqlite3.connect(self.db_path)
        weather_data_list = []
        for idx, row in weather_df.iterrows():
            weather_data_list.append((
                idx.strftime('%Y-%m-%d %H:%M:%S'),
                row['temperature'],
                row['humidity'],
                row['wind_speed'],
                row['solar_radiation'],
                row['weather_condition']
            ))
        
        cursor = conn.cursor()
        cursor.executemany('''
            INSERT INTO weather_data 
            (timestamp, temperature, humidity, wind_speed, solar_radiation, weather_condition)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', weather_data_list)
        
        conn.commit()
        conn.close()
        
        # Generate energy consumption data based on weather
        energy_data = []
        np.random.seed(42)
        
        for idx, row in weather_df.iterrows():
            # Base consumption
            base_consumption = 50
            
            # Temperature effect (more energy for heating/cooling)
            temp_effect = 0.8 * abs(row['temperature'] - 20)
            
            # Time of day effect
            hour_effect = 15 * np.sin(2 * np.pi * idx.hour / 24 + np.pi/2)
            
            # Day of week effect
            weekday_effect = 5 if idx.weekday() < 5 else -3
            
            # Weather condition effect
            weather_effects = {
                'clear': 0,
                'partly_cloudy': 2,
                'cloudy': 5,
                'rainy': 8,
                'stormy': 15
            }
            weather_effect = weather_effects.get(row['weather_condition'], 0)
            
            # Solar radiation effect (less energy needed during high solar)
            solar_effect = -row['solar_radiation'] / 100
            
            # Random noise
            noise = np.random.normal(0, 3)
            
            # Calculate total consumption
            total_consumption = (base_consumption + temp_effect + hour_effect + 
                               weekday_effect + weather_effect + solar_effect + noise)
            
            # Ensure positive values
            total_consumption = max(10, total_consumption)
            
            energy_data.append({
                'timestamp': idx,
                'energy_consumption': round(total_consumption, 2),
                'temperature': row['temperature'],
                'humidity': row['humidity'],
                'weather_condition': row['weather_condition'],
                'is_weekend': int(idx.weekday() >= 5),
                'is_holiday': int(np.random.random() < 0.03)  # 3% chance of holiday
            })
        
        energy_df = pd.DataFrame(energy_data).set_index('timestamp')
        
        # Insert energy data
        records_inserted = self.insert_energy_data(energy_df)
        
        print(f"✅ Inserted {records_inserted} energy consumption records")
        print(f"✅ Inserted {len(weather_data_list)} weather records")
        
        return energy_df
    
    def get_database_stats(self):
        """Get statistics about the database"""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Energy consumption stats
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM energy_consumption')
        stats['energy_records'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM energy_consumption')
        date_range = cursor.fetchone()
        stats['date_range'] = date_range
        
        cursor.execute('SELECT AVG(energy_kwh), MIN(energy_kwh), MAX(energy_kwh) FROM energy_consumption')
        energy_stats = cursor.fetchone()
        stats['energy_stats'] = {
            'avg': round(energy_stats[0], 2) if energy_stats[0] else 0,
            'min': round(energy_stats[1], 2) if energy_stats[1] else 0,
            'max': round(energy_stats[2], 2) if energy_stats[2] else 0
        }
        
        # Weather data stats
        cursor.execute('SELECT COUNT(*) FROM weather_data')
        stats['weather_records'] = cursor.fetchone()[0]
        
        # Model performance stats
        cursor.execute('SELECT COUNT(*) FROM model_performance')
        stats['model_records'] = cursor.fetchone()[0]
        
        # Predictions stats
        cursor.execute('SELECT COUNT(*) FROM predictions')
        stats['prediction_records'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
    
    def export_data(self, table_name, filename=None):
        """Export table data to CSV"""
        if filename is None:
            filename = f"{table_name}_export.csv"
        
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        
        df.to_csv(filename, index=False)
        return filename
    
    def close(self):
        """Clean up database connection"""
        # SQLite connections are closed after each operation
        pass

def main():
    """Test the database functionality"""
    print("🗄️ Energy Database System - Testing")
    print("=" * 40)
    
    # Initialize database
    db = EnergyDatabase()
    
    # Populate with sample data
    sample_data = db.populate_sample_data(days=30)  # 30 days of data
    
    # Get database statistics
    stats = db.get_database_stats()
    
    print("\n📊 Database Statistics:")
    print(f"Energy Records: {stats['energy_records']}")
    print(f"Weather Records: {stats['weather_records']}")
    print(f"Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Energy Stats: Avg={stats['energy_stats']['avg']} kWh, "
          f"Min={stats['energy_stats']['min']} kWh, Max={stats['energy_stats']['max']} kWh")
    
    # Test data retrieval
    print("\n🔍 Testing data retrieval...")
    recent_data = db.get_energy_data(start_date=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
    print(f"Retrieved {len(recent_data)} records from last 7 days")
    
    # Show sample data
    print("\n📋 Sample Energy Data (last 5 records):")
    print(recent_data.tail().to_string())
    
    return db

if __name__ == "__main__":
    database = main()

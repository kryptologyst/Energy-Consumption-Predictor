# Energy Consumption Predictor

A modern, comprehensive energy consumption forecasting system using advanced machine learning techniques and real-time web dashboard.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

### Advanced Machine Learning
- **Multiple ML Models**: Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Comprehensive Feature Engineering**: Lag features, rolling statistics, time-based features, weather correlations
- **Automated Model Selection**: Cross-validation and performance comparison
- **Real-time Predictions**: Generate forecasts for 6-72 hours ahead

### Interactive Web Dashboard
- **Real-time Monitoring**: Live energy consumption tracking
- **Interactive Charts**: Plotly-powered visualizations
- **Model Performance Analytics**: Compare different ML models
- **Feature Importance Analysis**: Understand what drives energy consumption
- **Weather Correlation**: Analyze impact of temperature and weather conditions

### Database Integration
- **SQLite Database**: Efficient data storage and retrieval
- **Mock Data Generation**: Realistic energy and weather data simulation
- **Data Export**: CSV export functionality
- **Performance Tracking**: Store and compare model metrics

### Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Bootstrap 5**: Modern, clean interface
- **Real-time Updates**: Auto-refreshing dashboard
- **Interactive Elements**: Smooth animations and transitions

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/energy-consumption-predictor.git
cd energy-consumption-predictor
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the dashboard**
Open your browser and navigate to `http://localhost:5000`

## 📁 Project Structure

```
energy-consumption-predictor/
├── energy_predictor.py      # Core ML prediction engine
├── database.py              # Database management and mock data
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── templates/
│   └── index.html          # Main dashboard template
├── static/
│   ├── css/
│   │   └── style.css       # Custom styling
│   └── js/
│       └── app.js          # Frontend JavaScript
└── README.md               # This file
```

## 🔧 Usage

### Command Line Interface

**Basic prediction with enhanced features:**
```python
from energy_predictor import EnergyConsumptionPredictor

# Initialize predictor
predictor = EnergyConsumptionPredictor()

# Generate sample data
df = predictor.generate_enhanced_data(periods=8760)  # 1 year

# Create advanced features
df_features = predictor.create_advanced_features(df)

# Train models
X_train, X_test, y_train, y_test = predictor.prepare_data(df_features)
results = predictor.train_models(X_train, y_train)

# Evaluate and get best model
evaluation = predictor.evaluate_models(X_test, y_test, results)
print(f"Best model: {predictor.best_model}")

# Generate 24-hour forecast
forecast = predictor.generate_forecast(df_features, steps=24)
```

### Web Dashboard

1. **Dashboard Overview**: Real-time energy consumption metrics and trends
2. **Predictions**: Generate custom forecasts (6-72 hours)
3. **Analytics**: Model performance comparison and feature importance
4. **Data Export**: Download consumption data as CSV

### Database Operations

```python
from database import EnergyDatabase

# Initialize database
db = EnergyDatabase()

# Populate with sample data
db.populate_sample_data(days=365)

# Retrieve data
energy_data = db.get_energy_data(start_date='2023-01-01')

# Get statistics
stats = db.get_database_stats()
```

## Machine Learning Models

### Implemented Algorithms
1. **Linear Regression**: Baseline model for comparison
2. **Ridge Regression**: L2 regularization for overfitting prevention
3. **Lasso Regression**: L1 regularization with feature selection
4. **Random Forest**: Ensemble method with feature importance
5. **Gradient Boosting**: Sequential ensemble learning
6. **XGBoost**: Optimized gradient boosting framework
7. **LightGBM**: Fast gradient boosting with categorical features

### Feature Engineering
- **Lag Features**: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h (1 week)
- **Rolling Statistics**: Mean, std, min, max over multiple windows
- **Time Features**: Hour, day of week, month (cyclical encoding)
- **Weather Features**: Temperature, humidity, weather conditions
- **Interaction Features**: Temperature × hour, temperature × weekend

### Model Selection
- **Cross-validation**: 5-fold time series cross-validation
- **Metrics**: R² score, RMSE, MAE
- **Automatic Selection**: Best model based on validation performance

## Performance Metrics

The system tracks multiple performance metrics:

- **R² Score**: Coefficient of determination (higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)
- **Cross-validation Score**: Average performance across folds

Typical performance on simulated data:
- **Best Model**: Usually XGBoost or LightGBM
- **R² Score**: 0.85-0.95
- **RMSE**: 2-5 kWh
- **MAE**: 1.5-3 kWh

## Web API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/dashboard_data` | GET | Dashboard statistics and recent data |
| `/api/predict` | POST | Generate energy predictions |
| `/api/model_performance` | GET | Model comparison metrics |
| `/api/feature_importance` | GET | Feature importance analysis |
| `/api/weather_correlation` | GET | Weather correlation analysis |
| `/api/export_data` | GET | Export data as CSV |

## Future Enhancements

### Planned Features
- [ ] **Real-time Data Integration**: Connect to actual smart meters
- [ ] **Advanced Weather API**: Integration with weather services
- [ ] **Deep Learning Models**: LSTM, GRU for time series
- [ ] **Anomaly Detection**: Identify unusual consumption patterns
- [ ] **Multi-building Support**: Manage multiple locations
- [ ] **Energy Optimization**: Recommendations for reducing consumption
- [ ] **Mobile App**: Native iOS/Android applications
- [ ] **Docker Deployment**: Containerized deployment options

### Technical Improvements
- [ ] **Caching Layer**: Redis for improved performance
- [ ] **Background Tasks**: Celery for async processing
- [ ] **Authentication**: User management and security
- [ ] **API Rate Limiting**: Prevent abuse
- [ ] **Monitoring**: Application performance monitoring
- [ ] **Testing**: Comprehensive test suite

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/energy-consumption-predictor.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest

# Format code
black .
```

## Sample Data

The system includes realistic mock data generation:

- **Energy Consumption**: Seasonal, daily, and weekly patterns
- **Weather Data**: Temperature, humidity, wind speed, solar radiation
- **Time Features**: Holidays, weekends, business hours
- **Correlations**: Realistic relationships between variables

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=development
export DATABASE_URL=sqlite:///energy_data.db
export SECRET_KEY=your-secret-key
```

### Database Configuration
- **Default**: SQLite (energy_data.db)
- **Production**: PostgreSQL recommended
- **Backup**: Automated daily backups

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

**2. Database Issues**
```bash
# Delete and recreate database
rm energy_data.db
python database.py
```

**3. Port Already in Use**
```bash
# Change port in app.py
app.run(port=5001)
```

### Performance Optimization

- **Large Datasets**: Use data sampling for faster training
- **Memory Issues**: Reduce feature window sizes
- **Slow Predictions**: Cache model predictions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **scikit-learn**: Machine learning library
- **Flask**: Web framework
- **Plotly**: Interactive visualizations
- **Bootstrap**: UI framework
- **XGBoost & LightGBM**: Gradient boosting frameworks

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/energy-consumption-predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/energy-consumption-predictor/discussions)
- **Email**: your.email@example.com

# Energy-Consumption-Predictor

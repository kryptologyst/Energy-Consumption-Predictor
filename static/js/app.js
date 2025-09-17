// Energy Consumption Predictor - Frontend JavaScript

let dashboardData = null;
let predictionChart = null;
let modelPerformanceChart = null;
let featureImportanceChart = null;
let hourlyPatternChart = null;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    showLoadingModal();
    loadDashboardData();
    loadModelPerformance();
    loadFeatureImportance();
    loadWeatherCorrelation();
});

// Show loading modal
function showLoadingModal() {
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
}

// Hide loading modal
function hideLoadingModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadingModal'));
    if (modal) {
        modal.hide();
    }
}

// Load dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard_data');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        dashboardData = data;
        updateStatsCards(data.stats);
        updateBestModelInfo(data.best_model);
        createMainChart(data);
        
        hideLoadingModal();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        hideLoadingModal();
        showAlert('Error loading dashboard data: ' + error.message, 'danger');
    }
}

// Update statistics cards
function updateStatsCards(stats) {
    document.getElementById('current-usage').textContent = stats.current_consumption;
    document.getElementById('avg-usage').textContent = stats.avg_consumption;
    document.getElementById('peak-usage').textContent = stats.max_consumption;
}

// Update best model information
function updateBestModelInfo(bestModel) {
    if (bestModel) {
        document.getElementById('best-model').textContent = bestModel.model_name;
        document.getElementById('model-score').textContent = `R² Score: ${bestModel.r2_score.toFixed(3)}`;
    } else {
        document.getElementById('best-model').textContent = 'Not Available';
        document.getElementById('model-score').textContent = 'Training required';
    }
}

// Create main energy consumption chart
function createMainChart(data) {
    const trace1 = {
        x: data.timestamps,
        y: data.energy_values,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Energy Consumption',
        line: {
            color: '#667eea',
            width: 3
        },
        marker: {
            size: 6,
            color: '#667eea'
        }
    };
    
    const trace2 = {
        x: data.timestamps,
        y: data.temperature_values,
        type: 'scatter',
        mode: 'lines',
        name: 'Temperature',
        yaxis: 'y2',
        line: {
            color: '#f093fb',
            width: 2
        }
    };
    
    const layout = {
        title: {
            text: 'Energy Consumption vs Temperature',
            font: { size: 18, color: '#2c3e50' }
        },
        xaxis: {
            title: 'Time',
            gridcolor: '#f0f0f0'
        },
        yaxis: {
            title: 'Energy Consumption (kWh)',
            gridcolor: '#f0f0f0'
        },
        yaxis2: {
            title: 'Temperature (°C)',
            overlaying: 'y',
            side: 'right',
            gridcolor: '#f0f0f0'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { t: 50, r: 80, b: 50, l: 80 },
        hovermode: 'x unified'
    };
    
    Plotly.newPlot('main-chart', [trace1, trace2], layout, {responsive: true});
}

// Generate predictions
async function generatePredictions() {
    const hours = document.getElementById('prediction-hours').value;
    const loadingDiv = document.getElementById('prediction-loading');
    const chartDiv = document.getElementById('prediction-chart');
    
    // Show loading
    loadingDiv.classList.remove('d-none');
    chartDiv.innerHTML = '';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ hours: parseInt(hours) })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        createPredictionChart(data);
        loadingDiv.classList.add('d-none');
        
    } catch (error) {
        console.error('Error generating predictions:', error);
        loadingDiv.classList.add('d-none');
        showAlert('Error generating predictions: ' + error.message, 'danger');
    }
}

// Create prediction chart
function createPredictionChart(data) {
    const trace = {
        x: data.timestamps,
        y: data.predictions,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Predicted Energy Consumption',
        line: {
            color: '#28a745',
            width: 3
        },
        marker: {
            size: 6,
            color: '#28a745'
        }
    };
    
    const layout = {
        title: {
            text: `Energy Forecast - ${data.model_used}`,
            font: { size: 16, color: '#2c3e50' }
        },
        xaxis: {
            title: 'Time',
            gridcolor: '#f0f0f0'
        },
        yaxis: {
            title: 'Energy Consumption (kWh)',
            gridcolor: '#f0f0f0'
        },
        plot_bgcolor: 'white',
        paper_bgcolor: 'white',
        margin: { t: 50, r: 50, b: 50, l: 80 }
    };
    
    Plotly.newPlot('prediction-chart', [trace], layout, {responsive: true});
}

// Load model performance data
async function loadModelPerformance() {
    try {
        const response = await fetch('/api/model_performance');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        createModelPerformanceChart(data.models);
        
    } catch (error) {
        console.error('Error loading model performance:', error);
    }
}

// Create model performance chart
function createModelPerformanceChart(models) {
    const ctx = document.getElementById('model-performance-chart').getContext('2d');
    
    if (modelPerformanceChart) {
        modelPerformanceChart.destroy();
    }
    
    modelPerformanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models.map(m => m.name),
            datasets: [{
                label: 'R² Score',
                data: models.map(m => m.r2),
                backgroundColor: 'rgba(102, 126, 234, 0.8)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Load feature importance
async function loadFeatureImportance() {
    try {
        const response = await fetch('/api/feature_importance');
        const data = await response.json();
        
        if (data.error) {
            console.log('Feature importance not available:', data.error);
            return;
        }
        
        createFeatureImportanceChart(data);
        
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

// Create feature importance chart
function createFeatureImportanceChart(data) {
    const ctx = document.getElementById('feature-importance-chart').getContext('2d');
    
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    featureImportanceChart = new Chart(ctx, {
        type: 'horizontalBar',
        data: {
            labels: data.features,
            datasets: [{
                label: 'Importance',
                data: data.importances,
                backgroundColor: 'rgba(40, 167, 69, 0.8)',
                borderColor: 'rgba(40, 167, 69, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Load weather correlation
async function loadWeatherCorrelation() {
    try {
        const response = await fetch('/api/weather_correlation');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        updateCorrelationInfo(data.correlations);
        createHourlyPatternChart(data.hourly_pattern);
        
    } catch (error) {
        console.error('Error loading weather correlation:', error);
    }
}

// Update correlation information
function updateCorrelationInfo(correlations) {
    const correlationDiv = document.getElementById('correlation-info');
    let html = '<h6>Weather Correlations:</h6>';
    
    for (const [factor, value] of Object.entries(correlations)) {
        const absValue = Math.abs(value);
        let className = 'correlation-neutral';
        
        if (absValue > 0.5) {
            className = value > 0 ? 'correlation-positive' : 'correlation-negative';
        }
        
        html += `
            <div class="correlation-item">
                <span>${factor.charAt(0).toUpperCase() + factor.slice(1)}:</span>
                <span class="correlation-value ${className}">${value.toFixed(3)}</span>
            </div>
        `;
    }
    
    correlationDiv.innerHTML = html;
}

// Create hourly pattern chart
function createHourlyPatternChart(hourlyData) {
    const ctx = document.getElementById('hourly-pattern-chart').getContext('2d');
    
    if (hourlyPatternChart) {
        hourlyPatternChart.destroy();
    }
    
    hourlyPatternChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hourlyData.hours,
            datasets: [{
                label: 'Average Consumption',
                data: hourlyData.consumption,
                backgroundColor: 'rgba(255, 193, 7, 0.2)',
                borderColor: 'rgba(255, 193, 7, 1)',
                borderWidth: 2,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Energy (kWh)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// Export data
async function exportData() {
    try {
        const response = await fetch('/api/export_data');
        
        if (!response.ok) {
            throw new Error('Export failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'energy_consumption_data.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showAlert('Data exported successfully!', 'success');
        
    } catch (error) {
        console.error('Error exporting data:', error);
        showAlert('Error exporting data: ' + error.message, 'danger');
    }
}

// Show alert message
function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.top = '20px';
    alertDiv.style.right = '20px';
    alertDiv.style.zIndex = '9999';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.parentNode.removeChild(alertDiv);
        }
    }, 5000);
}

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Auto-refresh dashboard data every 5 minutes
setInterval(() => {
    loadDashboardData();
}, 300000);

// Responsive chart resizing
window.addEventListener('resize', () => {
    if (predictionChart) {
        Plotly.Plots.resize('prediction-chart');
    }
    Plotly.Plots.resize('main-chart');
});

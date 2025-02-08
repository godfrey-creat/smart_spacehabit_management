# utils/habitat_utils.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def simulate_sensor_data(num_astronauts, simulation_time):
    """Simulate sensor data for the habitat."""
    time = np.arange(0, simulation_time, 0.1)
    
    # Simulate oxygen levels
    oxygen_level = 20 + np.sin(time) * 0.5  # Normal range: 19.5% to 20.5%
    
    # Simulate temperature
    temperature = 22 + np.cos(time) * 3  # Normal range: 19°C to 25°C
    
    # Simulate water usage
    water_usage = np.random.uniform(1, 5, len(time))  # Random usage between 1 and 5 liters
    
    # Simulate CO2 levels
    co2_level = 0.04 + np.sin(time) * 0.02  # Normal range: 0.03% to 0.05%
    
    # Simulate humidity
    humidity = 50 + np.cos(time) * 10  # Normal range: 40% to 60%
    
    # Simulate power usage
    power_usage = num_astronauts * 100 + np.random.uniform(-10, 10, len(time))  # Power in watts
    
    # Create a DataFrame to store all data
    data = pd.DataFrame({
        "Time": time,
        "Oxygen Level": oxygen_level,
        "Temperature": temperature,
        "Water Usage": water_usage,
        "CO2 Level": co2_level,
        "Humidity": humidity,
        "Power Usage": power_usage,
    })
    return data

def check_health_status(data):
    """Check astronaut health and habitat conditions."""
    avg_oxygen = data["Oxygen Level"].mean()
    avg_temp = data["Temperature"].mean()
    avg_co2 = data["CO2 Level"].mean()
    avg_humidity = data["Humidity"].mean()
    
    health_status = []
    
    # Oxygen check
    if avg_oxygen < 19.5:
        health_status.append("🚨 Warning: Low oxygen levels detected. Provide additional oxygen supply.")
    elif avg_oxygen > 20.5:
        health_status.append("🚨 Warning: High oxygen levels detected. Check oxygen system.")
    
    # Temperature check
    if avg_temp < 19:
        health_status.append("🚨 Warning: Low temperature detected. Activate heating systems.")
    elif avg_temp > 25:
        health_status.append("🚨 Warning: High temperature detected. Activate cooling systems.")
    
    # CO2 check
    if avg_co2 > 0.05:
        health_status.append("🚨 Warning: High CO2 levels detected. Activate CO2 scrubbers.")
    
    # Humidity check
    if avg_humidity < 40:
        health_status.append("🚨 Warning: Low humidity detected. Increase humidity levels.")
    elif avg_humidity > 60:
        health_status.append("🚨 Warning: High humidity detected. Decrease humidity levels.")
    
    # If everything is normal
    if not health_status:
        health_status.append("✅ All systems normal. Astronauts are safe and healthy.")
    
    return "\n".join(health_status)

def predict_failures(data):
    """Predict potential system failures using ML."""
    # Features: Time, Oxygen Level, Temperature, Water Usage, CO2 Level, Humidity, Power Usage
    X = data[["Time", "Oxygen Level", "Temperature", "Water Usage", "CO2 Level", "Humidity", "Power Usage"]]
    
    # Target: Simulate failure risk (higher values mean higher risk)
    y = np.random.uniform(0, 1, len(data))  # Random failure risk for demonstration
    
    # Train a Random Forest model
    model = RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict failure risk
    predictions = model.predict(X_test)
    
    # Calculate error (for demonstration)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error: {mse}")
    
    # Return predictions
    return predictions

def detect_anomalies(data):
    """Detect anomalies in sensor data."""
    # Calculate mean and standard deviation for each sensor
    mean = data.mean()
    std = data.std()
    
    # Detect anomalies (data points outside 2 standard deviations)
    anomalies = data[(np.abs(data - mean) > 2 * std).any(axis=1)]
    return anomalies
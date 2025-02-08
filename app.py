# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
#from utils.habitat_utils import simulate_sensor_data, check_health_status, predict_failures, detect_anomalies
from utils.habitat_utils1 import simulate_sensor_data, check_health_status, predict_failures, detect_anomalies

# Page configuration
st.set_page_config(page_title="Smart Space Habitat Manager", layout="wide")

# Title
st.title("🏠🚀 Smart Space Habitat Manager")
st.markdown("**AI-powered system for managing space habitats.**")

# Sidebar for user input
st.sidebar.header("Settings")
num_astronauts = st.sidebar.slider("Number of Astronauts", 1, 10, 4)
simulation_time = st.sidebar.slider("Simulation Time (hours)", 1, 24, 12)

# Simulate sensor data
sensor_data = simulate_sensor_data(num_astronauts, simulation_time)

# Display sensor data
st.header("📊 Habitat Sensor Data")
st.dataframe(sensor_data)

# Visualize oxygen levels
st.subheader("Oxygen Levels Over Time")
fig_oxygen = px.line(sensor_data, x="Time", y="Oxygen Level", title="Oxygen Level Monitoring")
st.plotly_chart(fig_oxygen)

# Visualize temperature
st.subheader("Temperature Over Time")
fig_temp = px.line(sensor_data, x="Time", y="Temperature", title="Temperature Monitoring")
st.plotly_chart(fig_temp)

# Visualize CO2 levels
st.subheader("CO2 Levels Over Time")
fig_co2 = px.line(sensor_data, x="Time", y="CO2 Level", title="CO2 Level Monitoring")
st.plotly_chart(fig_co2)

# Visualize humidity
st.subheader("Humidity Over Time")
fig_humidity = px.line(sensor_data, x="Time", y="Humidity", title="Humidity Monitoring")
st.plotly_chart(fig_humidity)

# Visualize power usage
st.subheader("Power Usage Over Time")
fig_power = px.line(sensor_data, x="Time", y="Power Usage", title="Power Usage Monitoring")
st.plotly_chart(fig_power)

# Health Monitoring
st.header("❤️ Astronaut Health Monitoring")
health_status = check_health_status(sensor_data)
st.write(health_status)

# Predictive Maintenance
st.header("🛠️ Predictive Maintenance")
failure_predictions = predict_failures(sensor_data)
st.write("Failure Risk Predictions:")
st.write(failure_predictions)

# Anomaly Detection
st.header("🚨 Anomaly Detection")
anomalies = detect_anomalies(sensor_data)
st.write("Detected Anomalies:")
st.write(anomalies)

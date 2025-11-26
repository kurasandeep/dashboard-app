# dashboard_app.py - Final, Stable, Error-Free Code for Streamlit Deployment

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# FIX 1: Import 'time' for robust date conversion
from datetime import datetime, timedelta, time 
# Libraries for Data Acquisition and Modeling
from meteostat import Point, Daily
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 0. CONFIGURATION AND CONSTANTS ---

# Define mapping for selected cities to their coordinates (Lat, Lon, Elevation, Name)
CITY_MAP = {
    "New York": (40.71, -74.01, 10, "New York City"),
    "London": (51.5, -0.12, 25, "London, UK"),
    "Sydney": (33.86, 151.2, 39, "Sydney, Australia"),
    "Tokyo": (35.68, 139.75, 40, "Tokyo, Japan"),
    "Miami": (25.76, -80.19, 2, "Miami, USA"),
    "Berlin": (52.52, 13.4, 34, "Berlin, Germany"),
}
DEFAULT_CITY = "New York"
DAYS_TO_ANALYZE = 365
NUM_TRACKS = 5 
END_DATE = datetime.now().date() 
START_DATE = END_DATE - timedelta(days=DAYS_TO_ANALYZE)

# Features used by the Random Forest Model
FEATURES = [
    'popularity_lag_1', 'energy', 'valence', 'tempo', 'danceability', 
    'tavg', 'prcp', 'daylight_hours', 
    'month_sin', 'month_cos', 
    'is_weekend'
]

# --- SIMULATION FUNCTION (STEP 1) ---

def get_simulated_spotify_data(end_date, days, tracks, location_name):
    """
    Simulates daily chart data and audio features for multiple tracks over a year, 
    with a seasonal popularity bias using a sine wave.
    """
    date_range = [end_date - timedelta(days=d) for d in range(days)]
    data = []

    track_features = {
        f'track_{i}': {
            'energy': np.random.uniform(0.3, 0.9), 
            'valence': np.random.uniform(0.2, 0.8), 
            'tempo': np.random.uniform(90, 150),
            'danceability': np.random.uniform(0.4, 0.8),
        } for i in range(1, tracks + 1)
    }

    for date in date_range:
        for track_id, features in track_features.items():
            # Seasonal Sine Wave Component 

#[Image of sine cosine wave plot]

            base_pop = 50 + 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
            popularity = np.clip(base_pop + np.random.normal(0, 5), 10, 100)

            record = {
                'date': date,  # Fixed: date is already a datetime.date object
                'track_id': track_id,
                'location': location_name,
                'popularity': int(popularity),
                **features
            }
            data.append(record)

    return pd.DataFrame(data)

# --- 1. DATA ACQUISITION & INTEGRATION (STEPS 1-3) ---

@st.cache_data(show_spinner="1. Acquiring Weather and Simulating Music Data...")
def get_integrated_data(lat, lon, elevation, location_name, start_date, end_date):
    """Acquires, simulates, and integrates all project data."""
    
    # Internal function to fetch weather data from Meteostat
    def get_meteostat_weather_data(lat, lon, start, end, location_name, elevation):
        
        # FIX 2: Convert input datetime.date objects to datetime.datetime 
        # objects before passing to Daily() to prevent internal Meteostat 
        # TypeError when checking cache age (datetime - date).
        start_dt = datetime.combine(start, time(0, 0))
        end_dt = datetime.combine(end, time(0, 0))
        
        location = Point(lat, lon, elevation)
        # Pass the fully qualified datetime objects
        data = Daily(location, start_dt, end_dt) 
        weather_df = data.fetch()
        weather_df['location'] = location_name
        
        weather_df = weather_df.reset_index().rename(columns={'time': 'date'})
        weather_df = weather_df[['date', 'location', 'tavg', 'prcp', 'tsun']].copy()
        weather_df['daylight_hours'] = weather_df['tsun'].fillna(0) / 60 
        weather_df.drop(columns=['tsun'], inplace=True)
        return weather_df.dropna(subset=['tavg'])

    days_to_analyze = (end_date - start_date).days + 1
    
    # 1. Get Simulated Spotify Data
    music_df = get_simulated_spotify_data(end_date, days_to_analyze, NUM_TRACKS, location_name)
    
    # 2. Get Meteostat Weather Data (Dynamic Location)
    weather_df = get_meteostat_weather_data(lat, lon, start_date, end_date, location_name, elevation)

    # 3. Merge and Feature Engineering
    
    # Standardize to Pandas datetime objects (datetime64)
    music_df['date'] = pd.to_datetime(music_df['date']).dt.normalize()
    weather_df['date'] = pd.to_datetime(weather_df['date']).dt.normalize()
    
    master_df = pd.merge(music_df, weather_df, on=['date', 'location'], how='inner')

    # Time Features (Accessing date components from datetime objects)
    master_df['month'] = master_df['date'].dt.month
    master_df['day_of_week'] = master_df['date'].dt.dayofweek
    master_df['is_weekend'] = master_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Cyclical Features
    master_df['month_sin'] = np.sin(2 * np.pi * master_df['month'] / 12)
    master_df['month_cos'] = np.cos(2 * np.pi * master_df['month'] / 12)
    
    # Lagged Popularity
    master_df = master_df.sort_values(by=['track_id', 'date']).reset_index(drop=True)
    master_df['popularity_lag_1'] = master_df.groupby('track_id')['popularity'].shift(1)
    master_df.dropna(subset=['popularity_lag_1'], inplace=True)
    
    return master_df

# --- 2. MODEL TRAINING (STEP 4) ---

@st.cache_resource(show_spinner="2. Training Random Forest Regressor...")
def train_model(data_df):
    """Trains the Random Forest Regressor and extracts validation data."""
    
    X = data_df[FEATURES]
    y = data_df['popularity']

    # Chronological Split (80% Train, 20% Test)
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    # Train Model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict and Validate
    y_pred = model.predict(X_test)
    
    validation_df = pd.DataFrame({
        # Safely convert datetime64 to date object for display
        'date': data_df['date'].dt.date.tail(len(y_test)), 
        'actual_popularity': y_test,
        'predicted_popularity': y_pred
    })
    
    return model, validation_df

# --- 3. FORECASTING (STEP 7) ---

@st.cache_data(show_spinner="3. Generating 30-Day Forecast...")
def generate_forecast(model, data_df, days=30):
    """Generates a forward 30-day forecast based on the trained model."""
    
    last_date = data_df['date'].max() # Returns a timestamp (datetime64)
    # Convert last_date to a date object for clean addition with timedelta
    future_dates = [last_date.date() + timedelta(days=d) for d in range(1, days + 1)] 
    forecast_df = pd.DataFrame({'date': future_dates})

    # Prepare stats for simulating future weather/lag
    seasonal_stats = data_df[['tavg', 'prcp', 'daylight_hours']].mean() 

    # Simulate Future Features
    forecast_df['tavg'] = seasonal_stats['tavg'] + np.random.normal(0, 3, days)
    forecast_df['prcp'] = seasonal_stats['prcp'] * np.random.uniform(0.5, 1.5, days)
    forecast_df['daylight_hours'] = seasonal_stats['daylight_hours'] + np.random.normal(0, 1, days)

    # Feature Engineering for Forecast
    # date column is Python date object here, so use .apply()
    forecast_df['month'] = forecast_df['date'].apply(lambda x: x.month)
    forecast_df['is_weekend'] = forecast_df['date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)
    forecast_df['month_sin'] = np.sin(2 * np.pi * forecast_df['month'] / 12)
    forecast_df['month_cos'] = np.cos(2 * np.pi * forecast_df['month'] / 12)

    # Lagged Popularity (Using the mean of the training set)
    last_lag_value = data_df['popularity_lag_1'].mean()
    forecast_df['popularity_lag_1'] = last_lag_value

    # Song Attributes (Average values from the training set)
    for feature in ['energy', 'valence', 'tempo', 'danceability']:
        forecast_df[feature] = data_df[feature].mean()

    # Make Forecast
    X_forecast = forecast_df[FEATURES]
    forecast_predictions = _model.predict(X_forecast)

    forecast_df['predicted_popularity'] = np.round(forecast_predictions).astype(int)
    return forecast_df[['date', 'predicted_popularity', 'tavg', 'daylight_hours', 'month']]


# --- 4. STREAMLIT APPLICATION UI & VISUALIZATION (STEP 8) ---

st.set_page_config(layout="wide", page_title="Song Trend Forecasting 🎶")
st.title("🎶 Weather-Driven Song Trend Forecasting Dashboard")

# --- UI INPUT (Sidebar) ---
st.sidebar.header("Location & Analysis Settings")

selected_city = st.sidebar.selectbox(
    "Select a City to Analyze:",
    options=list(CITY_MAP.keys()),
    index=list(CITY_MAP.keys()).index(DEFAULT_CITY)
)

# Resolve coordinates based on user selection
lat, lon, elevation, location_name = CITY_MAP[selected_city]

# Removed redundant .date() call from END_DATE in the f-string
st.sidebar.markdown(f"""
    ---
    **Weather Data Source:** {location_name}  
    **Training Period:** 1 Year ({START_DATE} to {END_DATE}) 
""")
st.sidebar.info("The model trains on simulated daily song data, correlating popularity with the selected city's historical weather to predict future trends.")
# --- End UI Input ---

# Execution Order: Load Data -> Train Model -> Generate Forecast
master_df = get_integrated_data(lat, lon, elevation, location_name, START_DATE, END_DATE)
model, validation_df = train_model(master_df)
forecast_df = generate_forecast(model, master_df)

# Calculate final R2 metric for display
r2 = r2_score(validation_df['actual_popularity'], validation_df['predicted_popularity'])

# --- Dashboard Header ---
st.markdown(f"### Location: {location_name} | R² Score: {r2:.4f}")

# ----------------------------------------------------
## 1. Historical Model Validation
# ----------------------------------------------------
st.header("1. Historical Model Validation (Actual vs. Predicted)")

fig_validation = go.Figure()
fig_validation.add_trace(go.Scatter(x=validation_df['date'], y=validation_df['actual_popularity'],
                                    mode='lines', name='Actual Popularity (Simulated)', line=dict(color='red', width=3)))
fig_validation.add_trace(go.Scatter(x=validation_df['date'], y=validation_df['predicted_popularity'],
                                    mode='lines', name='Predicted Popularity', line=dict(color='blue', dash='dot', width=2)))

fig_validation.update_layout(xaxis_title='Date', yaxis_title='Popularity Score', hovermode="x unified", height=400)
st.plotly_chart(fig_validation, use_container_width=True)

# ----------------------------------------------------
## 2. 30-Day Forward Trend Forecast
# ----------------------------------------------------
st.header("2. 30-Day Forward Trend Forecast")

col1, col2 = st.columns(2)

with col1:
    # Chart A: Predicted Popularity
    fig_forecast_pop = go.Figure()
    fig_forecast_pop.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_popularity'],
                                           mode='lines+markers', name='Predicted Popularity', line=dict(color='red', width=3)))
    fig_forecast_pop.update_layout(title='Predicted Song Popularity Trend', xaxis_title='Date', yaxis_title='Popularity Score', hovermode="x unified", height=400)
    st.plotly_chart(fig_forecast_pop, use_container_width=True)

with col2:
    # Chart B: Forecasted Weather Drivers
    fig_weather = go.Figure()
    fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['tavg'],
                                     mode='lines', name='Avg. Temp (°C)', yaxis='y1', line=dict(color='orange', width=2)))
    
    fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['daylight_hours'],
                                     mode='lines', name='Daylight Hours', yaxis='y2', line=dict(color='skyblue', dash='dot', width=2)))

    fig_weather.update_layout(title='Forecasted Weather Drivers', xaxis_title='Date',
        yaxis=dict(title='Avg. Temp (°C)', color='orange'),
        yaxis2=dict(title='Daylight Hours', overlaying='y', side='right', color='skyblue'),
        hovermode="x unified", height=400)
    st.plotly_chart(fig_weather, use_container_width=True)

# ----------------------------------------------------
## 3. Seasonal Trends of Key Features
# ----------------------------------------------------
st.header("3. Seasonal Trends of Key Features")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Prepare data for seasonal visualization
seasonal_df = master_df.groupby(master_df['date'].dt.month)[['valence', 'energy', 'tavg']].mean().reset_index()
seasonal_df['Month'] = seasonal_df['date'].apply(lambda x: months[x - 1])

# Plot Valence (Mood) and Energy vs. Month
fig_seasonal = px.line(seasonal_df, x='Month', y=['valence', 'energy'], 
                       title=f'Average Song Valence & Energy vs. Temperature in {selected_city}',
                       labels={'value': 'Score (0-1)', 'Month': 'Month'},
                       color_discrete_map={'valence': 'green', 'energy': 'blue'})

# Add Temperature as a secondary axis for seasonal context
fig_seasonal.add_trace(go.Scatter(x=seasonal_df['Month'], y=seasonal_df['tavg'], 
                                  name='Avg. Temp (°C)', yaxis='y2', mode='lines', 
                                  line=dict(color='red', dash='dash')))

fig_seasonal.update_layout(
    yaxis=dict(title='Score (0-1)'),
    yaxis2=dict(title='Avg. Temp (°C)', overlaying='y', side='right', showgrid=False),
    hovermode="x unified"
)
st.plotly_chart(fig_seasonal, use_container_width=True)

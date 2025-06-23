import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
import holidays
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Gold Price Prediction - GOLDEN by Aniysah Fauziyyah Alfa",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (same as app_final.py)
st.markdown("""
<style>
    /* Main container styles */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header styles */
    .main-header {
        font-family: 'Helvetica', sans-serif;
        font-size: 38px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: #EFF6FF;
        border-left: 5px solid #BFDBFE;
    }
    
    /* Subheader styles */
    .sub-header {
        font-family: 'Helvetica', sans-serif;
        font-size: 24px;
        font-weight: bold;
        color: #1E40AF;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #BFDBFE;
    }
    
    /* Card styles */
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Success message */
    .success-box {
        background-color: #ECFDF5;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #10B981;
        margin-bottom: 20px;
    }
    
    /* Warning message */
    .warning-box {
        background-color: #FFFBEB;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 20px;
    }
    
    /* Error message */
    .error-box {
        background-color: #FEF2F2;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
        margin-bottom: 20px;
    }
    
    /* Info box */
    .info-box {
        background-color: #EFF6FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        padding: 10px 15px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #F5F7FF;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 14px;
        color: #4B5563;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1E40AF;
    }
    
    /* Dashboard card */
    .dashboard-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: 100%;
        display: flex;
        flex-direction: column;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .dashboard-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .dashboard-card-title {
        font-size: 18px;
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 15px;
        text-align: center;
    }
    .dashboard-card-icon {
        font-size: 48px;
        text-align: center;
        margin-bottom: 15px;
    }
    .dashboard-card-description {
        font-size: 14px;
        color: #4B5563;
        flex-grow: 1;
        text-align: center;
    }
    
    /* About card styling */
    .about-card {
        background-color: #F5F7FF;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    .about-avatar {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 20px;
        border: 3px solid #BFDBFE;
    }
    .about-details {
        flex-grow: 1;
    }
    .about-name {
        font-size: 24px;
        font-weight: bold;
        color: #1E40AF;
        margin-bottom: 5px;
    }
    .about-role {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 10px;
    }
    .about-description {
        font-size: 14px;
        color: #4B5563;
    }
    
    /* Download button */
    .download-button {
        display: inline-block;
        background-color: #3B82F6;
        color: white !important;
        text-decoration: none;
        padding: 8px 15px;
        border-radius: 6px;
        font-weight: bold;
        margin-top: 10px;
        transition: all 0.3s;
    }
    .download-button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #E5E7EB;
        font-size: 12px;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'xgb_model' not in st.session_state:
    st.session_state.xgb_model = None
if 'svr_model' not in st.session_state:
    st.session_state.svr_model = None
if 'xgb_scaler_X' not in st.session_state:
    st.session_state.xgb_scaler_X = None
if 'xgb_scaler_y' not in st.session_state:
    st.session_state.xgb_scaler_y = None
if 'svr_scaler_X' not in st.session_state:
    st.session_state.svr_scaler_X = None
if 'svr_scaler_y' not in st.session_state:
    st.session_state.svr_scaler_y = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'page' not in st.session_state:
    st.session_state.page = "Dashboard"
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None

# Function to preprocess data (same as Google Colab)
def load_and_preprocess_data(data):
    """Load and preprocess the gold price data - SAME AS COLAB"""
    st.text(f"📊 Loading and preprocessing data...")
    
    data_copy = data.copy()
    
    # Display basic info
    st.text(f"Dataset loaded with shape: {data_copy.shape}")
    st.text(f"Columns: {list(data_copy.columns)}")
    
    # Check required columns
    required_columns = ['Tanggal', 'Harga_Emas']
    for column in required_columns:
        if column not in data_copy.columns:
            st.error(f"Kolom {column} tidak ditemukan di data Anda!")
            return None
    
    # Convert 'Inflasi' from percentage string to float if needed
    if 'Inflasi' in data_copy.columns:
        if data_copy['Inflasi'].dtype == 'object':
            data_copy["Inflasi"] = data_copy["Inflasi"].str.replace("%", "").astype(float)
        # Convert percentage to decimal
        if data_copy['Inflasi'].max() > 1:  # If values are in percentage format
            data_copy['Inflasi'] /= 100
    
    # Convert 'Suku_Bunga' to decimal if needed
    if 'Suku_Bunga' in data_copy.columns:
        if data_copy['Suku_Bunga'].max() > 1:  # If values are in percentage format
            data_copy['Suku_Bunga'] /= 100
    
    # Convert 'Tanggal' to datetime format
    if 'Tanggal' in data_copy.columns:
        data_copy["Tanggal"] = pd.to_datetime(data_copy["Tanggal"], dayfirst=True)
        data_copy.set_index('Tanggal', inplace=True)
    
    st.text(f"Starting date: {data_copy.index[0]}")
    st.text(f"Ending date: {data_copy.index[-1]}")
    st.text(f"Duration: {data_copy.index[-1] - data_copy.index[0]}")
    
    # Handle missing values and holidays - SAME AS COLAB
    ind_holidays = holidays.Indonesia()
    
    # Add holiday column
    data_copy['is_holiday'] = data_copy.index.map(lambda x: 1 if x in ind_holidays else 0)
    
    # Remove holidays
    data_cleaned = data_copy[data_copy['is_holiday'] == 0].drop(columns=['is_holiday'])
    
    # Fill missing values
    data_filled = data_cleaned.ffill()
    
    st.text(f"Dataset after handling missing values and holidays:")
    st.text(f"Missing values: {data_filled.isna().sum().sum()}")
    st.text(f"Total data points: {len(data_filled)}")
    
    return data_filled

# Function to add technical features (same as Google Colab)
def add_technical_features(df):
    """Add technical features for both XGBoost and SVR - SAME AS COLAB"""
    df = df.copy()
    
    # Moving averages
    df['MA7'] = df['Harga_Emas'].rolling(window=7).mean()
    df['MA30'] = df['Harga_Emas'].rolling(window=30).mean()
    
    # Lag features
    df['Harga_Emas_Lag1'] = df['Harga_Emas'].shift(1)
    df['Harga_Emas_Lag7'] = df['Harga_Emas'].shift(7)
    
    # Volatility
    df['Volatility'] = df['Harga_Emas'].rolling(window=7).std()
    
    # Trend
    df['Trend'] = df['Harga_Emas'].diff()
    
    # Momentum
    df['Momentum'] = df['Harga_Emas'] - df['Harga_Emas'].shift(7)
    
    # Rate of Change (ROC)
    df['ROC'] = df['Harga_Emas'].pct_change(periods=7) * 100
    
    # Remove NaN values
    df = df.dropna()
    
    st.text(f"✅ Feature engineering completed. Shape: {df.shape}")
    return df

# Function to split data (same as Google Colab)
def split_data(data):
    """Split data into training and testing sets (time-based) - SAME AS COLAB"""
    # Sort data by date
    data_sorted = data.sort_index()
    
    # Use 90% for training, 10% for testing - SAME AS COLAB
    cutoff_point = int(len(data_sorted) * 0.9)
    train_data = data_sorted.iloc[:cutoff_point]
    test_data = data_sorted.iloc[cutoff_point:]
    
    st.text(f"📊 Data Split:")
    st.text(f"Train Data Range: {train_data.index.min()} to {train_data.index.max()}")
    st.text(f"Test Data Range: {test_data.index.min()} to {test_data.index.max()}")
    st.text(f"Train shape: {train_data.shape}")
    st.text(f"Test shape: {test_data.shape}")
    
    # Define features and target
    X_train = train_data.drop(columns=['Harga_Emas'])
    y_train = train_data['Harga_Emas']
    X_test = test_data.drop(columns=['Harga_Emas'])
    y_test = test_data['Harga_Emas']
    
    return X_train, X_test, y_train, y_test, data_sorted

# Function to evaluate model (modified - removed R²)
def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, model_name):
    """Evaluate model performance - SAME AS COLAB but without R²"""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Inverse transform
    y_train_original = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    y_train_pred_original = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_pred_original = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
    
    # Calculate metrics (removed R²)
    train_rmse = np.sqrt(mean_squared_error(y_train_original, y_train_pred_original))
    train_mae = mean_absolute_error(y_train_original, y_train_pred_original)
    train_mape = np.mean(np.abs((y_train_original - y_train_pred_original) / y_train_original)) * 100
    
    test_rmse = np.sqrt(mean_squared_error(y_test_original, y_test_pred_original))
    test_mae = mean_absolute_error(y_test_original, y_test_pred_original)
    test_mape = np.mean(np.abs((y_test_original - y_test_pred_original) / y_test_original)) * 100
    
    # Display results - SAME FORMAT AS COLAB but without R²
    st.text(f"📊 {model_name} Model Evaluation")
    st.text("-" * 60)
    st.text(f"📌 Training Set:")
    st.text(f"   - MAE   : {train_mae:.2f}")
    st.text(f"   - RMSE  : {train_rmse:.2f}")
    st.text(f"   - MAPE  : {train_mape:.2f}%")
    st.text("-" * 60)
    st.text(f"📌 Testing Set:")
    st.text(f"   - MAE   : {test_mae:.2f}")
    st.text(f"   - RMSE  : {test_rmse:.2f}")
    st.text(f"   - MAPE  : {test_mape:.2f}%")
    
    return {
        'model': model,
        'y_train_pred': y_train_pred_original,
        'y_test_pred': y_test_pred_original,
        'y_test_original': y_test_original,
        'train_metrics': {
            'rmse': train_rmse, 'mae': train_mae, 'mape': train_mape
        },
        'test_metrics': {
            'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape
        }
    }

# Function to train XGBoost model (same as Google Colab)
def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with Bayesian optimization - SAME AS COLAB"""
    st.text("🔄 Training XGBoost Model with Bayesian Optimization...")
    
    # Normalization - SAME AS COLAB
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Time Series Cross-Validation - SAME AS COLAB
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Parameter space - SAME AS COLAB
    xgb_param_space = {
        'n_estimators': Integer(100, 300),
        'max_depth': Integer(3, 8),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'subsample': Real(0.7, 1.0),
        'colsample_bytree': Real(0.7, 1.0),
        'gamma': Real(0, 0.5)
    }
    
    # Bayesian optimization - SAME AS COLAB
    xgb_bayes = BayesSearchCV(
        estimator=XGBRegressor(objective='reg:squarederror', random_state=42),
        search_spaces=xgb_param_space,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_absolute_percentage_error',
        verbose=0,  # Changed to 0 for Streamlit
        n_jobs=-1,
        random_state=42
    )
    
    xgb_bayes.fit(X_train_scaled, y_train_scaled)
    
    # Best model - SAME AS COLAB
    best_xgb_params = xgb_bayes.best_params_
    st.text(f"Best XGBoost parameters: {best_xgb_params}")
    
    best_xgb = XGBRegressor(objective='reg:squarederror', random_state=42, **best_xgb_params)
    best_xgb.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate model - SAME AS COLAB
    xgb_results = evaluate_model(
        best_xgb, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
        scaler_y, "XGBoost"
    )
    
    return best_xgb, scaler_X, scaler_y, xgb_results

# Function to train SVR model (same as Google Colab)
def train_svr_model(X_train, y_train, X_test, y_test):
    """Train SVR model with Bayesian optimization - SAME AS COLAB"""
    st.text("🔄 Training SVR Model with Bayesian Optimization...")
    
    # Normalization - SAME AS COLAB
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    # Time Series Cross-Validation - SAME AS COLAB
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Parameter space - SAME AS COLAB
    svr_param_space = {
        'C': Real(0.1, 50, prior='log-uniform'),
        'gamma': Real(0.001, 1.0, prior='log-uniform'),
        'epsilon': Real(0.01, 0.5)
    }
    
    # Bayesian optimization - SAME AS COLAB
    svr_bayes = BayesSearchCV(
        estimator=SVR(),
        search_spaces=svr_param_space,
        n_iter=20,
        cv=tscv,
        scoring='neg_mean_absolute_percentage_error',
        verbose=0,  # Changed to 0 for Streamlit
        n_jobs=-1,
        random_state=42
    )
    
    svr_bayes.fit(X_train_scaled, y_train_scaled)
    
    # Best model - SAME AS COLAB
    best_svr_params = svr_bayes.best_params_
    st.text(f"Best SVR parameters: {best_svr_params}")
    
    best_svr = SVR(**best_svr_params)
    best_svr.fit(X_train_scaled, y_train_scaled)
    
    # Evaluate model - SAME AS COLAB
    svr_results = evaluate_model(
        best_svr, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, 
        scaler_y, "SVR"
    )
    
    return best_svr, scaler_X, scaler_y, svr_results

# Function for recursive forecasting (same as Google Colab)
def forecast_recursive(model, last_data, scaler_X, scaler_y, feature_names, data_sorted, days=30):
    """Recursive forecasting for both XGBoost and SVR - SAME AS COLAB"""
    base_data = last_data.copy()
    forecasts = []
    forecast_dates = []
    last_date = last_data.name
    recent_prices = data_sorted['Harga_Emas'].iloc[-30:].values
    
    for i in range(days):
        current_date = last_date + timedelta(days=i+1)
        forecast_dates.append(current_date)
        
        # Prepare features
        current_features = base_data.values.reshape(1, -1)
        current_features_scaled = scaler_X.transform(current_features)
        
        # Predict
        price_scaled = model.predict(current_features_scaled)[0]
        price_actual = scaler_y.inverse_transform([[price_scaled]])[0][0]
        forecasts.append(price_actual)
        
        # Update recent prices
        recent_prices = np.append(recent_prices[1:], price_actual)
        
        # Update features
        updated_data = base_data.copy()
        
        # Update lag features
        if 'Harga_Emas_Lag1' in feature_names:
            updated_data['Harga_Emas_Lag1'] = price_actual
        
        if 'Harga_Emas_Lag7' in feature_names and i >= 6:
            updated_data['Harga_Emas_Lag7'] = forecasts[i-6]
        
        # Update moving averages
        if 'MA7' in feature_names:
            updated_data['MA7'] = np.mean(recent_prices[-7:])
        
        if 'MA30' in feature_names:
            updated_data['MA30'] = np.mean(recent_prices[-30:])
        
        # Update volatility
        if 'Volatility' in feature_names:
            updated_data['Volatility'] = np.std(recent_prices[-7:])
        
        # Update trend
        if 'Trend' in feature_names:
            if i > 0:
                updated_data['Trend'] = price_actual - forecasts[i-1]
            else:
                updated_data['Trend'] = price_actual - recent_prices[-1]
        
        # Update momentum
        if 'Momentum' in feature_names:
            if i >= 6:
                updated_data['Momentum'] = price_actual - forecasts[i-6]
            else:
                lag_idx = min(6, i+1)
                updated_data['Momentum'] = price_actual - recent_prices[-lag_idx]
        
        # Update ROC
        if 'ROC' in feature_names:
            if i >= 6:
                updated_data['ROC'] = (price_actual / forecasts[i-6] - 1) * 100
            else:
                lag_idx = min(6, i+1)
                updated_data['ROC'] = (price_actual / recent_prices[-lag_idx] - 1) * 100
        
        base_data = updated_data
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Tanggal': forecast_dates,
        'Prediksi': forecasts
    })
    forecast_df.set_index('Tanggal', inplace=True)
    return forecast_df

# Function to create downloadable link
def create_download_link(df, filename):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="download-button">⬇️ Download {filename}</a>'
    return href

# Function to create plotly visualizations (same as app_final.py)
def plot_historical_and_forecast_plotly(historical_prices, xgb_forecast, svr_forecast, history_days=60):
    # Get the last n days of historical data
    if len(historical_prices) >= history_days:
        last_n_days = historical_prices.iloc[-history_days:]
    else:
        last_n_days = historical_prices
        
    # Safely get the last actual price for connecting lines
    if len(last_n_days) > 0:
        last_actual_price = last_n_days.iloc[-1]
        last_actual_date = last_n_days.index[-1]
    else:
        last_actual_price = 0
        last_actual_date = pd.Timestamp.now()
    
    # Prepare forecast data
    if len(xgb_forecast) > 0:
        xgb_forecast_dates = [last_actual_date] + list(xgb_forecast.index)
        xgb_forecast_values = [last_actual_price] + list(xgb_forecast['Prediksi'])
    else:
        xgb_forecast_dates = [last_actual_date]
        xgb_forecast_values = [last_actual_price]
    
    if len(svr_forecast) > 0:
        svr_forecast_dates = [last_actual_date] + list(svr_forecast.index)
        svr_forecast_values = [last_actual_price] + list(svr_forecast['Prediksi'])
    else:
        svr_forecast_dates = [last_actual_date]
        svr_forecast_values = [last_actual_price]
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add historical price line
    if len(last_n_days) > 0:
        fig.add_trace(go.Scatter(
            x=last_n_days.index, 
            y=last_n_days, 
            mode='lines',
            name=f'Harga Aktual ({len(last_n_days)} hari terakhir)',
            line=dict(color='#3B82F6', width=3)
        ))
    
    # Add XGBoost forecast line
    if len(xgb_forecast) > 0:
        fig.add_trace(go.Scatter(
            x=xgb_forecast_dates, 
            y=xgb_forecast_values, 
            mode='lines',
            name=f'Prediksi XGBoost ({len(xgb_forecast)} hari)',
            line=dict(color='#EF4444', width=3, dash='dash')
        ))
    
    # Add SVR forecast line
    if len(svr_forecast) > 0:
        fig.add_trace(go.Scatter(
            x=svr_forecast_dates, 
            y=svr_forecast_values, 
            mode='lines',
            name=f'Prediksi SVR ({len(svr_forecast)} hari)',
            line=dict(color='#10B981', width=3, dash='dash')
        ))
    
    # Customize layout
    fig.update_layout(
        title='Prediksi Harga Emas: Perbandingan Model XGBoost dan SVR',
        xaxis_title='Tanggal',
        yaxis_title='Harga Emas (Rupiah)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(
            tickformat=',',
            gridcolor='#E5E7EB'
        ),
        xaxis=dict(
            gridcolor='#E5E7EB'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=20, r=20, t=60, b=20),
        height=500,
    )
    
    return fig

# Function to create sample data
def create_sample_data(days=150):
    """Create synthetic gold price data for demonstration"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create synthetic gold prices with trend and noise
    base_price = 1000000  # Base price in Rupiah
    trend = np.random.normal(2000, 1000, len(date_range))  # Variable trend
    noise = np.random.normal(0, 10000, len(date_range))  # Random noise
    
    prices = [base_price]
    for i in range(1, len(date_range)):
        price = prices[-1] + trend[i] + noise[i]
        prices.append(max(price, 500000))  # Minimum price floor
    
    # Create DataFrame
    data = pd.DataFrame({
        'Tanggal': date_range.strftime('%d-%m-%Y'),
        'Harga_Emas': prices
    })
    
    return data

# Dashboard page
def dashboard_page():
    st.markdown('<div class="main-header">✨ Gold Price Prediction - GOLDEN ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">🏆 Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h3>👋 Selamat Datang!</h3>
        <p>Aplikasi GOLDEN memungkinkan Anda memprediksi harga emas menggunakan model machine learning XGBoost dan SVR.</p>
        <p>Upload data CSV Anda atau gunakan data sampel untuk memulai.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create layout for dashboard cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-card-icon">📈</div>
            <div class="dashboard-card-title">Prediksi Harga Emas</div>
            <div class="dashboard-card-description">
                Prediksi harga emas untuk masa depan menggunakan model XGBoost dan SVR.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Mulai Prediksi", key="btn_prediction"):
            st.session_state.page = "Prediction"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-card-icon">📊</div>
            <div class="dashboard-card-title">Historical Data</div>
            <div class="dashboard-card-description">
                Lihat dan analisis data historis harga emas dengan visualisasi interaktif.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Lihat Data Historis", key="btn_historical"):
            st.session_state.page = "Historical"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-card-icon">🔍</div>
            <div class="dashboard-card-title">Evaluasi Model</div>
            <div class="dashboard-card-description">
                Evaluasi performa model dengan membandingkan prediksi dan data aktual.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Evaluasi Model", key="btn_evaluation"):
            st.session_state.page = "Evaluation"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div class="dashboard-card">
            <div class="dashboard-card-icon">ℹ️</div>
            <div class="dashboard-card-title">About</div>
            <div class="dashboard-card-description">
                Informasi tentang aplikasi dan cara penggunaan.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Tentang Aplikasi", key="btn_about"):
            st.session_state.page = "About"
            st.rerun()
    
    # Show stats if data is loaded
    if st.session_state.data_loaded and st.session_state.historical_data is not None:
        st.markdown('<div class="sub-header">📊 Statistik Data</div>', unsafe_allow_html=True)
        
        try:
            data = st.session_state.historical_data.copy()
            data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=True)
            data.sort_values('Tanggal', inplace=True)
            
            if len(data) > 0:
                latest_price = data['Harga_Emas'].iloc[-1]
                avg_price = data['Harga_Emas'].mean()
                min_price = data['Harga_Emas'].min()
                max_price = data['Harga_Emas'].max()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Harga Terakhir</div>
                        <div class="metric-value">Rp {latest_price:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Harga Rata-Rata</div>
                        <div class="metric-value">Rp {avg_price:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Harga Terendah</div>
                        <div class="metric-value">Rp {min_price:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">Harga Tertinggi</div>
                        <div class="metric-value">Rp {max_price:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick chart
                st.markdown('<div class="sub-header">📈 Grafik Harga Emas</div>', unsafe_allow_html=True)
                
                fig = px.line(
                    data, 
                    x='Tanggal', 
                    y='Harga_Emas',
                    title='Perkembangan Harga Emas'
                )
                
                fig.update_layout(
                    xaxis=dict(gridcolor='#E5E7EB'),
                    yaxis=dict(gridcolor='#E5E7EB', tickformat=','),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan statistik data: {str(e)}")

# Prediction page
def prediction_page():
    st.markdown('<div class="main-header">📈 Prediksi Harga Emas</div>', unsafe_allow_html=True)
    
    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["🔄 Load & Process Data", "🤖 Train Models", "🔮 Generate Predictions"])

    with tab1:
        st.markdown('<div class="sub-header">📊 Load Data</div>', unsafe_allow_html=True)
        
        # Data source selection
        data_option = st.radio("Pilih sumber data:", ["Upload CSV File", "Use Sample Data"])
        
        if data_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file:
                try:
                    data = pd.read_csv(uploaded_file)
                    if 'Tanggal' not in data.columns or 'Harga_Emas' not in data.columns:
                        st.error("CSV harus memiliki kolom 'Tanggal' dan 'Harga_Emas'")
                        return
                    
                    st.session_state.data_loaded = True
                    st.session_state.historical_data = data
                    st.success("Data berhasil dimuat!")
                    
                except Exception as e:
                    st.error(f"Gagal memuat data: {str(e)}")
                    return
        else:
            if st.button("Load Sample Data"):
                data = create_sample_data(days=150)
                st.session_state.data_loaded = True
                st.session_state.historical_data = data
                st.success("Data sampel berhasil dimuat!")
        
        # Process data if loaded
        if st.session_state.data_loaded and st.session_state.historical_data is not None:
            st.markdown('<div class="sub-header">📋 Preview Data</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.historical_data.head(), use_container_width=True)
            
            if st.button("🔄 Process Data"):
                with st.spinner('🔄 Memproses data...'):
                    try:
                        # Preprocess data - SAME AS COLAB
                        data_cleaned = load_and_preprocess_data(st.session_state.historical_data)
                        
                        if data_cleaned is None:
                            return
                            
                        # Add features - SAME AS COLAB
                        data_with_features = add_technical_features(data_cleaned)
                        
                        if len(data_with_features) == 0:
                            return
                            
                        # Store processed data
                        st.session_state.processed_data = data_with_features
                        
                        st.success("✅ Data berhasil diproses!")
                        st.dataframe(data_with_features.head(), use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during data processing: {str(e)}")

    with tab2:
        st.markdown('<div class="sub-header">🤖 Train Models</div>', unsafe_allow_html=True)
        
        if st.session_state.processed_data is not None:
            st.info("📊 Using same data split as Google Colab: 90% training, 10% testing")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Training Data</div>
                    <div class="metric-value">90%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Testing Data</div>
                    <div class="metric-value">10%</div>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🚀 Train Models"):
                with st.spinner('🤖 Training models... This may take a few minutes.'):
                    try:
                        # Split data - SAME AS COLAB (90% train, 10% test)
                        X_train, X_test, y_train, y_test, data_sorted = split_data(st.session_state.processed_data)
                        
                        # Train XGBoost model - SAME AS COLAB
                        xgb_model, xgb_scaler_X, xgb_scaler_y, xgb_results = train_xgboost_model(
                            X_train, y_train, X_test, y_test
                        )
                        
                        # Train SVR model - SAME AS COLAB
                        svr_model, svr_scaler_X, svr_scaler_y, svr_results = train_svr_model(
                            X_train, y_train, X_test, y_test
                        )
                        
                        # Store models
                        st.session_state.xgb_model = xgb_model
                        st.session_state.svr_model = svr_model
                        st.session_state.xgb_scaler_X = xgb_scaler_X
                        st.session_state.xgb_scaler_y = xgb_scaler_y
                        st.session_state.svr_scaler_X = svr_scaler_X
                        st.session_state.svr_scaler_y = svr_scaler_y
                        st.session_state.models_trained = True
                        
                        # Compare models - SAME AS COLAB
                        st.text("🏆 MODEL COMPARISON")
                        st.text("=" * 50)
                        st.text(f"XGBoost MAPE: {xgb_results['test_metrics']['mape']:.2f}%")
                        st.text(f"SVR MAPE: {svr_results['test_metrics']['mape']:.2f}%")
                        
                        best_model_name = "XGBoost" if xgb_results['test_metrics']['mape'] < svr_results['test_metrics']['mape'] else "SVR"
                        st.text(f"🎯 Best Model: {best_model_name}")
                        
                        # Store metrics for evaluation page (without R²)
                        st.session_state.model_metrics = {
                            'XGBoost': xgb_results['test_metrics'],
                            'SVR': svr_results['test_metrics']
                        }
                        
                        # Display metrics in styled format (without R²)
                        st.success("✅ Models trained successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="card">
                                <h3 style="color: #EF4444;">XGBoost Model</h3>
                                <p><strong>MAPE:</strong> {xgb_results['test_metrics']['mape']:.2f}%</p>
                                <p><strong>RMSE:</strong> Rp {xgb_results['test_metrics']['rmse']:,.2f}</p>
                                <p><strong>MAE:</strong> Rp {xgb_results['test_metrics']['mae']:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="card">
                                <h3 style="color: #10B981;">SVR Model</h3>
                                <p><strong>MAPE:</strong> {svr_results['test_metrics']['mape']:.2f}%</p>
                                <p><strong>RMSE:</strong> Rp {svr_results['test_metrics']['rmse']:,.2f}</p>
                                <p><strong>MAE:</strong> Rp {svr_results['test_metrics']['mae']:,.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Best model conclusion
                        st.success(f"🏆 Best performing model: {best_model_name}")
                        
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
        else:
            st.warning("Silakan process data terlebih dahulu di tab 'Load & Process Data'.")

    with tab3:
        st.markdown('<div class="sub-header">🔮 Generate Predictions</div>', unsafe_allow_html=True)
        
        if st.session_state.models_trained and st.session_state.processed_data is not None:
            # Forecast settings
            forecast_days = st.slider("Jumlah Hari Prediksi", 7, 90, 30)
            
            st.info(f"📅 Memprediksi harga emas untuk {forecast_days} hari ke depan")
            
            if st.button("🔮 Generate Predictions"):
                with st.spinner('🔮 Generating predictions...'):
                    try:
                        # Get the last data point
                        data_sorted = st.session_state.processed_data.sort_index()
                        last_data_point = data_sorted.drop(columns=['Harga_Emas']).iloc[-1]
                        
                        # Display forecast summary like in Google Colab
                        st.text("🔮 Generating 30-day forecasts...")
                        
                        # Generate XGBoost forecasts using XGBoost scalers
                        xgb_forecast = forecast_recursive(
                            st.session_state.xgb_model,
                            last_data_point,
                            st.session_state.xgb_scaler_X,
                            st.session_state.xgb_scaler_y,
                            data_sorted.drop(columns=['Harga_Emas']).columns,
                            data_sorted,
                            days=forecast_days
                        )
                        
                        # Generate SVR forecasts using SVR scalers
                        svr_forecast = forecast_recursive(
                            st.session_state.svr_model,
                            last_data_point,
                            st.session_state.svr_scaler_X,
                            st.session_state.svr_scaler_y,
                            data_sorted.drop(columns=['Harga_Emas']).columns,
                            data_sorted,
                            days=forecast_days
                        )
                        
                        st.success(f"✅ Prediksi berhasil untuk {forecast_days} hari ke depan!")
                        
                        # Visualization
                        fig = plot_historical_and_forecast_plotly(
                            data_sorted['Harga_Emas'],
                            xgb_forecast,
                            svr_forecast,
                            history_days=min(60, len(data_sorted))
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast tables - SAME FORMAT AS COLAB
                        st.text(f"📋 XGBoost {forecast_days}-Day Forecast:")  
                        st.text(f"📋 SVR {forecast_days}-Day Forecast:")
                        
                        tab1, tab2 = st.tabs(["📊 XGBoost Predictions", "📊 SVR Predictions"])
                        
                        with tab1:
                            xgb_display = xgb_forecast.copy()
                            xgb_display['Prediksi'] = xgb_display['Prediksi'].apply(lambda x: f"{x:,.0f}")
                            xgb_display = xgb_display.reset_index()
                            xgb_display['Tanggal'] = xgb_display['Tanggal'].dt.strftime('%d-%m-%Y')
                            xgb_display.columns = ['Tanggal', 'Prediksi (Rp)']
                            
                            st.dataframe(xgb_display, use_container_width=True)
                            st.markdown(create_download_link(xgb_forecast, "xgboost_forecast_30days"), unsafe_allow_html=True)
                        
                        with tab2:
                            svr_display = svr_forecast.copy()
                            svr_display['Prediksi'] = svr_display['Prediksi'].apply(lambda x: f"{x:,.0f}")
                            svr_display = svr_display.reset_index()
                            svr_display['Tanggal'] = svr_display['Tanggal'].dt.strftime('%d-%m-%Y')
                            svr_display.columns = ['Tanggal', 'Prediksi (Rp)']
                            
                            st.dataframe(svr_display, use_container_width=True)
                            st.markdown(create_download_link(svr_forecast, "svr_forecast_30days"), unsafe_allow_html=True)
                            
                        # Final summary like in Google Colab
                        st.text("🎯 FINAL SUMMARY")
                        st.text("=" * 50)
                        st.text(f"✅ XGBoost Model - MAPE: {st.session_state.model_metrics['XGBoost']['mape']:.2f}%")
                        st.text(f"✅ SVR Model - MAPE: {st.session_state.model_metrics['SVR']['mape']:.2f}%")
                        best_model_name = "XGBoost" if st.session_state.model_metrics['XGBoost']['mape'] < st.session_state.model_metrics['SVR']['mape'] else "SVR"
                        st.text(f"🏆 Best performing model: {best_model_name}")
                        st.text("📁 Forecast files available for download!")
                        st.text("🚀 Analysis completed!")
                        
                    except Exception as e:
                        st.error(f"Error during prediction generation: {str(e)}")
        else:
            st.warning("Silakan train models terlebih dahulu di tab 'Train Models'.")

# Historical data page
def historical_page():
    # Main header
    st.markdown('<div class="main-header">📊 Data Historis Harga Emas</div>', unsafe_allow_html=True)
    
    # Back button
    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()

    # Check if historical data is loaded
    if st.session_state.data_loaded and st.session_state.historical_data is not None:
        # Process data with better error handling
        try:
            data = st.session_state.historical_data.copy()
            
            # More robust date conversion
            if 'Tanggal' in data.columns:
                # Try multiple date formats
                date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%d-%b-%Y', '%Y-%b-%d']
                
                # First try pandas auto-detection
                try:
                    data['Tanggal'] = pd.to_datetime(data['Tanggal'], dayfirst=True, errors='coerce')
                except:
                    # If auto-detection fails, try each format
                    for fmt in date_formats:
                        try:
                            data['Tanggal'] = pd.to_datetime(data['Tanggal'], format=fmt, errors='coerce')
                            if not data['Tanggal'].isna().all():
                                break
                        except:
                            continue
                
                # Remove rows with invalid dates
                data = data.dropna(subset=['Tanggal'])
                
                if len(data) == 0:
                    st.error("Tidak dapat memproses tanggal dalam data CSV. Pastikan format tanggal adalah DD-MM-YYYY, YYYY-MM-DD, atau format standar lainnya.")
                    return
                
                data.sort_values('Tanggal', inplace=True)
            
            # Ensure Harga_Emas is numeric
            if 'Harga_Emas' in data.columns:
                data['Harga_Emas'] = pd.to_numeric(data['Harga_Emas'], errors='coerce')
                data = data.dropna(subset=['Harga_Emas'])
                
                if len(data) == 0:
                    st.error("Tidak ada data harga yang valid. Pastikan kolom 'Harga_Emas' berisi angka.")
                    return
            
        except Exception as e:
            st.error(f"Error memproses data: {str(e)}")
            return
        
        # Tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📈 Tren Harga", "📊 Statistik", "📋 Tabel Data"])
        
        with tab1:
            st.markdown('<div class="sub-header">📈 Tren Harga Emas</div>', unsafe_allow_html=True)
            
            # Date range selector with safety check
            if len(data) > 0:
                min_date = data['Tanggal'].min().date()
                max_date = data['Tanggal'].max().date()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Tanggal Mulai", min_date, min_value=min_date, max_value=max_date, key="hist_start_date")
                with col2:
                    end_date = st.date_input("Tanggal Akhir", max_date, min_value=min_date, max_value=max_date, key="hist_end_date")
                
                # Filter data based on date range
                filtered_data = data[(data['Tanggal'].dt.date >= start_date) & (data['Tanggal'].dt.date <= end_date)]
                
                # Main trend chart
                if len(filtered_data) > 0:
                    fig = px.line(
                        filtered_data, 
                        x='Tanggal', 
                        y='Harga_Emas',
                        title='Tren Harga Emas',
                        labels={'Harga_Emas': 'Harga (Rp)', 'Tanggal': 'Tanggal'}
                    )
                    
                    fig.update_layout(
                        xaxis=dict(gridcolor='#E5E7EB'),
                        yaxis=dict(gridcolor='#E5E7EB', tickformat=','),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and show trend info with improved safety checks
                    if len(filtered_data) > 1:
                        try:
                            first_price = filtered_data['Harga_Emas'].iloc[0]
                            last_price = filtered_data['Harga_Emas'].iloc[-1]
                            
                            # More robust price change calculation
                            if pd.notna(first_price) and pd.notna(last_price) and first_price != 0:
                                price_change = last_price - first_price
                                price_change_pct = (price_change / first_price) * 100
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">Perubahan Harga</div>
                                        <div class="metric-value" style="color: {'#10B981' if price_change >= 0 else '#EF4444'}">
                                            Rp {price_change:,.0f} ({'+'if price_change >= 0 else ''}{price_change_pct:.2f}%)
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    avg_price = filtered_data['Harga_Emas'].mean()
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">Rata-rata Harian</div>
                                        <div class="metric-value">
                                            Rp {avg_price:,.0f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("Tidak dapat menghitung perubahan harga karena data tidak valid.")
                                
                        except Exception as e:
                            st.error(f"Error menghitung perubahan harga: {str(e)}")
                        
                        # Show volatility chart (price changes) with safety check
                        try:
                            st.markdown('<div class="sub-header">📊 Volatilitas Harga</div>', unsafe_allow_html=True)
                            
                            # Calculate daily price changes with better error handling
                            filtered_data_copy = filtered_data.copy()
                            filtered_data_copy['price_change'] = filtered_data_copy['Harga_Emas'].pct_change() * 100
                            filtered_data_clean = filtered_data_copy.dropna(subset=['price_change'])
                            
                            if len(filtered_data_clean) > 0:
                                # Create plot
                                fig = go.Figure()
                                
                                colors = ['#10B981' if x >= 0 else '#EF4444' for x in filtered_data_clean['price_change']]
                                
                                fig.add_trace(go.Bar(
                                    x=filtered_data_clean['Tanggal'],
                                    y=filtered_data_clean['price_change'],
                                    name='Perubahan Harga (%)',
                                    marker_color=colors
                                ))
                                
                                fig.update_layout(
                                    title='Perubahan Harga Emas Harian (%)',
                                    xaxis_title='Tanggal',
                                    yaxis_title='Perubahan (%)',
                                    xaxis=dict(gridcolor='#E5E7EB'),
                                    yaxis=dict(gridcolor='#E5E7EB'),
                                    plot_bgcolor='white',
                                    paper_bgcolor='white',
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Statistics about volatility
                                volatility = filtered_data_clean['price_change'].std()
                                max_gain = filtered_data_clean['price_change'].max()
                                max_loss = filtered_data_clean['price_change'].min()
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">Volatilitas (Std)</div>
                                        <div class="metric-value">{volatility:.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">Kenaikan Terbesar</div>
                                        <div class="metric-value" style="color: #10B981">+{max_gain:.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col3:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">Penurunan Terbesar</div>
                                        <div class="metric-value" style="color: #EF4444">{max_loss:.2f}%</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("Tidak dapat menghitung volatilitas - data tidak cukup.")
                        except Exception as e:
                            st.error(f"Error menghitung volatilitas: {str(e)}")
                    else:
                        st.info("Perlu minimal 2 data point untuk menghitung perubahan harga.")
                else:
                    st.warning("Tidak ada data dalam rentang tanggal yang dipilih.")
            else:
                st.warning("Data historis kosong.")
            
        with tab2:
            st.markdown('<div class="sub-header">📊 Statistik Harga Emas</div>', unsafe_allow_html=True)
            
            # Safety check for data
            if len(data) > 0:
                # General statistics
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Statistik Deskriptif</h3>', unsafe_allow_html=True)
                
                # Safely create statistics
                try:
                    stats_df = data['Harga_Emas'].describe().reset_index()
                    stats_df.columns = ['Statistik', 'Nilai']
                    
                    # Format all numeric values
                    stats_df['Nilai'] = stats_df['Nilai'].apply(lambda x: f"{x:,.0f}")
                    
                    # Map stat names to more readable names
                    stat_map = {
                        'count': 'Jumlah Data',
                        'mean': 'Rata-rata',
                        'std': 'Standar Deviasi',
                        'min': 'Minimum',
                        '25%': 'Kuartil 1 (25%)',
                        '50%': 'Median (50%)',
                        '75%': 'Kuartil 3 (75%)',
                        'max': 'Maksimum'
                    }
                    
                    stats_df['Statistik'] = stats_df['Statistik'].map(lambda x: stat_map.get(x, x))
                    
                    st.dataframe(stats_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error saat menghitung statistik: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Plot distribusi with safety check
                st.markdown('<div class="sub-header">📊 Distribusi Harga Emas</div>', unsafe_allow_html=True)
                
                fig = go.Figure()
                
                # Check data validity before creating histogram
                if data['Harga_Emas'].notna().any() and len(data['Harga_Emas'].dropna()) > 0:
                    # Use only valid data (not NaN or infinite)
                    valid_data = data['Harga_Emas'][np.isfinite(data['Harga_Emas'])]
                    
                    if len(valid_data) > 0:
                        # Add histogram with valid data
                        fig.add_trace(go.Histogram(
                            x=valid_data, 
                            nbinsx=30,
                            marker_color='#3B82F6',
                            opacity=0.7,
                            name='Distribusi Harga'
                        ))
                    else:
                        st.warning("Tidak ada data numerik yang valid untuk membuat histogram.")
                else:
                    st.warning("Tidak ada data yang valid untuk membuat histogram.")
                
                # Update layout
                fig.update_layout(
                    title='Distribusi Harga Emas',
                    xaxis_title='Harga Emas (Rupiah)',
                    yaxis_title='Frekuensi',
                    xaxis=dict(gridcolor='#E5E7EB', tickformat=','),
                    yaxis=dict(gridcolor='#E5E7EB'),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Data historis kosong.")
        
        with tab3:
            st.markdown('<div class="sub-header">📋 Tabel Data Historis</div>', unsafe_allow_html=True)
            
            # Safety check for data
            if len(data) > 0:
                # Format data for display
                display_data = data.copy()
                display_data['Tanggal'] = display_data['Tanggal'].dt.strftime('%d-%m-%Y')
                display_data['Harga_Emas'] = display_data['Harga_Emas'].apply(lambda x: f"Rp {x:,.0f}")
                
                # Show dataframe
                st.dataframe(display_data, use_container_width=True)
                
                # Download button
                csv = data.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="gold_historical_data.csv" class="download-button">⬇️ Download Historical Data (CSV)</a>'
                st.markdown(href, unsafe_allow_html=True)
            else:
                st.warning("Data historis kosong.")
    
    else:
        st.warning("Silakan muat data terlebih dahulu dari halaman Prediction untuk melihat data historis.")
        
        # Show sample button
        if st.button("Muat Data Sampel"):
            # Create sample data
            data = create_sample_data(days=365)  # 1 year of sample data
            st.session_state.data_loaded = True
            st.session_state.historical_data = data
            st.rerun()

# Evaluation page
def evaluation_page():
    # Main header
    st.markdown('<div class="main-header">🔍 Evaluasi Out-Of-Sample</div>', unsafe_allow_html=True)
    
    # Back button
    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()
    
    # Introduction
    st.markdown("""
    <div class="info-box">
        <h3>📈 Evaluasi Out-Of-Sample</h3>
        <p>Halaman ini memungkinkan Anda membandingkan hasil prediksi dengan data aktual untuk mengevaluasi akurasi model prediksi.</p>
        <p>Upload file CSV prediksi dan file CSV data aktual, kemudian pilih periode yang ingin Anda evaluasi.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create upload forms
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h3>📤 Upload File Prediksi</h3>
            <p>Upload file CSV hasil prediksi dengan format: Tanggal, Prediksi</p>
        </div>
        """, unsafe_allow_html=True)
        pred_file = st.file_uploader("Upload file prediksi (CSV)", type=['csv'], key="pred_file")
        
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📤 Upload File Aktual</h3>
            <p>Upload file CSV data aktual dengan format: Tanggal, Harga_Emas</p>
        </div>
        """, unsafe_allow_html=True)
        actual_file = st.file_uploader("Upload file data aktual (CSV)", type=['csv'], key="actual_file")
    
    # Period selection
    st.markdown('<div class="sub-header">⏱️ Periode Evaluasi</div>', unsafe_allow_html=True)
    
    eval_periods = {
        "7": "7 Hari",
        "10": "10 Hari",
        "14": "14 Hari",
        "30": "30 Hari",
        "all": "Semua Data"
    }
    
    period = st.radio("Pilih periode evaluasi:", list(eval_periods.values()), horizontal=True)
    
    # Get period value
    period_days = next(key for key, value in eval_periods.items() if value == period)
    
    # Process files and calculate metrics
    if st.button("🔍 Evaluasi Prediksi"):
        if pred_file is not None and actual_file is not None:
            try:
                with st.spinner('🔄 Memproses dan mengevaluasi data...'):
                    # Read CSV files
                    df_pred = pd.read_csv(pred_file)
                    df_actual = pd.read_csv(actual_file)
                    
                    # Check if required columns exist
                    if 'Tanggal' not in df_pred.columns or 'Prediksi' not in df_pred.columns:
                        st.error("File prediksi harus memiliki kolom 'Tanggal' dan 'Prediksi'")
                        return
                    
                    if 'Tanggal' not in df_actual.columns or 'Harga_Emas' not in df_actual.columns:
                        # Try alternative column name
                        if 'Tanggal' in df_actual.columns and any(col for col in df_actual.columns if 'harga' in col.lower()):
                            price_col = next(col for col in df_actual.columns if 'harga' in col.lower())
                            df_actual.rename(columns={price_col: 'Harga_Emas'}, inplace=True)
                        else:
                            st.error("File data aktual harus memiliki kolom 'Tanggal' dan 'Harga_Emas'")
                            return
                    
                    # Convert dates with more flexible parsing
                    try:
                        # First try with format detection
                        df_pred['Tanggal'] = pd.to_datetime(df_pred['Tanggal'], format='mixed', errors='coerce')
                        df_actual['Tanggal'] = pd.to_datetime(df_actual['Tanggal'], format='mixed', errors='coerce')
                        
                        # Check if we have valid dates after conversion
                        if df_pred['Tanggal'].isna().any() or df_actual['Tanggal'].isna().any():
                            st.warning("Some dates couldn't be parsed. Trying alternative formats...")
                            
                            # Try common formats explicitly
                            date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
                            
                            for fmt in date_formats:
                                try:
                                    df_pred['Tanggal'] = pd.to_datetime(df_pred['Tanggal'], format=fmt, errors='coerce')
                                    df_actual['Tanggal'] = pd.to_datetime(df_actual['Tanggal'], format=fmt, errors='coerce')
                                    
                                    # If this format works for all dates, break out of the loop
                                    if not df_pred['Tanggal'].isna().any() and not df_actual['Tanggal'].isna().any():
                                        st.success(f"Successfully parsed dates using format: {fmt}")
                                        break
                                except:
                                    continue
                        
                        # Final check to make sure dates were parsed
                        if df_pred['Tanggal'].isna().any() or df_actual['Tanggal'].isna().any():
                            st.error("Could not parse all dates. Please ensure your dates are in a consistent format like DD-MM-YYYY or YYYY-MM-DD")
                            return
                            
                    except Exception as e:
                        st.error(f"Error parsing dates: {str(e)}")
                        st.info("Please ensure your CSV files have dates in a consistent format. Common formats include: DD-MM-YYYY, YYYY-MM-DD, MM/DD/YYYY")
                        return
                    
                    # Sort by date
                    df_pred.sort_values('Tanggal', inplace=True)
                    df_actual.sort_values('Tanggal', inplace=True)
                    
                    # Merge dataframes on date
                    df_combined = pd.merge(df_pred, df_actual, on='Tanggal', how='outer')
                    # Drop NA values
                    df_combined = df_combined.dropna()
                    
                    if len(df_combined) == 0:
                        st.error("Tidak ada tanggal yang cocok antara file prediksi dan data aktual")
                        return
                    
                    # Limit to selected period if not 'all'
                    if period_days != 'all':
                        max_rows = min(int(period_days), len(df_combined))
                        df_combined = df_combined.head(max_rows)
                    
                    # Calculate error metrics
                    df_combined['Error'] = df_combined['Harga_Emas'] - df_combined['Prediksi']
                    df_combined['Error_Abs'] = abs(df_combined['Error'])
                    
                    # Safe percentage calculation
                    df_combined['Error_Pct'] = 0.0  # Initialize with zeros
                    mask = df_combined['Harga_Emas'] != 0  # Avoid division by zero
                    df_combined.loc[mask, 'Error_Pct'] = (df_combined.loc[mask, 'Error_Abs'] / df_combined.loc[mask, 'Harga_Emas']) * 100
                    
                    # Calculate metrics safely
                    mae = df_combined['Error_Abs'].mean()
                    mse = (df_combined['Error'] ** 2).mean()
                    rmse = np.sqrt(mse)
                    mape = df_combined['Error_Pct'].mean()
                    
                    # Display success message
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>✅ Evaluasi Berhasil!</h3>
                        <p>Berhasil mengevaluasi {len(df_combined)} data dari {df_combined['Tanggal'].min().strftime('%d-%m-%Y')} hingga {df_combined['Tanggal'].max().strftime('%d-%m-%Y')}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display metrics (without R²)
                    st.markdown('<div class="sub-header">📊 Metrik Error</div>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)  # Changed from 4 to 3 columns
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">MAE</div>
                            <div class="metric-value">Rp {mae:,.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">RMSE</div>
                            <div class="metric-value">Rp {rmse:,.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-label">MAPE</div>
                            <div class="metric-value">{mape:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualization of prediction vs actual with safety check
                    if len(df_combined) > 0:
                        st.markdown('<div class="sub-header">📈 Visualisasi Prediksi vs Aktual</div>', unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        
                        # Add actual price line
                        fig.add_trace(go.Scatter(
                            x=df_combined['Tanggal'], 
                            y=df_combined['Harga_Emas'], 
                            mode='lines+markers', 
                            name='Harga Aktual',
                            line=dict(color='#3B82F6', width=3)
                        ))
                        
                        # Add prediction line
                        fig.add_trace(go.Scatter(
                            x=df_combined['Tanggal'], 
                            y=df_combined['Prediksi'], 
                            mode='lines+markers', 
                            name='Prediksi',
                            line=dict(color='#EF4444', width=2, dash='dash')
                        ))
                        
                        # Customize layout
                        fig.update_layout(
                            title='Perbandingan Prediksi vs Aktual',
                            xaxis_title='Tanggal',
                            yaxis_title='Harga Emas (Rupiah)',
                            hovermode='x unified',
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5
                            ),
                            yaxis=dict(
                                tickformat=',',
                                gridcolor='#E5E7EB'
                            ),
                            xaxis=dict(
                                gridcolor='#E5E7EB'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=20, r=20, t=60, b=20),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Error visualization with safety check
                    if len(df_combined) > 0:
                        st.markdown('<div class="sub-header">📊 Visualisasi Error</div>', unsafe_allow_html=True)
                        
                        # Error bar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=df_combined['Tanggal'],
                            y=df_combined['Error'],
                            name='Error (Rp)',
                            marker_color=df_combined['Error'].apply(
                                lambda x: '#10B981' if x >= 0 else '#EF4444'
                            )
                        ))
                        
                        fig.update_layout(
                            title='Error Prediksi per Hari',
                            xaxis_title='Tanggal',
                            yaxis_title='Error (Rupiah)',
                            hovermode='x unified',
                            yaxis=dict(
                                tickformat=',',
                                gridcolor='#E5E7EB'
                            ),
                            xaxis=dict(
                                gridcolor='#E5E7EB'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=20, r=20, t=60, b=20),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Percentage error bar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=df_combined['Tanggal'],
                            y=df_combined['Error_Pct'],
                            name='Error (%)',
                            marker_color='#6366F1'
                        ))
                        
                        # Add horizontal line for average
                        fig.add_shape(
                            type='line',
                            x0=df_combined['Tanggal'].min(),
                            y0=mape,
                            x1=df_combined['Tanggal'].max(),
                            y1=mape,
                            line=dict(
                                color='#EF4444',
                                width=2,
                                dash='dash',
                            )
                        )
                        
                        # Add annotation for average line safely
                        if len(df_combined) > 0:
                            # Find a safe mid index
                            mid_idx = min(len(df_combined) - 1, len(df_combined) // 2)
                            
                            fig.add_annotation(
                                x=df_combined['Tanggal'].iloc[mid_idx],
                                y=mape,
                                text=f"Rata-rata: {mape:.2f}%",
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=-40
                            )
                        
                        fig.update_layout(
                            title='Persentase Error Prediksi per Hari',
                            xaxis_title='Tanggal',
                            yaxis_title='Error (%)',
                            hovermode='x unified',
                            yaxis=dict(
                                gridcolor='#E5E7EB'
                            ),
                            xaxis=dict(
                                gridcolor='#E5E7EB'
                            ),
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=20, r=20, t=60, b=20),
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed data table with safety check
                    if len(df_combined) > 0:
                        st.markdown('<div class="sub-header">📋 Tabel Detail Error</div>', unsafe_allow_html=True)
                        
                        # Prepare display dataframe
                        display_df = df_combined.copy()
                        display_df['Tanggal'] = display_df['Tanggal'].dt.strftime('%d-%m-%Y')
                        display_df['Prediksi'] = display_df['Prediksi'].apply(lambda x: f"Rp {x:,.0f}")
                        display_df['Harga_Emas'] = display_df['Harga_Emas'].apply(lambda x: f"Rp {x:,.0f}")
                        display_df['Error'] = display_df['Error'].apply(lambda x: f"Rp {x:,.0f}")
                        display_df['Error_Abs'] = display_df['Error_Abs'].apply(lambda x: f"Rp {x:,.0f}")
                        display_df['Error_Pct'] = display_df['Error_Pct'].apply(lambda x: f"{x:.2f}%")
                        
                        # Rename columns for display
                        display_df.columns = ['Tanggal', 'Prediksi', 'Harga Aktual', 'Error', 'Error Absolut', 'Error (%)']
                        
                        # Show dataframe
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Create download link
                        st.markdown(create_download_link(df_combined, "error_analysis"), unsafe_allow_html=True)
                    
                    # Summary and conclusions
                    st.markdown('<div class="sub-header">📝 Kesimpulan Evaluasi</div>', unsafe_allow_html=True)
                    
                    # Evaluation conclusion based on MAPE
                    conclusion = ""
                    if mape < 10:
                        conclusion = f"""
                        <div class="success-box">
                            <h3>✅ Prediksi Sangat Akurat</h3>
                            <p>Model menunjukkan performa yang sangat baik dengan MAPE hanya {mape:.2f}%.</p>
                            <p>Rata-rata error absolut adalah Rp {mae:,.2f} dari harga aktual.</p>
                        </div>
                        """
                    elif mape < 20:
                        conclusion = f"""
                        <div class="info-box">
                            <h3>✅ Prediksi Cukup Akurat</h3>
                            <p>Model menunjukkan performa yang baik dengan MAPE {mape:.2f}%.</p>
                            <p>Rata-rata error absolut adalah Rp {mae:,.2f} dari harga aktual.</p>
                        </div>
                        """
                    else:
                        conclusion = f"""
                        <div class="warning-box">
                            <h3>⚠️ Prediksi Kurang Akurat</h3>
                            <p>Model menunjukkan performa yang kurang baik dengan MAPE {mape:.2f}%.</p>
                            <p>Rata-rata error absolut adalah Rp {mae:,.2f} dari harga aktual.</p>
                            <p>Pertimbangkan untuk meningkatkan model atau menambah data pelatihan.</p>
                        </div>
                        """
                    
                    st.markdown(conclusion, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error selama evaluasi: {str(e)}")
        else:
            st.warning("Silakan upload file prediksi dan data aktual terlebih dahulu.")

# About page
def about_page():
    # Main header
    st.markdown('<div class="main-header">ℹ️ About Gold Price Prediction - GOLDEN </div>', unsafe_allow_html=True)
    
    # Back button
    if st.button("🔙 Kembali ke Dashboard"):
        st.session_state.page = "Dashboard"
        st.rerun()
    
    # About the app section
    st.markdown('<div class="sub-header">📱 Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    st.write("""
    <div class="card">
        <h3>Gold Price Prediction - GOLDEN</h3>
        <p>Aplikasi ini dikembangkan untuk memprediksi harga emas menggunakan model machine learning yang canggih. 
        Dengan menggabungkan kekuatan algoritma XGBoost dan SVR, aplikasi ini dapat memprediksi harga emas hingga 90 hari ke depan dengan tingkat akurasi yang tinggi.</p>
        <h4>Fitur Utama:</h4>
        <ul>
            <li>Prediksi harga emas hingga 90 hari ke depan</li>
            <li>Visualisasi data historis dan prediksi</li>
            <li>Analisis statistik harga emas</li>
            <li>Perbandingan model XGBoost dan SVR</li>
            <li>Export hasil prediksi dalam format CSV</li>
            <li>Evaluasi performa prediksi dengan data aktual</li>
        </ul>
        <h4>Teknologi yang Digunakan:</h4>
        <ul>
            <li>Python</li>
            <li>Streamlit</li>
            <li>Pandas & NumPy</li>
            <li>Plotly</li>
            <li>Scikit-learn</li>
            <li>XGBoost</li>
        </ul>
    """, unsafe_allow_html=True)
    
    # FUNGSI UNTUK ENCODE GAMBAR KE BASE64
    def get_base64_image(image_path):
        """Convert image to base64 string"""
        try:
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except FileNotFoundError:
            # Fallback ke placeholder jika file tidak ditemukan
            return ""

    # About the author section
    st.markdown("""
    <div class="about-card">
        <img src="data:image/jpeg;base64,{}" class="about-avatar">
        <div class="about-details">
            <div class="about-name">Aniysah Fauziyyah Alfa</div>
            <div class="about-role">Data Scientist & Machine Learning Engineer</div>
            <div class="about-description">
                <p>Seorang data scientist dan machine learning engineer yang berfokus pada pengembangan model prediktif untuk analisis data keuangan dan ekonomi.</p>
                <p>Aplikasi ini merupakan bagian dari proyek penelitian tentang penggunaan algoritma machine learning untuk memprediksi harga komoditas.</p>
            </div>
        </div>
    </div>
    """.format(get_base64_image("assets/profile.jpg")), unsafe_allow_html=True)
    
        
    # How it works section
    st.markdown('<div class="sub-header">🔬 Cara Kerja Aplikasi</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Metodologi Prediksi</h3>
        <p>Aplikasi ini menggunakan pendekatan hybrid dengan dua algoritma machine learning yang berbeda:</p>
        <h4>1. XGBoost (Extreme Gradient Boosting)</h4>
        <p>XGBoost adalah algoritma ensemble learning yang menggunakan prinsip gradient boosting. 
        Algoritma ini sangat efektif untuk data deret waktu dan bisa menangkap pola non-linear yang kompleks dalam data harga emas.</p>
        <h4>2. SVR (Support Vector Regression)</h4>
        <p>SVR menggunakan prinsip support vector machines untuk regresi. 
        Algoritma ini efektif untuk generalisasi dan bisa bekerja dengan baik bahkan dengan jumlah data yang terbatas.</p>
        <h4>Feature Engineering</h4>
        <p>Untuk meningkatkan akurasi prediksi, aplikasi melakukan feature engineering dengan menambahkan indikator-indikator teknikal seperti:</p>
        <ul>
            <li>Moving Average (MA7, MA30)</li>
            <li>Lag Features</li>
            <li>Volatility</li>
            <li>Trend</li>
            <li>Momentum</li>
            <li>Rate of Change (ROC)</li>
        </ul>
        <h4>Evaluasi Model</h4>
        <p>Performa model dievaluasi menggunakan beberapa metrik utama:</p>
        <ul>
            <li>MAPE (Mean Absolute Percentage Error) - mengukur rata-rata persentase kesalahan</li>
            <li>RMSE (Root Mean Square Error) - mengukur akar kuadrat rata-rata kesalahan</li>
            <li>MAE (Mean Absolute Error) - mengukur rata-rata kesalahan absolut</li>
        </ul>
    """, unsafe_allow_html=True)

    # Contact/Questions section
    st.markdown("""
    <div class="card">
        <h3>📞 Questions or Feedback?</h3>
        <p>Untuk pertanyaan, masukan, atau saran, silakan hubungi:</p>
        <ul>
            <li>Email: aniysahalfa@gmail.com</li>
            <li>GitHub: github.com/AniysahFauziyyahAlfa</li>
        </ul>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2025 Gold Price Prediction - GOLDEN  by Aniysah Fauziyyah Alfa ❤️. All rights reserved.</p>
        <p>Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Sidebar for navigation
    st.sidebar.title("🔮 Gold Price Prediction - GOLDEN")
    st.sidebar.markdown("by Aniysah Fauziyyah Alfa")
    st.sidebar.markdown("---")
    
    # Navigation
    pages = {
        "Dashboard": "🏠 Dashboard",
        "Prediction": "📈 Prediction",
        "Historical": "📊 Historical Data",
        "Evaluation": "🔍 Model Evaluation",
        "About": "ℹ️ About"
    }
    
    # Display navigation buttons
    st.sidebar.markdown('<div class="sub-header">📍 Navigation</div>', unsafe_allow_html=True)
    
    for page, name in pages.items():
        if st.sidebar.button(name, key=f"nav_{page}"):
            st.session_state.page = page
            st.rerun()
    
    # Display current page based on state
    if st.session_state.page == "Dashboard":
        dashboard_page()
    elif st.session_state.page == "Prediction":
        prediction_page()
    elif st.session_state.page == "Historical":
        historical_page()
    elif st.session_state.page == "Evaluation":
        evaluation_page()
    elif st.session_state.page == "About":
        about_page()

# Run the app
if __name__ == "__main__":
    main()
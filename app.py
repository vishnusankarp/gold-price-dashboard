import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Gold Price Intelligence", layout="wide", page_icon="🏆")

# --- 1. DATA INGESTION (Robust Version) ---
@st.cache_data
def load_data():
    tickers = {
        'Gold': 'GC=F', 'USD_Index': 'DX-Y.NYB', 
        '10Y_Treasury': '^TNX', 'VIX': '^VIX', 'SP500': '^GSPC'
    }
    # Fetch 5 years of data
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    
    # DOWNLOAD FIX: Force simple structure
    raw_data = yf.download(list(tickers.values()), start=start_date, progress=False)
    
    # 1. Handle "MultiIndex" (The most common error source)
    # yfinance often returns columns like: ('Adj Close', 'GC=F')
    if isinstance(raw_data.columns, pd.MultiIndex):
        try:
            # Try to grab just the 'Adj Close' level
            raw_data = raw_data.xs('Adj Close', axis=1, level=0, drop_level=True)
        except KeyError:
            # Fallback if 'Adj Close' is missing, try 'Close'
            raw_data = raw_data.xs('Close', axis=1, level=0, drop_level=True)

    # 2. Rename columns
    # Invert the dictionary to map 'GC=F' -> 'Gold'
    symbol_to_name = {v: k for k, v in tickers.items()}
    df = raw_data.rename(columns=symbol_to_name)
    
    # 3. Validation Check
    # If 'Gold' is still missing, it means the download failed silently
    if 'Gold' not in df.columns:
        st.error("⚠️ Data Error: Yahoo Finance returned data, but the 'Gold' column is missing. This is usually a temporary API issue.")
        st.stop()

    # Clean data
    df = df.fillna(method='ffill').dropna()
    
    if '10Y_Treasury' in df.columns:
        df['10Y_Treasury'] = df['10Y_Treasury'] / 10
        
    return df

# --- 2. FEATURE ENGINEERING ---
def add_features(df):
    df = df.copy()
    # RSI
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages & Bands
    df['SMA_50'] = df['Gold'].rolling(50).mean()
    df['SMA_200'] = df['Gold'].rolling(200).mean()
    df['Rolling_Std'] = df['Gold'].rolling(20).std()
    
    # Dynamic Macro Correlations
    df['Corr_USD'] = df['Gold'].rolling(60).corr(df['USD_Index'])
    df['Corr_Yield'] = df['Gold'].rolling(60).corr(df['10Y_Treasury'])
    
    # Target: Log Returns (Stationary)
    df['Log_Return_30d'] = np.log(df['Gold']).shift(-30) - np.log(df['Gold'])
    
    return df.dropna()

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_model(data):
    features = ['RSI', 'SMA_50', 'SMA_200', 'Rolling_Std', 'Corr_USD', 'Corr_Yield', 'USD_Index', '10Y_Treasury']
    target = 'Log_Return_30d'
    
    # Train on all available data except the last 30 days (where target is NaN)
    train_data = data.dropna(subset=[target])
    
    X = train_data[features]
    y = train_data[target]
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model, features

# --- MAIN APP LOGIC ---
st.title("🏆 Retail Procurement Dashboard: Gold (XAU)")
st.markdown("### Strategic Inventory Planning | 30-Day Horizon")

# Load & Process
with st.spinner('Fetching live market data...'):
    df_raw = load_data()
    df = add_features(df_raw)
    model, feature_list = train_model(df)

# Prediction Logic
last_row = df.iloc[-1]
current_price = last_row['Gold']
current_date = df.index[-1]

# Predict
latest_features = df[feature_list].iloc[[-1]]
pred_log_return = model.predict(latest_features)[0]
predicted_price = current_price * np.exp(pred_log_return)
pct_change = (predicted_price - current_price) / current_price

# Define Signal
if pct_change > 0.02:
    signal = "AGGRESSIVE BUY"
    signal_color = "green"
    advice = "Price surging. Maximize inventory."
elif pct_change < -0.02:
    signal = "WAIT / LIQUIDATE"
    signal_color = "red"
    advice = "Price softening. Delay procurement."
else:
    signal = "HOLD"
    signal_color = "gray"
    advice = "Market flat. Maintain standard stock."

# --- TOP KPI ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"${current_price:,.2f}", f"{current_date.strftime('%Y-%m-%d')}")
col2.metric("Predicted (30 Days)", f"${predicted_price:,.2f}", f"{pct_change:.2%}")
col3.markdown(f"**Signal:**")
col3.markdown(f":{signal_color}[**{signal}**]")
col4.markdown(f"**Advice:**\n{advice}")

st.divider()

# --- CHARTING SECTION ---
tab1, tab2 = st.tabs(["Price Forecast", "Macro Drivers"])

with tab1:
    st.subheader("Price Trend Analysis")
    # Interactive Plotly Chart
    fig = go.Figure()
    # Plot last 1 year of actual data
    subset = df.tail(365)
    fig.add_trace(go.Scatter(x=subset.index, y=subset['Gold'], mode='lines', name='Actual Price', line=dict(color='black')))
    
    # Plot Projection Line (Connecting today to +30 days)
    future_date = current_date + timedelta(days=30)
    fig.add_trace(go.Scatter(
        x=[current_date, future_date], 
        y=[current_price, predicted_price],
        mode='lines+markers', name='AI Prediction',
        line=dict(color=signal_color, width=4, dash='dash')
    ))
    
    fig.update_layout(height=400, template="simple_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What is driving this prediction?")
    # Feature Importance
    importance = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=True)
    
    fig_imp = go.Figure(go.Bar(
        x=importance['Importance'], y=importance['Feature'], orientation='h',
        marker=dict(color='orange')
    ))
    fig_imp.update_layout(height=400, title="Model Feature Importance", template="simple_white")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Correlation check
    st.info(f"Current Gold/USD Correlation (60d rolling): **{last_row['Corr_USD']:.2f}**")

# --- SIDEBAR: SIMULATION ---
st.sidebar.header("Scenario Planning")
st.sidebar.write("Adjust market conditions to see impact:")
shock_usd = st.sidebar.slider("Shock: USD Index Change", -5.0, 5.0, 0.0)
st.sidebar.markdown(f"*Impact on Model:* Simulating a **{shock_usd}%** shift in dollar strength.")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Procurement Intelligence", 
    layout="wide", 
    page_icon="🌍"
)

# --- 1. DATA INGESTION (Bulletproof Loop Method) ---
@st.cache_data(ttl="2h")
def load_data():
    tickers = {
        'Gold_USD': 'XAUUSD=X',   # Global Spot Price (Physical Gold)
        'USD_Index': 'DX-Y.NYB',  # Strength of US Dollar
        '10Y_Treasury': '^TNX',   # Opportunity Cost (Bond Yields)
        'VIX': '^VIX'             # Market Fear/Volatility
    }
    
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    df_list = []
    
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, progress=False)
            if not data.empty:
                if 'Close' in data.columns:
                    series = data['Close']
                elif 'Adj Close' in data.columns:
                    series = data['Adj Close']
                else:
                    series = data.iloc[:, 0]
                
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                
                series.name = name
                df_list.append(series)
        except Exception:
            continue

    if not df_list:
        st.error("⚠️ Complete API Failure. Loading fallback data...")
        st.stop()
        
    df = pd.concat(df_list, axis=1)
    
    # Validation Check
    if 'Gold_USD' not in df.columns:
        st.error("⚠️ Data Error: Could not retrieve Gold Spot prices.")
        st.stop()

    df = df.ffill().dropna()
    
    # Treasury yields are usually reported in percentages (e.g., 40 = 4.0%)
    if '10Y_Treasury' in df.columns:
        df['10Y_Treasury'] = df['10Y_Treasury'] / 10
        
    return df

# --- 2. FEATURE ENGINEERING ---
def add_features(df):
    df = df.copy()
    
    # RSI (Momentum)
    delta = df['Gold_USD'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages & Volatility
    df['SMA_50'] = df['Gold_USD'].rolling(50).mean()
    df['SMA_200'] = df['Gold_USD'].rolling(200).mean()
    df['Rolling_Std'] = df['Gold_USD'].rolling(20).std()
    
    # Macro Correlations
    if 'USD_Index' in df.columns:
        df['Corr_USD'] = df['Gold_USD'].rolling(60).corr(df['USD_Index'])
    else:
        df['Corr_USD'] = 0
    
    # Target: 30-Day Future Return
    df['Log_Return_30d'] = np.log(df['Gold_USD']).shift(-30) - np.log(df['Gold_USD'])
    
    return df 

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_model(data):
    features = ['RSI', 'SMA_50', 'SMA_200', 'Rolling_Std', 'Corr_USD', 'USD_Index', '10Y_Treasury', 'VIX']
    target = 'Log_Return_30d'
    
    # Ensure features exist in dataframe
    features = [f for f in features if f in data.columns]
    
    train_df = data.dropna(subset=[target, *features])
    
    X = train_df[features]
    y = train_df[target]
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model, features

# --- MAIN DASHBOARD UI ---
st.title("🌍 Enterprise Procurement Intelligence")
st.markdown("### Global Gold Spot (XAU/USD) | 30-Day Predictive Outlook")

st.caption(f"Last Model Calibration: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} (Local Time)")

with st.spinner('Ingesting global market data...'):
    df_raw = load_data()
    df = add_features(df_raw)
    model, feature_list = train_model(df)

# Prediction Logic
last_row = df.iloc[-1]
current_price = last_row['Gold_USD']
current_date = df.index[-1]

latest_features = df[feature_list].iloc[[-1]]
pred_log_return = model.predict(latest_features)[0]

predicted_price = current_price * np.exp(pred_log_return)
pct_change = (predicted_price - current_price) / current_price

# Decision Engine
if pct_change > 0.02:
    signal = "ACCELERATE PROCUREMENT"
    signal_color = "green"
    advice = "Global spot price projected to rise. Secure inventory to protect margins."
elif pct_change < -0.02:
    signal = "DELAY PROCUREMENT"
    signal_color = "red"
    advice = "Downward price pressure detected. Delay large stock orders."
else:
    signal = "NEUTRAL / SPOT BUYING"
    signal_color = "gray"
    advice = "Market rangebound. Procure strictly for immediate retail demand."

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Spot Price (USD/oz)", f"${current_price:,.2f}", f"{current_date.strftime('%d-%b-%Y')}")
col2.metric("30-Day Forecast", f"${predicted_price:,.2f}", f"{pct_change:.2%}")
col3.markdown(f"**Strategic Action:**")
col3.markdown(f":{signal_color}[**{signal}**]")
col4.markdown(f"**Guidance:**\n{advice}")

st.divider()

# --- INTERACTIVE CHARTS ---
tab1, tab2 = st.tabs(["Global Price Trend", "Market Drivers (AI Insight)"])

with tab1:
    st.subheader("Gold Spot Target Trajectory")
    fig = go.Figure()
    
    subset = df.tail(365)
    fig.add_trace(go.Scatter(
        x=subset.index, y=subset['Gold_USD'], 
        mode='lines', name='Actual (USD)', 
        line=dict(color='#B8860B') # Gold color
    ))
    
    future_date = current_date + timedelta(days=30)
    fig.add_trace(go.Scatter(
        x=[current_date, future_date], 
        y=[current_price, predicted_price],
        mode='lines+markers', name='AI Projection',
        line=dict(color=signal_color, width=4, dash='dash')
    ))
    
    fig.update_layout(height=400, template="simple_white", hovermode="x unified", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What is driving the current forecast?")
    
    # Dictionary mapping code variables to clean business labels
    feature_labels = {
        'RSI': 'Relative Strength Index (Momentum)',
        'SMA_50': '50-Day Short-Term Trend',
        'SMA_200': '200-Day Macro Trend',
        'Rolling_Std': '30-Day Market Volatility',
        'Corr_USD': 'Correlation to US Dollar',
        'USD_Index': 'US Dollar Strength (DXY)',
        '10Y_Treasury': 'US 10-Year Treasury Yields',
        'VIX': 'Global Market Fear Index (VIX)'
    }
    
    importance = pd.DataFrame({
        'Feature_Code': feature_list, 
        'Importance': model.feature_importances_
    })
    
    # Apply the clean names
    importance['Readable_Name'] = importance['Feature_Code'].map(feature_labels)
    importance = importance.sort_values(by='Importance', ascending=True)
    
    fig_imp = go.Figure(go.Bar(
        x=importance['Importance'], 
        y=importance['Readable_Name'], 
        orientation='h',
        marker=dict(color='#2C3E50') # Corporate Slate Blue
    ))
    fig_imp.update_layout(height=400, title="Algorithm Feature Weighting", template="simple_white")
    st.plotly_chart(fig_imp, use_container_width=True)

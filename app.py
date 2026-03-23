import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gold Price Intelligence (USD)", 
    layout="wide", 
    page_icon="🏆"
)

# --- 1. DATA INGESTION (Auto-Updates Every 2 Hours) ---
@st.cache_data(ttl="2h")
def load_data():
    tickers = {
        'Gold_USD': 'GC=F',       
        'USD_Index': 'DX-Y.NYB',  
        '10Y_Treasury': '^TNX',   
        'VIX': '^VIX'             
    }
    
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    df_list = []
    
    # Download one by one to avoid MultiIndex issues
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
        st.error("⚠️ Complete API Failure: Yahoo Finance is blocking the server IP.")
        st.stop()
        
    df = pd.concat(df_list, axis=1)
    
    if 'Gold_USD' not in df.columns:
        st.error(f"⚠️ Data Error: Yahoo Finance could not retrieve Gold data. Columns found: {list(df.columns)}")
        st.stop()

    df = df.ffill().dropna()
    
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
    
    # Moving Averages
    df['SMA_50'] = df['Gold_USD'].rolling(50).mean()
    df['SMA_200'] = df['Gold_USD'].rolling(200).mean()
    df['Rolling_Std'] = df['Gold_USD'].rolling(20).std()
    
    # Correlation with USD Index
    df['Corr_USD'] = df['Gold_USD'].rolling(60).corr(df['USD_Index'])
    
    # Target: 30-Day Future Return
    df['Log_Return_30d'] = np.log(df['Gold_USD']).shift(-30) - np.log(df['Gold_USD'])
    
    return df 

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_model(data):
    features = ['RSI', 'SMA_50', 'SMA_200', 'Rolling_Std', 'Corr_USD', 'USD_Index', '10Y_Treasury']
    target = 'Log_Return_30d'
    
    train_df = data.dropna(subset=[target, *features])
    X = train_df[features]
    y = train_df[target]
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model, features

# --- MAIN DASHBOARD UI ---
st.title("Gold Procurement Dashboard (USD)")
st.markdown("### Inventory Intelligence (XAU/USD) | 30-Day Outlook")

st.caption(f"Last Live Data Fetch: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} (New York Time)")

with st.spinner('Fetching market data...'):
    df_raw = load_data()
    df = add_features(df_raw)
    model, feature_list = train_model(df)

# Prediction Logic
last_row = df.iloc[-1]
current_price_usd = last_row['Gold_USD']
current_date = df.index[-1]

latest_features = df[feature_list].iloc[[-1]]
pred_log_return = model.predict(latest_features)[0]

predicted_price_usd = current_price_usd * np.exp(pred_log_return)
pct_change = (predicted_price_usd - current_price_usd) / current_price_usd

# Decision Engine
if pct_change > 0.02:
    signal = "AGGRESSIVE BUY"
    signal_color = "green"
    advice = "Price expected to rise. Secure inventory early."
elif pct_change < -0.02:
    signal = "WAIT / LIQUIDATE"
    signal_color = "red"
    advice = "Price softening. Delay procurement."
else:
    signal = "HOLD"
    signal_color = "gray"
    advice = "Market stable. Maintain standard stock."

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gold Spot ($/oz)", f"${current_price_usd:,.2f}", f"{current_date.strftime('%d-%b-%Y')}")
col2.metric("30-Day Forecast", f"${predicted_price_usd:,.2f}", f"{pct_change:.2%}")
col3.markdown(f"**Action Signal:**")
col3.markdown(f":{signal_color}[**{signal}**]")
col4.markdown(f"**Guidance:**\n{advice}")

st.divider()

# --- INTERACTIVE CHARTS ---
tab1, tab2 = st.tabs(["USD Price Trend", "Macro Drivers"])

with tab1:
    st.subheader("Gold Price in USD ($)")
    fig = go.Figure()
    
    subset = df.tail(365)
    fig.add_trace(go.Scatter(
        x=subset.index, y=subset['Gold_USD'], 
        mode='lines', name='Actual ($)', 
        line=dict(color='black')
    ))
    
    future_date = current_date + timedelta(days=30)
    fig.add_trace(go.Scatter(
        x=[current_date, future_date], 
        y=[current_price_usd, predicted_price_usd],
        mode='lines+markers', name='AI Forecast',
        line=dict(color=signal_color, width=4, dash='dash')
    ))
    
    fig.update_layout(height=400, template="simple_white", hovermode="x unified", yaxis_tickprefix="$")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What drives the USD price?")
    importance = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=True)
    
    fig_imp = go.Figure(go.Bar(
        x=importance['Importance'], y=importance['Feature'], orientation='h',
        marker=dict(color='orange')
    ))
    fig_imp.update_layout(height=400, title="Model Drivers", template="simple_white")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    st.info(f"📊 Current Gold/USD Correlation (60d rolling): **{last_row['Corr_USD']:.2f}**")

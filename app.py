import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gold Price Intelligence (UK)", 
    layout="wide", 
    page_icon="🇬🇧"
)

# --- 1. DATA INGESTION (Auto-Updates Every 2 Hours) ---
# ttl="2h" forces the app to re-download data if the cache is older than 2 hours
# --- 1. DATA INGESTION (Bulletproof Loop Method) ---
@st.cache_data(ttl="2h")
def load_data():
    tickers = {
        'Gold_USD': 'GC=F',       
        'GBP_USD': 'GBPUSD=X',    
        'USD_Index': 'DX-Y.NYB',  
        '10Y_Treasury': '^TNX',   
        'VIX': '^VIX'             
    }
    
    start_date = (pd.Timestamp.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')
    df_list = []
    
    # 1. Download one by one to avoid MultiIndex completely
    for name, ticker in tickers.items():
        try:
            # Download individual ticker
            data = yf.download(ticker, start=start_date, progress=False)
            
            if not data.empty:
                # Extract just the Close price, no matter how yfinance formats it
                if 'Close' in data.columns:
                    series = data['Close']
                elif 'Adj Close' in data.columns:
                    series = data['Adj Close']
                else:
                    series = data.iloc[:, 0] # Ultimate fallback
                
                # If yfinance still forces a DataFrame, squeeze it to a Series
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                
                series.name = name
                df_list.append(series)
        except Exception:
            continue # If one fails, keep trying the others

    # 2. Glue them together
    if not df_list:
        st.error("⚠️ Complete API Failure: Yahoo Finance is blocking the server IP.")
        st.stop()
        
    df = pd.concat(df_list, axis=1)
    
    # Validation Check
    if 'Gold_USD' not in df.columns or 'GBP_USD' not in df.columns:
        st.error(f"⚠️ Data Error: Yahoo Finance could not retrieve Gold or Currency data. Columns found: {list(df.columns)}")
        st.stop()

    # Clean data 
    df = df.ffill().dropna()
    
    # --- 🇬🇧 CRITICAL CALCULATION ---
    df['Gold_GBP'] = df['Gold_USD'] / df['GBP_USD']
    
    if '10Y_Treasury' in df.columns:
        df['10Y_Treasury'] = df['10Y_Treasury'] / 10
        
    return df

# --- 2. FEATURE ENGINEERING ---
def add_features(df):
    df = df.copy()
    
    # All technicals are calculated on the POUND PRICE
    # This ensures the model predicts the trend relevant to UK buyers
    
    # RSI (Momentum)
    delta = df['Gold_GBP'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Averages
    df['SMA_50'] = df['Gold_GBP'].rolling(50).mean()
    df['SMA_200'] = df['Gold_GBP'].rolling(200).mean()
    df['Rolling_Std'] = df['Gold_GBP'].rolling(20).std()
    
    # Correlations (Does the Pound drop make Gold jump?)
    df['Corr_GBP'] = df['Gold_GBP'].rolling(60).corr(df['GBP_USD'])
    
    # Target: 30-Day Future Return
    # We KEEP the last 30 days (NaN targets) so we can display today's price
    df['Log_Return_30d'] = np.log(df['Gold_GBP']).shift(-30) - np.log(df['Gold_GBP'])
    
    return df 

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_model(data):
    features = ['RSI', 'SMA_50', 'SMA_200', 'Rolling_Std', 'Corr_GBP', 'GBP_USD', '10Y_Treasury']
    target = 'Log_Return_30d'
    
    # Filter NaNs ONLY for training the AI
    train_df = data.dropna(subset=[target, *features])
    
    X = train_df[features]
    y = train_df[target]
    
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=4, random_state=42)
    model.fit(X, y)
    
    return model, features

# --- MAIN DASHBOARD UI ---
st.title("🇬🇧 UK Jewellers Procurement Dashboard")
st.markdown("### Inventory Intelligence (XAU/GBP) | 30-Day Outlook")

# Display current fetch time
st.caption(f"Last Live Data Fetch: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')} (London Time)")

# Load & Process
with st.spinner('Calibrating to Sterling markets...'):
    df_raw = load_data()
    df = add_features(df_raw)
    model, feature_list = train_model(df)

# Prediction Logic (Using TODAY's data)
last_row = df.iloc[-1]
current_price_gbp = last_row['Gold_GBP']
current_date = df.index[-1]

latest_features = df[feature_list].iloc[[-1]]
pred_log_return = model.predict(latest_features)[0]

# Calculate Forecast
predicted_price_gbp = current_price_gbp * np.exp(pred_log_return)
pct_change = (predicted_price_gbp - current_price_gbp) / current_price_gbp

# Decision Engine
if pct_change > 0.02:
    signal = "STOCK UP"
    signal_color = "green"
    advice = "Sterling price set to rise. Buy ahead of inflation."
elif pct_change < -0.02:
    signal = "WAIT / HOLD CASH"
    signal_color = "red"
    advice = "Price softening in GBP terms. Delay orders."
else:
    signal = "MAINTAIN STOCK"
    signal_color = "gray"
    advice = "Market rangebound. Buy only for immediate sales."

# --- METRICS ROW ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Gold Spot (£/oz)", f"£{current_price_gbp:,.2f}", f"{current_date.strftime('%d-%b-%Y')}")
col2.metric("30-Day Forecast", f"£{predicted_price_gbp:,.2f}", f"{pct_change:.2%}")
col3.markdown(f"**Action Signal:**")
col3.markdown(f":{signal_color}[**{signal}**]")
col4.markdown(f"**Guidance:**\n{advice}")

st.divider()

# --- INTERACTIVE CHARTS ---
tab1, tab2 = st.tabs(["GBP Price Trend", "Currency Impact"])

with tab1:
    st.subheader("Gold Price in Sterling (£)")
    fig = go.Figure()
    
    # Plot last 1 year of Actual History
    subset = df.tail(365)
    fig.add_trace(go.Scatter(
        x=subset.index, y=subset['Gold_GBP'], 
        mode='lines', name='Actual (£)', 
        line=dict(color='#00247D') # UK Blue
    ))
    
    # Plot Projection
    future_date = current_date + timedelta(days=30)
    fig.add_trace(go.Scatter(
        x=[current_date, future_date], 
        y=[current_price_gbp, predicted_price_gbp],
        mode='lines+markers', name='AI Forecast',
        line=dict(color=signal_color, width=4, dash='dash')
    ))
    
    fig.update_layout(height=400, template="simple_white", hovermode="x unified", yaxis_tickprefix="£")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("What drives the UK price?")
    importance = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=True)
    
    fig_imp = go.Figure(go.Bar(
        x=importance['Importance'], y=importance['Feature'], orientation='h',
        marker=dict(color='#CF142B') # UK Red
    ))
    fig_imp.update_layout(height=400, title="Model Drivers", template="simple_white")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Currency Insight Box
    gbp_rate = last_row['GBP_USD']
    st.info(f"💷 **Currency Check:** £1 = ${gbp_rate:.3f}. \n\n"
            f"If the Pound drops below ${gbp_rate*0.95:.3f}, Gold prices in UK will rise "
            f"even if the Global Price stays flat.")

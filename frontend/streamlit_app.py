"""
Streamlit dashboard for LangGraph trading analysis
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

API_URL = os.getenv('API_URL', 'http://localhost:8000/analyze')

st.set_page_config(page_title="LangGraph Stock Analysis", layout="wide")

st.title("📈 LangGraph Stock Analysis Dashboard")

with st.sidebar:
    st.header("Analysis Parameters")
    symbol = st.text_input("Stock Symbol", value="AAPL")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "1d"], index=4)
    analysis_period = st.slider("Analysis Period (days)", 5, 120, 30)
    run_button = st.button("Run Analysis")

if run_button:
    with st.spinner("Running analysis..."):
        payload = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "analysis_period": analysis_period
        }
        try:
            response = requests.post(API_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()

            st.subheader(f"Results for {symbol.upper()}")
            st.write(data)

            # Plot price chart if data available
            if data.get('price_data') and data['price_data'].get('Close'):
                # Convert to DataFrame
                price_df = pd.DataFrame(data['price_data'])
                price_df.index = pd.to_datetime(price_df.index)

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=price_df.index,
                    open=price_df['Open'],
                    high=price_df['High'],
                    low=price_df['Low'],
                    close=price_df['Close'],
                    name='Price'
                ))
                fig.update_layout(title=f"{symbol.upper()} Price Chart", xaxis_title="Date", yaxis_title="Price ($)")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

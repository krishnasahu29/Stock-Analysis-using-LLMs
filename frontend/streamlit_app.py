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

            # Trade Recommendation
            st.header("Trade Recommendation")
            col1, col2, col3 = st.columns(3)
            col1.metric("Trade Signal", data.get('trade_signal', 'N/A'))
            col2.metric("Confidence Score", f"{data.get('confidence_score', 0)*100:.2f}%")
            col3.metric("Entry Price", f"${data.get('entry_price', 0):.2f}")
            st.text_area("Trade Rationale", data.get('trade_rationale', 'N/A'), height=100)

            # Price Chart
            if data.get('price_data') and data['price_data'].get('Close'):
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

            # Technical Analysis
            st.header("Technical Analysis")
            if data.get('technical_signals'):
                st.write("### Technical Signals")
                st.json(data['technical_signals'])
            if data.get('support_resistance'):
                st.write("### Support and Resistance")
                st.json(data['support_resistance'])
            with st.expander("Technical Indicators"):
                st.json(data.get('technical_indicators', {}))

            # Sentiment Analysis
            st.header("Sentiment Analysis")
            if data.get('news_sentiment') is not None:
                st.metric("News Sentiment", f"{data['news_sentiment']:.2f}")
            if data.get('social_sentiment') is not None:
                st.metric("Social Sentiment", f"{data['social_sentiment']:.2f}")
            if data.get('sentiment_sources'):
                st.write("### News Articles")
                st.dataframe(pd.DataFrame(data['sentiment_sources']))

            # Risk Analysis
            st.header("Risk Analysis")
            if data.get('position_size') is not None:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Position Size", f"{data['position_size']:.2f}")
                col2.metric("Stop Loss", f"${data['stop_loss']:.2f}")
                col3.metric("Take Profit", f"${data['take_profit']:.2f}")
                col4.metric("Risk/Reward Ratio", f"{data['risk_reward_ratio']:.2f}")

            # Errors and Warnings
            if data.get('errors'):
                st.error("### Errors")
                st.write(data['errors'])
            if data.get('warnings'):
                st.warning("### Warnings")
                st.write(data['warnings'])


        except Exception as e:
            st.error(f"Error: {str(e)}")

"""
State management for the LangGraph trading system
"""
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
import pandas as pd


class TradingState(TypedDict):
    """State object for the trading analysis workflow"""

    # Input parameters
    symbol: str
    timeframe: str  # e.g., '1m', '5m', '15m', '30m', '1h'
    analysis_period: int  # days of historical data

    # Market data
    price_data: Optional[pd.DataFrame]
    volume_data: Optional[pd.DataFrame]
    current_price: Optional[float]

    # Technical analysis results
    technical_indicators: Optional[Dict[str, Any]]
    technical_signals: Optional[Dict[str, float]]  # signal strength -1 to 1
    support_resistance: Optional[Dict[str, List[float]]]

    # Sentiment analysis results
    news_sentiment: Optional[float]  # -1 to 1
    social_sentiment: Optional[float]  # -1 to 1
    sentiment_sources: Optional[List[Dict[str, Any]]]

    # Risk analysis results
    position_size: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    portfolio_risk: Optional[float]

    # Final recommendations
    trade_signal: Optional[str]  # 'BUY', 'SELL', 'HOLD'
    confidence_score: Optional[float]  # 0 to 1
    entry_price: Optional[float]
    trade_rationale: Optional[str]

    # Metadata
    timestamp: Optional[datetime]
    errors: Optional[List[str]]
    warnings: Optional[List[str]]

"""
Portfolio Management Agent for LangGraph Trading System
"""
from typing import Dict, Any
import logging
from agents.state import TradingState

logger = logging.getLogger(__name__)


class PortfolioManagementAgent:
    """Agent that synthesizes analysis and produces final trade recommendation"""

    def decide(self, state: TradingState) -> TradingState:
        try:
            symbol = state["symbol"]
            logger.info(f"Making portfolio decision for {symbol}")

            signal_strength = state.get("technical_signals", {}).get('signal_strength', 0.0)
            news_sentiment = state.get("news_sentiment", 0.0)
            combined_score = (signal_strength * 0.6) + (news_sentiment * 0.4)

            if combined_score > 0.1:
                trade_signal = 'BUY'
            elif combined_score < -0.1:
                trade_signal = 'SELL'
            else:
                trade_signal = 'HOLD'

            state["trade_signal"] = trade_signal
            state["confidence_score"] = min(1.0, abs(combined_score))
            state["entry_price"] = state.get("current_price")
            state["trade_rationale"] = (
                f"Technical signal strength {signal_strength:.2f}, "
                f"news sentiment {news_sentiment:.2f}, "
                f"combined score {combined_score:.2f} -> {trade_signal}"
            )

            logger.info(f"Decision completed for {symbol}: {trade_signal}")
            return state
        except Exception as e:
            error_msg = f"Portfolio decision error: {str(e)}"
            logger.error(error_msg)
            state["errors"] = state.get("errors", []) + [error_msg]
            return state

# Singleton instance
portfolio_management_agent = PortfolioManagementAgent()

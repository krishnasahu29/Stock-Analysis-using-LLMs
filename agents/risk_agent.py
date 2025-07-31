"""
Risk Management Agent for LangGraph Trading System
"""
from typing import Dict, Any
import logging
from tools.risk_calculator import risk_calculator
from tools.technical_indicators import technical_indicators
from agents.state import TradingState

logger = logging.getLogger(__name__)


class RiskManagementAgent:
    """Agent for calculating risk metrics and position sizing"""
    def __init__(self, account_balance: float = 100000.0):
        self.account_balance = account_balance

    def analyze(self, state: TradingState) -> TradingState:
        try:
            symbol = state["symbol"]
            logger.info(f"Starting risk analysis for {symbol}")

            price_data = state.get("price_data")
            if price_data is None or price_data.empty:
                state["errors"] = state.get("errors", []) + ["No price data for risk analysis"]
                return state

            # Calculate volatility and ATR
            volatility = risk_calculator.calculate_volatility(price_data)
            atr = risk_calculator.calculate_atr(price_data)

            # Determine trade direction from technical signal
            signal_strength = state.get("technical_signals", {}).get('signal_strength', 0.0)
            trade_direction = 'BUY' if signal_strength > 0 else 'SELL'

            # Entry price is current price
            entry_price = state.get("current_price", price_data['Close'].iloc[-1])
            stop_loss_price = risk_calculator.calculate_stop_loss(entry_price, atr, trade_direction)
            take_profit_price = risk_calculator.calculate_take_profit(entry_price, stop_loss_price, trade_direction)

            # Position sizing
            confidence = min(1.0, abs(signal_strength) + abs(state.get("news_sentiment", 0.0)))
            position_info = risk_calculator.position_sizing(
                account_balance=self.account_balance,
                risk_per_trade=0.02,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                confidence_score=confidence
            )

            # Update state
            state["position_size"] = position_info.get('position_size', 0.0)
            state["stop_loss"] = stop_loss_price
            state["take_profit"] = take_profit_price
            state["risk_reward_ratio"] = 2.0
            state["portfolio_risk"] = position_info.get('risk_amount', 0.0)
            state["volatility"] = volatility
            state["atr"] = atr

            logger.info(f"Risk analysis completed for {symbol}")
            return state
        except Exception as e:
            error_msg = f"Risk analysis error: {str(e)}"
            logger.error(error_msg)
            state["errors"] = state.get("errors", []) + [error_msg]
            return state

# Singleton instance
risk_management_agent = RiskManagementAgent()

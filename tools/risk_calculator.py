"""
Risk management and position sizing calculations
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskCalculator:
    """Calculate risk metrics and position sizing for trading"""

    def __init__(self):
        self.default_risk_free_rate = 0.02  # 2% annual risk-free rate

    def calculate_volatility(self, price_data: pd.DataFrame, window: int = 20) -> float:
        """Calculate historical volatility (annualized)"""
        try:
            if len(price_data) < window:
                return 0.0

            returns = price_data['Close'].pct_change().dropna()
            if len(returns) < 2:
                return 0.0

            # Calculate rolling standard deviation
            volatility = returns.rolling(window=window).std().iloc[-1]

            # Annualize volatility (assuming 252 trading days)
            annualized_vol = volatility * np.sqrt(252)

            return annualized_vol if not np.isnan(annualized_vol) else 0.0

        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def calculate_atr(self, price_data: pd.DataFrame, window: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            if len(price_data) < window or len(price_data) < 2:
                return 0.0

            high = price_data['High']
            low = price_data['Low']
            close = price_data['Close'].shift(1)

            tr1 = high - low
            tr2 = np.abs(high - close)
            tr3 = np.abs(low - close)

            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = true_range.rolling(window=window).mean().iloc[-1]

            return atr if not np.isnan(atr) else 0.0

        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return 0.0

    def kelly_criterion(self, win_probability: float, avg_win: float, avg_loss: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if avg_loss <= 0 or win_probability <= 0 or win_probability >= 1:
                return 0.0

            win_loss_ratio = avg_win / avg_loss
            kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)

            # Cap Kelly fraction at 25% for safety
            return max(0.0, min(0.25, kelly_fraction))

        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {str(e)}")
            return 0.0

    def position_sizing(self, 
                       account_balance: float,
                       risk_per_trade: float,
                       entry_price: float,
                       stop_loss_price: float,
                       confidence_score: float = 1.0) -> Dict[str, Any]:
        """
        Calculate position size based on risk management rules

        Args:
            account_balance: Total account balance
            risk_per_trade: Risk per trade as decimal (e.g., 0.02 for 2%)
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            confidence_score: Confidence in the trade (0-1)

        Returns:
            Dict with position sizing information
        """
        try:
            if entry_price <= 0 or stop_loss_price <= 0:
                return {'position_size': 0, 'shares': 0, 'risk_amount': 0}

            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss_price)

            if risk_per_share == 0:
                return {'position_size': 0, 'shares': 0, 'risk_amount': 0}

            # Adjust risk based on confidence
            adjusted_risk = risk_per_trade * confidence_score

            # Calculate maximum risk amount
            max_risk_amount = account_balance * adjusted_risk

            # Calculate number of shares
            shares = int(max_risk_amount / risk_per_share)

            # Calculate actual position size in dollars
            position_size = shares * entry_price

            # Apply position size limits (e.g., max 10% of account)
            max_position_size = account_balance * 0.1
            if position_size > max_position_size:
                shares = int(max_position_size / entry_price)
                position_size = shares * entry_price

            # Minimum position size check
            min_position_size = 100  # $100 minimum
            if position_size < min_position_size:
                shares = 0
                position_size = 0

            return {
                'position_size': position_size,
                'shares': shares,
                'risk_amount': shares * risk_per_share,
                'risk_percentage': (shares * risk_per_share) / account_balance * 100,
                'position_percentage': position_size / account_balance * 100,
                'risk_per_share': risk_per_share,
                'adjusted_risk_per_trade': adjusted_risk
            }

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return {'position_size': 0, 'shares': 0, 'risk_amount': 0}

    def calculate_stop_loss(self, 
                          entry_price: float,
                          atr: float,
                          trade_direction: str,
                          atr_multiplier: float = 2.0) -> float:
        """Calculate stop loss based on ATR"""
        try:
            if entry_price <= 0 or atr <= 0:
                return 0.0

            stop_distance = atr * atr_multiplier

            if trade_direction.upper() == 'BUY':
                stop_loss = entry_price - stop_distance
            else:  # SELL
                stop_loss = entry_price + stop_distance

            return max(0.01, stop_loss)  # Ensure positive stop loss

        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return 0.0

    def calculate_take_profit(self,
                            entry_price: float,
                            stop_loss_price: float,
                            trade_direction: str,
                            risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit target based on risk-reward ratio"""
        try:
            if entry_price <= 0 or stop_loss_price <= 0:
                return 0.0

            risk_amount = abs(entry_price - stop_loss_price)
            reward_amount = risk_amount * risk_reward_ratio

            if trade_direction.upper() == 'BUY':
                take_profit = entry_price + reward_amount
            else:  # SELL
                take_profit = entry_price - reward_amount

            return max(0.01, take_profit)  # Ensure positive price

        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return 0.0

    def portfolio_risk_analysis(self,
                              positions: List[Dict[str, Any]],
                              account_balance: float) -> Dict[str, Any]:
        """Analyze overall portfolio risk"""
        try:
            if not positions:
                return {
                    'total_exposure': 0.0,
                    'total_risk': 0.0,
                    'risk_percentage': 0.0,
                    'position_count': 0,
                    'diversification_score': 1.0
                }

            total_position_value = sum(pos.get('position_size', 0) for pos in positions)
            total_risk_amount = sum(pos.get('risk_amount', 0) for pos in positions)

            # Calculate portfolio metrics
            exposure_percentage = (total_position_value / account_balance) * 100
            risk_percentage = (total_risk_amount / account_balance) * 100

            # Simple diversification score based on position count
            position_count = len(positions)
            diversification_score = min(1.0, position_count / 10.0)  # Ideal: 10+ positions

            return {
                'total_exposure': total_position_value,
                'total_risk': total_risk_amount,
                'exposure_percentage': exposure_percentage,
                'risk_percentage': risk_percentage,
                'position_count': position_count,
                'diversification_score': diversification_score,
                'max_risk_warning': risk_percentage > 10.0,  # Warning if total risk > 10%
                'max_exposure_warning': exposure_percentage > 50.0  # Warning if exposure > 50%
            }

        except Exception as e:
            logger.error(f"Error analyzing portfolio risk: {str(e)}")
            return {}

    def risk_warnings(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate risk warnings based on analysis"""
        warnings = []

        try:
            # High volatility warning
            volatility = analysis_results.get('volatility', 0)
            if volatility > 0.4:  # 40% annualized volatility
                warnings.append(f"High volatility detected: {volatility:.1%} annualized")

            # Large position size warning
            position_pct = analysis_results.get('position_percentage', 0)
            if position_pct > 10:
                warnings.append(f"Large position size: {position_pct:.1f}% of account")

            # High portfolio risk warning
            portfolio_risk = analysis_results.get('portfolio_risk_percentage', 0)
            if portfolio_risk > 10:
                warnings.append(f"High portfolio risk: {portfolio_risk:.1f}% total risk")

            # Low confidence warning
            confidence = analysis_results.get('confidence_score', 1.0)
            if confidence < 0.5:
                warnings.append(f"Low confidence score: {confidence:.1%}")

        except Exception as e:
            logger.error(f"Error generating risk warnings: {str(e)}")

        return warnings


# Create singleton instance
risk_calculator = RiskCalculator()

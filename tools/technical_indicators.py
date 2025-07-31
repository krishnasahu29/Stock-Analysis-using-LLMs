"""
Technical analysis indicators for stock trading
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Calculate various technical indicators for trading analysis"""

    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()

    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD Indicator"""
        exp1 = data.ewm(span=fast).mean()
        exp2 = data.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()

        return {
            'k': k_percent,
            'd': d_percent
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    @staticmethod
    def support_resistance_levels(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Identify support and resistance levels"""
        high_col = 'High'
        low_col = 'Low'

        if high_col not in data.columns or low_col not in data.columns:
            return {'support': [], 'resistance': []}

        # Local maxima and minima
        highs = data[high_col]
        lows = data[low_col]

        # Find peaks and troughs
        resistance_levels = []
        support_levels = []

        for i in range(window, len(data) - window):
            # Check if current high is a local maximum
            if highs.iloc[i] == highs.iloc[i-window:i+window+1].max():
                resistance_levels.append(highs.iloc[i])

            # Check if current low is a local minimum
            if lows.iloc[i] == lows.iloc[i-window:i+window+1].min():
                support_levels.append(lows.iloc[i])

        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))

        # Keep only the most significant levels (top 3 of each)
        return {
            'resistance': resistance_levels[:3],
            'support': support_levels[-3:]
        }

    @staticmethod
    def volume_analysis(data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """Analyze volume patterns"""
        if 'Volume' not in data.columns:
            return {}

        volume = data['Volume']
        volume_sma = volume.rolling(window=window).mean()

        current_volume = volume.iloc[-1] if len(volume) > 0 else 0
        avg_volume = volume_sma.iloc[-1] if len(volume_sma) > 0 else 0

        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        return {
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': volume_ratio,
            'volume_trend': 'high' if volume_ratio > 1.5 else 'normal' if volume_ratio > 0.8 else 'low'
        }

    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive set of technical indicators"""
        try:
            if data.empty or len(data) < 26:  # Need enough data for indicators
                return {}

            close = data['Close']
            high = data['High']
            low = data['Low']

            indicators = {
                # Moving Averages
                'sma_20': self.sma(close, 20).iloc[-1] if len(close) >= 20 else None,
                'sma_50': self.sma(close, 50).iloc[-1] if len(close) >= 50 else None,
                'ema_12': self.ema(close, 12).iloc[-1] if len(close) >= 12 else None,
                'ema_26': self.ema(close, 26).iloc[-1] if len(close) >= 26 else None,

                # Momentum Indicators
                'rsi': self.rsi(close).iloc[-1] if len(close) >= 14 else None,

                # MACD
                'macd_data': self.macd(close),

                # Bollinger Bands
                'bollinger': self.bollinger_bands(close),

                # Stochastic
                'stochastic': self.stochastic(high, low, close) if len(close) >= 14 else None,

                # Volatility
                'atr': self.atr(high, low, close).iloc[-1] if len(close) >= 14 else None,

                # Support/Resistance
                'levels': self.support_resistance_levels(data),

                # Volume Analysis
                'volume': self.volume_analysis(data)
            }

            # Get current values for key indicators
            current_indicators = {}

            if indicators['macd_data']:
                macd_data = indicators['macd_data']
                if len(macd_data['macd']) > 0:
                    current_indicators['macd'] = macd_data['macd'].iloc[-1]
                    current_indicators['macd_signal'] = macd_data['signal'].iloc[-1]
                    current_indicators['macd_histogram'] = macd_data['histogram'].iloc[-1]

            if indicators['bollinger']:
                bb_data = indicators['bollinger']
                if len(bb_data['middle']) > 0:
                    current_indicators['bb_upper'] = bb_data['upper'].iloc[-1]
                    current_indicators['bb_middle'] = bb_data['middle'].iloc[-1]
                    current_indicators['bb_lower'] = bb_data['lower'].iloc[-1]

            if indicators['stochastic']:
                stoch_data = indicators['stochastic']
                if len(stoch_data['k']) > 0:
                    current_indicators['stoch_k'] = stoch_data['k'].iloc[-1]
                    current_indicators['stoch_d'] = stoch_data['d'].iloc[-1]

            # Add other current values
            for key, value in indicators.items():
                if key not in ['macd_data', 'bollinger', 'stochastic'] and value is not None:
                    current_indicators[key] = value

            return current_indicators

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}


# Create singleton instance
technical_indicators = TechnicalIndicators()

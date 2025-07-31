"""
Technical Analysis Agent for LangGraph Trading System
"""
from typing import Dict, Any
import logging
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from tools.data_fetcher import data_fetcher
from tools.technical_indicators import technical_indicators
from agents.state import TradingState

logger = logging.getLogger(__name__)


class TechnicalAnalysisAgent:
    """Agent for performing technical analysis on stock data"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical analyst specializing in day trading.

            Your task is to analyze the provided technical indicators and generate trading signals.

            Consider the following in your analysis:
            - Current price trends and momentum
            - Support and resistance levels
            - Volume patterns
            - Multiple timeframe convergence
            - Risk-reward scenarios

            Provide your analysis in a structured format with:
            1. Signal strength (-1 to 1, where -1 is strong sell, 1 is strong buy)
            2. Key technical levels (support/resistance)
            3. Rationale for your analysis
            4. Short-term price targets

            Focus on intraday trading opportunities with clear entry/exit points."""),
            ("human", "Analyze the following technical data for {symbol}:\n\n{technical_data}")
        ])

    def analyze(self, state: TradingState) -> Dict[str, Any]:
        """Perform technical analysis and update state"""
        try:
            symbol = state["symbol"]
            timeframe = state.get("timeframe", "1h")
            analysis_period = state.get("analysis_period", 30)

            logger.info(f"Starting technical analysis for {symbol}")

            # Fetch price data
            price_data = data_fetcher.fetch_stock_data(
                symbol=symbol,
                period=f"{analysis_period}d",
                interval=timeframe
            )

            if price_data is None or price_data.empty:
                return {"errors": state.get("errors", []) + [f"No price data available for {symbol}"]}

            # Calculate technical indicators
            indicators = technical_indicators.calculate_all_indicators(price_data)

            if not indicators:
                return {"errors": state.get("errors", []) + ["Failed to calculate technical indicators"]}

            # Generate trading signals using LLM
            technical_analysis = self._generate_llm_analysis(symbol, indicators, price_data)

            price_data.index = price_data.index.strftime('%Y-%m-%d %H:%M:%S')
            return {
                "price_data": price_data,
                "current_price": float(price_data['Close'].iloc[-1]),
                "technical_indicators": indicators,
                "technical_signals": {
                    'signal_strength': technical_analysis.get('signal_strength', 0.0),
                    'trend_direction': technical_analysis.get('trend_direction', 'NEUTRAL'),
                    'momentum': technical_analysis.get('momentum', 'NEUTRAL')
                },
                "support_resistance": {
                    'support': indicators.get('levels', {}).get('support', []),
                    'resistance': indicators.get('levels', {}).get('resistance', [])
                }
            }

        except Exception as e:
            error_msg = f"Technical analysis error: {str(e)}"
            logger.error(error_msg)
            return {"errors": state.get("errors", []) + [error_msg]}

    def _generate_llm_analysis(self, symbol: str, indicators: Dict[str, Any], price_data) -> Dict[str, Any]:
        """Use LLM to interpret technical indicators"""
        try:
            # Format technical data for LLM
            current_price = price_data['Close'].iloc[-1]
            formatted_data = self._format_technical_data(indicators, current_price)

            # Generate analysis using LLM
            chain = self.prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "technical_data": formatted_data
            })

            # Parse LLM response
            analysis_text = response.content if hasattr(response, 'content') else str(response)

            # Extract signal strength (simplified parsing)
            signal_strength = self._extract_signal_strength(analysis_text, indicators)
            trend_direction = self._determine_trend_direction(indicators)
            momentum = self._determine_momentum(indicators)

            return {
                'signal_strength': signal_strength,
                'trend_direction': trend_direction,
                'momentum': momentum,
                'analysis_text': analysis_text,
                'technical_summary': self._create_technical_summary(indicators)
            }

        except Exception as e:
            logger.error(f"Error in LLM technical analysis: {str(e)}")
            return {
                'signal_strength': 0.0,
                'trend_direction': 'NEUTRAL',
                'momentum': 'NEUTRAL',
                'analysis_text': f"Analysis error: {str(e)}",
                'technical_summary': {}
            }

    def _format_technical_data(self, indicators: Dict[str, Any], current_price: float) -> str:
        """Format technical indicators for LLM analysis"""
        try:
            formatted = f"Current Price: ${current_price:.2f}\n\n"

            # Moving Averages
            if indicators.get('sma_20'):
                formatted += f"SMA(20): ${indicators['sma_20']:.2f}\n"
            if indicators.get('sma_50'):
                formatted += f"SMA(50): ${indicators['sma_50']:.2f}\n"
            if indicators.get('ema_12'):
                formatted += f"EMA(12): ${indicators['ema_12']:.2f}\n"

            # Momentum Indicators
            if indicators.get('rsi'):
                formatted += f"RSI(14): {indicators['rsi']:.2f}\n"

            # MACD
            if indicators.get('macd'):
                formatted += f"MACD: {indicators['macd']:.4f}\n"
            if indicators.get('macd_signal'):
                formatted += f"MACD Signal: {indicators['macd_signal']:.4f}\n"

            # Bollinger Bands
            if indicators.get('bb_upper') and indicators.get('bb_lower'):
                formatted += f"Bollinger Bands: ${indicators['bb_lower']:.2f} - ${indicators['bb_upper']:.2f}\n"

            # Stochastic
            if indicators.get('stoch_k') and indicators.get('stoch_d'):
                formatted += f"Stochastic K: {indicators['stoch_k']:.2f}, D: {indicators['stoch_d']:.2f}\n"

            # Volume
            volume_info = indicators.get('volume', {})
            if volume_info:
                formatted += f"Volume Trend: {volume_info.get('volume_trend', 'unknown')}\n"
                formatted += f"Volume Ratio: {volume_info.get('volume_ratio', 0):.2f}\n"

            # Support/Resistance
            levels = indicators.get('levels', {})
            if levels.get('support'):
                formatted += f"Support Levels: {', '.join(f'${level:.2f}' for level in levels['support'])}\n"
            if levels.get('resistance'):
                formatted += f"Resistance Levels: {', '.join(f'${level:.2f}' for level in levels['resistance'])}\n"

            return formatted

        except Exception as e:
            logger.error(f"Error formatting technical data: {str(e)}")
            return f"Current Price: ${current_price:.2f}\nError formatting additional data"

    def _extract_signal_strength(self, analysis_text: str, indicators: Dict[str, Any]) -> float:
        """Extract signal strength from analysis"""
        try:
            # Simplified signal strength calculation based on key indicators
            signal_strength = 0.0

            # RSI signals
            rsi = indicators.get('rsi', 50)
            if rsi > 70:
                signal_strength -= 0.3  # Overbought
            elif rsi < 30:
                signal_strength += 0.3  # Oversold

            # MACD signals
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if macd > macd_signal:
                signal_strength += 0.2
            else:
                signal_strength -= 0.2

            # Moving average signals
            current_price = indicators.get('current_price', 0)
            sma_20 = indicators.get('sma_20', current_price)
            if current_price > sma_20:
                signal_strength += 0.2
            else:
                signal_strength -= 0.2

            # Volume confirmation
            volume_info = indicators.get('volume', {})
            if volume_info.get('volume_trend') == 'high':
                signal_strength *= 1.2  # Boost signal with high volume

            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, signal_strength))

        except Exception as e:
            logger.error(f"Error extracting signal strength: {str(e)}")
            return 0.0

    def _determine_trend_direction(self, indicators: Dict[str, Any]) -> str:
        """Determine overall trend direction"""
        try:
            sma_20 = indicators.get('sma_20', 0)
            sma_50 = indicators.get('sma_50', 0)

            if sma_20 > sma_50 * 1.01:  # 1% threshold
                return 'UPTREND'
            elif sma_20 < sma_50 * 0.99:
                return 'DOWNTREND'
            else:
                return 'SIDEWAYS'

        except Exception:
            return 'NEUTRAL'

    def _determine_momentum(self, indicators: Dict[str, Any]) -> str:
        """Determine momentum based on indicators"""
        try:
            rsi = indicators.get('rsi', 50)
            macd_histogram = indicators.get('macd_histogram', 0)

            if rsi > 55 and macd_histogram > 0:
                return 'BULLISH'
            elif rsi < 45 and macd_histogram < 0:
                return 'BEARISH'
            else:
                return 'NEUTRAL'

        except Exception:
            return 'NEUTRAL'

    def _create_technical_summary(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of key technical points"""
        try:
            return {
                'rsi_level': indicators.get('rsi', 50),
                'rsi_signal': 'Overbought' if indicators.get('rsi', 50) > 70 else 'Oversold' if indicators.get('rsi', 50) < 30 else 'Neutral',
                'trend_strength': abs(indicators.get('macd', 0)),
                'volume_trend': indicators.get('volume', {}).get('volume_trend', 'unknown'),
                'support_levels': indicators.get('levels', {}).get('support', []),
                'resistance_levels': indicators.get('levels', {}).get('resistance', [])
            }

        except Exception:
            return {}


# Create singleton instance
technical_analysis_agent = TechnicalAnalysisAgent()

"""
Sentiment Analysis Agent for LangGraph Trading System
"""
from typing import Dict, Any
import logging
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from tools.data_fetcher import data_fetcher
from tools.sentiment_analysis import sentiment_analyzer
from agents.state import TradingState

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    """Agent for analyzing news and market sentiment"""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial sentiment analyst specializing in market psychology and news impact.

            Your task is to analyze news sentiment and its potential impact on stock prices, especially for day trading.

            Consider the following:
            - Market sentiment trends and shifts
            - News relevance and credibility
            - Potential price impact magnitude and duration
            - Market timing and catalyst events

            Provide analysis including:
            1. Sentiment score (-1 to 1)
            2. Impact assessment (high/medium/low)
            3. Time horizon for impact
            4. Key sentiment drivers

            Focus on sentiment that could affect intraday price movements."""),
            ("human", "Analyze the sentiment for {symbol} based on the following data:\n\n{sentiment_data}")
        ])

    def analyze(self, state: TradingState) -> TradingState:
        """Perform sentiment analysis and update state"""
        try:
            symbol = state["symbol"]

            logger.info(f"Starting sentiment analysis for {symbol}")

            # Fetch news articles
            news_articles = data_fetcher.fetch_news(symbol, days=3)  # Last 3 days for day trading

            if not news_articles:
                logger.warning(f"No news articles found for {symbol}")
                state["news_sentiment"] = 0.0
                state["social_sentiment"] = 0.0
                state["sentiment_sources"] = []
                return state

            # Analyze sentiment using sentiment analyzer
            sentiment_results = sentiment_analyzer.analyze_news_articles(news_articles, symbol)

            # Store results in state
            state["news_sentiment"] = sentiment_results.get('overall_sentiment', 0.0)
            state["sentiment_sources"] = sentiment_results.get('articles_analysis', [])

            # Generate enhanced sentiment analysis using LLM
            llm_analysis = self._generate_llm_sentiment_analysis(
                symbol, 
                sentiment_results, 
                news_articles[:10]  # Top 10 articles
            )

            # Store enhanced analysis
            state["social_sentiment"] = llm_analysis.get('enhanced_sentiment', 0.0)

            logger.info(f"Sentiment analysis completed for {symbol}")
            return state

        except Exception as e:
            error_msg = f"Sentiment analysis error: {str(e)}"
            logger.error(error_msg)
            state["errors"] = state.get("errors", []) + [error_msg]
            state["news_sentiment"] = 0.0
            state["social_sentiment"] = 0.0
            return state

    def _generate_llm_sentiment_analysis(self, symbol: str, sentiment_results: Dict[str, Any], 
                                       top_articles: list) -> Dict[str, Any]:
        """Use LLM to enhance sentiment analysis"""
        try:
            # Format sentiment data for LLM
            formatted_data = self._format_sentiment_data(sentiment_results, top_articles)

            # Generate analysis using LLM
            chain = self.prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "sentiment_data": formatted_data
            })

            # Parse LLM response
            analysis_text = response.content if hasattr(response, 'content') else str(response)

            # Calculate enhanced sentiment score
            base_sentiment = sentiment_results.get('overall_sentiment', 0.0)
            enhanced_sentiment = self._calculate_enhanced_sentiment(
                base_sentiment, analysis_text, sentiment_results
            )

            return {
                'enhanced_sentiment': enhanced_sentiment,
                'impact_assessment': self._assess_impact(sentiment_results),
                'sentiment_confidence': sentiment_results.get('confidence', 0.0),
                'analysis_text': analysis_text,
                'key_themes': self._extract_themes(top_articles)
            }

        except Exception as e:
            logger.error(f"Error in LLM sentiment analysis: {str(e)}")
            return {
                'enhanced_sentiment': sentiment_results.get('overall_sentiment', 0.0),
                'impact_assessment': 'MEDIUM',
                'sentiment_confidence': 0.5,
                'analysis_text': f"Analysis error: {str(e)}",
                'key_themes': []
            }

    def _format_sentiment_data(self, sentiment_results: Dict[str, Any], articles: list) -> str:
        """Format sentiment data for LLM analysis"""
        try:
            formatted = f"Overall Sentiment Score: {sentiment_results.get('overall_sentiment', 0.0):.3f}\n"
            formatted += f"Confidence: {sentiment_results.get('confidence', 0.0):.3f}\n"
            formatted += f"Total Articles: {sentiment_results.get('article_count', 0)}\n"
            formatted += f"Positive: {sentiment_results.get('positive_count', 0)}, "
            formatted += f"Negative: {sentiment_results.get('negative_count', 0)}, "
            formatted += f"Neutral: {sentiment_results.get('neutral_count', 0)}\n\n"

            # Add top article headlines
            formatted += "Recent Headlines:\n"
            for i, article in enumerate(articles[:5], 1):
                title = article.get('title', 'No title')[:100]
                source = article.get('source', 'Unknown')
                formatted += f"{i}. {title} ({source})\n"

            return formatted

        except Exception as e:
            logger.error(f"Error formatting sentiment data: {str(e)}")
            return "Error formatting sentiment data"

    def _calculate_enhanced_sentiment(self, base_sentiment: float, analysis_text: str, 
                                    sentiment_results: Dict[str, Any]) -> float:
        """Calculate enhanced sentiment incorporating LLM insights"""
        try:
            enhanced = base_sentiment

            # Adjust based on confidence
            confidence = sentiment_results.get('confidence', 0.5)
            enhanced *= confidence

            # Adjust based on article count (more articles = more reliable)
            article_count = sentiment_results.get('article_count', 0)
            if article_count > 10:
                enhanced *= 1.1  # Boost for high article count
            elif article_count < 3:
                enhanced *= 0.8  # Reduce for low article count

            # Check for strong positive/negative keywords in LLM analysis
            analysis_lower = analysis_text.lower()
            strong_positive = ['very bullish', 'strong positive', 'extremely positive', 'highly bullish']
            strong_negative = ['very bearish', 'strong negative', 'extremely negative', 'highly bearish']

            if any(phrase in analysis_lower for phrase in strong_positive):
                enhanced = min(1.0, enhanced * 1.2)
            elif any(phrase in analysis_lower for phrase in strong_negative):
                enhanced = max(-1.0, enhanced * 1.2)

            return max(-1.0, min(1.0, enhanced))

        except Exception as e:
            logger.error(f"Error calculating enhanced sentiment: {str(e)}")
            return base_sentiment

    def _assess_impact(self, sentiment_results: Dict[str, Any]) -> str:
        """Assess potential market impact of sentiment"""
        try:
            sentiment_score = abs(sentiment_results.get('overall_sentiment', 0.0))
            confidence = sentiment_results.get('confidence', 0.0)
            article_count = sentiment_results.get('article_count', 0)

            # Calculate impact score
            impact_score = sentiment_score * confidence * min(1.0, article_count / 10.0)

            if impact_score > 0.6:
                return 'HIGH'
            elif impact_score > 0.3:
                return 'MEDIUM'
            else:
                return 'LOW'

        except Exception:
            return 'MEDIUM'

    def _extract_themes(self, articles: list) -> list:
        """Extract key themes from article titles"""
        try:
            themes = []
            keywords = {
                'earnings': ['earnings', 'quarterly', 'revenue', 'profit'],
                'analyst': ['upgrade', 'downgrade', 'rating', 'analyst', 'target'],
                'product': ['product', 'launch', 'announcement', 'partnership'],
                'regulatory': ['regulation', 'fda', 'approval', 'compliance'],
                'market': ['market', 'sector', 'industry', 'competition']
            }

            for article in articles:
                title = article.get('title', '').lower()
                for theme, theme_keywords in keywords.items():
                    if any(keyword in title for keyword in theme_keywords):
                        themes.append(theme)
                        break

            # Return unique themes with counts
            theme_counts = {}
            for theme in themes:
                theme_counts[theme] = theme_counts.get(theme, 0) + 1

            return sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)

        except Exception:
            return []


# Create singleton instance
sentiment_analysis_agent = SentimentAnalysisAgent()

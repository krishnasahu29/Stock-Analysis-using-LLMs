"""
Data fetching tools for market data and news
"""
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import logging
from newsdataapi import NewsDataApiClient
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles fetching of market data and news"""

    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if self.news_api_key:
            self.news_client = NewsDataApiClient(apikey=self.news_api_key)

    def fetch_stock_data(self, symbol: str, period: str = "30d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """
        Fetch stock data from Yahoo Finance

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                logger.error(f"No data found for symbol {symbol}")
                return None

            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"Missing required columns in data for {symbol}")
                return None

            return data

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def fetch_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'float_shares': info.get('floatShares', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {str(e)}")
            return None

    def fetch_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Fetch recent news for a stock symbol

        Args:
            symbol: Stock symbol
            days: Number of days to look back

        Returns:
            List of news articles
        """
        news_list = []

        # Try NewsAPI first
        if self.news_client:
            try:
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

                # Search for company news
                ticker = yf.Ticker(symbol)
                company_name = ticker.info.get('longName', symbol)

                response = self.news_client.news_api(
                    q=f"{symbol} OR {company_name}",
                    from_date=from_date,
                    language='en'
                )

                for article in response.get('results', []):
                    news_list.append({
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'url': article.get('link'),
                        'published_at': article.get('pubDate'),
                        'source': article.get('source_id'),
                        'content': article.get('content', '')
                    })

            except Exception as e:
                logger.error(f"Error fetching news from NewsAPI: {str(e)}")

        # Fallback: Try Yahoo Finance news
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news

            for item in news[:20]:  # Limit to 20 articles
                news_list.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'url': item.get('link', ''),
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat(),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'content': item.get('summary', '')
                })

        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news: {str(e)}")

        return news_list

    def fetch_market_overview(self) -> Dict[str, Any]:
        """Fetch overall market indicators"""
        try:
            # Major indices
            indices = {
                '^GSPC': 'S&P 500',
                '^DJI': 'Dow Jones',
                '^IXIC': 'NASDAQ',
                '^VIX': 'VIX'
            }

            market_data = {}
            for symbol, name in indices.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d", interval="1d")
                if len(hist) >= 2:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = ((current - previous) / previous) * 100

                    market_data[name] = {
                        'current': current,
                        'change_percent': change,
                        'symbol': symbol
                    }

            return market_data

        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            return {}


# Create singleton instance
data_fetcher = DataFetcher()

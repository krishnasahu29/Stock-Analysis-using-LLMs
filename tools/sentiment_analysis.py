"""
Sentiment analysis tools for news and social media
"""
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import Dict, List, Any, Optional
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from news articles and social media"""

    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_keywords = {
            'positive': [
                'bullish', 'rally', 'surge', 'gain', 'profit', 'growth', 'strong', 
                'upgrade', 'beat', 'exceed', 'outperform', 'positive', 'buy',
                'revenue growth', 'earnings beat', 'new high', 'breakout'
            ],
            'negative': [
                'bearish', 'decline', 'fall', 'loss', 'weak', 'downgrade', 
                'miss', 'underperform', 'negative', 'sell', 'crash', 'plunge',
                'earnings miss', 'revenue decline', 'new low', 'breakdown'
            ]
        }

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis"""
        if not text:
            return ""
 
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Remove special characters but keep important punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.!?,-]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def financial_keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment based on financial keywords"""
        if not text:
            return 0.0

        text_lower = text.lower()
        positive_count = sum(1 for keyword in self.financial_keywords['positive'] 
                           if keyword in text_lower)
        negative_count = sum(1 for keyword in self.financial_keywords['negative'] 
                           if keyword in text_lower)

        total_keywords = positive_count + negative_count
        if total_keywords == 0:
            return 0.0

        return (positive_count - negative_count) / total_keywords

    def textblob_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity  # 0 to 1
            }
        except Exception as e:
            logger.error(f"TextBlob sentiment analysis error: {str(e)}")
            return {'polarity': 0.0, 'subjectivity': 0.0}

    def vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment using VADER"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'compound': scores['compound'],
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
        except Exception as e:
            logger.error(f"VADER sentiment analysis error: {str(e)}")
            return {'compound': 0.0, 'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}

    def analyze_single_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """Analyze sentiment for a single text"""
        if not text:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'source': source,
                'length': 0
            }

        cleaned_text = self.preprocess_text(text)

        # Get sentiment from different methods
        textblob_scores = self.textblob_sentiment(cleaned_text)
        vader_scores = self.vader_sentiment(cleaned_text)
        keyword_score = self.financial_keyword_sentiment(cleaned_text)

        # Combine scores with weighted average
        # VADER compound score has good performance for financial text
        # TextBlob polarity provides additional perspective
        # Financial keywords add domain-specific insight
        combined_score = (
            vader_scores['compound'] * 0.4 +
            textblob_scores['polarity'] * 0.3 +
            keyword_score * 0.3
        )

        # Calculate confidence based on agreement between methods
        scores = [vader_scores['compound'], textblob_scores['polarity'], keyword_score]
        agreement = 1.0 - (max(scores) - min(scores)) / 2.0  # Higher when scores agree

        return {
            'sentiment_score': combined_score,
            'confidence': max(0.1, agreement),  # Minimum confidence of 0.1
            'textblob_polarity': textblob_scores['polarity'],
            'vader_compound': vader_scores['compound'],
            'keyword_sentiment': keyword_score,
            'source': source,
            'length': len(cleaned_text),
            'processed_text': cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
        }

    def analyze_news_articles(self, articles: List[Dict[str, Any]], 
                            symbol: str = "") -> Dict[str, Any]:
        """Analyze sentiment for a list of news articles"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'articles_analysis': []
            }

        article_sentiments = []
        articles_analysis = []

        for article in articles:
            # Combine title and description for analysis
            text_to_analyze = ""
            if article.get('title'):
                text_to_analyze += article['title'] + ". "
            if article.get('description'):
                text_to_analyze += article['description']

            # Analyze the article
            analysis = self.analyze_single_text(
                text_to_analyze, 
                article.get('source', 'unknown')
            )

            # Add article metadata
            analysis.update({
                'title': article.get('title', '')[:100],
                'url': article.get('url', ''),
                'published_at': article.get('published_at', ''),
                'relevance_score': self._calculate_relevance(text_to_analyze, symbol)
            })

            articles_analysis.append(analysis)
            article_sentiments.append({
                'score': analysis['sentiment_score'],
                'confidence': analysis['confidence'],
                'relevance': analysis['relevance_score']
            })

        # Calculate overall sentiment weighted by confidence and relevance
        if article_sentiments:
            weighted_sum = 0.0
            weight_sum = 0.0

            for sentiment in article_sentiments:
                weight = sentiment['confidence'] * sentiment['relevance']
                weighted_sum += sentiment['score'] * weight
                weight_sum += weight

            overall_sentiment = weighted_sum / weight_sum if weight_sum > 0 else 0.0

            # Count sentiment categories
            positive_count = sum(1 for s in article_sentiments if s['score'] > 0.1)
            negative_count = sum(1 for s in article_sentiments if s['score'] < -0.1)
            neutral_count = len(article_sentiments) - positive_count - negative_count

            # Calculate overall confidence
            avg_confidence = sum(s['confidence'] for s in article_sentiments) / len(article_sentiments)

        else:
            overall_sentiment = 0.0
            positive_count = negative_count = neutral_count = 0
            avg_confidence = 0.0

        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'article_count': len(articles),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles_analysis': articles_analysis
        }

    def _calculate_relevance(self, text: str, symbol: str) -> float:
        """Calculate how relevant an article is to the given symbol"""
        if not text or not symbol:
            return 0.5  # Default relevance

        text_lower = text.lower()
        symbol_lower = symbol.lower()

        # Direct symbol mention
        relevance_score = 0.5  # Base score

        if symbol_lower in text_lower:
            relevance_score += 0.3

        # Company name mentions would require company info lookup
        # For now, use simple keyword matching
        financial_mentions = sum(1 for keyword in 
                               self.financial_keywords['positive'] + self.financial_keywords['negative'] 
                               if keyword in text_lower)

        # Boost relevance based on financial keyword density
        text_length = len(text_lower.split())
        if text_length > 0:
            keyword_density = financial_mentions / text_length
            relevance_score += min(0.2, keyword_density * 10)  # Cap at 0.2 boost

        return min(1.0, relevance_score)  # Cap at 1.0


# Create singleton instance
sentiment_analyzer = SentimentAnalyzer()

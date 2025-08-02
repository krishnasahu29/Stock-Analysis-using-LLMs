import React from 'react';
import PriceChart from './PriceChart';

const Results = ({ data }) => {
  if (!data) {
    return null;
  }

  const final_state = data[data.length - 1]['__end__'];

  return (
    <div className="results">
      <h2>Results</h2>
      <div className="trade-recommendation">
        <h3>Trade Recommendation</h3>
        <p>
          <strong>Trade Signal:</strong> {final_state.trade_signal}
        </p>
        <p>
          <strong>Confidence Score:</strong>{' '}
          {(final_state.confidence_score * 100).toFixed(2)}%
        </p>
        <p>
          <strong>Entry Price:</strong> ${final_state.entry_price.toFixed(2)}
        </p>
        <p>
          <strong>Trade Rationale:</strong> {final_state.trade_rationale}
        </p>
      </div>
      <div className="price-chart">
        <h3>Price Chart</h3>
        <PriceChart data={final_state} />
      </div>
      <div className="technical-analysis">
        <h3>Technical Analysis</h3>
        <pre>{JSON.stringify(final_state.technical_signals, null, 2)}</pre>
        <pre>{JSON.stringify(final_state.support_resistance, null, 2)}</pre>
        <pre>{JSON.stringify(final_state.technical_indicators, null, 2)}</pre>
      </div>
      <div className="sentiment-analysis">
        <h3>Sentiment Analysis</h3>
        <p>News Sentiment: {final_state.news_sentiment.toFixed(2)}</p>
        <p>Social Sentiment: {final_state.social_sentiment.toFixed(2)}</p>
        <h4>News Articles</h4>
        <ul>
          {final_state.sentiment_sources.map((article, index) => (
            <li key={index}>
              <a href={article.url} target="_blank" rel="noopener noreferrer">
                {article.title}
              </a>
            </li>
          ))}
        </ul>
      </div>
      <div className="risk-analysis">
        <h3>Risk Analysis</h3>
        <p>Position Size: {final_state.position_size.toFixed(2)}</p>
        <p>Stop Loss: ${final_state.stop_loss.toFixed(2)}</p>
        <p>Take Profit: ${final_state.take_profit.toFixed(2)}</p>
        <p>Risk/Reward Ratio: {final_state.risk_reward_ratio.toFixed(2)}</p>
      </div>
    </div>
  );
};

export default Results;

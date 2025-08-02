import React, { useState } from 'react';

const InputForm = ({ onAnalysis }) => {
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('1h');
  const [analysisPeriod, setAnalysisPeriod] = useState(30);

  const handleSubmit = (e) => {
    e.preventDefault();
    onAnalysis({ symbol, timeframe, analysisPeriod });
  };

  return (
    <form onSubmit={handleSubmit} className="input-form">
      <div className="form-group">
        <label htmlFor="symbol">Stock Symbol</label>
        <input
          type="text"
          id="symbol"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
        />
      </div>
      <div className="form-group">
        <label htmlFor="timeframe">Timeframe</label>
        <select
          id="timeframe"
          value={timeframe}
          onChange={(e) => setTimeframe(e.target.value)}
        >
          <option value="1m">1m</option>
          <option value="5m">5m</option>
          <option value="15m">15m</option>
          <option value="30m">30m</option>
          <option value="1h">1h</option>
          <option value="1d">1d</option>
        </select>
      </div>
      <div className="form-group">
        <label htmlFor="analysis-period">Analysis Period (days)</label>
        <input
          type="number"
          id="analysis-period"
          value={analysisPeriod}
          onChange={(e) => setAnalysisPeriod(parseInt(e.target.value))}
        />
      </div>
      <button type="submit">Run Analysis</button>
    </form>
  );
};

export default InputForm;

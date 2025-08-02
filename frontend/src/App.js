import React, { useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import InputForm from './components/InputForm';
import Results from './components/Results';
import './App.css';

const API_URL = 'http://localhost:8000/analyze';

function App() {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalysis = async (params) => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(API_URL, params);
      setAnalysisData(response.data);
    } catch (error) {
      setError(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <Header />
      <main>
        <InputForm onAnalysis={handleAnalysis} />
        {loading && <p>Loading...</p>}
        {error && <p>Error: {error.message}</p>}
        <Results data={analysisData} />
      </main>
    </div>
  );
}

export default App;

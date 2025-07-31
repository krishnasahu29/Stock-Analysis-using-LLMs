"""
FastAPI server for LangGraph trading analysis
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv

from ..trading_graph import build_trading_graph

load_dotenv()

app = FastAPI(title="LangGraph Stock Analysis API")

graph = build_trading_graph()

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    analysis_period: int = 30  # days


@app.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        state = {
            'symbol': request.symbol.upper(),
            'timeframe': request.timeframe,
            'analysis_period': request.analysis_period
        }
        result = graph.invoke(state)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for local run
if __name__ == "__main__":
    host = os.getenv('API_HOST', '127.0.0.1')
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host=host, port=port)

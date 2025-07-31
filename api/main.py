"""
FastAPI server for LangGraph trading analysis
"""
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from dotenv import load_dotenv
import json
import numpy as np
from fastapi.responses import JSONResponse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_graph import build_trading_graph

load_dotenv()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = FastAPI(title="LangGraph Stock Analysis API")

graph = build_trading_graph()
compiled_graph = graph.compile()

class AnalysisRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    analysis_period: int = 30  # days


@app.get("/")
def read_root():
    return {"message": "Welcome to the LangGraph Stock Analysis API"}


import logging
import traceback
import numpy as np

logger = logging.getLogger(__name__)

@app.post("/analyze", response_class=JSONResponse)
async def analyze_stock(request: AnalysisRequest):
    try:
        state = {
            'symbol': request.symbol.upper(),
            'timeframe': request.timeframe,
            'analysis_period': request.analysis_period
        }
        result = compiled_graph.stream(state)
        return json.loads(json.dumps(result, cls=NumpyEncoder))
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for local run
if __name__ == "__main__":
    host = os.getenv('API_HOST', '127.0.0.1')
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host=host, port=port)

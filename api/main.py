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
from fastapi.middleware.cors import CORSMiddleware

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_graph import build_trading_graph

load_dotenv()

import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super(NumpyEncoder, self).default(obj)

app = FastAPI(title="LangGraph Stock Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def sanitize_for_tracer(obj):
    """
    Recursively convert pandas Timestamp keys/values to strings.
    """
    if isinstance(obj, dict):
        safe_dict = {}
        for k, v in obj.items():
            # Convert keys if they are Timestamp
            safe_key = k.isoformat() if isinstance(k, pd.Timestamp) else k
            # Recurse on the value
            safe_dict[safe_key] = sanitize_for_tracer(v)
        return safe_dict
    elif isinstance(obj, list):
        return [sanitize_for_tracer(v) for v in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj

@app.post("/analyze", response_class=JSONResponse)
async def analyze_stock(request: AnalysisRequest):
    try:
        state = {
            'symbol': request.symbol.upper(),
            'timeframe': request.timeframe,
            'analysis_period': request.analysis_period
        }
        safe_state = sanitize_for_tracer(state)
        result = list(compiled_graph.stream(safe_state))
        final_state = result[-1]
        print(final_state)
        return json.loads(json.dumps(final_state, cls=NumpyEncoder))
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for local run
if __name__ == "__main__":
    host = os.getenv('API_HOST', '127.0.0.1')
    port = int(os.getenv('API_PORT', 8000))
    uvicorn.run(app, host=host, port=port)

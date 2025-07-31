"""
LangGraph workflow for the stock analysis system
"""
from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import Dict, Any
import logging

from agents.state import TradingState
from agents.technical_agent import technical_analysis_agent
from agents.sentiment_agent import sentiment_analysis_agent
from agents.risk_agent import risk_management_agent
from agents.portfolio_agent import portfolio_management_agent

logger = logging.getLogger(__name__)


def build_trading_graph() -> StateGraph:
    """Construct LangGraph for the trading workflow"""
    graph = StateGraph(TradingState)

    # Register nodes
    graph.add_node("technical_analysis", technical_analysis_agent.analyze)
    graph.add_node("sentiment_analysis", sentiment_analysis_agent.analyze)
    graph.add_node("risk_analysis", risk_management_agent.analyze)
    graph.add_node("portfolio_decision", portfolio_management_agent.decide)

    # Initial step: run technical and sentiment in parallel
    graph.add_parallel_edges([
        ("technical_analysis", "risk_analysis"),
        ("sentiment_analysis", "risk_analysis")
    ])

    # After risk analysis, go to portfolio decision
    graph.add_edge("risk_analysis", "portfolio_decision")

    # End
    graph.add_edge("portfolio_decision", END)

    return graph

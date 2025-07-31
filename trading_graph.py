"""
LangGraph workflow for the stock analysis system
"""
from langgraph.graph import StateGraph
from langgraph.graph import END
from typing import Dict, Any, List
import logging

from agents.state import TradingState
from agents.technical_agent import technical_analysis_agent
from agents.sentiment_agent import sentiment_analysis_agent
from agents.risk_agent import risk_management_agent
from agents.portfolio_agent import portfolio_management_agent

logger = logging.getLogger(__name__)


def entry_point(state: TradingState) -> Dict[str, Any]:
    """
    This is the entry point for the trading graph.
    It returns the initial state.
    """
    return {"messages": []}


def should_continue(state: TradingState) -> str:
    """
    This function determines which node to go to next.
    In this case, it will always return "continue".
    """
    return "continue"


def build_trading_graph() -> StateGraph:
    """Construct LangGraph for the trading workflow"""
    graph = StateGraph(TradingState)

    # Register nodes
    graph.add_node("entry", entry_point)
    graph.add_node("technical_analysis", technical_analysis_agent.analyze)
    graph.add_node("sentiment_analysis", sentiment_analysis_agent.analyze)
    graph.add_node("risk_analysis", risk_management_agent.analyze)
    graph.add_node("portfolio_decision", portfolio_management_agent.decide)

    # Set the entry point
    graph.set_entry_point("entry")

    # Define the edges
    graph.add_conditional_edges(
        "entry",
        lambda x: ["technical_analysis", "sentiment_analysis"],
    )
    graph.add_edge("technical_analysis", "risk_analysis")
    graph.add_edge("sentiment_analysis", "risk_analysis")
    graph.add_edge("risk_analysis", "portfolio_decision")
    graph.add_edge("portfolio_decision", END)

    return graph

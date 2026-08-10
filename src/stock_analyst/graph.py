from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .agents import (
    consensus_agent,
    edit_report,
    fundamental_agent,
    news_agent,
    technical_agent,
)
from .data import MarketData, load_market_data


class AnalystResult(TypedDict):
    role: str
    text: str


class State(TypedDict, total=False):
    ticker: str
    data: MarketData
    briefings: Annotated[list[AnalystResult], operator.add]
    report: str


def validate_and_load(state: State):
    ticker = state["ticker"].strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    data = load_market_data(ticker)
    if data.history.empty and not data.profile:
        raise ValueError(f"No usable Yahoo Finance data returned for {ticker}.")
    return {"ticker": ticker, "data": data}


def fan_out(state: State):
    # Each Send carries the same immutable market snapshot to one specialist.
    # LangGraph schedules these branches as parallel work in the same graph step.
    return [
        Send("technical", {"ticker": state["ticker"], "data": state["data"]}),
        Send("fundamental", {"ticker": state["ticker"], "data": state["data"]}),
        Send("news", {"ticker": state["ticker"], "data": state["data"]}),
        Send("consensus", {"ticker": state["ticker"], "data": state["data"]}),
    ]


def technical_node(state: State):
    return {"briefings": [{
        "role": "Technical Analyst",
        "text": technical_agent(state["data"]),
    }]}


def fundamental_node(state: State):
    return {"briefings": [{
        "role": "Fundamental Analyst",
        "text": fundamental_agent(state["data"]),
    }]}


def news_node(state: State):
    return {"briefings": [{
        "role": "News Analyst",
        "text": news_agent(state["data"]),
    }]}


def consensus_node(state: State):
    return {"briefings": [{
        "role": "Consensus Analyst",
        "text": consensus_agent(state["data"]),
    }]}


def editor_node(state: State):
    return {"report": edit_report(state["ticker"], state["briefings"])}


builder = StateGraph(State)
builder.add_node("validate_and_load", validate_and_load)
builder.add_node("technical", technical_node)
builder.add_node("fundamental", fundamental_node)
builder.add_node("news", news_node)
builder.add_node("consensus", consensus_node)
builder.add_node("editor", editor_node)

builder.add_edge(START, "validate_and_load")
builder.add_conditional_edges("validate_and_load", fan_out)
builder.add_edge("technical", "editor")
builder.add_edge("fundamental", "editor")
builder.add_edge("news", "editor")
builder.add_edge("consensus", "editor")
builder.add_edge("editor", END)

graph = builder.compile()

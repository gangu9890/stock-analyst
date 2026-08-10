from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import GEMINI_API_KEY, MODEL_ANALYST, MODEL_EDITOR
from .data import (
    MarketData,
    build_consensus_context,
    build_fundamental_context,
    build_news_context,
    build_technical_context,
)

ANALYST_SYSTEM = """You are one specialist in a multi-agent equity research team.
Use only the supplied market data. Do not invent numbers or sources.
Return a concise analyst briefing with:
1) key observations
2) bullish signals
3) bearish signals
4) what could invalidate the view
5) a directional stance: bullish, neutral, or bearish
This is research, not personalized financial advice.
"""


def _model(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )


def _run(role: str, context: str) -> str:
    prompt = f"""{ANALYST_SYSTEM}

Role: {role}

DATA:
{context}
"""
    return _model(MODEL_ANALYST).invoke(prompt).content


def technical_agent(data: MarketData) -> str:
    return _run("Technical Analyst — price action, moving averages, momentum, volatility and drawdown.",
                build_technical_context(data))


def fundamental_agent(data: MarketData) -> str:
    return _run("Fundamental Analyst — revenue, earnings, margins, leverage, valuation and quality.",
                build_fundamental_context(data))


def news_agent(data: MarketData) -> str:
    return _run("News Analyst — identify material recent headlines, catalysts, risks and uncertainty.",
                build_news_context(data))


def consensus_agent(data: MarketData) -> str:
    return _run("Consensus Analyst — analyst ratings, price targets and disagreement/dispersion.",
                build_consensus_context(data))


def edit_report(ticker: str, briefings: list[dict[str, str]]) -> str:
    joined = "\n\n".join(
        f"## {b['role']}\n{b['text']}" for b in briefings
    )
    prompt = f"""You are the Head of Equity Research.

Write a decision-useful research note for {ticker} from the four independent
briefings below.

Required sections:
# {ticker} Equity Research Note
## Verdict
Give Bullish / Neutral / Bearish and 2-3 sentences explaining why.
## Bull Case
## Bear Case
## Where Analysts Disagree
## Key Evidence
Use bullets and preserve important numbers.
## Catalysts
## Risks
## What Would Change the View
## Analyst Coverage
List the four specialist views.

Do not fabricate data. Clearly distinguish reported facts from interpretation.
End with: "Research only — not personalized financial advice."

BRIEFINGS:
{joined}
"""
    return _model(MODEL_EDITOR).invoke(prompt).content

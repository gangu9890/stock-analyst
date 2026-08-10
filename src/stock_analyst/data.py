from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class MarketData:
    ticker: str
    profile: dict[str, Any]
    history: pd.DataFrame
    financials: dict[str, pd.DataFrame]
    news: list[dict[str, Any]]
    recommendations: pd.DataFrame | None
    targets: dict[str, Any]


def load_market_data(ticker: str) -> MarketData:
    t = yf.Ticker(ticker)

    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    try:
        history = t.history(period="1y", auto_adjust=False)
    except Exception:
        history = pd.DataFrame()

    financials = {}
    for name, getter in [
        ("income", lambda: t.income_stmt),
        ("balance", lambda: t.balance_sheet),
        ("cashflow", lambda: t.cashflow),
    ]:
        try:
            financials[name] = getter()
        except Exception:
            financials[name] = pd.DataFrame()

    try:
        news = t.news or []
    except Exception:
        news = []

    try:
        recommendations = t.recommendations
    except Exception:
        recommendations = None

    try:
        targets = t.analyst_price_targets or {}
    except Exception:
        targets = {}

    return MarketData(
        ticker=ticker,
        profile=info,
        history=history,
        financials=financials,
        news=news,
        recommendations=recommendations,
        targets=targets,
    )


def compact_number(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except Exception:
        return str(x)
    for suffix, divisor in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(x) >= divisor:
            return f"{x/divisor:.2f}{suffix}"
    return f"{x:.2f}"


def build_technical_context(data: MarketData) -> str:
    h = data.history
    if h.empty or "Close" not in h:
        return "Technical data unavailable."

    close = h["Close"].dropna()
    last = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else None
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else None
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else None

    def change(n: int):
        if len(close) <= n:
            return None
        return (last / float(close.iloc[-n-1]) - 1) * 100

    peak = float(close.max())
    drawdown = (last / peak - 1) * 100 if peak else None

    return "\n".join([
        f"Ticker: {data.ticker}",
        f"Last close: {last:.2f}",
        f"1W change: {change(5)!s}%",
        f"1M change: {change(21)!s}%",
        f"6M change: {change(126)!s}%",
        f"50D MA: {ma50}",
        f"20D MA: {ma20}",
        f"200D MA: {ma200}",
        f"Distance from 1Y high: {drawdown:.2f}%" if drawdown is not None else "Distance from 1Y high: n/a",
        f"1Y volatility (annualized): {(close.pct_change().dropna().std() * (252 ** 0.5) * 100):.2f}%",
    ])


def build_fundamental_context(data: MarketData) -> str:
    p = data.profile
    income = data.financials.get("income", pd.DataFrame())

    lines = [
        f"Ticker: {data.ticker}",
        f"Company: {p.get('longName', data.ticker)}",
        f"Sector: {p.get('sector', 'n/a')}",
        f"Industry: {p.get('industry', 'n/a')}",
        f"Market cap: {compact_number(p.get('marketCap'))}",
        f"Trailing P/E: {p.get('trailingPE', 'n/a')}",
        f"Forward P/E: {p.get('forwardPE', 'n/a')}",
        f"Profit margin: {p.get('profitMargins', 'n/a')}",
        f"ROE: {p.get('returnOnEquity', 'n/a')}",
        f"Debt/equity: {p.get('debtToEquity', 'n/a')}",
        f"Revenue growth: {p.get('revenueGrowth', 'n/a')}",
    ]

    if not income.empty:
        for label in ["Total Revenue", "Net Income", "Operating Income", "EBITDA"]:
            if label in income.index:
                vals = income.loc[label].dropna()
                if len(vals):
                    lines.append(f"{label}: {compact_number(vals.iloc[0])}")

    return "\n".join(lines)


def build_news_context(data: MarketData) -> str:
    if not data.news:
        return "No recent news returned by yfinance."

    lines = []
    for item in data.news[:8]:
        content = item.get("content", item)
        title = content.get("title") or content.get("headline") or "Untitled"
        publisher = content.get("provider", {}).get("displayName", "")
        lines.append(f"- {title} ({publisher})")
    return "\n".join(lines)


def build_consensus_context(data: MarketData) -> str:
    lines = [
        f"Ticker: {data.ticker}",
        f"Analyst target summary: {data.targets or 'n/a'}",
    ]
    rec = data.recommendations
    if rec is not None and not rec.empty:
        lines.append("Recent recommendation data:")
        lines.append(rec.tail(8).to_string())
    else:
        lines.append("No recommendation history returned.")
    return "\n".join(lines)

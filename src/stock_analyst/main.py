from __future__ import annotations

import argparse
from pathlib import Path

from .graph import graph


def main():
    parser = argparse.ArgumentParser(description="Parallel LangGraph stock analyst")
    parser.add_argument("ticker", nargs="?", default="RELIANCE.NS")
    args = parser.parse_args()

    result = graph.invoke({"ticker": args.ticker})
    report = result["report"]

    out = Path("output")
    out.mkdir(exist_ok=True)
    path = out / f"{args.ticker.upper()}-report.md"
    path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved to: {path}")


if __name__ == "__main__":
    main()

"""
StockTrader — Main Entry Point
Runs the full analysis pipeline for all selected tickers,
then generates summary.json.

Usage:
    python main.py                        # run default tickers
    python main.py AAPL TSLA NVDA GME     # run custom tickers
"""
import sys
import json
from pathlib import Path
from orchestration import run_for_ticker

# ── Default stock selection ────────────────────────────────────────────────────
# Chosen to represent four distinct market conditions:
#   MSFT  → steady large-cap, consistent grower (should produce clearer momentum)
#   NVDA  → high-momentum growth stock (momentum vs contrarian clash expected)
#   PFE   → large-cap that has declined significantly from its highs
#   INTC  → sideways / underperforming stock with no clear direction
DEFAULT_TICKERS = ["MSFT", "NVDA", "PFE", "INTC"]

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def build_summary(results: list[dict]) -> dict:
    total_agree = sum(1 for r in results if r["evaluator"]["agents_agree"])
    total_disagree = len(results) - total_agree

    return {
        "strategies": [
            results[0]["strategy_a"]["name"] if results else "Momentum Trader",
            results[0]["strategy_b"]["name"] if results else "Value Contrarian",
        ],
        "stocks_analyzed": [r["ticker"] for r in results],
        "total_agreements": total_agree,
        "total_disagreements": total_disagree,
        "results": [
            {
                "ticker": r["ticker"],
                "a_decision": r["strategy_a"]["decision"],
                "a_confidence": r["strategy_a"]["confidence"],
                "b_decision": r["strategy_b"]["decision"],
                "b_confidence": r["strategy_b"]["confidence"],
                "agree": r["evaluator"]["agents_agree"],
            }
            for r in results
        ],
    }


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TICKERS

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║              StockTrader — Agent Analysis            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Tickers   : {', '.join(tickers)}")
    print(f"  Strategies: Momentum Trader  vs.  Value Contrarian")
    print(f"  LLM       : Ollama (local)")

    results = []
    errors = []

    for ticker in tickers:
        try:
            result = run_for_ticker(ticker)
            results.append(result)
        except Exception as e:
            print(f"\n  ✗ ERROR on {ticker}: {e}")
            errors.append({"ticker": ticker, "error": str(e)})

    if not results:
        print("\n  No results to summarize. Check your setup and try again.")
        return

    # ── Summary ──────────────────────────────────────────────
    summary = build_summary(results)
    summary_path = OUTPUTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print results table ──────────────────────────────────
    print(f"\n\n{'─'*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'─'*55}")
    print(f"  {'Ticker':<8} {'Momentum':>10} {'Contrarian':>12} {'Agree':>7}")
    print(f"  {'─'*6:<8} {'─'*8:>10} {'─'*10:>12} {'─'*5:>7}")
    for r in summary["results"]:
        agree_str = "✓" if r["agree"] else "✗"
        print(f"  {r['ticker']:<8} {r['a_decision']:>10} {r['b_decision']:>12} {agree_str:>7}")
    print(f"{'─'*55}")
    print(f"  Agreements: {summary['total_agreements']}  |  Disagreements: {summary['total_disagreements']}")
    print(f"  Summary saved → {summary_path}")

    if errors:
        print(f"\n  Failed tickers: {[e['ticker'] for e in errors]}")

    print(f"\n{'═'*55}\n")


if __name__ == "__main__":
    main()

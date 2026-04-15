"""
Orchestration
Wires all components together for a single ticker run.
Strategy A and Strategy B are logically parallel:
  - same input, no cross-visibility before evaluation.
"""
import json
import concurrent.futures
from datetime import datetime
from pathlib import Path

from market_data import fetch_market_data, format_data_for_prompt
from strategies import run_momentum_trader, run_value_contrarian
from evaluator import evaluate

OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def run_for_ticker(ticker: str) -> dict:
    """
    Full pipeline for one stock ticker.
    Returns the complete result dict and saves it to outputs/<TICKER>.json.
    """
    ticker = ticker.upper()
    print(f"\n{'='*55}")
    print(f"  Analyzing {ticker}")
    print(f"{'='*55}")

    # ── Step 1: Market Data (no LLM) ──────────────────────────
    print(f"  [1/4] Fetching market data...")
    data = fetch_market_data(ticker)
    data_text = format_data_for_prompt(data)
    print(f"        Current price: ${data['current_price']}  |  RSI: {data['rsi_14d']}  |  30d: {data['pct_change_30d']:+.1f}%")

    # ── Step 2: Run both strategies in parallel ────────────────
    print(f"  [2/4] Running strategy agents in parallel...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(run_momentum_trader, data_text)
        future_b = executor.submit(run_value_contrarian, data_text)
        result_a = future_a.result()
        result_b = future_b.result()

    print(f"        Momentum Trader  → {result_a['decision']} (confidence {result_a['confidence']}/10)")
    print(f"        Value Contrarian → {result_b['decision']} (confidence {result_b['confidence']}/10)")

    # ── Step 3: Evaluate ───────────────────────────────────────
    print(f"  [3/4] Running evaluator...")
    eval_result = evaluate(result_a, result_b, data_text)
    agreement_label = "✓ AGREE" if eval_result["agents_agree"] else "✗ DISAGREE"
    print(f"        {agreement_label}")

    # ── Step 4: Build and save output ─────────────────────────
    print(f"  [4/4] Saving output...")

    # Build the market_data_summary for JSON (key fields only)
    market_summary = {
        "current_price": data["current_price"],
        "price_30d_ago": data["price_30d_ago"],
        "pct_change_30d": data["pct_change_30d"],
        "pct_change_7d": data["pct_change_7d"],
        "avg_daily_volume": int(data["avg_daily_volume"]),
        "volume_trend_pct": data["volume_trend_pct"],
        "volatility_30d": data["volatility_30d"],
        "moving_avg_20d": data["moving_avg_20d"],
        "moving_avg_50d": data["moving_avg_50d"],
        "above_20d_ma": data["above_20d_ma"],
        "above_50d_ma": data["above_50d_ma"],
        "ma_crossover_bullish": data["ma_crossover_bullish"],
        "rsi_14d": data["rsi_14d"],
        "week52_high": data["week52_high"],
        "week52_low": data["week52_low"],
        "pct_from_52w_high": data["pct_from_52w_high"],
        "pct_from_52w_low": data["pct_from_52w_low"],
        "max_drawdown_90d": data["max_drawdown_90d"],
        "current_drawdown_pct": data["current_drawdown_pct"],
        "momentum_score": data["momentum_score"],
    }

    output = {
        "ticker": ticker,
        "run_date": datetime.today().strftime("%Y-%m-%d"),
        "market_data_summary": market_summary,
        "strategy_a": {
            "name": result_a["name"],
            "decision": result_a["decision"],
            "confidence": result_a["confidence"],
            "justification": result_a["justification"],
        },
        "strategy_b": {
            "name": result_b["name"],
            "decision": result_b["decision"],
            "confidence": result_b["confidence"],
            "justification": result_b["justification"],
        },
        "evaluator": eval_result,
    }

    out_path = OUTPUTS_DIR / f"{ticker}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"        Saved → {out_path}")

    return output

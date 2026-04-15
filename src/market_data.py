"""
Market Data Component
Fetches real stock data via yfinance and computes indicators.
Zero LLM calls in this module.
"""
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


def fetch_market_data(ticker: str) -> dict:
    """
    Fetch historical price data and compute indicators for both
    Momentum Trader and Value Contrarian strategies.

    Returns a structured dict with all derived features.
    """
    stock = yf.Ticker(ticker)

    # Fetch 1 year of history (gives us 52-week high/low + enough for MAs)
    end = datetime.today()
    start = end - timedelta(days=365)
    hist = stock.history(start=start, end=end)

    if hist.empty:
        raise ValueError(f"No price history found for ticker: {ticker}")

    info = {}
    try:
        info = stock.info or {}
    except Exception:
        pass

    closes = hist["Close"]
    volumes = hist["Volume"]
    highs = hist["High"]
    lows = hist["Low"]

    # ── Current price ──────────────────────────────────────────
    current_price = float(closes.iloc[-1])

    # ── Price changes ──────────────────────────────────────────
    price_30d_ago = float(closes.iloc[-31]) if len(closes) >= 31 else float(closes.iloc[0])
    pct_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100

    price_7d_ago = float(closes.iloc[-8]) if len(closes) >= 8 else float(closes.iloc[0])
    pct_change_7d = ((current_price - price_7d_ago) / price_7d_ago) * 100

    # ── Moving averages ────────────────────────────────────────
    moving_avg_20d = float(closes.iloc[-20:].mean()) if len(closes) >= 20 else float(closes.mean())
    moving_avg_50d = float(closes.iloc[-50:].mean()) if len(closes) >= 50 else float(closes.mean())

    # MA trend signal: price vs MAs
    above_20d_ma = current_price > moving_avg_20d
    above_50d_ma = current_price > moving_avg_50d
    ma_crossover = moving_avg_20d > moving_avg_50d  # golden cross if True

    # ── Volume ─────────────────────────────────────────────────
    avg_daily_volume = float(volumes.iloc[-30:].mean()) if len(volumes) >= 30 else float(volumes.mean())
    recent_volume = float(volumes.iloc[-5:].mean()) if len(volumes) >= 5 else float(volumes.mean())
    volume_trend = ((recent_volume - avg_daily_volume) / avg_daily_volume) * 100  # % above/below avg

    # ── Volatility ─────────────────────────────────────────────
    daily_returns = closes.pct_change().dropna()
    volatility_30d = float(daily_returns.iloc[-30:].std()) if len(daily_returns) >= 30 else float(daily_returns.std())

    # ── 52-week high/low ───────────────────────────────────────
    week52_high = float(closes.max())
    week52_low = float(closes.min())
    pct_from_52w_high = ((current_price - week52_high) / week52_high) * 100
    pct_from_52w_low = ((current_price - week52_low) / week52_low) * 100

    # ── Drawdown ───────────────────────────────────────────────
    rolling_max = closes.cummax()
    drawdown_series = (closes - rolling_max) / rolling_max * 100
    max_drawdown_90d = float(drawdown_series.iloc[-90:].min()) if len(drawdown_series) >= 90 else float(drawdown_series.min())
    current_drawdown = float(drawdown_series.iloc[-1])

    # ── RSI (14-day) ───────────────────────────────────────────
    rsi = _compute_rsi(closes, period=14)

    # ── Daily return stats ─────────────────────────────────────
    avg_daily_return_30d = float(daily_returns.iloc[-30:].mean() * 100) if len(daily_returns) >= 30 else float(daily_returns.mean() * 100)
    best_day_30d = float(daily_returns.iloc[-30:].max() * 100) if len(daily_returns) >= 30 else float(daily_returns.max() * 100)
    worst_day_30d = float(daily_returns.iloc[-30:].min() * 100) if len(daily_returns) >= 30 else float(daily_returns.min() * 100)

    # ── Momentum score (simple: returns weighted across timeframes) ─
    momentum_score = (pct_change_7d * 0.4) + (pct_change_30d * 0.6)

    return {
        "ticker": ticker,
        "fetch_date": datetime.today().strftime("%Y-%m-%d"),
        # Core price
        "current_price": round(current_price, 2),
        "price_30d_ago": round(price_30d_ago, 2),
        "price_7d_ago": round(price_7d_ago, 2),
        "pct_change_30d": round(pct_change_30d, 2),
        "pct_change_7d": round(pct_change_7d, 2),
        # Moving averages
        "moving_avg_20d": round(moving_avg_20d, 2),
        "moving_avg_50d": round(moving_avg_50d, 2),
        "above_20d_ma": above_20d_ma,
        "above_50d_ma": above_50d_ma,
        "ma_crossover_bullish": ma_crossover,
        # Volume
        "avg_daily_volume": round(avg_daily_volume, 0),
        "recent_5d_volume": round(recent_volume, 0),
        "volume_trend_pct": round(volume_trend, 2),
        # Volatility
        "volatility_30d": round(volatility_30d, 4),
        # 52-week range
        "week52_high": round(week52_high, 2),
        "week52_low": round(week52_low, 2),
        "pct_from_52w_high": round(pct_from_52w_high, 2),
        "pct_from_52w_low": round(pct_from_52w_low, 2),
        # Drawdown
        "max_drawdown_90d": round(max_drawdown_90d, 2),
        "current_drawdown_pct": round(current_drawdown, 2),
        # RSI
        "rsi_14d": round(rsi, 2),
        # Daily return stats
        "avg_daily_return_30d_pct": round(avg_daily_return_30d, 4),
        "best_day_30d_pct": round(best_day_30d, 2),
        "worst_day_30d_pct": round(worst_day_30d, 2),
        # Composite
        "momentum_score": round(momentum_score, 2),
    }


def _compute_rsi(closes, period: int = 14) -> float:
    """Compute RSI using Wilder's smoothing method."""
    delta = closes.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    if len(gain) < period:
        return 50.0  # neutral fallback

    avg_gain = gain.iloc[:period].mean()
    avg_loss = loss.iloc[:period].mean()

    for i in range(period, len(gain)):
        avg_gain = (avg_gain * (period - 1) + gain.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss.iloc[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def format_data_for_prompt(data: dict) -> str:
    """Format market data dict as a clean text block for LLM prompts."""
    return f"""
STOCK: {data['ticker']}  |  Date: {data['fetch_date']}

PRICE ACTION
  Current price:        ${data['current_price']}
  7-day change:         {data['pct_change_7d']:+.2f}%
  30-day change:        {data['pct_change_30d']:+.2f}%
  52-week high:         ${data['week52_high']}  ({data['pct_from_52w_high']:+.1f}% from current)
  52-week low:          ${data['week52_low']}  ({data['pct_from_52w_low']:+.1f}% from current)

TREND INDICATORS
  20-day MA:            ${data['moving_avg_20d']}  (price is {'ABOVE' if data['above_20d_ma'] else 'BELOW'})
  50-day MA:            ${data['moving_avg_50d']}  (price is {'ABOVE' if data['above_50d_ma'] else 'BELOW'})
  MA crossover signal:  {'Bullish (20d > 50d)' if data['ma_crossover_bullish'] else 'Bearish (20d < 50d)'}
  RSI (14-day):         {data['rsi_14d']} {'(overbought >70)' if data['rsi_14d'] > 70 else '(oversold <30)' if data['rsi_14d'] < 30 else '(neutral)'}
  Momentum score:       {data['momentum_score']:+.2f}

VOLUME
  Avg daily volume:     {int(data['avg_daily_volume']):,}
  Recent 5-day avg:     {int(data['recent_5d_volume']):,}
  Volume vs avg:        {data['volume_trend_pct']:+.1f}%

VOLATILITY & RISK
  30-day volatility:    {data['volatility_30d']:.4f} (daily std dev)
  Max drawdown (90d):   {data['max_drawdown_90d']:.1f}%
  Current drawdown:     {data['current_drawdown_pct']:.1f}% from recent peak
  Best day (30d):       {data['best_day_30d_pct']:+.2f}%
  Worst day (30d):      {data['worst_day_30d_pct']:+.2f}%
""".strip()

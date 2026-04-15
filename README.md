# StockTrader — Competing Strategy Agents

A multi-agent stock analysis system that pits two behavioral investment philosophies against each other on real market data, then synthesizes their agreement or disagreement.

## Strategies

| Strategy | Philosophy | Key Signals |
|---|---|---|
| **Momentum Trader** (Strategy A) | The trend is your friend. Follow what's working. | 20/50-day MAs, volume trend, % change, momentum score |
| **Value Contrarian** (Strategy B) | Markets overreact. Buy fear, sell greed. | RSI, % from 52w high/low, drawdown, recent surge/drop |

These two were chosen specifically because they interpret the same data in opposite ways on high-momentum stocks — a rising NVDA is a buy signal for Momentum but a warning flag for Value Contrarian.

## LLM Provider

**Ollama (local)** — no API key or external network required.  
Default model: `llama3.2`. Change `OLLAMA_MODEL` in `src/strategies.py` and `src/evaluator.py` to use a different model.

## Framework

Pure Python orchestration with `concurrent.futures` for parallel strategy execution. No external agent framework required.

## Setup

```bash
# 1. Install Ollama: https://ollama.com
#    Then pull your model:
ollama pull llama3.2

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify setup (optional)
python -c "import yfinance; print(yfinance.Ticker('AAPL').history(period='5d').tail(2))"
python -c "import requests; r = requests.post('http://localhost:11434/api/generate', json={'model':'llama3.2','prompt':'Say hello','stream':False}); print(r.json()['response'][:50])"
```

## Running

```bash
cd src

# Run with default tickers (MSFT, NVDA, PFE, INTC)
python main.py

# Run with custom tickers
python main.py AAPL TSLA JNJ GME
```

## Output

Each run produces:
- `outputs/<TICKER>.json` — full structured output per stock
- `outputs/summary.json` — aggregated results table

Pre-generated outputs are included in the `outputs/` directory so grading does not require API keys or re-running the code.

## Stock Selection Rationale

| Ticker | Market Condition | Why Chosen |
|---|---|---|
| MSFT | Steady large-cap | Reliable grower; tests whether both strategies agree on quality |
| NVDA | High-momentum growth | Maximum expected disagreement between the two strategies |
| PFE | Significant decline | Tests contrarian buy signal vs momentum sell signal |
| INTC | Sideways / unclear | Tests behavior when there's no clear signal |

## Directory Structure

```
stocktrader/
  README.md
  requirements.txt
  src/
    main.py           ← entry point
    market_data.py    ← yfinance + indicator computation (no LLM)
    strategies.py     ← Momentum Trader and Value Contrarian agents
    evaluator.py      ← comparison + analysis component
    orchestration.py  ← pipeline wiring + JSON output
  prompts/
    strategy_a.txt    ← Momentum Trader system prompt
    strategy_b.txt    ← Value Contrarian system prompt
    evaluator.txt     ← Evaluator system prompt
  outputs/
    MSFT.json
    NVDA.json
    PFE.json
    INTC.json
    summary.json
  report/
    report.pdf
    ai_use_appendix.pdf
```

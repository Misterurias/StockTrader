"""
Strategy Agents
Two independent LLM-powered agents with distinct behavioral philosophies.
Each reads the same market data and produces a structured decision independently.
"""
import json
import re
import requests
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # change to your preferred local model


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def _call_ollama(system_prompt: str, user_message: str, model: str = OLLAMA_MODEL) -> str:
    """Call local Ollama and return the raw response text."""
    full_prompt = f"{system_prompt}\n\n---\n\n{user_message}"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        }
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _parse_agent_response(raw: str, strategy_name: str) -> dict:
    """
    Parse the LLM response into structured output.
    Expects JSON somewhere in the response; falls back to regex extraction.
    """
    # Try to find JSON block
    json_match = re.search(r'\{[^{}]*"decision"[^{}]*\}', raw, re.DOTALL | re.IGNORECASE)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            decision = parsed.get("decision", "HOLD").upper().strip()
            confidence = int(parsed.get("confidence", 5))
            justification = parsed.get("justification", "").strip()
            if decision in ("BUY", "HOLD", "SELL") and 1 <= confidence <= 10 and justification:
                return {
                    "name": strategy_name,
                    "decision": decision,
                    "confidence": confidence,
                    "justification": justification,
                    "raw_response": raw,
                }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: regex extraction
    decision = "HOLD"
    for word in ["BUY", "SELL", "HOLD"]:
        if re.search(rf'\b{word}\b', raw, re.IGNORECASE):
            decision = word
            break

    conf_match = re.search(r'confidence["\s:]*(\d+)', raw, re.IGNORECASE)
    confidence = int(conf_match.group(1)) if conf_match else 5
    confidence = max(1, min(10, confidence))

    # Extract justification: everything after "justification" key or first sentence block
    just_match = re.search(r'justification["\s:]*(.+?)(?=\}|$)', raw, re.DOTALL | re.IGNORECASE)
    if just_match:
        justification = just_match.group(1).strip().strip('"').strip()
    else:
        # Use the whole response, trimmed
        justification = raw[:600].strip()

    return {
        "name": strategy_name,
        "decision": decision,
        "confidence": confidence,
        "justification": justification,
        "raw_response": raw,
    }


def run_momentum_trader(market_data_text: str) -> dict:
    """
    Momentum Trader agent.
    Philosophy: The trend is your friend. Follow what is already working.
    """
    system_prompt = _load_prompt("strategy_a.txt")
    user_message = f"Analyze the following stock data and provide your recommendation:\n\n{market_data_text}"
    raw = _call_ollama(system_prompt, user_message)
    return _parse_agent_response(raw, "Momentum Trader")


def run_value_contrarian(market_data_text: str) -> dict:
    """
    Value Contrarian agent.
    Philosophy: Markets overreact. Buy fear, sell greed.
    """
    system_prompt = _load_prompt("strategy_b.txt")
    user_message = f"Analyze the following stock data and provide your recommendation:\n\n{market_data_text}"
    raw = _call_ollama(system_prompt, user_message)
    return _parse_agent_response(raw, "Value Contrarian")

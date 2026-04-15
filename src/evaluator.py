"""
Evaluator Component
Compares both strategy outputs, identifies agreement/disagreement,
and synthesizes an analysis using the LLM.
"""
import requests
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


def _load_prompt(filename: str) -> str:
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _call_ollama(system_prompt: str, user_message: str) -> str:
    full_prompt = f"{system_prompt}\n\n---\n\n{user_message}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 400},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()


def evaluate(strategy_a: dict, strategy_b: dict, market_data_text: str) -> dict:
    """
    Compare both strategy outputs and produce a consensus or disagreement analysis.
    Returns a dict with: agents_agree (bool) and analysis (str).
    """
    agree = strategy_a["decision"] == strategy_b["decision"]

    # Build evaluator prompt context
    context = f"""
MARKET DATA:
{market_data_text}

STRATEGY A — {strategy_a['name']}:
  Decision:   {strategy_a['decision']}
  Confidence: {strategy_a['confidence']}/10
  Reasoning:  {strategy_a['justification']}

STRATEGY B — {strategy_b['name']}:
  Decision:   {strategy_b['decision']}
  Confidence: {strategy_b['confidence']}/10
  Reasoning:  {strategy_b['justification']}

The strategies {'AGREE' if agree else 'DISAGREE'} on this stock.
""".strip()

    system_prompt = _load_prompt("evaluator.txt")
    analysis = _call_ollama(system_prompt, context)

    return {
        "agents_agree": agree,
        "analysis": analysis,
    }

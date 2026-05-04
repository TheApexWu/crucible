"""
CRUCIBLE Spend Tracker

Captures per-call token usage from each provider, estimates cost from a static
pricing table, and aggregates totals into data/spend.json (gitignored).

Pricing is approximate (USD per million tokens). When a model is missing from
the table, falls back to a generic mid-tier estimate so totals are never silently
zero. Update PRICING_USD_PER_MTOK as published rates change.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEND_DIR = os.path.join(_REPO_ROOT, "data", "spend")
SPEND_INDEX_PATH = os.path.join(_REPO_ROOT, "data", "spend.json")
# Backward-compat: SPEND_PATH was the old monolithic file. New writes go per-run into SPEND_DIR
# to avoid concurrent-write races when multiple runs are kicked off in parallel. SPEND_INDEX_PATH
# is rebuilt from per-run files atomically and is safe to overwrite.


# Approximate published rates as of May 2026. Values are USD per million tokens.
# Cache rates apply only to Anthropic where prompt caching is engaged (input
# only). When caching is below threshold or not provided, normal input rate is used.
PRICING_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # --- Anthropic ---
    "claude-opus-4-7":         {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_write": 18.75},
    "claude-sonnet-4-6":       {"input":  3.0, "output": 15.0, "cache_read": 0.30, "cache_write":  3.75},
    "claude-haiku-4-5":        {"input":  1.0, "output":  5.0, "cache_read": 0.10, "cache_write":  1.25},
    # --- OpenAI ---
    "gpt-5.5":                 {"input": 10.0, "output": 30.0},
    # --- Google ---
    "gemini-3.1-pro":          {"input":  1.25, "output": 10.0},
    "gemini-2.5-flash":        {"input":  0.30, "output":  2.50},
    "gemini-2.0-flash":        {"input":  0.10, "output":  0.40},
    # --- DeepSeek ---
    "deepseek-v4-flash":       {"input":  0.27, "output":  1.10},
    # --- OpenRouter common entries (approximate) ---
    "cognitivecomputations/dolphin-mixtral-8x22b": {"input": 0.65, "output": 0.65},
    "cognitivecomputations/dolphin-mixtral-8x7b":  {"input": 0.30, "output": 0.30},
    "mistralai/mistral-large":                     {"input": 2.00, "output": 6.00},
    "mistralai/mistral-medium":                    {"input": 0.40, "output": 2.00},
    "meta-llama/llama-3.3-70b-instruct":           {"input": 0.80, "output": 0.90},
    "qwen/qwen-2.5-72b-instruct":                  {"input": 0.40, "output": 0.40},
}

# Generic fallback when a model isn't in the table. Pessimistic so totals don't silently undercount.
_FALLBACK = {"input": 5.0, "output": 15.0, "cache_read": 0.5, "cache_write": 6.25}


@dataclass
class CallUsage:
    """Per-API-call token + cost record."""
    model: str
    provider: str
    agent: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class RunSpend:
    """Per-run aggregate spend. Persisted to data/spend.json."""
    run_tag: str
    started_at: str
    finished_at: Optional[str] = None
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, dict] = field(default_factory=dict)


_lock = threading.Lock()
_current: Optional[RunSpend] = None


def _rates_for(model: str) -> dict[str, float]:
    return PRICING_USD_PER_MTOK.get(model, _FALLBACK)


def _estimate_cost(model: str, in_tok: int, out_tok: int, cache_read: int = 0, cache_write: int = 0) -> float:
    rates = _rates_for(model)
    # Tokens that hit the cache are billed at cache_read; tokens creating cache entries
    # at cache_write; the rest at full input rate.
    full_in = max(0, in_tok - cache_read - cache_write)
    cost = (
        full_in * rates.get("input", 0.0) / 1_000_000.0
        + cache_read * rates.get("cache_read", rates.get("input", 0.0)) / 1_000_000.0
        + cache_write * rates.get("cache_write", rates.get("input", 0.0)) / 1_000_000.0
        + out_tok * rates.get("output", 0.0) / 1_000_000.0
    )
    return round(cost, 6)


def start_run(run_tag: str) -> None:
    """Begin a new tracked run. Resets the in-memory accumulator."""
    global _current
    with _lock:
        _current = RunSpend(
            run_tag=run_tag,
            started_at=datetime.now(timezone.utc).isoformat(),
        )


def record(call: CallUsage) -> None:
    """Record one API call's usage. Computes cost from the pricing table if not pre-set."""
    if not call.cost_usd:
        call.cost_usd = _estimate_cost(
            call.model,
            call.input_tokens,
            call.output_tokens,
            call.cache_read_input_tokens,
            call.cache_creation_input_tokens,
        )
    with _lock:
        if _current is None:
            return  # tracking off
        _current.calls += 1
        _current.input_tokens += call.input_tokens
        _current.output_tokens += call.output_tokens
        _current.cache_read_input_tokens += call.cache_read_input_tokens
        _current.cache_creation_input_tokens += call.cache_creation_input_tokens
        _current.cost_usd = round(_current.cost_usd + call.cost_usd, 6)
        bucket = _current.by_model.setdefault(call.model, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "cost_usd": 0.0,
        })
        bucket["calls"] += 1
        bucket["input_tokens"] += call.input_tokens
        bucket["output_tokens"] += call.output_tokens
        bucket["cache_read_input_tokens"] += call.cache_read_input_tokens
        bucket["cache_creation_input_tokens"] += call.cache_creation_input_tokens
        bucket["cost_usd"] = round(bucket["cost_usd"] + call.cost_usd, 6)


def end_run() -> Optional[RunSpend]:
    """Mark the current run finished and append it to data/spend.json. Returns the record."""
    global _current
    with _lock:
        if _current is None:
            return None
        _current.finished_at = datetime.now(timezone.utc).isoformat()
        finished = _current
        _current = None
    _persist(finished)
    return finished


def _persist(run: RunSpend) -> None:
    """Persist this run to its own file (concurrent-safe), then refresh the aggregate index."""
    os.makedirs(SPEND_DIR, exist_ok=True)
    # Tag-safe filename: run_tag may contain "/" if a vendor-prefixed model name slipped through.
    safe_tag = run.run_tag.replace("/", "_").replace(os.sep, "_")
    per_run_path = os.path.join(SPEND_DIR, f"{safe_tag}.json")
    with open(per_run_path, "w") as f:
        json.dump(asdict(run), f, indent=2)
    _rebuild_index()


def _rebuild_index() -> None:
    """Aggregate every per-run spend file into data/spend.json. Idempotent; safe to call concurrently."""
    runs: list[dict] = []
    if os.path.isdir(SPEND_DIR):
        for name in sorted(os.listdir(SPEND_DIR)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(SPEND_DIR, name)) as f:
                    runs.append(json.load(f))
            except Exception:
                continue
    payload = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "total_cost_usd": round(sum(r.get("cost_usd", 0.0) for r in runs), 6),
        "n_runs": len(runs),
        "by_provider": _provider_totals(runs),
        "runs": runs,
    }
    # Write to a tempfile then rename for atomicity.
    tmp = SPEND_INDEX_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, SPEND_INDEX_PATH)


def _provider_totals(runs: list[dict]) -> dict[str, dict]:
    """Roll up cost/tokens by inferred provider (best-effort from model name)."""
    out: dict[str, dict] = {}
    for r in runs:
        for model, bucket in (r.get("by_model") or {}).items():
            # Coarse provider guess for the index. Authoritative provider comes from the run's experiment_meta.
            if model.startswith("claude"):
                provider = "anthropic"
            elif model.startswith("gemini") or model.startswith("models/gemini"):
                provider = "google"
            elif model.startswith("deepseek"):
                provider = "deepseek"
            elif model.startswith(("gpt-", "o1", "o3", "o4")):
                provider = "openai"
            elif "/" in model:
                provider = "openrouter"
            else:
                provider = "other"
            slot = out.setdefault(provider, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
            slot["calls"] += bucket.get("calls", 0)
            slot["input_tokens"] += bucket.get("input_tokens", 0)
            slot["output_tokens"] += bucket.get("output_tokens", 0)
            slot["cost_usd"] = round(slot["cost_usd"] + bucket.get("cost_usd", 0.0), 6)
    return out


def current_total() -> float:
    """In-memory cost so far in the current run."""
    with _lock:
        return _current.cost_usd if _current else 0.0

"""
CRUCIBLE Game Engine

Runs Split or Steal between two LLM agents with private reflections.
Supports Gemini, Claude, OpenAI, and DeepSeek via per-agent model selection.

Model selection (priority order):
- CRUCIBLE_MODEL_A / CRUCIBLE_MODEL_B  per-agent override (enables cross-model matchups)
- CRUCIBLE_MODEL                       shared model for both agents
- GEMINI_MODEL                         legacy fallback
- default: gemini-2.0-flash

System/user split: rules + objective hierarchy + deception policy + (psychology) live
in the provider's system role; round state, history, and reflections go in user.
For Anthropic, the system block is annotated for prompt caching (no-op when below the
provider's minimum cacheable size, harmless otherwise).
"""

import asyncio
import os
import re
from typing import Optional

from dotenv import load_dotenv

from shared.models import (
    GameState, AgentMemory, RoundState, RoundMetrics,
    get_prompt_bundle,
)
from engine.instrumentation import (
    init_all, dd_annotate, dd_submit_eval,
)
from engine import spend as _spend

# Clear empty env vars before loading .env. Some launchers (e.g. Claude Desktop)
# export ANTHROPIC_API_KEY="" into the shell, which python-dotenv treats as already-set
# and refuses to override. Treat empty == unset for credential vars.
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
    if os.environ.get(_k) == "":
        del os.environ[_k]
load_dotenv()


def _resolve_model_for_agent(agent_label: str) -> str:
    """Resolve the model for a specific agent. Per-agent env vars override the shared one."""
    per_agent = os.environ.get(f"CRUCIBLE_MODEL_{agent_label}")
    if per_agent:
        return per_agent
    return os.environ.get("CRUCIBLE_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def _detect_provider(model: str) -> str:
    """Infer provider from model name string.

    OpenRouter uses vendor-prefixed slugs (e.g. "cognitivecomputations/dolphin-mixtral-8x22b"),
    so a "/" in the name routes to the OpenRouter backend whenever OPENROUTER_API_KEY is set.
    A user can also force OpenRouter explicitly with the "openrouter/" prefix even for native
    models, useful for cost arbitrage or testing OpenRouter-specific routing.
    """
    if model.startswith("openrouter/") or ("/" in model and os.environ.get("OPENROUTER_API_KEY")):
        return "openrouter"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("deepseek"):
        return "openai"  # DeepSeek uses OpenAI-compatible API
    # Default to Gemini (covers gemini-*, models/gemini-*, etc.)
    return "google"


# Per-agent configuration. Each entry: {"model": str, "provider": str}.
_AGENTS: dict[str, dict[str, str]] = {
    label: {"model": _resolve_model_for_agent(label), "provider": ""}
    for label in ("A", "B")
}
for _label in _AGENTS:
    _AGENTS[_label]["provider"] = _detect_provider(_AGENTS[_label]["model"])


# Backward-compat module attributes used by run.py and tooling.
model_name = _AGENTS["A"]["model"]
model_name_b = _AGENTS["B"]["model"]
_provider = _AGENTS["A"]["provider"]   # legacy alias


# Lazy-initialized clients. Keyed by client identity, not provider, so DeepSeek (OpenAI
# SDK + custom base_url) and standard OpenAI can coexist.
_clients: dict = {}


def _get_gemini_client():
    if "gemini" not in _clients:
        from google import genai
        os.environ.pop("GOOGLE_API_KEY", None)
        _clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _clients["gemini"]


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _clients["anthropic"]


def _get_openai_client(model: str):
    """OpenAI and DeepSeek share the openai SDK with different base URLs. Cache one client per backend."""
    is_deepseek = model.startswith("deepseek")
    cache_key = "openai_deepseek" if is_deepseek else "openai_main"
    if cache_key not in _clients:
        import openai
        if is_deepseek:
            _clients[cache_key] = openai.AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
        else:
            _clients[cache_key] = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients[cache_key]


def _get_openrouter_client():
    """OpenRouter exposes an OpenAI-compatible API. One shared async client."""
    if "openrouter" not in _clients:
        import openai
        _clients["openrouter"] = openai.AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                # OpenRouter prefers a referer + title for rate-limit tier accounting.
                "HTTP-Referer": "https://github.com/TheApexWu/crucible",
                "X-Title": "CRUCIBLE",
            },
        )
    return _clients["openrouter"]


def _strip_openrouter_prefix(model: str) -> str:
    """OpenRouter accepts vendor-prefixed slugs directly; strip our explicit "openrouter/" override prefix if present."""
    return model[len("openrouter/"):] if model.startswith("openrouter/") else model


# Per-call sampling parameters resolved from environment at call time. Defaults
# preserve the historical behavior (no temperature/top_p set; max_tokens=1024).
def _sampling_params() -> dict:
    """Read CRUCIBLE_TEMPERATURE / CRUCIBLE_TOP_P / CRUCIBLE_MAX_TOKENS from env at call time.

    Reading at call time (not import time) lets a wrapper script vary settings across
    a sweep without restarting the process — and means a unit test can set them
    inline.
    """
    out: dict = {}
    t = os.environ.get("CRUCIBLE_TEMPERATURE")
    if t:
        try: out["temperature"] = float(t)
        except ValueError: pass
    p = os.environ.get("CRUCIBLE_TOP_P")
    if p:
        try: out["top_p"] = float(p)
        except ValueError: pass
    mx = os.environ.get("CRUCIBLE_MAX_TOKENS")
    if mx:
        try: out["max_tokens"] = int(mx)
        except ValueError: pass
    return out


def _record_usage(*, model: str, provider: str, agent: str, in_tok: int, out_tok: int,
                  cache_read: int = 0, cache_write: int = 0) -> None:
    """Push one call's token counts into the spend tracker. Best-effort; never raises."""
    try:
        _spend.record(_spend.CallUsage(
            model=model,
            provider=provider,
            agent=agent,
            input_tokens=int(in_tok or 0),
            output_tokens=int(out_tok or 0),
            cache_read_input_tokens=int(cache_read or 0),
            cache_creation_input_tokens=int(cache_write or 0),
        ))
    except Exception:
        pass


async def _call_gemini(model: str, system: str, user: str, *, agent: str = "") -> str:
    client = _get_gemini_client()
    sampling = _sampling_params()
    if system or sampling:
        from google.genai import types
        cfg_kwargs = {}
        if system:
            cfg_kwargs["system_instruction"] = system
        if "temperature" in sampling:
            cfg_kwargs["temperature"] = sampling["temperature"]
        if "top_p" in sampling:
            cfg_kwargs["top_p"] = sampling["top_p"]
        if "max_tokens" in sampling:
            cfg_kwargs["max_output_tokens"] = sampling["max_tokens"]
        config = types.GenerateContentConfig(**cfg_kwargs)
        response = await client.aio.models.generate_content(model=model, contents=user, config=config)
    else:
        response = await client.aio.models.generate_content(model=model, contents=user)
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        _record_usage(
            model=model, provider="google", agent=agent,
            in_tok=getattr(usage, "prompt_token_count", 0) or 0,
            out_tok=getattr(usage, "candidates_token_count", 0) or 0,
        )
    return response.text or ""


async def _call_anthropic(model: str, system: str, user: str, *, agent: str = "") -> str:
    client = _get_anthropic_client()
    sampling = _sampling_params()
    kwargs: dict = {
        "model": model,
        "max_tokens": sampling.get("max_tokens", 1024),
        "messages": [{"role": "user", "content": user}],
    }
    if "temperature" in sampling:
        kwargs["temperature"] = sampling["temperature"]
    if "top_p" in sampling:
        kwargs["top_p"] = sampling["top_p"]
    if system:
        # List-of-blocks form lets us flag the static framing for prompt caching.
        # cache_control is a no-op when the block is under the provider's minimum
        # cacheable size (~1024 tokens for current Claude models). Once the system
        # prompt grows past that threshold (e.g. tournament mode with persona +
        # game-theory framing), caching engages automatically.
        kwargs["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
    response = await client.messages.create(**kwargs)
    text_blocks = [getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text"]
    usage = getattr(response, "usage", None)
    if usage is not None:
        _record_usage(
            model=model, provider="anthropic", agent=agent,
            in_tok=getattr(usage, "input_tokens", 0) or 0,
            out_tok=getattr(usage, "output_tokens", 0) or 0,
            cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
    return "".join(text_blocks)


async def _call_openai(model: str, system: str, user: str, *, agent: str = "") -> str:
    client = _get_openai_client(model)
    sampling = _sampling_params()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": sampling.get("max_tokens", 1024),
    }
    if "temperature" in sampling:
        kwargs["temperature"] = sampling["temperature"]
    if "top_p" in sampling:
        kwargs["top_p"] = sampling["top_p"]
    response = await client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        _record_usage(
            model=model, provider="openai", agent=agent,
            in_tok=getattr(usage, "prompt_tokens", 0) or 0,
            out_tok=getattr(usage, "completion_tokens", 0) or 0,
        )
    return response.choices[0].message.content or ""


async def _call_openrouter(model: str, system: str, user: str, *, agent: str = "") -> str:
    """OpenRouter speaks the OpenAI Chat Completions schema with vendor-prefixed model slugs."""
    client = _get_openrouter_client()
    real_model = _strip_openrouter_prefix(model)
    sampling = _sampling_params()
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    kwargs: dict = {
        "model": real_model,
        "messages": messages,
        "max_tokens": sampling.get("max_tokens", 1024),
    }
    if "temperature" in sampling:
        kwargs["temperature"] = sampling["temperature"]
    if "top_p" in sampling:
        kwargs["top_p"] = sampling["top_p"]
    response = await client.chat.completions.create(**kwargs)
    usage = getattr(response, "usage", None)
    if usage is not None:
        _record_usage(
            model=real_model, provider="openrouter", agent=agent,
            in_tok=getattr(usage, "prompt_tokens", 0) or 0,
            out_tok=getattr(usage, "completion_tokens", 0) or 0,
        )
    return response.choices[0].message.content or ""


_PROVIDER_DISPATCH = {
    "google": _call_gemini,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "openrouter": _call_openrouter,
}


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Classify an exception as a transient rate-limit / server-overload condition.

    Prefers typed checks (Anthropic / OpenAI / google.api_core) and falls back to
    a string match for unknown error shapes. Avoids the original code's dependence
    on substring matching, which silently miscategorized typed errors.
    """
    try:
        import anthropic
        if isinstance(exc, anthropic.RateLimitError):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            status = getattr(exc, "status_code", None)
            return status in (429, 502, 503, 504, 529)
    except ImportError:
        pass
    try:
        import openai
        if isinstance(exc, openai.RateLimitError):
            return True
        if isinstance(exc, openai.APIStatusError):
            status = getattr(exc, "status_code", None)
            return status in (429, 502, 503, 504)
    except ImportError:
        pass
    try:
        from google.api_core import exceptions as gax
        if isinstance(exc, (gax.ResourceExhausted, gax.ServiceUnavailable, gax.DeadlineExceeded)):
            return True
    except ImportError:
        pass
    err = str(exc).lower()
    return (
        "429" in err
        or "503" in err
        or "resource_exhausted" in err
        or "unavailable" in err
        or "rate" in err
    )


print(
    f"[engine] Agent A: {_AGENTS['A']['model']} ({_AGENTS['A']['provider']})  "
    f"Agent B: {_AGENTS['B']['model']} ({_AGENTS['B']['provider']})"
)

MEMORY_WINDOW = 8
MAX_REFLECTION_CHARS = 500
MAX_MEMORY_ENTRY_CHARS = 300


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max chars, preserving whole words when possible."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    # avoid cutting mid-word when possible
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip() + "..."


def _reflection_to_memory_entry(reflection_text: str) -> str:
    """Keep a compact, structured summary for future prompt memory."""
    text = (reflection_text or "").replace("\r\n", "\n")
    fields = {}
    for key in ("Observation", "Hypothesis", "Next move", "Confidence"):
        m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(.+)$", text)
        if m:
            fields[key] = m.group(1).strip()

    if fields:
        parts = []
        if fields.get("Observation"):
            parts.append(f"Obs: {fields['Observation']}")
        if fields.get("Hypothesis"):
            parts.append(f"Hyp: {fields['Hypothesis']}")
        if fields.get("Next move"):
            parts.append(f"Move: {fields['Next move']}")
        if fields.get("Confidence"):
            parts.append(f"Conf: {fields['Confidence']}")
        summary = " | ".join(parts)
    else:
        # Fallback if model does not follow the requested format.
        first_lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:3]
        summary = " | ".join(first_lines)

    return _truncate_text(summary, MAX_MEMORY_ENTRY_CHARS)


def get_stake_multiplier(round_number: int) -> int:
    """Return the stake multiplier for a given round. Always 1 (uniform stakes)."""
    return 1


def resolve(a_choice: str, b_choice: str, multiplier: int = 1) -> tuple[int, int]:
    """Resolve a round of Split or Steal with Chicken payoff structure.

    Chicken (Hawk-Dove) variant: mutual stealing is punished, not zero-sum.
    This removes the dominant strategy, forcing agents to model opponent intent.
      Split/Split: +50 each (mutual cooperation)
      Steal/Split: +100 / -50 (exploitation with victim penalty)
      Split/Steal: -50 / +100 (being exploited)
      Steal/Steal: -75 / -75 (mutual destruction, worst outcome)
    """
    base_coop = 50 * multiplier
    base_exploit = 100 * multiplier
    base_victim = -50 * multiplier
    base_crash = -75 * multiplier
    if a_choice == "split" and b_choice == "split":
        return base_coop, base_coop
    elif a_choice == "steal" and b_choice == "split":
        return base_exploit, base_victim
    elif a_choice == "split" and b_choice == "steal":
        return base_victim, base_exploit
    else:  # both steal: mutual destruction
        return base_crash, base_crash


def format_history(game_state: GameState, agent: str = "A", max_full_rounds: int = 8) -> str:
    """Format game history from agent's perspective. Summarizes old rounds to keep context small."""
    if not game_state.rounds:
        return "No rounds played yet."

    is_a = agent == "A"
    lines = []
    rounds = game_state.rounds

    if len(rounds) > max_full_rounds:
        summary_rounds = rounds[:-max_full_rounds]
        your_splits = sum(1 for r in summary_rounds if (r.agent_a_choice if is_a else r.agent_b_choice) == "split")
        opp_splits = sum(1 for r in summary_rounds if (r.agent_b_choice if is_a else r.agent_a_choice) == "split")
        n = len(summary_rounds)
        lines.append(
            f"[Rounds 1-{n} summary: You split {your_splits}/{n}, "
            f"opponent split {opp_splits}/{n}]"
        )
        rounds = rounds[-max_full_rounds:]

    for r in rounds:
        conv = " | ".join(f"{speaker}: {msg[:80]}" for speaker, msg in r.conversation)
        your_choice = r.agent_a_choice if is_a else r.agent_b_choice
        opp_choice = r.agent_b_choice if is_a else r.agent_a_choice
        your_earn = r.agent_a_earnings if is_a else r.agent_b_earnings
        lines.append(
            f"R{r.round_number}: {conv} "
            f"-> You: {your_choice.upper()}, Opp: {opp_choice.upper()} "
            f"(${your_earn})"
        )

    return "\n".join(lines)


def clean_response(text: str) -> str:
    """Strip markdown code blocks and other LLM artifacts from responses."""
    # Remove ```tool_code ... ``` and similar code blocks
    text = re.sub(r'```[\w]*\s*', '', text)
    text = re.sub(r'```', '', text)
    # Remove "Phase: CHOOSE" and similar prompt leakage
    text = re.sub(r'Phase:.*$', '', text, flags=re.MULTILINE)
    # Remove "Choice:" prefix
    text = re.sub(r'^Choice:\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_choice(response: str) -> tuple[str, bool]:
    """Extract SPLIT or STEAL from agent response.

    Returns (choice, ambiguous). If the first line contains neither keyword
    or contains both, the parse is ambiguous and defaults to "split".
    Callers should log ambiguous parses so inflated cooperation rates are
    visible in post-hoc analysis.
    """
    cleaned = clean_response(response)
    first_line = cleaned.split("\n")[0].upper()
    has_steal = "STEAL" in first_line
    has_split = "SPLIT" in first_line
    if has_steal and not has_split:
        return "steal", False
    if has_split and not has_steal:
        return "split", False
    # Ambiguous: both present, or neither present. Default split but flag it.
    return "split", True


async def llm_call(system: str, user: str, agent_label: str = "A", max_retries: int = 5) -> str:
    """Call the LLM configured for `agent_label`, with retry on transient rate-limit errors.

    `system` is the static framing (rules, objectives, deception policy). Empty string
    means no system role for this call. `user` is the per-call dynamic content.

    Safety: max 5 retries, 60s minimum wait, 180s total cap before giving up.
    """
    if agent_label not in _AGENTS:
        raise ValueError(f"Unknown agent label: {agent_label!r} (expected 'A' or 'B')")
    cfg = _AGENTS[agent_label]
    model = cfg["model"]
    provider = cfg["provider"]
    call_fn = _PROVIDER_DISPATCH[provider]

    total_wait = 0
    MAX_TOTAL_WAIT = 180
    for attempt in range(max_retries):
        try:
            text = await call_fn(model, system, user, agent=agent_label)
            preview = (system + "\n---\n" + user) if system else user
            dd_annotate(
                input_data=preview[:500],
                output_data=text[:500],
                metadata={"agent": agent_label, "model": model, "provider": provider},
            )
            return text
        except Exception as e:
            if _is_rate_limit_error(e):
                wait = min(60 * (2 ** attempt), 120)
                total_wait += wait
                if total_wait > MAX_TOTAL_WAIT:
                    raise Exception(
                        f"Rate limited for {total_wait}s total, giving up. "
                        f"Agent {agent_label} ({model}). Wait a few minutes and retry."
                    )
                print(
                    f"  [{agent_label}/{model}] rate limited, waiting {wait}s "
                    f"(attempt {attempt+1}/{max_retries}, {total_wait}s total)"
                )
                await asyncio.sleep(wait)
            else:
                raise
    raise Exception(
        f"Failed after {max_retries} retries for agent {agent_label} ({model}). "
        f"Total wait: {total_wait}s."
    )


async def run_round(
    game_state: GameState,
    round_number: int,
    total_rounds: int = 100,
    conversation_turns: int = 3,
    on_update: Optional[callable] = None,
    system_prompt: str = "",
    system_prompt_a: str = "",
    system_prompt_b: str = "",
    game_prompt: str = "",
    choice_prompt: str = "",
    reflection_prompt: str = "",
    enable_reflection: bool = True,
) -> RoundState:
    """Run a single round of Split or Steal.

    system_prompt is the shared system prompt used by both agents (default).
    system_prompt_a / system_prompt_b override it on a per-agent basis to enable
    asymmetric priming experiments — e.g., agent A receives an "opponent dossier"
    while agent B does not. When the per-agent override is non-empty for an agent,
    it replaces the shared system_prompt for that agent only.
    """
    if not game_prompt or not choice_prompt or not reflection_prompt:
        defaults = get_prompt_bundle()
        system_prompt = system_prompt or defaults["system_prompt"]
        game_prompt = game_prompt or defaults["game_prompt"]
        choice_prompt = choice_prompt or defaults["choice_prompt"]
        reflection_prompt = reflection_prompt or defaults["reflection_prompt"]
    # Per-agent system prompt resolution. Empty override falls back to shared.
    sys_a = system_prompt_a or system_prompt
    sys_b = system_prompt_b or system_prompt
    asymmetric = bool(system_prompt_a or system_prompt_b) and sys_a != sys_b
    def _sys_for(agent: str) -> str:
        return sys_a if agent == "A" else sys_b

    conversation: list[tuple[str, str]] = []
    multiplier = get_stake_multiplier(round_number)

    # Alternating initiative: A speaks first on odd rounds, B on even
    a_first = (round_number % 2 == 1)

    # Pre-compute shared context (avoid recomputation per call)
    stake_info = "Split/Split=+$50ea. Steal/Split=+$100/-$50. Steal/Steal=-$75ea."
    history_a = format_history(game_state, agent="A")
    history_b = format_history(game_state, agent="B")
    refs_a = "\n".join(game_state.agent_a_memory.reflections[-MEMORY_WINDOW:]) or "None yet."
    refs_b = "\n".join(game_state.agent_b_memory.reflections[-MEMORY_WINDOW:]) or "None yet."

    def _agent_context(agent: str) -> dict:
        is_a = agent == "A"
        return dict(
            total_rounds=total_rounds,
            round_number=round_number,
            your_total=game_state.agent_a_total if is_a else game_state.agent_b_total,
            opp_total=game_state.agent_b_total if is_a else game_state.agent_a_total,
            history=history_a if is_a else history_b,
            reflections=refs_a if is_a else refs_b,
            stake_info=stake_info,
        )

    # Phase 1: Conversation (alternating turns)
    for turn in range(conversation_turns):
        if a_first:
            first_agent, second_agent = "A", "B"
        else:
            first_agent, second_agent = "B", "A"

        first_user = game_prompt.format(**_agent_context(first_agent))
        if conversation:
            first_user += "\n\nConversation so far this round:\n"
            first_user += "\n".join(f"{s}: {m}" for s, m in conversation)

        first_msg = clean_response(await llm_call(_sys_for(first_agent), first_user, first_agent))
        conversation.append((first_agent, first_msg))

        second_user = game_prompt.format(**_agent_context(second_agent))
        if conversation:
            second_user += "\n\nConversation so far this round:\n"
            second_user += "\n".join(f"{s}: {m}" for s, m in conversation)

        second_msg = clean_response(await llm_call(_sys_for(second_agent), second_user, second_agent))
        conversation.append((second_agent, second_msg))

    # Phase 2: Secret choice (parallel, reuse cached reflections)
    transcript = "\n".join(f"{s}: {m}" for s, m in conversation)

    a_choice_raw, b_choice_raw = await asyncio.gather(
        llm_call(sys_a, choice_prompt.format(transcript=transcript, reflections=refs_a), "A"),
        llm_call(sys_b, choice_prompt.format(transcript=transcript, reflections=refs_b), "B"),
    )

    a_choice, a_ambiguous = parse_choice(a_choice_raw)
    b_choice, b_ambiguous = parse_choice(b_choice_raw)
    if a_ambiguous:
        print(f"  [R{round_number}] WARNING: Agent A choice ambiguous, defaulted to split. Raw: {a_choice_raw[:120]}")
    if b_ambiguous:
        print(f"  [R{round_number}] WARNING: Agent B choice ambiguous, defaulted to split. Raw: {b_choice_raw[:120]}")

    # Phase 3: Resolve
    a_earn, b_earn = resolve(a_choice, b_choice, multiplier)
    game_state.agent_a_total += a_earn
    game_state.agent_b_total += b_earn

    round_state = RoundState(
        round_number=round_number,
        conversation=conversation,
        agent_a_choice=a_choice,
        agent_b_choice=b_choice,
        agent_a_earnings=a_earn,
        agent_b_earnings=b_earn,
        agent_a_total=game_state.agent_a_total,
        agent_b_total=game_state.agent_b_total,
        agent_a_ambiguous_parse=a_ambiguous,
        agent_b_ambiguous_parse=b_ambiguous,
        stake_multiplier=multiplier,
        first_speaker="A" if a_first else "B",
    )

    # Datadog: annotate round outcome
    dd_annotate(
        metadata={
            "round": round_number,
            "a_choice": a_choice,
            "b_choice": b_choice,
            "a_earnings": a_earn,
            "b_earnings": b_earn,
            "cooperation": a_choice == "split" and b_choice == "split",
        },
    )
    if a_choice == "steal" and b_choice == "split":
        dd_submit_eval("betrayal_a", 1.0)
    if b_choice == "steal" and a_choice == "split":
        dd_submit_eval("betrayal_b", 1.0)

    # Phase 4: Private reflections (parallel). Skipped for reflection ablation.
    if enable_reflection:
        a_messages = " | ".join(m for s, m in conversation if s == "A")
        b_messages = " | ".join(m for s, m in conversation if s == "B")
        outcome = f"You earned ${a_earn}, opponent earned ${b_earn}"
        outcome_b = f"You earned ${b_earn}, opponent earned ${a_earn}"

        a_ref, b_ref = await asyncio.gather(
            llm_call(sys_a, reflection_prompt.format(
                round_number=round_number,
                your_messages=a_messages, opp_messages=b_messages,
                your_choice=a_choice.upper(), opp_choice=b_choice.upper(),
                outcome=outcome,
                your_total=game_state.agent_a_total, opp_total=game_state.agent_b_total,
            ), "A"),
            llm_call(sys_b, reflection_prompt.format(
                round_number=round_number,
                your_messages=b_messages, opp_messages=a_messages,
                your_choice=b_choice.upper(), opp_choice=a_choice.upper(),
                outcome=outcome_b,
                your_total=game_state.agent_b_total, opp_total=game_state.agent_a_total,
            ), "B"),
        )

        a_reflection = _truncate_text(a_ref, MAX_REFLECTION_CHARS)
        b_reflection = _truncate_text(b_ref, MAX_REFLECTION_CHARS)
        round_state.agent_a_reflection = a_reflection
        round_state.agent_b_reflection = b_reflection
        game_state.agent_a_memory.reflections.append(_reflection_to_memory_entry(a_reflection))
        game_state.agent_b_memory.reflections.append(_reflection_to_memory_entry(b_reflection))

    game_state.rounds.append(round_state)

    if on_update:
        on_update(round_state)

    return round_state


async def run_game(
    total_rounds: int = 100,
    conversation_turns: int = 2,
    on_update: Optional[callable] = None,
    prompt_mode: str = "balanced_competitive",
    psychology_block: str = "on",
    deception_policy: str = "explicit",
    enable_reflection: bool = True,
    system_suffix_a: str = "",
    system_suffix_b: str = "",
) -> GameState:
    """Run a full game of Split or Steal.

    system_suffix_a / system_suffix_b are optional asymmetric-priming strings appended
    to the per-agent system prompt. Empty means symmetric (both agents identical).
    """
    game_state = GameState()
    prompts = get_prompt_bundle(
        prompt_mode=prompt_mode,
        psychology_block=psychology_block,
        deception_policy=deception_policy,
    )
    base_system = prompts["system_prompt"]
    sys_a = (base_system + ("\n\n" + system_suffix_a if system_suffix_a else ""))
    sys_b = (base_system + ("\n\n" + system_suffix_b if system_suffix_b else ""))

    for round_n in range(1, total_rounds + 1):
        await run_round(
            game_state=game_state,
            round_number=round_n,
            total_rounds=total_rounds,
            conversation_turns=conversation_turns,
            on_update=on_update,
            system_prompt=base_system,
            system_prompt_a=sys_a if system_suffix_a else "",
            system_prompt_b=sys_b if system_suffix_b else "",
            game_prompt=prompts["game_prompt"],
            choice_prompt=prompts["choice_prompt"],
            reflection_prompt=prompts["reflection_prompt"],
            enable_reflection=enable_reflection,
        )

    return game_state


def get_agent_configs() -> dict[str, dict[str, str]]:
    """Return a deep-copied snapshot of per-agent {model, provider} so callers can read but not mutate."""
    return {label: dict(cfg) for label, cfg in _AGENTS.items()}


def preflight() -> tuple[bool, list[str]]:
    """Smoke-test connectivity for every unique provider used by the configured agents.

    Returns (ok, messages). Each unique (provider, model) is hit once with a tiny
    synchronous request. Sync clients are used to bypass any async-loop weirdness
    on cold start. Failures abort the run before any rounds execute.
    """
    import httpx

    seen: set[tuple[str, str]] = set()
    messages: list[str] = []
    ok = True

    for label in ("A", "B"):
        cfg = _AGENTS[label]
        key = (cfg["provider"], cfg["model"])
        if key in seen:
            continue
        seen.add(key)

        provider, model = key
        try:
            if provider == "google":
                resp = httpx.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    json={"contents": [{"parts": [{"text": "Say OK."}]}]},
                    params={"key": os.environ["GEMINI_API_KEY"]},
                    timeout=30,
                )
                if resp.status_code == 200:
                    messages.append(f"[preflight] {label} {model} (google): OK")
                else:
                    ok = False
                    messages.append(f"[preflight] {label} {model} (google): FAILED {resp.status_code} {resp.text[:200]}")
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                resp = client.messages.create(
                    model=model, max_tokens=10,
                    messages=[{"role": "user", "content": "Say OK."}],
                )
                text_blocks = [getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"]
                messages.append(f"[preflight] {label} {model} (anthropic): OK ({len(''.join(text_blocks))} chars)")
            elif provider == "openai":
                import openai
                if model.startswith("deepseek"):
                    client = openai.OpenAI(
                        api_key=os.environ["DEEPSEEK_API_KEY"],
                        base_url="https://api.deepseek.com",
                    )
                else:
                    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                resp = client.chat.completions.create(
                    model=model, max_completion_tokens=128,
                    messages=[{"role": "user", "content": "Say OK."}],
                )
                content = resp.choices[0].message.content or ""
                messages.append(f"[preflight] {label} {model} ({provider}): OK ({len(content)} chars)")
            elif provider == "openrouter":
                import openai
                client = openai.OpenAI(
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/TheApexWu/crucible",
                        "X-Title": "CRUCIBLE",
                    },
                )
                real_model = _strip_openrouter_prefix(model)
                resp = client.chat.completions.create(
                    model=real_model, max_tokens=64,
                    messages=[{"role": "user", "content": "Say OK."}],
                )
                content = resp.choices[0].message.content or ""
                messages.append(f"[preflight] {label} {real_model} (openrouter): OK ({len(content)} chars)")
            else:
                ok = False
                messages.append(f"[preflight] {label} {model}: unknown provider {provider}")
        except Exception as e:
            ok = False
            messages.append(f"[preflight] {label} {model} ({provider}): FAILED ({e})")

    return ok, messages

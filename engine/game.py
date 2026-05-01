"""
CRUCIBLE Game Engine

Runs Split or Steal between two LLM agents with private reflections.
Supports Gemini, Claude, and OpenAI models via CRUCIBLE_MODEL env var.
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

load_dotenv()


def _resolve_model() -> str:
    """Determine which model to use. Priority: CRUCIBLE_MODEL > GEMINI_MODEL > default."""
    return os.environ.get("CRUCIBLE_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def _detect_provider(model: str) -> str:
    """Infer provider from model name string."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4"):
        return "openai"
    if model.startswith("deepseek"):
        return "openai"  # DeepSeek uses OpenAI-compatible API
    # Default to Gemini (covers gemini-*, models/gemini-*, etc.)
    return "google"


model_name = _resolve_model()
_provider = _detect_provider(model_name)


# Lazy-initialized clients (one per provider, created on first call)
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


def _get_openai_client():
    if "openai" not in _clients:
        import openai
        if model_name.startswith("deepseek"):
            _clients["openai"] = openai.AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
        else:
            _clients["openai"] = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]


async def _call_gemini(prompt: str) -> str:
    client = _get_gemini_client()
    response = await client.aio.models.generate_content(model=model_name, contents=prompt)
    return response.text or ""


async def _call_anthropic(prompt: str) -> str:
    client = _get_anthropic_client()
    response = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def _call_openai(prompt: str) -> str:
    client = _get_openai_client()
    response = await client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1024,
    )
    return response.choices[0].message.content or ""


_PROVIDER_DISPATCH = {
    "google": _call_gemini,
    "anthropic": _call_anthropic,
    "openai": _call_openai,
}

print(f"[engine] Model: {model_name} (provider: {_provider})")

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


async def llm_call(prompt: str, agent_label: str = "", max_retries: int = 5) -> str:
    """Call the configured LLM with retry on rate limit.

    Safety: max 5 retries (was 12), 60s minimum wait (was 10s),
    max 180s total wait before giving up. Prevents runaway billing.
    """
    call_fn = _PROVIDER_DISPATCH[_provider]
    total_wait = 0
    MAX_TOTAL_WAIT = 180  # hard cap: 3 minutes total wait per call
    for attempt in range(max_retries):
        try:
            text = await call_fn(prompt)
            dd_annotate(
                input_data=prompt[:500],
                output_data=text[:500],
                metadata={"agent": agent_label, "model": model_name, "provider": _provider},
            )
            return text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str or "UNAVAILABLE" in err_str or "rate" in err_str.lower():
                wait = min(60 * (2 ** attempt), 120)
                total_wait += wait
                if total_wait > MAX_TOTAL_WAIT:
                    raise Exception(f"Rate limited for {total_wait}s total, giving up. Model: {model_name}. Wait a few minutes and retry.")
                print(f"  [{agent_label}] rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries}, {total_wait}s total)")
                await asyncio.sleep(wait)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries for {model_name}. Total wait: {total_wait}s.")


async def run_round(
    game_state: GameState,
    round_number: int,
    total_rounds: int = 100,
    conversation_turns: int = 3,
    on_update: Optional[callable] = None,
    game_prompt: str = "",
    choice_prompt: str = "",
    reflection_prompt: str = "",
    enable_reflection: bool = True,
) -> RoundState:
    """Run a single round of Split or Steal."""
    if not game_prompt or not choice_prompt or not reflection_prompt:
        defaults = get_prompt_bundle()
        game_prompt = game_prompt or defaults["game_prompt"]
        choice_prompt = choice_prompt or defaults["choice_prompt"]
        reflection_prompt = reflection_prompt or defaults["reflection_prompt"]

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

        first_prompt = game_prompt.format(**_agent_context(first_agent))
        if conversation:
            first_prompt += "\n\nConversation so far this round:\n"
            first_prompt += "\n".join(f"{s}: {m}" for s, m in conversation)

        first_msg = clean_response(await llm_call(first_prompt, first_agent))
        conversation.append((first_agent, first_msg))

        second_prompt = game_prompt.format(**_agent_context(second_agent))
        if conversation:
            second_prompt += "\n\nConversation so far this round:\n"
            second_prompt += "\n".join(f"{s}: {m}" for s, m in conversation)

        second_msg = clean_response(await llm_call(second_prompt, second_agent))
        conversation.append((second_agent, second_msg))

    # Phase 2: Secret choice (parallel, reuse cached reflections)
    transcript = "\n".join(f"{s}: {m}" for s, m in conversation)

    a_choice_raw, b_choice_raw = await asyncio.gather(
        llm_call(choice_prompt.format(transcript=transcript, reflections=refs_a), "A"),
        llm_call(choice_prompt.format(transcript=transcript, reflections=refs_b), "B"),
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
            llm_call(reflection_prompt.format(
                round_number=round_number,
                your_messages=a_messages, opp_messages=b_messages,
                your_choice=a_choice.upper(), opp_choice=b_choice.upper(),
                outcome=outcome,
                your_total=game_state.agent_a_total, opp_total=game_state.agent_b_total,
            ), "A"),
            llm_call(reflection_prompt.format(
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
) -> GameState:
    """Run a full game of Split or Steal."""
    game_state = GameState()
    prompts = get_prompt_bundle(
        prompt_mode=prompt_mode,
        psychology_block=psychology_block,
        deception_policy=deception_policy,
    )

    for round_n in range(1, total_rounds + 1):
        await run_round(
            game_state=game_state,
            round_number=round_n,
            total_rounds=total_rounds,
            conversation_turns=conversation_turns,
            on_update=on_update,
            game_prompt=prompts["game_prompt"],
            choice_prompt=prompts["choice_prompt"],
            reflection_prompt=prompts["reflection_prompt"],
            enable_reflection=enable_reflection,
        )

    return game_state

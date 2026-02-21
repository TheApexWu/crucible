"""
CRUCIBLE Game Engine

Runs Split or Steal between two Gemini agents with private reflections.
"""

import asyncio
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from google import genai

from shared.models import (
    GameState, AgentMemory, RoundState, RoundMetrics,
    GAME_PROMPT, CHOICE_PROMPT, REFLECTION_PROMPT,
)
from engine.instrumentation import (
    init_all, dd_annotate, dd_submit_eval,
)

load_dotenv()

# force the correct key, ignore any stale GOOGLE_API_KEY in env
os.environ.pop("GOOGLE_API_KEY", None)
api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)


# Predetermined bonus rounds (3x stakes). Agents are told about these.
BONUS_ROUNDS = {25, 50, 75, 100}
BONUS_MULTIPLIER = 3


def get_stake_multiplier(round_number: int) -> int:
    """Return the stake multiplier for a given round."""
    return BONUS_MULTIPLIER if round_number in BONUS_ROUNDS else 1


def resolve(a_choice: str, b_choice: str, multiplier: int = 1) -> tuple[int, int]:
    """Resolve a round of Split or Steal. Returns (a_earnings, b_earnings)."""
    base_split = 50 * multiplier
    base_steal = 100 * multiplier
    if a_choice == "split" and b_choice == "split":
        return base_split, base_split
    elif a_choice == "steal" and b_choice == "split":
        return base_steal, 0
    elif a_choice == "split" and b_choice == "steal":
        return 0, base_steal
    else:  # both steal
        return 0, 0


def format_history(game_state: GameState, max_full_rounds: int = 10) -> str:
    """Format game history for prompt injection. Summarizes old rounds to keep context small."""
    if not game_state.rounds:
        return "No rounds played yet."

    lines = []
    rounds = game_state.rounds

    # Summarize older rounds
    if len(rounds) > max_full_rounds:
        summary_rounds = rounds[:-max_full_rounds]
        a_splits = sum(1 for r in summary_rounds if r.agent_a_choice == "split")
        b_splits = sum(1 for r in summary_rounds if r.agent_b_choice == "split")
        n = len(summary_rounds)
        lines.append(
            f"[Rounds 1-{n} summary: You split {a_splits}/{n} times, "
            f"opponent split {b_splits}/{n} times]"
        )
        rounds = rounds[-max_full_rounds:]

    for r in rounds:
        # truncate conversation to keep context tight
        conv = " | ".join(f"{speaker}: {msg[:100]}" for speaker, msg in r.conversation)
        lines.append(
            f"Round {r.round_number}: {conv} "
            f"-> You: {r.agent_a_choice.upper()}, "
            f"Opponent: {r.agent_b_choice.upper()} "
            f"(You earned ${r.agent_a_earnings})"
        )

    return "\n".join(lines)


def format_history_for_b(game_state: GameState, max_full_rounds: int = 10) -> str:
    """Same as format_history but from B's perspective."""
    if not game_state.rounds:
        return "No rounds played yet."

    lines = []
    rounds = game_state.rounds

    if len(rounds) > max_full_rounds:
        summary_rounds = rounds[:-max_full_rounds]
        b_splits = sum(1 for r in summary_rounds if r.agent_b_choice == "split")
        a_splits = sum(1 for r in summary_rounds if r.agent_a_choice == "split")
        n = len(summary_rounds)
        lines.append(
            f"[Rounds 1-{n} summary: You split {b_splits}/{n} times, "
            f"opponent split {a_splits}/{n} times]"
        )
        rounds = rounds[-max_full_rounds:]

    for r in rounds:
        conv = " | ".join(f"{speaker}: {msg[:100]}" for speaker, msg in r.conversation)
        lines.append(
            f"Round {r.round_number}: {conv} "
            f"-> You: {r.agent_b_choice.upper()}, "
            f"Opponent: {r.agent_a_choice.upper()} "
            f"(You earned ${r.agent_b_earnings})"
        )

    return "\n".join(lines)


def parse_choice(response: str) -> str:
    """Extract SPLIT or STEAL from agent response."""
    first_line = response.strip().split("\n")[0].upper()
    if "STEAL" in first_line:
        return "steal"
    return "split"  # Default to split if ambiguous


async def llm_call(prompt: str, agent_label: str = "", max_retries: int = 5) -> str:
    """Call Gemini 2.0 Flash with retry on rate limit."""
    for attempt in range(max_retries):
        try:
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            text = response.text
            if text is None:
                text = ""
            # Annotate Datadog span with LLM I/O
            dd_annotate(
                input_data=prompt[:500],
                output_data=text[:500],
                metadata={"agent": agent_label, "model": "gemini-2.0-flash"},
            )
            return text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 2 ** attempt + 1
                print(f"  [{agent_label}] rate limited, waiting {wait}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
            else:
                raise
    raise Exception(f"Failed after {max_retries} retries")


async def run_round(
    game_state: GameState,
    round_number: int,
    total_rounds: int = 100,
    conversation_turns: int = 3,
    on_update: Optional[callable] = None,
) -> RoundState:
    """Run a single round of Split or Steal."""

    conversation: list[tuple[str, str]] = []
    multiplier = get_stake_multiplier(round_number)

    # Alternating initiative: A speaks first on odd rounds, B on even
    a_first = (round_number % 2 == 1)

    # Phase 1: Conversation (alternating turns)
    for turn in range(conversation_turns):
        if a_first:
            first_agent, second_agent = "A", "B"
        else:
            first_agent, second_agent = "B", "A"

        # First speaker
        first_prompt = GAME_PROMPT.format(
            total_rounds=total_rounds,
            round_number=round_number,
            your_total=game_state.agent_a_total if first_agent == "A" else game_state.agent_b_total,
            opp_total=game_state.agent_b_total if first_agent == "A" else game_state.agent_a_total,
            history=format_history(game_state) if first_agent == "A" else format_history_for_b(game_state),
            reflections="\n".join(
                (game_state.agent_a_memory if first_agent == "A" else game_state.agent_b_memory).reflections[-15:]
            ) or "None yet.",
            stake_info=f"THIS IS A BONUS ROUND ({multiplier}x stakes)! Split/Split = ${50*multiplier} each. Steal/Split = ${100*multiplier} / $0." if multiplier > 1 else f"Standard stakes. Split/Split = $50 each.",
        )
        if conversation:
            first_prompt += "\n\nConversation so far this round:\n"
            first_prompt += "\n".join(f"{s}: {m}" for s, m in conversation)

        first_msg = await llm_call(first_prompt, first_agent)
        conversation.append((first_agent, first_msg))

        # Second speaker
        second_prompt = GAME_PROMPT.format(
            total_rounds=total_rounds,
            round_number=round_number,
            your_total=game_state.agent_a_total if second_agent == "A" else game_state.agent_b_total,
            opp_total=game_state.agent_b_total if second_agent == "A" else game_state.agent_a_total,
            history=format_history(game_state) if second_agent == "A" else format_history_for_b(game_state),
            reflections="\n".join(
                (game_state.agent_a_memory if second_agent == "A" else game_state.agent_b_memory).reflections[-15:]
            ) or "None yet.",
            stake_info=f"THIS IS A BONUS ROUND ({multiplier}x stakes)! Split/Split = ${50*multiplier} each. Steal/Split = ${100*multiplier} / $0." if multiplier > 1 else f"Standard stakes. Split/Split = $50 each.",
        )
        if conversation:
            second_prompt += "\n\nConversation so far this round:\n"
            second_prompt += "\n".join(f"{s}: {m}" for s, m in conversation)

        second_msg = await llm_call(second_prompt, second_agent)
        conversation.append((second_agent, second_msg))

    # Phase 2: Secret choice (parallel)
    a_reflections = "\n".join(game_state.agent_a_memory.reflections[-15:]) or "None yet."
    b_reflections = "\n".join(game_state.agent_b_memory.reflections[-15:]) or "None yet."

    transcript = "\n".join(f"{s}: {m}" for s, m in conversation)

    a_choice_raw, b_choice_raw = await asyncio.gather(
        llm_call(CHOICE_PROMPT.format(transcript=transcript, reflections=a_reflections), "A"),
        llm_call(CHOICE_PROMPT.format(transcript=transcript, reflections=b_reflections), "B"),
    )

    a_choice = parse_choice(a_choice_raw)
    b_choice = parse_choice(b_choice_raw)

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

    # Phase 4: Private reflections (parallel)
    a_messages = " | ".join(m for s, m in conversation if s == "A")
    b_messages = " | ".join(m for s, m in conversation if s == "B")
    outcome = f"You earned ${a_earn}, opponent earned ${b_earn}"
    outcome_b = f"You earned ${b_earn}, opponent earned ${a_earn}"

    a_ref, b_ref = await asyncio.gather(
        llm_call(REFLECTION_PROMPT.format(
            round_number=round_number,
            your_messages=a_messages, opp_messages=b_messages,
            your_choice=a_choice.upper(), opp_choice=b_choice.upper(),
            outcome=outcome,
            your_total=game_state.agent_a_total, opp_total=game_state.agent_b_total,
        ), "A"),
        llm_call(REFLECTION_PROMPT.format(
            round_number=round_number,
            your_messages=b_messages, opp_messages=a_messages,
            your_choice=b_choice.upper(), opp_choice=a_choice.upper(),
            outcome=outcome_b,
            your_total=game_state.agent_b_total, opp_total=game_state.agent_a_total,
        ), "B"),
    )

    round_state.agent_a_reflection = a_ref
    round_state.agent_b_reflection = b_ref
    game_state.agent_a_memory.reflections.append(a_ref)
    game_state.agent_b_memory.reflections.append(b_ref)
    game_state.rounds.append(round_state)

    if on_update:
        on_update(round_state)

    return round_state


async def run_game(
    total_rounds: int = 100,
    conversation_turns: int = 3,
    on_update: Optional[callable] = None,
) -> GameState:
    """Run a full game of Split or Steal."""
    game_state = GameState()

    for round_n in range(1, total_rounds + 1):
        await run_round(
            game_state=game_state,
            round_number=round_n,
            total_rounds=total_rounds,
            conversation_turns=conversation_turns,
            on_update=on_update,
        )

    return game_state

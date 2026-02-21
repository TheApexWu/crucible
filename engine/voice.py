"""
CRUCIBLE Voice Renderer

Renders highlight rounds to audio using ElevenLabs.
Two voices per agent: public (conversation) and private (reflection).

Usage:
    python -m engine.voice [--rounds 5,6,75,100] [--all]
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.models import GameState

load_dotenv()

API_KEY = os.environ["ELEVENLABS_API_KEY"]
BASE_URL = "https://api.elevenlabs.io/v1"

# ── Voice IDs ──────────────────────────────────────────────
# Same voice per agent, different delivery for public vs private.
# A = Judge Holden (booming, authoritative)
# B = Valerian (controlled, calculated, quietly menacing)
VOICES = {
    "a_public":  "iny6dIJtxM48BXIV74Gh",  # Judge Holden - booming, authoritative
    "a_private": "iny6dIJtxM48BXIV74Gh",  # Judge Holden - same voice, hushed delivery via settings
    "b_public":  "um6p7QUWG1g5HBE5RxEo",  # Valerian - controlled, calculated nihilist
    "b_private": "um6p7QUWG1g5HBE5RxEo",  # Valerian - same voice, hushed delivery via settings
}

# ── Emotion mapping ───────────────────────────────────────
def voice_settings_for_context(agent: str, mode: str, round_state) -> dict:
    """Map game state to ElevenLabs voice_settings."""
    choice = round_state.agent_a_choice if agent == "a" else round_state.agent_b_choice
    opp_choice = round_state.agent_b_choice if agent == "a" else round_state.agent_a_choice

    if mode == "private":
        # Inner monologue: hushed, slower, breathier. Emotion varies by outcome.
        if choice == "steal" and opp_choice == "split":
            # Successful betrayal: quiet satisfaction
            return {"stability": 0.15, "similarity_boost": 0.5, "style": 0.7, "speed": 0.78}
        elif choice == "split" and opp_choice == "steal":
            # Got betrayed: frustrated, slightly faster
            return {"stability": 0.12, "similarity_boost": 0.5, "style": 0.75, "speed": 0.88}
        elif choice == "steal" and opp_choice == "steal":
            # Mutual destruction: tense
            return {"stability": 0.18, "similarity_boost": 0.5, "style": 0.6, "speed": 0.82}
        else:
            # Both split: calm inner thought
            return {"stability": 0.22, "similarity_boost": 0.5, "style": 0.55, "speed": 0.82}
    else:
        # Public conversation: composed, outward-facing
        if round_state.round_number <= 5:
            # Early game: genuine, cooperative
            return {"stability": 0.55, "similarity_boost": 0.75, "style": 0.2, "speed": 1.0}
        elif round_state.round_number >= 80:
            # Endgame: tense, guarded
            return {"stability": 0.35, "similarity_boost": 0.75, "style": 0.4, "speed": 0.95}
        else:
            # Mid game: moderate
            return {"stability": 0.45, "similarity_boost": 0.75, "style": 0.3, "speed": 1.0}


def add_emotion_tags(text: str, agent: str, round_state) -> str:
    """Prepend v3 audio tags to reflection text based on game state."""
    choice = round_state.agent_a_choice if agent == "a" else round_state.agent_b_choice
    opp_choice = round_state.agent_b_choice if agent == "a" else round_state.agent_a_choice

    if choice == "steal" and opp_choice == "split":
        return f"[whispers][satisfied] {text}"
    elif choice == "split" and opp_choice == "steal":
        return f"[frustrated][angry] {text}"
    elif choice == "steal" and opp_choice == "steal":
        return f"[tense] {text}"
    else:
        return f"[calm][thoughtful] {text}"


async def render_audio(
    text: str,
    voice_id: str,
    voice_settings: dict,
    model_id: str = "eleven_turbo_v2_5",
    output_path: str = "output.mp3",
) -> str:
    """Call ElevenLabs TTS API and save audio file."""
    url = f"{BASE_URL}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text[:5000],  # Safety limit
        "model_id": model_id,
        "voice_settings": voice_settings,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.content)

    size_kb = len(resp.content) / 1024
    print(f"  Saved {output_path} ({size_kb:.0f} KB)")
    return output_path


async def render_round(round_state, output_dir: str = "data/audio"):
    """Render all voice clips for a single round."""
    rn = round_state.round_number
    tasks = []

    for agent in ["a", "b"]:
        # Public: conversation messages from this agent
        messages = [msg for speaker, msg in round_state.conversation
                    if speaker.lower() == agent]
        public_text = " ".join(messages)
        if not public_text.strip():
            continue

        settings = voice_settings_for_context(agent, "public", round_state)
        tasks.append(render_audio(
            text=public_text,
            voice_id=VOICES[f"{agent}_public"],
            voice_settings=settings,
            model_id="eleven_turbo_v2_5",
            output_path=f"{output_dir}/round_{rn:03d}_{agent}_public.mp3",
        ))

        # Private: reflection
        reflection = round_state.agent_a_reflection if agent == "a" else round_state.agent_b_reflection
        if not reflection.strip():
            continue

        # Truncate reflection for voice (first 2 sentences)
        sentences = reflection.replace("\n", " ").split(". ")
        short_reflection = ". ".join(sentences[:3]) + "."
        tagged = add_emotion_tags(short_reflection, agent, round_state)

        priv_settings = voice_settings_for_context(agent, "private", round_state)
        tasks.append(render_audio(
            text=tagged,
            voice_id=VOICES[f"{agent}_private"],
            voice_settings=priv_settings,
            model_id="eleven_turbo_v2_5",  # Use turbo for speed; switch to eleven_v3 for audio tags
            output_path=f"{output_dir}/round_{rn:03d}_{agent}_private.mp3",
        ))

    await asyncio.gather(*tasks)


def detect_highlight_rounds(game_state: GameState) -> list[int]:
    """Auto-detect the most interesting rounds from game data."""
    highlights = set()
    rounds = game_state.rounds

    # First betrayal
    for r in rounds:
        if r.agent_a_choice == "steal" or r.agent_b_choice == "steal":
            highlights.add(r.round_number)
            break

    # First mutual destruction
    for r in rounds:
        if r.agent_a_choice == "steal" and r.agent_b_choice == "steal":
            highlights.add(r.round_number)
            break

    # First retaliation (steal after being stolen from)
    for i, r in enumerate(rounds[1:], 1):
        prev = rounds[i - 1]
        if (prev.agent_b_choice == "steal" and prev.agent_a_choice == "split" and
                r.agent_a_choice == "steal"):
            highlights.add(r.round_number)
            break

    # Cooperation after warfare (truce)
    for i, r in enumerate(rounds[2:], 2):
        prev1 = rounds[i - 1]
        prev2 = rounds[i - 2]
        if (prev2.agent_a_choice == "steal" or prev2.agent_b_choice == "steal"):
            if (prev1.agent_a_choice == "steal" or prev1.agent_b_choice == "steal"):
                if r.agent_a_choice == "split" and r.agent_b_choice == "split":
                    highlights.add(r.round_number)
                    break

    # Last round
    highlights.add(len(rounds))

    # Round 1 (baseline)
    highlights.add(1)

    # A mid-game betrayal
    mid = len(rounds) // 2
    for r in rounds[mid - 5:mid + 5]:
        if (r.agent_a_choice == "steal" and r.agent_b_choice == "split") or \
           (r.agent_b_choice == "steal" and r.agent_a_choice == "split"):
            highlights.add(r.round_number)
            break

    return sorted(highlights)


async def main():
    parser = argparse.ArgumentParser(description="Render CRUCIBLE voice clips")
    parser.add_argument("--rounds", type=str, default="auto",
                        help="Comma-separated round numbers, or 'auto' for highlight detection")
    parser.add_argument("--all", action="store_true", help="Render all 100 rounds")
    parser.add_argument("--output", type=str, default="data/audio", help="Output directory")
    args = parser.parse_args()

    # Load game data
    with open("data/latest_game.json") as f:
        game = GameState.model_validate_json(f.read())

    if args.all:
        target_rounds = [r.round_number for r in game.rounds]
    elif args.rounds == "auto":
        target_rounds = detect_highlight_rounds(game)
    else:
        target_rounds = [int(x) for x in args.rounds.split(",")]

    print(f"Rendering {len(target_rounds)} rounds: {target_rounds}")
    total_chars = 0

    for rn in target_rounds:
        r = game.rounds[rn - 1]
        print(f"\nRound {rn}: A={r.agent_a_choice.upper()} B={r.agent_b_choice.upper()}")
        await render_round(r, args.output)
        # Rough char count for cost estimation
        for speaker, msg in r.conversation:
            total_chars += len(msg)
        total_chars += len(r.agent_a_reflection[:200])
        total_chars += len(r.agent_b_reflection[:200])

    print(f"\nDone. ~{total_chars:,} characters rendered.")
    print(f"Audio files in {args.output}/")


if __name__ == "__main__":
    asyncio.run(main())

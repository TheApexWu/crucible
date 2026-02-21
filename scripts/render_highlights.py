"""Render per-turn voice clips for highlight rounds. Neutral delivery, no emotion leakage."""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["ELEVENLABS_API_KEY"]
BASE_URL = "https://api.elevenlabs.io/v1"

VOICES = {
    "a": "iny6dIJtxM48BXIV74Gh",  # Judge Holden
    "b": "um6p7QUWG1g5HBE5RxEo",  # Valerian
}

NEUTRAL_PUBLIC = {"stability": 0.50, "similarity_boost": 0.75, "style": 0.15, "speed": 1.0}
NEUTRAL_PRIVATE = {"stability": 0.35, "similarity_boost": 0.60, "style": 0.25, "speed": 0.9}

HIGHLIGHTS = [1, 6, 13, 23, 62, 89, 100]


async def render_audio(text, voice_id, settings, output_path):
    url = f"{BASE_URL}/text-to-speech/{voice_id}"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {"text": text[:5000], "model_id": "eleven_turbo_v2_5", "voice_settings": settings}

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"  {output_path} ({len(resp.content)/1024:.0f} KB)")


async def main():
    with open("data/latest_game.json") as f:
        data = json.load(f)

    total_chars = 0
    total_clips = 0

    for rn in HIGHLIGHTS:
        r = data["rounds"][rn - 1]
        ac, bc = r["agent_a_choice"], r["agent_b_choice"]
        print(f"\nRound {rn}: A={ac.upper()} B={bc.upper()} ({len(r['conversation'])} turns)")

        tasks = []

        # Per-turn public clips
        for i, (speaker, msg) in enumerate(r["conversation"]):
            agent = speaker.lower()
            path = f"data/audio/round_{rn:03d}_turn_{i:02d}_{agent}.mp3"
            tasks.append(render_audio(msg, VOICES[agent], NEUTRAL_PUBLIC, path))
            total_chars += len(msg)
            total_clips += 1

        # Private reflections (keep as single clips per agent)
        for agent in ["a", "b"]:
            ref = r.get(f"agent_{agent}_reflection", "")
            if ref.strip():
                sentences = ref.replace("\n", " ").split(". ")
                short = ". ".join(sentences[:3]) + "."
                path = f"data/audio/round_{rn:03d}_{agent}_private.mp3"
                tasks.append(render_audio(short, VOICES[agent], NEUTRAL_PRIVATE, path))
                total_chars += len(short)
                total_clips += 1

        await asyncio.gather(*tasks)

    print(f"\nDone. {total_clips} clips, ~{total_chars:,} chars. No emotion tags.")


if __name__ == "__main__":
    asyncio.run(main())

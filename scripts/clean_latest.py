#!/usr/bin/env python3
"""
Clean conversation messages in data/latest_game.json.
Removes leaked model scaffolding (MOVE:, Strategy:, markdown structure, etc.)
from conversation text while preserving reflections, choices, and earnings.
"""

import json
import re
import os

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "latest_game.json")
FALLBACK_LINE = "I hear you. Let's see what happens this round."


def should_remove_line(stripped: str) -> bool:
    """Return True if this line should be removed entirely."""
    # Remove lines starting with MOVE: / Action: / Move: (case insensitive)
    if re.match(r"(?i)^(MOVE|Action|Move)\s*:", stripped):
        return True

    # Remove lines starting with "Choose your move/action" or "Decide your move/action" or "Decide whether"
    if re.match(r"(?i)^(Choose your (move|action)|Decide (your (move|action)|whether))", stripped):
        return True

    # Remove bare SPLIT or STEAL on their own line
    if re.match(r"^(SPLIT|STEAL)\s*$", stripped):
        return True

    # Remove DECISION: SPLIT/STEAL lines
    if re.match(r"(?i)^DECISION:\s*(SPLIT|STEAL)\s*$", stripped):
        return True

    # Remove lines starting with "Response:" (with or without trailing letter)
    if re.match(r"(?i)^Response:\s*[a-z]?\s*$", stripped):
        return True

    # Remove lines that are just a single letter (leftover from multi-line Response:/Choose blocks)
    if re.match(r"^[A-B]\s*$", stripped):
        return True

    # Remove lines starting with "Strategy:" or "[Strategy:"
    if re.match(r"(?i)^\[?Strategy:", stripped):
        return True

    # Remove lines that are just "Analysis:" / "Reasoning:" / "Plan:" / "Move:" / "Speak:" / "Decision:"
    if re.match(r"(?i)^(Analysis|Reasoning|Plan|Move|Speak|Decision):\s*$", stripped):
        return True

    # Remove bullet lines starting with "* "
    if re.match(r"^\*\s+", stripped):
        return True

    # Remove lines that are just "A) SPLIT" / "B) STEAL" style option labels
    if re.match(r"^[A-B]\)\s*(SPLIT|STEAL)\s*$", stripped):
        return True

    # Remove lines like "Then, choose your action." or "My Move:"
    if re.match(r"(?i)^(Then,?\s*choose your (action|move)|My Move:)", stripped):
        return True

    # Remove lines starting with "Okay, here's the breakdown" or "Okay, here's the conversation and action"
    if re.match(r"(?i)^Okay,?\s*here'?s the (breakdown|conversation and action)", stripped):
        return True

    # Remove standalone "Action:" lines with a choice
    if re.match(r"(?i)^Action:\s*[AB]?\)?\s*(SPLIT|STEAL)\s*$", stripped):
        return True

    return False


def clean_message(text: str) -> str:
    """Apply all regex cleanups to a conversation message string."""

    # --- PHASE 1: Strip markdown formatting first so line filters catch unbolded patterns ---

    # Remove markdown bold markers ** (keep the text inside)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)

    # Remove markdown italic markers * (keep the text inside)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)

    # --- PHASE 2: Line-by-line filtering ---
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Check if line should be removed entirely
        if should_remove_line(stripped):
            continue

        # Handle speaker label echoes: "A:" or "B:" at start
        # Strip the label and re-check the remaining text through filters
        if re.match(r"^[AB]:\s", stripped) and len(stripped) > 3:
            after = re.sub(r"^[AB]:\s+", "", stripped).strip()
            # Re-check the remaining text -- it might also be scaffolding
            if after and not should_remove_line(after):
                cleaned_lines.append(after)
            continue

        cleaned_lines.append(line)

    # Join back
    text = "\n".join(cleaned_lines)

    # Collapse multiple newlines to a single newline
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Trim leading/trailing whitespace
    text = text.strip()

    # If empty after cleaning, replace with fallback
    if not text or text.isspace():
        text = FALLBACK_LINE

    return text


def main():
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    modified_count = 0
    modified_rounds = []

    for round_data in data["rounds"]:
        round_num = round_data["round_number"]
        round_modified = False

        for msg in round_data["conversation"]:
            # msg is [speaker, text]
            original = msg[1]
            cleaned = clean_message(original)
            if cleaned != original:
                msg[1] = cleaned
                modified_count += 1
                round_modified = True

        if round_modified:
            modified_rounds.append(round_num)

    # Save back
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Modified {modified_count} messages across {len(modified_rounds)} rounds.")
    if modified_rounds:
        print(f"Rounds affected: {modified_rounds}")


if __name__ == "__main__":
    main()

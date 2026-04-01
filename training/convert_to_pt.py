#!/usr/bin/env python3
"""
convert_to_pt.py — Converts JSONL episodes from run_bootstrap.ts into .pt files
                    consumable by train.py's EpisodeDataset.

Usage:
  python convert_to_pt.py --input ./episodes --output ./episodes

This reads all *.jsonl files from --input, parses each line as a JSON episode
(list of step dicts), groups them into batches, and saves as .pt files.

Each .pt file contains: list[list[dict]]  (a list of episodes, each episode
is a list of step dicts matching the data contract in train.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_KEYS = {
    "global_features", "self_characters", "oppo_characters",
    "hand_cards", "hand_mask", "summons", "summons_mask",
    "action_mask", "mcts_policy", "reward", "is_terminal",
}

OPTIONAL_KEYS_DEFAULTS: dict[str, Any] = {
    "self_supports": [0.0] * (4 * 16),
    "self_supports_mask": [0.0] * 4,
    "oppo_supports": [0.0] * (4 * 16),
    "oppo_supports_mask": [0.0] * 4,
    "self_combat_statuses": [0.0] * (10 * 16),
    "self_combat_statuses_mask": [0.0] * 10,
    "oppo_combat_statuses": [0.0] * (10 * 16),
    "oppo_combat_statuses_mask": [0.0] * 10,
    "self_char_entities": [0.0] * (3 * 8 * 16),
    "self_char_entities_mask": [0.0] * (3 * 8),
    "oppo_char_entities": [0.0] * (3 * 8 * 16),
    "oppo_char_entities_mask": [0.0] * (3 * 8),
    "hp_after_5_turns": 0.0,
    "cards_playable_next": [0.0] * 10,
    "oppo_hand_features": [0.0] * 16,
    "kill_within_3": [0.0] * 6,
    "reaction_next_attack": 0.0,
    "dice_effective_actions": 0.0,
}


def validate_and_normalize(step: dict[str, Any]) -> dict[str, Any]:
    """Ensure all required keys exist and fill optional defaults."""
    for key in REQUIRED_KEYS:
        if key not in step:
            raise KeyError(f"Missing required key '{key}' in step")
    for key, default in OPTIONAL_KEYS_DEFAULTS.items():
        if key not in step:
            step[key] = default
    return step


def convert(input_dir: Path, output_dir: Path, episodes_per_file: int = 500) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in {input_dir}")
        return

    all_episodes: list[list[dict]] = []
    total_steps = 0
    skipped = 0

    for fpath in jsonl_files:
        print(f"Reading {fpath.name}...")
        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    episode = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"  WARN: skip line {line_num} in {fpath.name}: {e}")
                    skipped += 1
                    continue

                if not isinstance(episode, list) or len(episode) == 0:
                    skipped += 1
                    continue

                try:
                    normalized = [validate_and_normalize(step) for step in episode]
                except KeyError as e:
                    print(f"  WARN: skip episode at line {line_num}: {e}")
                    skipped += 1
                    continue

                all_episodes.append(normalized)
                total_steps += len(normalized)

    if not all_episodes:
        print("No valid episodes found.")
        return

    file_count = 0
    for i in range(0, len(all_episodes), episodes_per_file):
        chunk = all_episodes[i : i + episodes_per_file]
        out_path = output_dir / f"bootstrap_{file_count:04d}.pt"
        torch.save(chunk, str(out_path))
        print(f"  Saved {len(chunk)} episodes ({sum(len(e) for e in chunk)} steps) → {out_path.name}")
        file_count += 1

    print(
        f"\nDone: {len(all_episodes)} episodes, {total_steps} steps "
        f"→ {file_count} .pt files ({skipped} lines skipped)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSONL episodes to .pt for train.py")
    parser.add_argument("--input", type=str, default="./episodes", help="Directory with .jsonl files")
    parser.add_argument("--output", type=str, default="./episodes", help="Directory to write .pt files")
    parser.add_argument("--batch", type=int, default=500, help="Episodes per .pt file")
    args = parser.parse_args()

    convert(Path(args.input), Path(args.output), args.batch)


if __name__ == "__main__":
    main()

"""
train.py — TD(λ) + Prioritized Experience Replay training loop.

Data contract:
  Each episode is a list of step dicts.  A step dict contains:
    - All keys from NeuralStateEncoder output (float lists)
    - "action_mask"              : [128]
    - "mcts_policy"              : [128]  MCTS visit-count distribution
    - "reward"                   : float  (0 mid-game, ±1 terminal)
    - "is_terminal"              : bool
    - "hp_after_5_turns"         : float  HP of active char in 5 turns (/ 10)
    - "cards_playable_next"      : [10]   binary: is each hand card playable next turn?
    - "oppo_hand_features"       : [16]   mean-pooled CARD_FEATURE_DIM of oppo's hand
    - "kill_within_3"            : [6]    binary: char dies within 3 turns (3 self + 3 oppo)
    - "reaction_next_attack"     : float  binary: reaction triggers on next attack
    - "dice_effective_actions"   : float  effective action count from current dice (/ 10)

  Episodes are stored as .pt files: each file is a list[list[dict]].

Usage:
  python train.py --data_dir ./episodes --epochs 30 --lr 1e-3

Key features:
  • TD(λ) returns for every transition (not just terminal outcomes)
  • Prioritized Experience Replay with IS correction (Schaul et al. 2015):
    - priority = (|TD_error| + ε)^α          (α=0.6)
    - IS weight = (N · P(i))^{-β} / max(w)  (β anneals 0.4 → 1.0)
  • 6 auxiliary heads for sample-efficient representation learning
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from model import (
    TCGNeuralEvaluator,
    ModelConfig,
    ModelOutput,
    LossOutput,
    compute_loss,
    count_parameters,
    GLOBAL_FEATURE_DIM,
    CHARACTER_FEATURE_DIM,
    CARD_FEATURE_DIM,
    ENTITY_FEATURE_DIM,
    MAX_CHARACTERS,
    MAX_HAND_CARDS,
    MAX_SUMMONS,
    MAX_SUPPORTS,
    MAX_COMBAT_STATUSES,
    MAX_CHARACTER_ENTITIES,
    MAX_ACTION_SLOTS,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TD(λ) Target Computation                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def compute_td_lambda_targets(
    rewards: torch.Tensor,       # [T]
    values: torch.Tensor,        # [T]  V(s_t) detached
    is_terminals: torch.Tensor,  # [T]  1.0 if episode ends after step t
    gamma: float = 0.99,
    lam: float = 0.7,
) -> torch.Tensor:
    """
    GAE-style TD(λ) returns.

        δ_t = r_t + γ·(1−d_t)·V(s_{t+1}) − V(s_t)
        G^λ_t = V(s_t) + Σ (γλ)^l · δ_{t+l}

    Returns [T] value targets.
    """
    T = rewards.size(0)
    targets = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else torch.tensor(0.0, device=rewards.device)
        not_done = 1.0 - is_terminals[t]
        delta = rewards[t] + gamma * not_done * next_value - values[t]
        gae = delta + gamma * lam * not_done * gae
        targets[t] = gae + values[t]

    return targets


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Episode Dataset  with Prioritized Experience Replay                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPSILON = 1e-5


def _to_tensor(x: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(x, dtype=dtype)


_CHAR_ENT_SLOTS = MAX_CHARACTERS * MAX_CHARACTER_ENTITIES  # 24


class EpisodeStep:
    """One decision point within a game episode."""

    __slots__ = (
        "global_features", "self_characters", "oppo_characters",
        "hand_cards", "hand_mask", "summons", "summons_mask",
        "self_supports", "self_supports_mask",
        "oppo_supports", "oppo_supports_mask",
        "self_combat_statuses", "self_combat_statuses_mask",
        "oppo_combat_statuses", "oppo_combat_statuses_mask",
        "self_char_entities", "self_char_entities_mask",
        "oppo_char_entities", "oppo_char_entities_mask",
        "action_mask", "mcts_policy", "reward", "is_terminal",
        "hp_after_5_turns", "cards_playable_next", "oppo_hand_features",
        "kill_within_3", "reaction_next_attack", "dice_effective_actions",
    )

    def __init__(self, raw: dict[str, Any]) -> None:
        self.global_features = _to_tensor(raw["global_features"])
        self.self_characters = _to_tensor(raw["self_characters"]).reshape(MAX_CHARACTERS, CHARACTER_FEATURE_DIM)
        self.oppo_characters = _to_tensor(raw["oppo_characters"]).reshape(MAX_CHARACTERS, CHARACTER_FEATURE_DIM)
        self.hand_cards = _to_tensor(raw["hand_cards"]).reshape(MAX_HAND_CARDS, CARD_FEATURE_DIM)
        self.hand_mask = _to_tensor(raw["hand_mask"])
        self.summons = _to_tensor(raw["summons"]).reshape(MAX_SUMMONS, ENTITY_FEATURE_DIM)
        self.summons_mask = _to_tensor(raw["summons_mask"])

        self.self_supports = _to_tensor(
            raw.get("self_supports", [0.0] * (MAX_SUPPORTS * ENTITY_FEATURE_DIM))
        ).reshape(MAX_SUPPORTS, ENTITY_FEATURE_DIM)
        self.self_supports_mask = _to_tensor(
            raw.get("self_supports_mask", [0.0] * MAX_SUPPORTS)
        )
        self.oppo_supports = _to_tensor(
            raw.get("oppo_supports", [0.0] * (MAX_SUPPORTS * ENTITY_FEATURE_DIM))
        ).reshape(MAX_SUPPORTS, ENTITY_FEATURE_DIM)
        self.oppo_supports_mask = _to_tensor(
            raw.get("oppo_supports_mask", [0.0] * MAX_SUPPORTS)
        )

        self.self_combat_statuses = _to_tensor(
            raw.get("self_combat_statuses", [0.0] * (MAX_COMBAT_STATUSES * ENTITY_FEATURE_DIM))
        ).reshape(MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM)
        self.self_combat_statuses_mask = _to_tensor(
            raw.get("self_combat_statuses_mask", [0.0] * MAX_COMBAT_STATUSES)
        )
        self.oppo_combat_statuses = _to_tensor(
            raw.get("oppo_combat_statuses", [0.0] * (MAX_COMBAT_STATUSES * ENTITY_FEATURE_DIM))
        ).reshape(MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM)
        self.oppo_combat_statuses_mask = _to_tensor(
            raw.get("oppo_combat_statuses_mask", [0.0] * MAX_COMBAT_STATUSES)
        )

        self.self_char_entities = _to_tensor(
            raw.get("self_char_entities", [0.0] * (_CHAR_ENT_SLOTS * ENTITY_FEATURE_DIM))
        ).reshape(_CHAR_ENT_SLOTS, ENTITY_FEATURE_DIM)
        self.self_char_entities_mask = _to_tensor(
            raw.get("self_char_entities_mask", [0.0] * _CHAR_ENT_SLOTS)
        )
        self.oppo_char_entities = _to_tensor(
            raw.get("oppo_char_entities", [0.0] * (_CHAR_ENT_SLOTS * ENTITY_FEATURE_DIM))
        ).reshape(_CHAR_ENT_SLOTS, ENTITY_FEATURE_DIM)
        self.oppo_char_entities_mask = _to_tensor(
            raw.get("oppo_char_entities_mask", [0.0] * _CHAR_ENT_SLOTS)
        )

        self.action_mask = _to_tensor(raw["action_mask"])
        self.mcts_policy = _to_tensor(raw["mcts_policy"])
        self.reward = float(raw["reward"])
        self.is_terminal = bool(raw["is_terminal"])
        self.hp_after_5_turns = float(raw.get("hp_after_5_turns", 0.0))
        self.cards_playable_next = _to_tensor(raw.get("cards_playable_next", [0.0] * MAX_HAND_CARDS))
        self.oppo_hand_features = _to_tensor(raw.get("oppo_hand_features", [0.0] * CARD_FEATURE_DIM))
        self.kill_within_3 = _to_tensor(raw.get("kill_within_3", [0.0] * (MAX_CHARACTERS * 2)))
        self.reaction_next_attack = float(raw.get("reaction_next_attack", 0.0))
        self.dice_effective_actions = float(raw.get("dice_effective_actions", 0.0))


class EpisodeDataset(Dataset):
    """
    Loads episodes, computes TD(λ) targets and initial PER priorities.

    Priorities are stored as a mutable tensor so the training loop can
    update them online after each backward pass.
    """

    def __init__(
        self,
        data_dir: str,
        model: TCGNeuralEvaluator,
        device: str = "cpu",
        gamma: float = 0.99,
        lam: float = 0.7,
        per_alpha: float = PER_ALPHA,
    ) -> None:
        super().__init__()
        self.steps: list[dict[str, torch.Tensor]] = []
        self.priorities: list[float] = []
        self.per_alpha = per_alpha
        data_path = Path(data_dir)
        if not data_path.exists():
            return

        model.eval()
        for fpath in sorted(data_path.glob("*.pt")):
            episodes: list[list[dict]] = torch.load(fpath, weights_only=False)
            for episode in episodes:
                self._process_episode(episode, model, device, gamma, lam)

    @torch.no_grad()
    def _process_episode(
        self,
        raw_steps: list[dict[str, Any]],
        model: TCGNeuralEvaluator,
        device: str,
        gamma: float,
        lam: float,
    ) -> None:
        if not raw_steps:
            return

        parsed = [EpisodeStep(s) for s in raw_steps]
        T = len(parsed)

        batched = self._collate_steps(parsed, device)
        out: ModelOutput = model(batched)
        values = out.value.squeeze(-1).cpu()  # [T]

        rewards = torch.tensor([s.reward for s in parsed])
        is_terminals = torch.tensor([1.0 if s.is_terminal else 0.0 for s in parsed])
        td_targets = compute_td_lambda_targets(rewards, values, is_terminals, gamma, lam)

        td_errors = (values - td_targets).abs()

        for t in range(T):
            step = parsed[t]
            self.steps.append({
                "global_features": step.global_features,
                "self_characters": step.self_characters,
                "oppo_characters": step.oppo_characters,
                "hand_cards": step.hand_cards,
                "hand_mask": step.hand_mask,
                "summons": step.summons,
                "summons_mask": step.summons_mask,
                "self_supports": step.self_supports,
                "self_supports_mask": step.self_supports_mask,
                "oppo_supports": step.oppo_supports,
                "oppo_supports_mask": step.oppo_supports_mask,
                "self_combat_statuses": step.self_combat_statuses,
                "self_combat_statuses_mask": step.self_combat_statuses_mask,
                "oppo_combat_statuses": step.oppo_combat_statuses,
                "oppo_combat_statuses_mask": step.oppo_combat_statuses_mask,
                "self_char_entities": step.self_char_entities,
                "self_char_entities_mask": step.self_char_entities_mask,
                "oppo_char_entities": step.oppo_char_entities,
                "oppo_char_entities_mask": step.oppo_char_entities_mask,
                "action_mask": step.action_mask,
                "target_policy": step.mcts_policy,
                "target_value": td_targets[t],
                "target_next_hp": torch.tensor(step.hp_after_5_turns),
                "target_card_played": step.cards_playable_next,
                "target_oppo_hand_features": step.oppo_hand_features,
                "target_kill": step.kill_within_3,
                "target_reaction": torch.tensor(step.reaction_next_attack),
                "target_dice_eff": torch.tensor(step.dice_effective_actions),
            })
            priority = (td_errors[t].item() + PER_EPSILON) ** self.per_alpha
            self.priorities.append(priority)

    @staticmethod
    def _collate_steps(
        steps: list[EpisodeStep], device: str
    ) -> dict[str, torch.Tensor]:
        return {
            "global_features": torch.stack([s.global_features for s in steps]).to(device),
            "self_characters": torch.stack([s.self_characters for s in steps]).to(device),
            "oppo_characters": torch.stack([s.oppo_characters for s in steps]).to(device),
            "hand_cards": torch.stack([s.hand_cards for s in steps]).to(device),
            "hand_mask": torch.stack([s.hand_mask for s in steps]).to(device),
            "summons": torch.stack([s.summons for s in steps]).to(device),
            "summons_mask": torch.stack([s.summons_mask for s in steps]).to(device),
            "self_supports": torch.stack([s.self_supports for s in steps]).to(device),
            "self_supports_mask": torch.stack([s.self_supports_mask for s in steps]).to(device),
            "oppo_supports": torch.stack([s.oppo_supports for s in steps]).to(device),
            "oppo_supports_mask": torch.stack([s.oppo_supports_mask for s in steps]).to(device),
            "self_combat_statuses": torch.stack([s.self_combat_statuses for s in steps]).to(device),
            "self_combat_statuses_mask": torch.stack([s.self_combat_statuses_mask for s in steps]).to(device),
            "oppo_combat_statuses": torch.stack([s.oppo_combat_statuses for s in steps]).to(device),
            "oppo_combat_statuses_mask": torch.stack([s.oppo_combat_statuses_mask for s in steps]).to(device),
            "self_char_entities": torch.stack([s.self_char_entities for s in steps]).to(device),
            "self_char_entities_mask": torch.stack([s.self_char_entities_mask for s in steps]).to(device),
            "oppo_char_entities": torch.stack([s.oppo_char_entities for s in steps]).to(device),
            "oppo_char_entities_mask": torch.stack([s.oppo_char_entities_mask for s in steps]).to(device),
            "action_mask": torch.stack([s.action_mask for s in steps]).to(device),
        }

    def get_sampler_weights(self) -> torch.Tensor:
        """Return a 1-D weight tensor for WeightedRandomSampler."""
        return torch.tensor(self.priorities, dtype=torch.float64)

    def compute_is_weights(self, indices: list[int], beta: float) -> torch.Tensor:
        """
        Importance-sampling correction weights (Schaul et al. 2015).

            P(i) = priority_i / sum(priorities)
            w_i  = (N * P(i))^{-beta}

        Weights are normalized so max(w) = 1 to avoid exploding gradients.
        """
        N = len(self.priorities)
        total_p = sum(self.priorities)
        weights = torch.zeros(len(indices))
        for i, idx in enumerate(indices):
            prob = self.priorities[idx] / total_p
            weights[i] = (N * prob) ** (-beta)
        weights /= weights.max().clamp(min=1e-8)
        return weights

    def update_priorities(self, indices: list[int], new_td_errors: torch.Tensor) -> None:
        """Online update: recalculate priorities from fresh TD errors."""
        for i, idx in enumerate(indices):
            self.priorities[idx] = (new_td_errors[i].item() + PER_EPSILON) ** self.per_alpha

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> tuple[int, dict[str, torch.Tensor]]:
        return idx, self.steps[idx]


def collate_fn(
    batch: list[tuple[int, dict[str, torch.Tensor]]]
) -> tuple[list[int], dict[str, torch.Tensor]]:
    """Collate (index, step_dict) pairs into (indices, batched_dict)."""
    indices = [b[0] for b in batch]
    dicts = [b[1] for b in batch]
    keys = dicts[0].keys()
    stacked = {k: torch.stack([d[k] for d in dicts]) for k in keys}
    return indices, stacked


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Training Loop                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TrainConfig:
    data_dir: str = "./episodes"
    checkpoint_dir: str = "./checkpoints"
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.7
    grad_clip: float = 1.0
    value_weight: float = 1.0
    policy_weight: float = 1.0
    aux_weight: float = 0.1
    per_alpha: float = PER_ALPHA
    per_beta_start: float = PER_BETA_START
    per_beta_end: float = PER_BETA_END
    log_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Model architecture (passed to ModelConfig) ──
    d_context: int = 256
    d_trunk: int = 256
    n_heads: int = 4

    # ── Arena evaluation ──
    eval_every: int = 5
    eval_games_per_side: int = 30
    eval_mode: str = "fast"
    eval_temperature: float = 0.5
    promotion_threshold: float = 0.54
    early_stop_patience: int = 3
    project_root: str = ""
    bun_path: str = "bun"


_MODEL_INPUT_KEYS = frozenset((
    "global_features", "self_characters", "oppo_characters",
    "hand_cards", "hand_mask", "summons", "summons_mask",
    "self_supports", "self_supports_mask",
    "oppo_supports", "oppo_supports_mask",
    "self_combat_statuses", "self_combat_statuses_mask",
    "oppo_combat_statuses", "oppo_combat_statuses_mask",
    "self_char_entities", "self_char_entities_mask",
    "oppo_char_entities", "oppo_char_entities_mask",
    "action_mask",
))


def train_one_epoch(
    model: TCGNeuralEvaluator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dataset: EpisodeDataset,
    cfg: TrainConfig,
    epoch: int,
    beta: float,
) -> dict[str, float]:
    model.train()
    device = cfg.device

    totals = {
        "total": 0.0, "value": 0.0, "policy": 0.0,
        "hp": 0.0, "card": 0.0, "belief": 0.0,
        "kill": 0.0, "react": 0.0, "dice": 0.0,
        "policy_entropy": 0.0,
    }
    n_batches = 0

    for batch_idx, (indices, batch) in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        model_input = {k: batch[k] for k in _MODEL_INPUT_KEYS}
        output: ModelOutput = model(model_input)

        loss: LossOutput = compute_loss(
            output=output,
            target_value=batch["target_value"],
            target_policy=batch["target_policy"],
            target_next_hp=batch["target_next_hp"],
            target_card_played=batch["target_card_played"],
            hand_mask=batch["hand_mask"],
            target_oppo_hand_features=batch["target_oppo_hand_features"],
            target_kill=batch["target_kill"],
            target_reaction=batch["target_reaction"],
            target_dice_eff=batch["target_dice_eff"],
            value_weight=cfg.value_weight,
            policy_weight=cfg.policy_weight,
            aux_weight=cfg.aux_weight,
            per_sample=True,
        )

        is_weights = dataset.compute_is_weights(indices, beta).to(device)
        weighted_loss = (loss.total * is_weights).mean()

        optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        with torch.no_grad():
            new_td_errors = (output.value.squeeze(-1) - batch["target_value"]).abs().cpu()
        dataset.update_priorities(indices, new_td_errors)

        with torch.no_grad():
            log_pi = output.log_policy
            pi = torch.exp(log_pi).clamp(min=1e-8)
            entropy = -(pi * log_pi).sum(dim=-1).mean().item()

        totals["total"] += weighted_loss.item()
        totals["value"] += loss.value_loss.item()
        totals["policy"] += loss.policy_loss.item()
        totals["hp"] += loss.hp_loss.item()
        totals["card"] += loss.card_play_loss.item()
        totals["belief"] += loss.belief_loss.item()
        totals["kill"] += loss.kill_loss.item()
        totals["react"] += loss.reaction_loss.item()
        totals["dice"] += loss.dice_eff_loss.item()
        totals["policy_entropy"] += entropy
        n_batches += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg = {k: v / n_batches for k, v in totals.items()}
            print(
                f"  [epoch {epoch+1} | batch {batch_idx+1}/{len(loader)}] "
                f"total={avg['total']:.4f}  V={avg['value']:.4f}  "
                f"P={avg['policy']:.4f}  HP={avg['hp']:.4f}  "
                f"C={avg['card']:.4f}  B={avg['belief']:.4f}  "
                f"K={avg['kill']:.4f}  R={avg['react']:.4f}  D={avg['dice']:.4f}"
            )

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def _build_per_loader(
    dataset: EpisodeDataset,
    cfg: TrainConfig,
) -> DataLoader:
    """Build a DataLoader with WeightedRandomSampler from PER priorities."""
    weights = dataset.get_sampler_weights()
    num_samples = len(dataset)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Arena Evaluation — auto-export ONNX, run bun arena, parse results     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _export_temp_onnx(
    model: TCGNeuralEvaluator,
    path: Path,
) -> bool:
    """Export model to a temporary ONNX file. Returns True on success."""
    try:
        from model import export_onnx
        model.eval()
        export_onnx(model, str(path))
        model.train()
        return True
    except Exception as e:
        print(f"  [EVAL] ONNX export failed: {e}")
        return False


def _find_project_root(cfg: TrainConfig) -> Path:
    """Locate the project root (where packages/core/scripts/run_arena.ts lives)."""
    if cfg.project_root:
        return Path(cfg.project_root)
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd().parent,
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "packages" / "core" / "scripts" / "run_arena.ts").exists():
            return c
    return Path.cwd()


@dataclass
class ArenaEvalResult:
    win_rate: float = 0.5
    ci_lo: float = 0.0
    ci_hi: float = 1.0
    elo_delta: float = 0.0
    is_significant: bool = False
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    total_games: int = 0
    avg_game_length: float = 0.0
    error: str = ""


def run_arena_eval(
    candidate_onnx: Path,
    baseline_onnx: Path,
    cfg: TrainConfig,
    project_root: Path,
) -> ArenaEvalResult:
    """
    Spawn `bun run run_arena.ts` as a subprocess and parse the JSON report.
    Returns ArenaEvalResult with win rate, CI, and Elo delta.
    """
    result = ArenaEvalResult()

    arena_script = project_root / "packages" / "core" / "scripts" / "run_arena.ts"
    if not arena_script.exists():
        result.error = f"Arena script not found: {arena_script}"
        return result

    arena_output_dir = candidate_onnx.parent / "arena_tmp"
    arena_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        cfg.bun_path, "run", str(arena_script),
        "--model-a", str(candidate_onnx),
        "--model-b", str(baseline_onnx),
        "--games-per-side", str(cfg.eval_games_per_side),
        "--mode", cfg.eval_mode,
        "--temperature", str(cfg.eval_temperature),
        "--output", str(arena_output_dir),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        result.error = "Arena subprocess timed out (600s)"
        return result
    except FileNotFoundError:
        result.error = f"bun not found at '{cfg.bun_path}'. Set --bun_path or add bun to PATH."
        return result
    except Exception as e:
        result.error = f"Arena subprocess error: {e}"
        return result

    if proc.returncode != 0:
        stderr_preview = (proc.stderr or "")[:500]
        result.error = f"Arena exited with code {proc.returncode}: {stderr_preview}"
        return result

    report_path = arena_output_dir / "arena_report.json"
    if not report_path.exists():
        result.error = "Arena finished but arena_report.json not found"
        return result

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        result.win_rate = report.get("winRateA", 0.5)
        ci = report.get("winRateCI95", [0.0, 1.0])
        result.ci_lo = ci[0]
        result.ci_hi = ci[1]
        result.elo_delta = report.get("eloDelta", 0.0)
        result.is_significant = report.get("isSignificant", False)
        result.wins_a = report.get("winsA", 0)
        result.wins_b = report.get("winsB", 0)
        result.draws = report.get("draws", 0)
        result.total_games = report.get("totalGames", 0)
        stats = report.get("gameStats", {})
        result.avg_game_length = stats.get("avgGameLength", 0.0)
    except Exception as e:
        result.error = f"Failed to parse arena report: {e}"

    # Clean up temp arena output
    try:
        shutil.rmtree(arena_output_dir, ignore_errors=True)
    except Exception:
        pass

    return result


def train(cfg: TrainConfig) -> None:
    print(f"Device: {cfg.device}")
    print(f"Data:   {cfg.data_dir}")

    model_cfg = ModelConfig(d_context=cfg.d_context, d_trunk=cfg.d_trunk, n_heads=cfg.n_heads)
    model = TCGNeuralEvaluator(model_cfg).to(cfg.device)
    print(
        f"Model parameters: {count_parameters(model):,}  "
        f"(d_context={cfg.d_context}, d_trunk={cfg.d_trunk}, n_heads={cfg.n_heads})"
    )

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest = ckpt_dir / "latest.pt"
    start_epoch = 0
    if latest.exists():
        state = torch.load(latest, weights_only=False, map_location=cfg.device)
        saved_arch = state.get("model_config", {})
        current_arch = {"d_context": cfg.d_context, "d_trunk": cfg.d_trunk, "n_heads": cfg.n_heads}
        if saved_arch and saved_arch != current_arch:
            print(
                f"WARNING: Architecture mismatch! Checkpoint has {saved_arch}, "
                f"but current config is {current_arch}. Starting fresh (ignoring checkpoint weights)."
            )
            start_epoch = 0
        else:
            model.load_state_dict(state["model"])
            start_epoch = state.get("epoch", 0) + 1
            print(f"Resumed from epoch {start_epoch}")

    print("Loading episodes & computing TD(λ) targets + PER priorities ...")
    t0 = time.time()
    dataset = EpisodeDataset(
        data_dir=cfg.data_dir,
        model=model,
        device=cfg.device,
        gamma=cfg.gamma,
        lam=cfg.lam,
        per_alpha=cfg.per_alpha,
    )
    print(f"  {len(dataset)} steps loaded in {time.time() - t0:.1f}s")

    if len(dataset) == 0:
        print("No data found — generating synthetic episodes for smoke test ...")
        dataset = _make_synthetic_dataset(n_episodes=20, steps_per_ep=30, model=model, cfg=cfg)
        print(f"  {len(dataset)} synthetic steps")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01,
    )

    metrics_log_path = ckpt_dir / "metrics.jsonl"
    metrics_log = open(metrics_log_path, "a", encoding="utf-8")
    print(f"Metrics log: {metrics_log_path}")

    # ── Arena evaluation setup ──
    project_root = _find_project_root(cfg)
    baseline_onnx = ckpt_dir / "baseline.onnx"
    models_dir = project_root / "models"

    if not baseline_onnx.exists():
        # Try to bootstrap baseline from existing model files
        for candidate in [
            models_dir / "tcg_evaluator.onnx",
            ckpt_dir / "tcg_evaluator.onnx",
        ]:
            if candidate.exists():
                shutil.copy2(candidate, baseline_onnx)
                print(f"[EVAL] Baseline initialized from {candidate}")
                break
        else:
            # Export current model as the initial baseline
            print("[EVAL] No baseline found. Exporting current model as baseline...")
            _export_temp_onnx(model, baseline_onnx)

    eval_enabled = baseline_onnx.exists() and cfg.eval_every > 0
    if eval_enabled:
        print(
            f"[EVAL] Arena evaluation every {cfg.eval_every} epochs, "
            f"{cfg.eval_games_per_side * 2} games/eval, "
            f"promotion threshold: {cfg.promotion_threshold:.0%}, "
            f"early-stop patience: {cfg.early_stop_patience}"
        )
    else:
        print("[EVAL] Arena evaluation disabled (no baseline or eval_every=0)")

    best_win_rate = 0.5
    epochs_without_improvement = 0
    promotion_count = 0
    early_stopped = False

    for epoch in range(start_epoch, cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        beta = cfg.per_beta_start + frac * (cfg.per_beta_end - cfg.per_beta_start)

        loader = _build_per_loader(dataset, cfg)

        t0 = time.time()
        metrics = train_one_epoch(model, loader, optimizer, dataset, cfg, epoch, beta)
        elapsed = time.time() - t0
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        ent = metrics.get("policy_entropy", 0.0)

        print(
            f"Epoch {epoch+1}/{cfg.epochs}  ({elapsed:.1f}s)  "
            f"total={metrics['total']:.4f}  V={metrics['value']:.4f}  "
            f"P={metrics['policy']:.4f}  HP={metrics['hp']:.4f}  "
            f"C={metrics['card']:.4f}  B={metrics['belief']:.4f}  "
            f"K={metrics['kill']:.4f}  R={metrics['react']:.4f}  D={metrics['dice']:.4f}  "
            f"Ent={ent:.4f}  lr={lr:.2e}  beta={beta:.3f}"
        )

        log_entry: dict[str, Any] = {
            "epoch": epoch + 1,
            **metrics,
            "lr": lr,
            "beta": beta,
            "elapsed_s": round(elapsed, 2),
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "model_config": {
                    "d_context": cfg.d_context,
                    "d_trunk": cfg.d_trunk,
                    "n_heads": cfg.n_heads,
                },
            },
            ckpt_dir / "latest.pt",
        )

        # ── Arena evaluation ──
        is_eval_epoch = (
            eval_enabled
            and (epoch + 1) >= cfg.eval_every
            and (epoch + 1) % cfg.eval_every == 0
        )

        if is_eval_epoch:
            print(f"\n{'─' * 60}")
            print(f"[EVAL] Epoch {epoch+1}: Exporting candidate ONNX...")
            candidate_onnx = ckpt_dir / f"candidate_e{epoch+1}.onnx"

            if _export_temp_onnx(model, candidate_onnx):
                print(
                    f"[EVAL] Running arena: candidate (epoch {epoch+1}) "
                    f"vs baseline ({cfg.eval_games_per_side * 2} games)..."
                )
                eval_t0 = time.time()
                arena = run_arena_eval(candidate_onnx, baseline_onnx, cfg, project_root)
                eval_elapsed = time.time() - eval_t0

                log_entry["arena"] = {
                    "win_rate": arena.win_rate,
                    "ci_lo": arena.ci_lo,
                    "ci_hi": arena.ci_hi,
                    "elo_delta": arena.elo_delta,
                    "is_significant": arena.is_significant,
                    "wins_a": arena.wins_a,
                    "wins_b": arena.wins_b,
                    "draws": arena.draws,
                    "total_games": arena.total_games,
                    "avg_game_length": arena.avg_game_length,
                    "eval_elapsed_s": round(eval_elapsed, 1),
                    "error": arena.error,
                }

                if arena.error:
                    print(f"[EVAL] ERROR: {arena.error}")
                else:
                    wr_pct = arena.win_rate * 100
                    ci_lo_pct = arena.ci_lo * 100
                    ci_hi_pct = arena.ci_hi * 100
                    print(
                        f"[EVAL] Result: {arena.wins_a}W / {arena.wins_b}L / {arena.draws}D "
                        f"({arena.total_games} games, {eval_elapsed:.0f}s)"
                    )
                    print(
                        f"[EVAL] Win rate: {wr_pct:.1f}% "
                        f"[{ci_lo_pct:.1f}%, {ci_hi_pct:.1f}%] (95% CI)  "
                        f"Elo: {arena.elo_delta:+.0f}  "
                        f"Significant: {'YES' if arena.is_significant else 'NO'}"
                    )

                    # ── Promotion: candidate beats baseline ──
                    if arena.win_rate >= cfg.promotion_threshold and arena.is_significant:
                        promotion_count += 1
                        best_win_rate = arena.win_rate
                        epochs_without_improvement = 0

                        shutil.copy2(candidate_onnx, baseline_onnx)
                        promoted_path = ckpt_dir / f"promoted_e{epoch+1}.onnx"
                        shutil.copy2(candidate_onnx, promoted_path)
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "epoch": epoch,
                                "metrics": metrics,
                                "model_config": {
                                    "d_context": cfg.d_context,
                                    "d_trunk": cfg.d_trunk,
                                    "n_heads": cfg.n_heads,
                                },
                            },
                            ckpt_dir / f"promoted_e{epoch+1}.pt",
                        )

                        print(
                            f"[EVAL] *** PROMOTED *** New baseline! "
                            f"(win rate {wr_pct:.1f}%, promotion #{promotion_count})"
                        )
                        print(f"[EVAL] Saved: {promoted_path}")
                    else:
                        epochs_without_improvement += 1
                        print(
                            f"[EVAL] No promotion "
                            f"(need {cfg.promotion_threshold:.0%} + significant). "
                            f"Patience: {epochs_without_improvement}/{cfg.early_stop_patience}"
                        )

                    # ── Early stopping ──
                    if (
                        cfg.early_stop_patience > 0
                        and epochs_without_improvement >= cfg.early_stop_patience
                    ):
                        print(
                            f"[EVAL] *** EARLY STOP *** No improvement for "
                            f"{cfg.early_stop_patience} consecutive evaluations. "
                            f"Stopping training."
                        )
                        log_entry["early_stopped"] = True
                        early_stopped = True

                # Clean up candidate ONNX
                try:
                    candidate_onnx.unlink(missing_ok=True)
                except Exception:
                    pass

            print(f"{'─' * 60}\n")

        metrics_log.write(json.dumps(log_entry) + "\n")
        metrics_log.flush()

        if early_stopped:
            break

    metrics_log.close()
    if early_stopped:
        print(f"Training stopped early at epoch {epoch+1}. Promotions: {promotion_count}")
    else:
        print(f"Training complete. Promotions: {promotion_count}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Synthetic Data (for smoke testing without real episodes)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _make_synthetic_dataset(
    n_episodes: int,
    steps_per_ep: int,
    model: TCGNeuralEvaluator,
    cfg: TrainConfig,
) -> EpisodeDataset:
    """Generate random episodes so train() can run end-to-end without data."""
    ds = EpisodeDataset.__new__(EpisodeDataset)
    ds.steps = []
    ds.priorities = []
    ds.per_alpha = cfg.per_alpha

    for ep_idx in range(n_episodes):
        raw_steps: list[dict[str, Any]] = []
        outcome = 1.0 if ep_idx % 2 == 0 else -1.0

        for t in range(steps_per_ep):
            is_last = t == steps_per_ep - 1
            action_mask = (torch.rand(MAX_ACTION_SLOTS) > 0.5).float()
            action_mask[0] = 1.0

            mcts_policy = torch.rand(MAX_ACTION_SLOTS) * action_mask
            mcts_policy = mcts_policy / mcts_policy.sum().clamp(min=1e-8)

            raw_steps.append({
                "global_features": torch.randn(GLOBAL_FEATURE_DIM).tolist(),
                "self_characters": torch.randn(MAX_CHARACTERS * CHARACTER_FEATURE_DIM).tolist(),
                "oppo_characters": torch.randn(MAX_CHARACTERS * CHARACTER_FEATURE_DIM).tolist(),
                "hand_cards": torch.randn(MAX_HAND_CARDS * CARD_FEATURE_DIM).tolist(),
                "hand_mask": [1.0] * MAX_HAND_CARDS,
                "summons": torch.randn(MAX_SUMMONS * ENTITY_FEATURE_DIM).tolist(),
                "summons_mask": [1.0] * MAX_SUMMONS,
                "self_supports": torch.randn(MAX_SUPPORTS * ENTITY_FEATURE_DIM).tolist(),
                "self_supports_mask": [1.0] * MAX_SUPPORTS,
                "oppo_supports": torch.randn(MAX_SUPPORTS * ENTITY_FEATURE_DIM).tolist(),
                "oppo_supports_mask": [1.0] * MAX_SUPPORTS,
                "self_combat_statuses": torch.randn(MAX_COMBAT_STATUSES * ENTITY_FEATURE_DIM).tolist(),
                "self_combat_statuses_mask": [1.0] * MAX_COMBAT_STATUSES,
                "oppo_combat_statuses": torch.randn(MAX_COMBAT_STATUSES * ENTITY_FEATURE_DIM).tolist(),
                "oppo_combat_statuses_mask": [1.0] * MAX_COMBAT_STATUSES,
                "self_char_entities": torch.randn(_CHAR_ENT_SLOTS * ENTITY_FEATURE_DIM).tolist(),
                "self_char_entities_mask": [1.0] * _CHAR_ENT_SLOTS,
                "oppo_char_entities": torch.randn(_CHAR_ENT_SLOTS * ENTITY_FEATURE_DIM).tolist(),
                "oppo_char_entities_mask": [1.0] * _CHAR_ENT_SLOTS,
                "action_mask": action_mask.tolist(),
                "mcts_policy": mcts_policy.tolist(),
                "reward": outcome if is_last else 0.0,
                "is_terminal": is_last,
                "hp_after_5_turns": max(0.0, 0.7 - t * 0.02),
                "cards_playable_next": (torch.rand(MAX_HAND_CARDS) > 0.5).float().tolist(),
                "oppo_hand_features": torch.rand(CARD_FEATURE_DIM).clamp(0, 1).tolist(),
                "kill_within_3": (torch.rand(MAX_CHARACTERS * 2) > 0.85).float().tolist(),
                "reaction_next_attack": float(torch.rand(1).item() > 0.6),
                "dice_effective_actions": max(0.0, 0.4 - t * 0.01),
            })

        ds._process_episode(raw_steps, model, cfg.device, cfg.gamma, cfg.lam)

    return ds


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI Entry Point                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train TCGNeuralEvaluator with TD(λ) + PER")
    p.add_argument("--data_dir", type=str, default="./episodes")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.7)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--aux_weight", type=float, default=0.1)
    p.add_argument("--per_alpha", type=float, default=PER_ALPHA)
    p.add_argument("--per_beta_start", type=float, default=PER_BETA_START)
    p.add_argument("--per_beta_end", type=float, default=PER_BETA_END)
    p.add_argument("--log_interval", type=int, default=50,
                   help="Print batch-level metrics every N batches")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Model architecture
    p.add_argument("--d_context", type=int, default=256,
                   help="Context MLP output dimension")
    p.add_argument("--d_trunk", type=int, default=256,
                   help="Trunk MLP dimension")
    p.add_argument("--n_heads", type=int, default=4,
                   help="Number of attention heads")
    # Arena evaluation
    p.add_argument("--eval_every", type=int, default=5,
                   help="Run arena evaluation every N epochs (0 to disable)")
    p.add_argument("--eval_games_per_side", type=int, default=30,
                   help="Games per side in arena evaluation (total = 2x)")
    p.add_argument("--eval_mode", type=str, default="fast",
                   choices=["fast", "quality"])
    p.add_argument("--eval_temperature", type=float, default=0.5)
    p.add_argument("--promotion_threshold", type=float, default=0.54,
                   help="Win rate threshold to promote candidate to new baseline")
    p.add_argument("--early_stop_patience", type=int, default=3,
                   help="Stop after N consecutive evals without promotion (0 to disable)")
    p.add_argument("--project_root", type=str, default="",
                   help="Path to project root (auto-detected if empty)")
    p.add_argument("--bun_path", type=str, default="bun",
                   help="Path to bun executable")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)

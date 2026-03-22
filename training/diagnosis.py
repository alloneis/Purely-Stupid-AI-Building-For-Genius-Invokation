"""
diagnosis.py — Loss curve analysis and training diagnosis for the orchestrator.

Reads metrics.jsonl produced by train.py and determines the training state:
  - "healthy"            : Loss still dropping, arena improving -> keep going
  - "need_more_data"     : Loss dropping but arena degrading -> likely overfit, add data
  - "converged"          : Loss plateaued + arena stagnant -> capacity exhausted, scale up
  - "near_saturation"    : Loss plateaued but arena still improving -> one more data gen
  - "severe_overfit"     : Loss plateaued + arena degrading -> reset with more data
  - "cold_start"         : No arena data yet (first generation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TransitionConfig:
    loss_plateau_window: int = 4
    loss_plateau_min_drop: float = 0.02
    arena_stagnation_count: int = 2
    max_generations: int = 5
    grace_generations: int = 0
    promotion_threshold: float = 0.54
    arena_degrade_threshold: float = 0.48


@dataclass
class DiagnosisResult:
    verdict: str
    loss_trend: str          # "dropping", "plateaued"
    arena_trend: str         # "improving", "stagnant", "degrading", "no_data"
    final_value_loss: float
    final_policy_loss: float
    final_entropy: float
    best_arena_win_rate: float
    latest_arena_win_rate: float
    promotions_this_gen: int
    epochs_completed: int
    detail: str


def load_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    """Load all epoch entries from metrics.jsonl."""
    entries: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return entries
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def _detect_loss_plateau(
    entries: list[dict[str, Any]],
    window: int,
    min_drop: float,
) -> tuple[bool, str]:
    """
    Check if the combined (value + policy) loss has plateaued over the
    last `window` epochs. Returns (is_plateau, detail_string).
    """
    if len(entries) < window:
        return False, f"Not enough epochs ({len(entries)}) for window={window}"

    recent = entries[-window:]
    start_total = recent[0].get("total", 0.0)
    end_total = recent[-1].get("total", 0.0)

    if start_total <= 0:
        return False, "Start loss is zero or negative"

    relative_drop = (start_total - end_total) / abs(start_total)

    if relative_drop < min_drop:
        return True, (
            f"Plateau: loss dropped only {relative_drop:.4f} "
            f"({start_total:.4f} -> {end_total:.4f}) over {window} epochs, "
            f"threshold={min_drop}"
        )
    return False, (
        f"Still dropping: {relative_drop:.4f} relative drop "
        f"({start_total:.4f} -> {end_total:.4f})"
    )


def _analyze_arena(
    entries: list[dict[str, Any]],
    promotion_threshold: float,
    degrade_threshold: float,
) -> tuple[str, float, float, int]:
    """
    Analyze arena results from metrics entries.
    Returns (trend, best_win_rate, latest_win_rate, promotion_count).
    """
    arena_entries = [e for e in entries if "arena" in e and not e["arena"].get("error")]

    if not arena_entries:
        return "no_data", 0.0, 0.0, 0

    win_rates = [e["arena"]["win_rate"] for e in arena_entries]
    latest_wr = win_rates[-1]
    best_wr = max(win_rates)
    promotions = sum(
        1 for e in arena_entries
        if e["arena"].get("win_rate", 0) >= promotion_threshold
        and e["arena"].get("is_significant", False)
    )

    if latest_wr >= promotion_threshold:
        trend = "improving"
    elif latest_wr < degrade_threshold:
        trend = "degrading"
    else:
        trend = "stagnant"

    return trend, best_wr, latest_wr, promotions


def diagnose(
    metrics_path: Path,
    transition: TransitionConfig,
) -> DiagnosisResult:
    """
    Read metrics.jsonl and produce a diagnosis for the orchestrator.

    Decision matrix (corrected per user spec):
      - dropping + improving   -> "healthy"         (keep going)
      - dropping + stagnant    -> "need_more_data"  (still learning but arena can't show it)
      - dropping + degrading   -> "need_more_data"  (true overfit, add data)
      - plateaued + improving  -> "near_saturation" (one more try)
      - plateaued + stagnant   -> "converged"       (capacity exhausted -> scale up)
      - plateaued + degrading  -> "severe_overfit"  (reset + double data)
      - * + no_data            -> "cold_start"
    """
    entries = load_metrics(metrics_path)

    if not entries:
        return DiagnosisResult(
            verdict="cold_start",
            loss_trend="unknown", arena_trend="no_data",
            final_value_loss=0.0, final_policy_loss=0.0, final_entropy=0.0,
            best_arena_win_rate=0.0, latest_arena_win_rate=0.0,
            promotions_this_gen=0, epochs_completed=0,
            detail="No metrics data found",
        )

    last = entries[-1]
    is_plateau, plateau_detail = _detect_loss_plateau(
        entries, transition.loss_plateau_window, transition.loss_plateau_min_drop,
    )
    loss_trend = "plateaued" if is_plateau else "dropping"

    arena_trend, best_wr, latest_wr, promotions = _analyze_arena(
        entries, transition.promotion_threshold, transition.arena_degrade_threshold,
    )

    # Decision matrix
    if arena_trend == "no_data":
        verdict = "cold_start"
        detail = "No arena results available yet"
    elif not is_plateau and arena_trend == "improving":
        verdict = "healthy"
        detail = f"Loss still dropping, arena improving ({latest_wr:.1%}). {plateau_detail}"
    elif not is_plateau and arena_trend == "stagnant":
        verdict = "need_more_data"
        detail = f"Loss dropping but arena stagnant ({latest_wr:.1%}). Model may need more diverse data."
    elif not is_plateau and arena_trend == "degrading":
        verdict = "need_more_data"
        detail = f"Loss dropping but arena degrading ({latest_wr:.1%}). True overfit — add more data."
    elif is_plateau and arena_trend == "improving":
        verdict = "near_saturation"
        detail = f"Loss plateaued but arena still improving ({latest_wr:.1%}). One more generation may help."
    elif is_plateau and arena_trend == "stagnant":
        verdict = "converged"
        detail = (
            f"Loss plateaued AND arena stagnant ({latest_wr:.1%}). "
            f"Model has converged at current capacity. Scale up architecture."
        )
    elif is_plateau and arena_trend == "degrading":
        verdict = "severe_overfit"
        detail = (
            f"Loss plateaued AND arena degrading ({latest_wr:.1%}). "
            f"Severe overfit — reset and double data volume."
        )
    else:
        verdict = "healthy"
        detail = f"Unclassified state: loss={loss_trend}, arena={arena_trend}"

    return DiagnosisResult(
        verdict=verdict,
        loss_trend=loss_trend,
        arena_trend=arena_trend,
        final_value_loss=last.get("value", 0.0),
        final_policy_loss=last.get("policy", 0.0),
        final_entropy=last.get("policy_entropy", 0.0),
        best_arena_win_rate=best_wr,
        latest_arena_win_rate=latest_wr,
        promotions_this_gen=promotions,
        epochs_completed=last.get("epoch", 0),
        detail=detail,
    )

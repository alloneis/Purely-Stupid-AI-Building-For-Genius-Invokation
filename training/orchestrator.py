#!/usr/bin/env python3
"""
orchestrator.py — Auto-scaling training pipeline for TCG Neural AI.

Reads a strategy.json that defines progressive scaling phases (A/B/C...),
then auto-chains the full pipeline:
  1. Self-play data generation  (bun run_neural_selfplay.ts or run_bootstrap.ts)
  2. JSONL -> .pt conversion    (python convert_to_pt.py)
  3. Training with arena eval   (python train.py)
  4. Loss curve diagnosis        (diagnosis.py)
  5. Decision: continue / add data / scale up / early stop

Key safety features:
  - Cold-start: first generation auto-promotes without arena
  - Grace period: after architecture upscale, skips arena for N generations
  - FIFO replay buffer: keeps last N generations of data
  - Corrected convergence detection: plateau + stagnant = scale up, NOT overfit

Usage:
  python orchestrator.py [--strategy strategy.json] [--resume]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diagnosis import (
    diagnose,
    DiagnosisResult,
    TransitionConfig,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Strategy Parsing                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class PhaseModelConfig:
    d_context: int = 256
    d_trunk: int = 256
    n_heads: int = 4


@dataclass
class PhaseDataConfig:
    games: int = 10000
    mode: str = "fast"
    batch: int = 16
    temperature: float = 1.0


@dataclass
class PhaseTrainingConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 256
    eval_every: int = 5
    early_stop_patience: int = 3


@dataclass
class PhaseArenaConfig:
    games_per_side: int = 30
    promotion_threshold: float = 0.54


@dataclass
class PhaseTransitionConfig:
    max_generations: int = 5
    loss_plateau_window: int = 4
    loss_plateau_min_drop: float = 0.02
    arena_stagnation_count: int = 2
    arena_degrade_threshold: float = 0.48
    grace_generations: int = 0


@dataclass
class Phase:
    name: str
    model: PhaseModelConfig
    data: PhaseDataConfig
    training: PhaseTrainingConfig
    arena: PhaseArenaConfig
    transition: PhaseTransitionConfig


@dataclass
class GlobalConfig:
    bun_path: str = "bun"
    device: str = "cuda"
    work_dir: str = "./training_runs"
    data_gen: str = "neural"
    buffer_generations: int = 3


@dataclass
class Strategy:
    phases: list[Phase]
    global_cfg: GlobalConfig


def _load_strategy(path: Path) -> Strategy:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    phases = []
    for p in raw["phases"]:
        phases.append(Phase(
            name=p["name"],
            model=PhaseModelConfig(**p.get("model", {})),
            data=PhaseDataConfig(**p.get("data", {})),
            training=PhaseTrainingConfig(**p.get("training", {})),
            arena=PhaseArenaConfig(**p.get("arena", {})),
            transition=PhaseTransitionConfig(**p.get("transition", {})),
        ))

    global_cfg = GlobalConfig(**raw.get("global", {}))
    return Strategy(phases=phases, global_cfg=global_cfg)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Project Path Resolution                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _find_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd().parent,
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "packages" / "core" / "scripts" / "run_arena.ts").exists():
            return c
    return Path(__file__).resolve().parent.parent


def _find_training_dir() -> Path:
    return Path(__file__).resolve().parent


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Structured Logging                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


class OrchestratorLog:
    def __init__(self, work_dir: Path):
        self.log_path = work_dir / "orchestrator.jsonl"
        self._f = open(self.log_path, "a", encoding="utf-8")

    def append(self, entry: dict[str, Any]) -> None:
        entry["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._f.write(json.dumps(entry, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIFO Replay Buffer                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _manage_replay_buffer(
    episodes_dir: Path,
    active_dir: Path,
    max_generations: int,
) -> None:
    """
    FIFO buffer: keep only the last `max_generations` generation subdirs.
    Symlink/copy all their .pt files into `active_dir` for training.
    """
    active_dir.mkdir(parents=True, exist_ok=True)

    # Clear active directory
    for f in active_dir.glob("*.pt"):
        f.unlink()

    # Find generation subdirectories sorted by name (timestamped)
    gen_dirs = sorted(
        [d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith("gen_")],
        key=lambda d: d.name,
    )

    # Evict oldest generations beyond buffer limit
    while len(gen_dirs) > max_generations:
        oldest = gen_dirs.pop(0)
        print(f"  [BUFFER] Evicting old generation: {oldest.name}")
        shutil.rmtree(oldest, ignore_errors=True)

    # Copy all .pt files from remaining generations into active/
    total_pt = 0
    for gd in gen_dirs:
        for pt_file in gd.glob("*.pt"):
            dest = active_dir / f"{gd.name}_{pt_file.name}"
            shutil.copy2(pt_file, dest)
            total_pt += 1

    print(f"  [BUFFER] {len(gen_dirs)} generations, {total_pt} .pt files in active/")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Subprocess Runners                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _run_subprocess(
    cmd: list[str],
    label: str,
    cwd: Path,
    timeout: int = 0,
) -> tuple[int, str]:
    """Run a subprocess, streaming its stdout to console. Returns (exit_code, stderr)."""
    print(f"\n  [{label}] Running: {' '.join(cmd[:6])}...")
    try:
        kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdout": sys.stdout,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if timeout > 0:
            kwargs["timeout"] = timeout

        proc = subprocess.run(cmd, **kwargs)
        return proc.returncode, proc.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, f"Timed out after {timeout}s"
    except FileNotFoundError as e:
        return -2, str(e)


def run_data_gen(
    phase: Phase,
    gen_label: str,
    output_dir: Path,
    model_onnx: Path | None,
    global_cfg: GlobalConfig,
    project_root: Path,
) -> tuple[bool, float]:
    """Generate self-play data. Returns (success, elapsed_seconds)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if global_cfg.data_gen == "bootstrap" or model_onnx is None or not model_onnx.exists():
        script = project_root / "packages" / "core" / "scripts" / "run_bootstrap.ts"
        cmd = [
            global_cfg.bun_path, "run", str(script),
            "--games", str(phase.data.games),
            "--parallel", str(phase.data.batch),
            "--output", str(output_dir),
        ]
    else:
        script = project_root / "packages" / "core" / "scripts" / "run_neural_selfplay.ts"
        cmd = [
            global_cfg.bun_path, "run", str(script),
            "--model", str(model_onnx),
            "--games", str(phase.data.games),
            "--batch", str(phase.data.batch),
            "--mode", phase.data.mode,
            "--temperature", str(phase.data.temperature),
            "--output", str(output_dir),
        ]

    code, stderr = _run_subprocess(cmd, f"DATA-GEN {gen_label}", project_root)
    elapsed = time.time() - t0

    if code != 0:
        print(f"  [DATA-GEN] FAILED (exit={code}): {stderr[:300]}")
        return False, elapsed

    jsonl_count = len(list(output_dir.glob("*.jsonl")))
    print(f"  [DATA-GEN] Done: {jsonl_count} .jsonl files in {elapsed:.0f}s")
    return True, elapsed


def run_convert(
    jsonl_dir: Path,
    pt_dir: Path,
    training_dir: Path,
) -> bool:
    """Convert JSONL episodes to .pt files."""
    pt_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(training_dir / "convert_to_pt.py"),
        "--input", str(jsonl_dir),
        "--output", str(pt_dir),
    ]
    code, stderr = _run_subprocess(cmd, "CONVERT", training_dir)
    if code != 0:
        print(f"  [CONVERT] FAILED: {stderr[:300]}")
        return False

    pt_count = len(list(pt_dir.glob("*.pt")))
    print(f"  [CONVERT] Done: {pt_count} .pt files")
    return True


def run_training(
    phase: Phase,
    data_dir: Path,
    ckpt_dir: Path,
    baseline_onnx: Path | None,
    global_cfg: GlobalConfig,
    training_dir: Path,
    project_root: Path,
) -> tuple[bool, float]:
    """Run train.py with the phase's configuration. Returns (success, elapsed)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cmd = [
        sys.executable, str(training_dir / "train.py"),
        "--data_dir", str(data_dir),
        "--checkpoint_dir", str(ckpt_dir),
        "--epochs", str(phase.training.epochs),
        "--batch_size", str(phase.training.batch_size),
        "--lr", str(phase.training.lr),
        "--device", global_cfg.device,
        "--d_context", str(phase.model.d_context),
        "--d_trunk", str(phase.model.d_trunk),
        "--n_heads", str(phase.model.n_heads),
        "--eval_every", str(phase.training.eval_every),
        "--early_stop_patience", str(phase.training.early_stop_patience),
        "--eval_games_per_side", str(phase.arena.games_per_side),
        "--promotion_threshold", str(phase.arena.promotion_threshold),
        "--project_root", str(project_root),
        "--bun_path", global_cfg.bun_path,
    ]

    code, stderr = _run_subprocess(cmd, "TRAIN", training_dir)
    elapsed = time.time() - t0

    if code != 0:
        print(f"  [TRAIN] FAILED (exit={code}): {stderr[:300]}")
        return False, elapsed

    print(f"  [TRAIN] Done in {elapsed:.0f}s")
    return True, elapsed


def export_onnx(
    ckpt_path: Path,
    output_path: Path,
    training_dir: Path,
) -> bool:
    """Export checkpoint to ONNX."""
    cmd = [
        sys.executable, str(training_dir / "export_onnx.py"),
        "--checkpoint", str(ckpt_path),
        "--output", str(output_path),
    ]
    code, stderr = _run_subprocess(cmd, "EXPORT", training_dir)
    if code != 0:
        print(f"  [EXPORT] FAILED: {stderr[:300]}")
        return False
    return True


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Architecture Change Detection                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _arch_changed(prev: Phase | None, current: Phase) -> bool:
    if prev is None:
        return False
    return (
        prev.model.d_context != current.model.d_context
        or prev.model.d_trunk != current.model.d_trunk
        or prev.model.n_heads != current.model.n_heads
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main Orchestration Loop                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def orchestrate(strategy: Strategy, resume: bool = False) -> None:
    global_cfg = strategy.global_cfg
    work_dir = Path(global_cfg.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    project_root = _find_project_root()
    training_dir = _find_training_dir()

    log = OrchestratorLog(work_dir)
    episodes_root = work_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)
    active_data = work_dir / "episodes" / "active"

    current_model_onnx = work_dir / "current_model.onnx"
    baseline_onnx = work_dir / "baseline.onnx"

    # Bootstrap from existing project-level ONNX if no work_dir model yet
    project_onnx = project_root / "models" / "tcg_evaluator.onnx"
    if not current_model_onnx.exists() and project_onnx.exists():
        shutil.copy2(project_onnx, current_model_onnx)
        print(f"Bootstrapped current_model from {project_onnx}")

    state_path = work_dir / "orchestrator_state.json"
    state: dict[str, Any] = {}
    if resume and state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        print(f"Resuming from state: phase={state.get('phase_idx', 0)}, gen={state.get('gen_idx', 0)}")

    start_phase = state.get("phase_idx", 0)
    total_promotions = state.get("total_promotions", 0)

    print(f"{'=' * 70}")
    print(f"  Auto-Scaling Training Orchestrator")
    print(f"  Phases: {[p.name for p in strategy.phases]}")
    print(f"  Work dir: {work_dir}")
    print(f"  Project root: {project_root}")
    print(f"  Buffer: keep last {global_cfg.buffer_generations} generations")
    print(f"{'=' * 70}\n")

    prev_phase: Phase | None = None
    if start_phase > 0:
        prev_phase = strategy.phases[start_phase - 1]

    for phase_idx in range(start_phase, len(strategy.phases)):
        phase = strategy.phases[phase_idx]
        is_arch_change = _arch_changed(prev_phase, phase)
        grace_remaining = phase.transition.grace_generations if is_arch_change else 0

        ckpt_dir = work_dir / f"checkpoints_{phase.name}"

        print(f"\n{'#' * 70}")
        print(f"  PHASE: {phase.name}")
        print(f"  Model: d_context={phase.model.d_context}, d_trunk={phase.model.d_trunk}, n_heads={phase.model.n_heads}")
        print(f"  Data: {phase.data.games} games, mode={phase.data.mode}")
        if is_arch_change:
            print(f"  ARCHITECTURE CHANGE detected. Grace period: {grace_remaining} generations")
        print(f"{'#' * 70}\n")

        if is_arch_change:
            # Archive the previous phase's best model
            if baseline_onnx.exists():
                archive_name = f"phase_{prev_phase.name}_best.onnx" if prev_phase else "prev_best.onnx"
                archive_path = work_dir / archive_name
                shutil.copy2(baseline_onnx, archive_path)
                print(f"  Archived previous baseline -> {archive_path}")
            # Clear baseline — new architecture starts fresh
            if baseline_onnx.exists():
                baseline_onnx.unlink()

        start_gen = 0
        if resume and phase_idx == start_phase:
            start_gen = state.get("gen_idx", 0)

        stagnation_count = 0

        for gen in range(start_gen, phase.transition.max_generations):
            gen_label = f"{phase.name}_gen{gen}"
            gen_data_dir = episodes_root / f"gen_{phase.name}_{gen:03d}"

            print(f"\n{'─' * 60}")
            print(f"  Generation: {gen_label} ({gen + 1}/{phase.transition.max_generations})")
            print(f"{'─' * 60}")

            # Save resume state
            state = {
                "phase_idx": phase_idx,
                "gen_idx": gen,
                "total_promotions": total_promotions,
            }
            with open(state_path, "w") as f:
                json.dump(state, f)

            # ── Step 1: Data generation ──
            model_for_selfplay = current_model_onnx if current_model_onnx.exists() else None
            data_ok, data_elapsed = run_data_gen(
                phase, gen_label, gen_data_dir,
                model_for_selfplay, global_cfg, project_root,
            )
            if not data_ok:
                log.append({"phase": phase.name, "generation": gen, "error": "data_gen_failed"})
                print("  FATAL: Data generation failed. Stopping.")
                log.close()
                return

            # ── Step 2: Convert JSONL -> .pt ──
            pt_dir = gen_data_dir  # .pt files go alongside .jsonl
            if not run_convert(gen_data_dir, pt_dir, training_dir):
                log.append({"phase": phase.name, "generation": gen, "error": "convert_failed"})
                print("  FATAL: Conversion failed. Stopping.")
                log.close()
                return

            # ── Step 3: Manage replay buffer (FIFO) ──
            _manage_replay_buffer(episodes_root, active_data, global_cfg.buffer_generations)

            # ── Step 4: Copy baseline into checkpoint dir for train.py's arena ──
            if baseline_onnx.exists():
                shutil.copy2(baseline_onnx, ckpt_dir / "baseline.onnx")

            # ── Step 5: Training ──
            train_ok, train_elapsed = run_training(
                phase, str(active_data), ckpt_dir,
                baseline_onnx if baseline_onnx.exists() else None,
                global_cfg, training_dir, project_root,
            )
            if not train_ok:
                log.append({"phase": phase.name, "generation": gen, "error": "training_failed"})
                print("  WARNING: Training failed. Continuing to next generation.")
                continue

            # ── Step 6: Export current model to ONNX ──
            latest_ckpt = ckpt_dir / "latest.pt"
            gen_onnx = work_dir / f"{gen_label}.onnx"
            if latest_ckpt.exists():
                if export_onnx(latest_ckpt, gen_onnx, training_dir):
                    shutil.copy2(gen_onnx, current_model_onnx)
                else:
                    log.append({"phase": phase.name, "generation": gen, "error": "onnx_export_failed"})
                    continue

            # ── Step 7: Cold-start handling ──
            if not baseline_onnx.exists():
                shutil.copy2(current_model_onnx, baseline_onnx)
                print(f"  [COLD-START] First model promoted to baseline (no arena needed)")
                total_promotions += 1
                log.append({
                    "phase": phase.name, "generation": gen,
                    "data_games": phase.data.games,
                    "data_elapsed_s": round(data_elapsed, 1),
                    "training_elapsed_s": round(train_elapsed, 1),
                    "diagnosis": "cold_start_promoted",
                    "action": "continue",
                    "promotions_total": total_promotions,
                })
                continue

            # ── Step 8: Grace period (after architecture change) ──
            if grace_remaining > 0:
                grace_remaining -= 1
                print(
                    f"  [GRACE] Architecture warmup: "
                    f"{grace_remaining} generations remaining before arena"
                )
                if grace_remaining == 0:
                    # Grace period over: promote current as new baseline for fair comparison
                    shutil.copy2(current_model_onnx, baseline_onnx)
                    shutil.copy2(baseline_onnx, ckpt_dir / "baseline.onnx")
                    print(f"  [GRACE] Grace period complete. New architecture baseline set.")

                log.append({
                    "phase": phase.name, "generation": gen,
                    "data_games": phase.data.games,
                    "data_elapsed_s": round(data_elapsed, 1),
                    "training_elapsed_s": round(train_elapsed, 1),
                    "diagnosis": "grace_period",
                    "grace_remaining": grace_remaining,
                    "action": "continue",
                })
                continue

            # ── Step 9: Diagnose ──
            metrics_path = ckpt_dir / "metrics.jsonl"
            transition_cfg = TransitionConfig(
                loss_plateau_window=phase.transition.loss_plateau_window,
                loss_plateau_min_drop=phase.transition.loss_plateau_min_drop,
                arena_stagnation_count=phase.transition.arena_stagnation_count,
                max_generations=phase.transition.max_generations,
                grace_generations=phase.transition.grace_generations,
                promotion_threshold=phase.arena.promotion_threshold,
                arena_degrade_threshold=phase.transition.arena_degrade_threshold,
            )
            diag = diagnose(metrics_path, transition_cfg)

            print(f"\n  [DIAGNOSIS] Verdict: {diag.verdict}")
            print(f"    Loss: {diag.loss_trend} | Arena: {diag.arena_trend}")
            print(f"    Value={diag.final_value_loss:.4f} Policy={diag.final_policy_loss:.4f} Ent={diag.final_entropy:.1f}")
            print(f"    Arena WR: latest={diag.latest_arena_win_rate:.1%} best={diag.best_arena_win_rate:.1%}")
            print(f"    Promotions this gen: {diag.promotions_this_gen}")
            print(f"    Detail: {diag.detail}")

            # Update promotion count from train.py's inline promotions
            if diag.promotions_this_gen > 0:
                total_promotions += diag.promotions_this_gen
                stagnation_count = 0
                # Copy train.py's promoted baseline back to orchestrator level
                train_baseline = ckpt_dir / "baseline.onnx"
                if train_baseline.exists():
                    shutil.copy2(train_baseline, baseline_onnx)

            log_entry: dict[str, Any] = {
                "phase": phase.name,
                "generation": gen,
                "data_games": phase.data.games,
                "data_elapsed_s": round(data_elapsed, 1),
                "training_elapsed_s": round(train_elapsed, 1),
                "epochs_completed": diag.epochs_completed,
                "final_value_loss": diag.final_value_loss,
                "final_policy_loss": diag.final_policy_loss,
                "final_entropy": diag.final_entropy,
                "arena_win_rate": diag.latest_arena_win_rate,
                "arena_best_win_rate": diag.best_arena_win_rate,
                "promotions_this_gen": diag.promotions_this_gen,
                "promotions_total": total_promotions,
                "diagnosis": diag.verdict,
                "loss_trend": diag.loss_trend,
                "arena_trend": diag.arena_trend,
                "detail": diag.detail,
            }

            # ── Step 10: Act on diagnosis ──
            if diag.verdict == "healthy":
                stagnation_count = 0
                log_entry["action"] = "continue"
                print(f"  [ACTION] Continue training (same phase)")

            elif diag.verdict == "near_saturation":
                stagnation_count += 1
                if stagnation_count >= phase.transition.arena_stagnation_count:
                    log_entry["action"] = "advance_phase"
                    print(f"  [ACTION] Near saturation x{stagnation_count} -> advancing to next phase")
                    log.append(log_entry)
                    break
                log_entry["action"] = "one_more_try"
                print(f"  [ACTION] Near saturation — one more data generation ({stagnation_count}/{phase.transition.arena_stagnation_count})")

            elif diag.verdict == "converged":
                log_entry["action"] = "advance_phase"
                print(f"  [ACTION] Model converged at current capacity -> advancing to next phase")
                log.append(log_entry)
                break

            elif diag.verdict == "need_more_data":
                stagnation_count = 0
                log_entry["action"] = "generate_more_data"
                print(f"  [ACTION] Need more data — regenerating")

            elif diag.verdict == "severe_overfit":
                stagnation_count += 1
                if stagnation_count >= phase.transition.arena_stagnation_count:
                    log_entry["action"] = "advance_phase"
                    print(f"  [ACTION] Severe overfit x{stagnation_count} -> giving up on this phase")
                    log.append(log_entry)
                    break
                log_entry["action"] = "reset_and_double_data"
                print(f"  [ACTION] Severe overfit — next gen will double data")

            elif diag.verdict == "cold_start":
                log_entry["action"] = "continue"
                print(f"  [ACTION] Cold start (no arena data) — continuing")

            else:
                log_entry["action"] = "continue"
                print(f"  [ACTION] Unknown verdict '{diag.verdict}' — continuing")

            log.append(log_entry)

        # Phase complete — clean up metrics.jsonl for next phase's fresh start
        metrics_file = ckpt_dir / "metrics.jsonl"
        if metrics_file.exists():
            archive = ckpt_dir / f"metrics_{phase.name}_final.jsonl"
            shutil.copy2(metrics_file, archive)
            metrics_file.unlink()

        prev_phase = phase
        print(f"\n  Phase {phase.name} complete. Total promotions: {total_promotions}")

    # All phases done
    state = {"phase_idx": len(strategy.phases), "gen_idx": 0, "total_promotions": total_promotions}
    with open(state_path, "w") as f:
        json.dump(state, f)

    log.append({
        "event": "pipeline_complete",
        "total_promotions": total_promotions,
        "phases_completed": len(strategy.phases),
    })
    log.close()

    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete!")
    print(f"  Total promotions: {total_promotions}")
    print(f"  Final model: {current_model_onnx}")
    print(f"  Baseline: {baseline_onnx}")
    print(f"  Logs: {work_dir / 'orchestrator.jsonl'}")
    print(f"{'=' * 70}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI Entry Point                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-scaling TCG AI training orchestrator",
    )
    parser.add_argument(
        "--strategy", type=str, default="strategy.json",
        help="Path to strategy.json config file",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved orchestrator state",
    )
    args = parser.parse_args()

    strategy_path = Path(args.strategy)
    if not strategy_path.exists():
        print(f"ERROR: Strategy file not found: {strategy_path}")
        sys.exit(1)

    strategy = _load_strategy(strategy_path)
    orchestrate(strategy, resume=args.resume)


if __name__ == "__main__":
    main()

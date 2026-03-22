#!/usr/bin/env python3
"""
export_onnx.py — Load a trained checkpoint and export to ONNX format.

Usage:
  python training/export_onnx.py [--checkpoint ./checkpoints/latest.pt] [--output ./models/tcg_evaluator.onnx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import TCGNeuralEvaluator, ModelConfig, export_onnx, make_dummy_batch, count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained model to ONNX")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/latest.pt")
    parser.add_argument("--output", type=str, default="./models/tcg_evaluator.onnx")
    parser.add_argument("--verify", action="store_true", help="Run verification after export")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")

    saved_arch = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
    if saved_arch:
        model_cfg = ModelConfig(
            d_context=saved_arch.get("d_context", 256),
            d_trunk=saved_arch.get("d_trunk", 256),
            n_heads=saved_arch.get("n_heads", 4),
        )
        print(f"  Using saved ModelConfig: {saved_arch}")
    else:
        model_cfg = ModelConfig()
        print("  No model_config in checkpoint, using defaults")

    model = TCGNeuralEvaluator(model_cfg)
    state_dict = None
    epoch = "?"
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        epoch = ckpt.get("epoch", "?")
    elif isinstance(ckpt, dict) and any(
        k.startswith("fixed_enc") or k.startswith("hand_enc") for k in ckpt
    ):
        state_dict = ckpt

    if state_dict is not None:
        model.load_state_dict(state_dict)
        print(f"  Loaded weights from epoch {epoch}")
    else:
        print("  WARNING: Could not identify checkpoint format, using untrained model")

    print(f"Parameters: {count_parameters(model):,}")

    export_onnx(model, str(out_path))

    if args.verify:
        verify_onnx(str(out_path))


def verify_onnx(onnx_path: str) -> None:
    """Verify the exported ONNX model produces matching outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skipping verification")
        print("  Install with: pip install onnxruntime")
        return

    import numpy as np

    print("\nVerifying ONNX output...")
    sess = ort.InferenceSession(onnx_path)

    batch = make_dummy_batch(batch_size=2)
    feeds = {}
    for inp in sess.get_inputs():
        name = inp.name
        feeds[name] = batch[name].numpy()

    outputs = sess.run(None, feeds)
    output_names = [o.name for o in sess.get_outputs()]

    print(f"  Inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"  Outputs: {output_names}")
    for name, arr in zip(output_names, outputs):
        print(f"    {name}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]")

    value = outputs[0]
    assert value.shape == (2, 1), f"Value shape mismatch: {value.shape}"
    assert -1.0 <= value.min() and value.max() <= 1.0, f"Value out of tanh range"

    log_policy = outputs[1]
    assert log_policy.shape[1] == 64, f"Policy dim mismatch: {log_policy.shape}"

    policy_probs = np.exp(log_policy)
    prob_sums = policy_probs.sum(axis=-1)
    for i, s in enumerate(prob_sums):
        assert abs(s - 1.0) < 0.01, f"Policy probs don't sum to 1: {s}"

    print("  All verifications passed!")


if __name__ == "__main__":
    main()

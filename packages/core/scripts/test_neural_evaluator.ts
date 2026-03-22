#!/usr/bin/env bun
/**
 * Smoke test for NeuralEvaluator + ONNX inference.
 *
 * Loads the exported ONNX model via onnxruntime-node, creates a
 * NeuralEvaluator, and runs evaluateAsync + getPolicy on a real game state.
 *
 * Usage:
 *   bun run packages/core/scripts/test_neural_evaluator.ts [--model ./models/tcg_evaluator.onnx]
 */

import { DEFAULT_ASSETS_MANAGER } from "@gi-tcg/assets-manager";
import getData from "@gi-tcg/data";
import { ActionValidity, DiceType } from "@gi-tcg/typings";
import * as ort from "onnxruntime-node";

import { CURRENT_VERSION } from "../src/index";
import { getDefaultGameConfig } from "../src/base/state";
import { PureGameEngine } from "../src/decoupled/pure_engine";
import { RuleEngine } from "../src/decoupled/rule_engine";
import { ISMCTSAgent } from "../src/decoupled/ismcts_agent";
import {
  NeuralEvaluator,
  type OnnxSession,
  type OnnxTensor,
  type OnnxTensorFactory,
} from "../src/decoupled/neural/neural_evaluator";
import { MAX_ACTION_SLOTS } from "../src/decoupled/neural/constants";

const DECK_1 = "FZDByRUNGRCB0WoNFlGgWpEPE0AB9TAPFGCB9kgWGIERCoEQDLFADcQQDPFgacYWDJAA";
const DECK_2 = "FdHxNj8TAWDQxFkMFkAhyWIYCYDA45wOCUDQ5J0PGEAh9IIPGWDh9p4YC6FgirYRCxAA";

// ─── Adapter: onnxruntime-node → our OnnxSession interface ────────────

function wrapSession(session: ort.InferenceSession): OnnxSession {
  return {
    async run(feeds, _outputNames) {
      const ortFeeds: Record<string, ort.Tensor> = {};
      for (const [key, val] of Object.entries(feeds)) {
        ortFeeds[key] = new ort.Tensor("float32", val.data as Float32Array, val.dims as number[]);
      }
      const result = await session.run(ortFeeds);
      const output: Record<string, OnnxTensor> = {};
      for (const [key, val] of Object.entries(result)) {
        output[key] = {
          data: val.data as Float32Array,
          dims: val.dims,
        };
      }
      return output;
    },
  };
}

const tensorFactory: OnnxTensorFactory = {
  create(data: Float32Array, dims: readonly number[]): OnnxTensor {
    return { data, dims };
  },
};

// ─── Main ──────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const modelPath = (() => {
    const idx = args.indexOf("--model");
    return idx >= 0 && args[idx + 1] ? args[idx + 1] : "./models/tcg_evaluator.onnx";
  })();

  console.log(`Loading ONNX model: ${modelPath}`);
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu","cuda"],
  });
  console.log(`  Inputs:  ${session.inputNames.join(", ")}`);
  console.log(`  Outputs: ${session.outputNames.join(", ")}`);

  const wrappedSession = wrapSession(session);
  const neural = new NeuralEvaluator({
    session: wrappedSession,
    tensorFactory,
  });
  console.log("NeuralEvaluator created\n");

  // Set up a game state
  const data = getData(CURRENT_VERSION);
  const config = { ...getDefaultGameConfig(), errorLevel: "toleratePreview" as const };
  const engine = new PureGameEngine(data, config, {
    rerollDice: (_s, _w, dice) => dice.filter((d) => d !== DiceType.Omni),
  });

  let state = engine.createInitialState(
    DEFAULT_ASSETS_MANAGER.decode(DECK_1),
    DEFAULT_ASSETS_MANAGER.decode(DECK_2),
  );

  // Advance a few steps so there's a non-trivial state
  for (let i = 0; i < 5 && !state.isFinished; i++) {
    const actions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
      .filter((a) => a.validity === ActionValidity.VALID);
    if (actions.length === 0) break;
    try {
      state = await engine.execute(state, actions[0]);
    } catch { break; }
  }

  const who = state.gameState.currentTurn;
  const legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
    .filter((a) => a.validity === ActionValidity.VALID);
  console.log(`Game state: round=${state.gameState.roundNumber}, turn=${who}, legal=${legalActions.length}`);

  // Test evaluateAsync
  console.log("\n--- evaluateAsync ---");
  const t0 = performance.now();
  const value = await neural.evaluateAsync(state, who);
  const t1 = performance.now();
  console.log(`  Value: ${value.toFixed(6)}  (${(t1 - t0).toFixed(1)}ms)`);

  // Test getPolicy
  console.log("\n--- getPolicy ---");
  const t2 = performance.now();
  const policy = await neural.getPolicy(state, legalActions, who);
  const t3 = performance.now();
  const policySum = policy.reduce((s, v) => s + v, 0);
  console.log(`  Policy sum: ${policySum.toFixed(6)}  (${(t3 - t2).toFixed(1)}ms)`);

  const topIndices = [...policy]
    .map((p, i) => ({ i, p }))
    .filter((x) => x.p > 0.01)
    .sort((a, b) => b.p - a.p)
    .slice(0, 5);
  for (const { i, p } of topIndices) {
    console.log(`    slot ${i}: ${(p * 100).toFixed(1)}%`);
  }

  // Test evaluateWithPolicy
  console.log("\n--- evaluateWithPolicy ---");
  const t4 = performance.now();
  const combined = await neural.evaluateWithPolicy(state, legalActions, who);
  const t5 = performance.now();
  console.log(`  Value: ${combined.value.toFixed(6)}, Policy sum: ${combined.policy.reduce((s, v) => s + v, 0).toFixed(6)}  (${(t5 - t4).toFixed(1)}ms)`);

  // Test with ISMCTSAgent using neural evaluator
  console.log("\n--- ISMCTSAgent + NeuralEvaluator (async) ---");
  const agent = new ISMCTSAgent({
    iterations: 30,
    determinizationCount: 1,
    maxDepth: 4,
    neuralEvaluator: neural,
  });

  const t6 = performance.now();
  const { action, debug } = await agent.getBestActionNeuralAsync(state, legalActions, who);
  const t7 = performance.now();
  console.log(`  Selected: ${debug.selected.visits} visits, mean=${debug.selected.meanValue.toFixed(4)}`);
  console.log(`  Top3:`);
  for (const top of debug.top3) {
    const payload = top.action.action;
    const desc = payload ? `${payload.$case}` : "?";
    console.log(`    ${desc} visits=${top.visits} mean=${top.meanValue.toFixed(4)}`);
  }
  console.log(`  Time: ${(t7 - t6).toFixed(1)}ms`);
  console.log(`  Process logs (last 3):`);
  for (const log of debug.processLogs.slice(-3)) {
    console.log(`    ${log}`);
  }

  console.log("\nAll checks passed!");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

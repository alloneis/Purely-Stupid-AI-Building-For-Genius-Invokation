#!/usr/bin/env bun
/**
 * Neural Self-Play Data Generator
 *
 * Runs ONNX-powered self-play games and outputs JSONL training data.
 *
 * Fast mode:  Neural policy sampling only.  Target: 4000+ games/hour.
 * Quality mode: Neural + light IS-MCTS.     Target: 300+ games/hour.
 *
 * Usage:
 *   bun run packages/core/scripts/run_neural_selfplay.ts \
 *     --model ./models/tcg_evaluator.onnx \
 *     --games 10000 --batch 8 --mode fast --output ./episodes
 */

import { DEFAULT_ASSETS_MANAGER } from "@gi-tcg/assets-manager";
import getData from "@gi-tcg/data";
import * as ort from "onnxruntime-node";
import { writeFileSync, appendFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";

import { CURRENT_VERSION } from "../src/index";
import { getDefaultGameConfig } from "../src/base/state";
import {
  NeuralSelfPlayEngine,
  type SelfPlayConfig,
  type StepRecord,
  type EpisodeResult,
} from "../src/decoupled/neural/self_play_engine";
import type { OnnxSession, OnnxTensor, OnnxTensorFactory } from "../src/decoupled/neural/neural_evaluator";

// ─── ONNX Adapters ─────────────────────────────────────────────────────

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
        output[key] = { data: val.data as Float32Array, dims: val.dims };
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

// ─── Deck Pool ─────────────────────────────────────────────────────────

const DECK_POOL: [string, string][] = [
  [
    "FZDByRUNGRCB0WoNFlGgWpEPE0AB9TAPFGCB9kgWGIERCoEQDLFADcQQDPFgacYWDJAA",
    "FdHxNj8TAWDQxFkMFkAhyWIYCYDA45wOCUDQ5J0PGEAh9IIPGWDh9p4YC6FgirYRCxAA",
  ],
];

// ─── Main ──────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, def: string) => {
    const idx = args.indexOf(`--${name}`);
    return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
  };

  const modelPath = getArg("model", "./models/tcg_evaluator.onnx");
  const numGames = parseInt(getArg("games", "100"), 10);
  const batchSize = parseInt(getArg("batch", "8"), 10);
  const mode = getArg("mode", "fast") as "fast" | "quality";
  const temperature = parseFloat(getArg("temperature", "1.0"));
  const outputDir = resolve(getArg("output", "./episodes"));
  const flushEvery = parseInt(getArg("flush", "200"), 10);

  if (!existsSync(outputDir)) mkdirSync(outputDir, { recursive: true });

  console.log(`Loading ONNX model: ${modelPath}`);
  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
  });
  console.log(`  Inputs:  ${session.inputNames.join(", ")}`);
  console.log(`  Outputs: ${session.outputNames.join(", ")}`);

  const data = getData(CURRENT_VERSION);
  const gameConfig = { ...getDefaultGameConfig(), errorLevel: "toleratePreview" as const };
  const selfPlayConfig: Partial<SelfPlayConfig> = {
    mode,
    batchSize,
    temperature,
  };

  const engine = new NeuralSelfPlayEngine(
    wrapSession(session),
    tensorFactory,
    data as any,
    gameConfig,
    selfPlayConfig,
  );

  console.log(
    `\nSelf-play: ${numGames} games, batch=${batchSize}, mode=${mode}, ` +
    `temp=${temperature}, output → ${outputDir}\n`,
  );

  let completedGames = 0;
  let totalStepsAll = 0;
  let totalEpisodes = 0;
  let totalP0Wins = 0;
  let totalP1Wins = 0;
  let totalDraws = 0;
  const allEpisodes: StepRecord[][] = [];
  const startTime = Date.now();
  let fileCount = 0;
  let batchIdx = 0;

  const progressFile = resolve(outputDir, "progress.jsonl");

  for (let batch = 0; batch < numGames; batch += batchSize) {
    const thisBatch = Math.min(batchSize, numGames - batch);
    const gameIds: number[] = [];
    for (let i = 0; i < thisBatch; i++) gameIds.push(batch + i);

    const results = await engine.runBatch(
      gameIds,
      DECK_POOL,
      (code) => DEFAULT_ASSETS_MANAGER.decode(code),
    );

    let batchSteps = 0;
    let batchP0 = 0;
    let batchP1 = 0;
    let batchDraws = 0;

    for (const result of results) {
      completedGames++;
      totalStepsAll += result.totalSteps;
      batchSteps += result.totalSteps;
      if (result.winner === 0) { batchP0++; totalP0Wins++; }
      else if (result.winner === 1) { batchP1++; totalP1Wins++; }
      else { batchDraws++; totalDraws++; }
      for (const ep of result.episodes) {
        if (ep.length > 0) {
          allEpisodes.push(ep);
          totalEpisodes++;
        }
      }
    }

    const elapsed = (Date.now() - startTime) / 1000;
    const avgMsNum = results.length > 0
      ? results.reduce((s, r) => s + r.elapsedMs, 0) / results.length
      : 0;
    const avgGameLen = results.length > 0
      ? batchSteps / results.length
      : 0;

    const progressEntry = {
      batch: batchIdx,
      completed: completedGames,
      total: numGames,
      episodes: totalEpisodes,
      steps: totalStepsAll,
      elapsed_s: +elapsed.toFixed(2),
      games_per_s: +(completedGames / elapsed).toFixed(2),
      avg_ms_per_game: +avgMsNum.toFixed(0),
      avg_game_length: +avgGameLen.toFixed(1),
      p0_wins: batchP0,
      p1_wins: batchP1,
      draws: batchDraws,
      cumulative_p0_wins: totalP0Wins,
      cumulative_p1_wins: totalP1Wins,
      cumulative_draws: totalDraws,
    };
    appendFileSync(progressFile, JSON.stringify(progressEntry) + "\n");
    batchIdx++;

    if (completedGames % Math.max(1, Math.min(50, batchSize * 5)) === 0 || completedGames === numGames) {
      const gps = (completedGames / elapsed).toFixed(2);
      const avgMs = avgMsNum.toFixed(0);
      console.log(
        `  [${completedGames}/${numGames}] episodes=${totalEpisodes} ` +
        `steps=${totalStepsAll} elapsed=${elapsed.toFixed(1)}s ` +
        `(${gps} games/s, ~${avgMs}ms/game) ` +
        `W: P0=${totalP0Wins} P1=${totalP1Wins} D=${totalDraws}`,
      );
    }

    if (allEpisodes.length >= flushEvery) {
      const chunkFile = resolve(outputDir, `neural_${mode}_${fileCount++}.jsonl`);
      const lines = allEpisodes.map((ep) => JSON.stringify(ep));
      writeFileSync(chunkFile, lines.join("\n") + "\n");
      console.log(`  flushed ${allEpisodes.length} episodes → ${chunkFile}`);
      allEpisodes.length = 0;
    }
  }

  if (allEpisodes.length > 0) {
    const chunkFile = resolve(outputDir, `neural_${mode}_${fileCount++}.jsonl`);
    const lines = allEpisodes.map((ep) => JSON.stringify(ep));
    writeFileSync(chunkFile, lines.join("\n") + "\n");
    console.log(`  flushed ${allEpisodes.length} episodes → ${chunkFile}`);
  }

  engine.dispose();

  const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  const finalGps = (completedGames / ((Date.now() - startTime) / 1000)).toFixed(2);
  console.log(
    `\nDone: ${completedGames} games, ${totalEpisodes} episodes, ` +
    `${totalStepsAll} total steps in ${totalElapsed}s (${finalGps} games/s)`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

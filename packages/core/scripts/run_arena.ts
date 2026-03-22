#!/usr/bin/env bun
/**
 * Model Arena — Pit two ONNX models against each other.
 *
 * Runs N games per side (A=P0 then A=P1) to eliminate first-player bias.
 * Reports win rate with Wilson score 95% CI, Elo delta, and game statistics.
 *
 * Usage:
 *   bun run packages/core/scripts/run_arena.ts \
 *     --model-a ./models/new_model.onnx \
 *     --model-b ./models/old_model.onnx \
 *     --games-per-side 100 --mode fast --output ./arena_results
 */

import { DEFAULT_ASSETS_MANAGER } from "@gi-tcg/assets-manager";
import getData from "@gi-tcg/data";
import * as ort from "onnxruntime-node";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, basename } from "node:path";

import { CURRENT_VERSION } from "../src/index";
import { getDefaultGameConfig } from "../src/base/state";
import {
  ArenaEngine,
  type ArenaConfig,
  type ArenaResult,
  type ArenaGameRecord,
} from "../src/decoupled/neural/arena_engine";
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

// ─── Report Formatter ──────────────────────────────────────────────────

function printReport(r: ArenaResult): void {
  const pctA = ((r.winsA / r.totalGames) * 100).toFixed(1);
  const pctB = ((r.winsB / r.totalGames) * 100).toFixed(1);
  const pctD = ((r.draws / r.totalGames) * 100).toFixed(1);
  const ciLo = (r.winRateCI95[0] * 100).toFixed(1);
  const ciHi = (r.winRateCI95[1] * 100).toFixed(1);
  const wrPct = (r.winRateA * 100).toFixed(1);

  console.log(`\n${"=".repeat(60)}`);
  console.log(`Arena: ${r.modelA} vs ${r.modelB}`);
  console.log(`  ${r.totalGames} games (${r.totalGames / 2} per side), ${r.totalGames > 0 ? "fast" : "?"} mode`);
  console.log(`${"=".repeat(60)}`);
  console.log();
  console.log(`  Model A wins: ${String(r.winsA).padStart(4)}/${r.totalGames} (${pctA}%)`);
  console.log(`  Model B wins: ${String(r.winsB).padStart(4)}/${r.totalGames} (${pctB}%)`);
  console.log(`  Draws:        ${String(r.draws).padStart(4)}/${r.totalGames} (${pctD}%)`);
  console.log();
  console.log(`  Win rate (excl draws): ${wrPct}% [${ciLo}%, ${ciHi}%] (95% CI)`);
  console.log(`  Elo delta: ${r.eloDelta >= 0 ? "+" : ""}${r.eloDelta.toFixed(0)}`);
  console.log(`  Significant improvement: ${r.isSignificant ? "YES" : "NO"}`);
  console.log();
  console.log(`  Game stats:`);
  console.log(`    Avg length:         ${r.gameStats.avgGameLength.toFixed(1)} steps`);
  console.log(`    Avg surviving chars: ${r.gameStats.avgSurvivingCharsWinner.toFixed(1)} (winner)`);
  console.log(`    Avg first kill:      step ${r.gameStats.avgTurnsToFirstKill.toFixed(0)}`);
  console.log(`    Avg actions/turn:    ${r.gameStats.avgActionsPerTurn.toFixed(1)}`);
  console.log(`${"=".repeat(60)}\n`);
}

// ─── Main ──────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, def: string) => {
    const idx = args.indexOf(`--${name}`);
    return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
  };

  const modelAPath = getArg("model-a", "./models/tcg_evaluator.onnx");
  const modelBPath = getArg("model-b", "./models/tcg_evaluator.onnx");
  const gamesPerSide = parseInt(getArg("games-per-side", "50"), 10);
  const mode = getArg("mode", "fast") as "fast" | "quality";
  const temperature = parseFloat(getArg("temperature", "0.5"));
  const outputDir = resolve(getArg("output", "./arena_results"));

  if (!existsSync(outputDir)) mkdirSync(outputDir, { recursive: true });

  console.log(`Loading Model A: ${modelAPath}`);
  const sessionA = await ort.InferenceSession.create(modelAPath, {
    executionProviders: ["cpu"],
  });

  console.log(`Loading Model B: ${modelBPath}`);
  const sessionB = await ort.InferenceSession.create(modelBPath, {
    executionProviders: ["cpu"],
  });

  const data = getData(CURRENT_VERSION);
  const gameConfig = { ...getDefaultGameConfig(), errorLevel: "toleratePreview" as const };

  const arenaConfig: Partial<ArenaConfig> = {
    gamesPerSide,
    mode,
    temperature,
  };

  const engine = new ArenaEngine(
    wrapSession(sessionA),
    wrapSession(sessionB),
    tensorFactory,
    tensorFactory,
    data as any,
    gameConfig,
    basename(modelAPath),
    basename(modelBPath),
    arenaConfig,
  );

  const totalGames = gamesPerSide * 2;
  console.log(
    `\nArena: ${basename(modelAPath)} vs ${basename(modelBPath)}` +
    `  (${totalGames} games, ${gamesPerSide}/side, mode=${mode}, temp=${temperature})\n`,
  );

  const decodedDecks = DECK_POOL.map(
    ([d1, d2]) => [DEFAULT_ASSETS_MANAGER.decode(d1), DEFAULT_ASSETS_MANAGER.decode(d2)] as [any, any],
  );

  const startTime = Date.now();
  let liveA = 0;
  let liveB = 0;
  let liveD = 0;

  const result = await engine.run(decodedDecks, (completed, total, record) => {
    if (record.winner === 0) liveA++;
    else if (record.winner === 1) liveB++;
    else liveD++;
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const w = record.winner === 0 ? "A" : record.winner === 1 ? "B" : "D";
    process.stdout.write(
      `\r  [${completed}/${total}] ${elapsed}s  ` +
      `A=${liveA} B=${liveB} D=${liveD}  last=${w} ${record.elapsedMs.toFixed(0)}ms`,
    );
  });

  console.log();

  printReport(result);

  const reportPath = resolve(outputDir, "arena_report.json");
  const { perGame, ...summary } = result;
  writeFileSync(reportPath, JSON.stringify(summary, null, 2));
  console.log(`Report saved: ${reportPath}`);

  const detailPath = resolve(outputDir, "arena_games.jsonl");
  const lines = perGame.map((g) => JSON.stringify(g));
  writeFileSync(detailPath, lines.join("\n") + "\n");
  console.log(`Game details: ${detailPath}`);

  engine.dispose();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

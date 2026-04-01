#!/usr/bin/env bun
/**
 * Phase 0 Bootstrap Data Collector
 *
 * Runs IS-MCTS self-play games and records training examples as JSONL.
 * Each line is a JSON-encoded episode (array of step dicts).
 *
 * Safety valves:
 *   - Max 15 rounds per game (force terminal on round overflow)
 *   - Max 15 actions per turn per player (force declareEnd)
 *   - Max 500 total steps per game (absolute backstop)
 *
 * Usage:
 *   bun run packages/core/scripts/run_bootstrap.ts [--games 100] [--output ./episodes] [--parallel 4]
 */

import { DEFAULT_ASSETS_MANAGER } from "@gi-tcg/assets-manager";
import getData from "@gi-tcg/data";
import { ActionValidity, DiceType, type Action } from "@gi-tcg/typings";
import { flip } from "@gi-tcg/utils";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";

import {
  CURRENT_VERSION,
  type PureGameState,
  type GameAction,
} from "../src/index";
import { getDefaultGameConfig } from "../src/base/state";
import { PureGameEngine } from "../src/decoupled/pure_engine";
import { ISMCTSAgent } from "../src/decoupled/ismcts_agent";
import { RuleEngine } from "../src/decoupled/rule_engine";
import { NeuralStateEncoder, type EncodedState } from "../src/decoupled/neural/state_encoder";
import { ActionIndexer } from "../src/decoupled/neural/action_encoder";
import {
  MAX_ACTION_SLOTS,
  CARD_FEATURE_DIM,
} from "../src/decoupled/neural/constants";

// ─── Config ────────────────────────────────────────────────────────────

const MAX_ROUNDS = 15;
const MAX_ACTIONS_PER_TURN = 15;
const MAX_TOTAL_STEPS = 500;

interface BootstrapConfig {
  iterations: number;
  determinizations: number;
  maxDepth: number;
}

const DECK_POOL = [
  "FZDByRUNGRCB0WoNFlGgWpEPE0AB9TAPFGCB9kgWGIERCoEQDLFADcQQDPFgacYWDJAA",
  "FdHxNj8TAWDQxFkMFkAhyWIYCYDA45wOCUDQ5J0PGEAh9IIPGWDh9p4YC6FgirYRCxAA",
];

// ─── Types ─────────────────────────────────────────────────────────────

interface StepRecord {
  global_features: number[];
  self_characters: number[];
  oppo_characters: number[];
  hand_cards: number[];
  hand_mask: number[];
  summons: number[];
  summons_mask: number[];
  self_supports: number[];
  self_supports_mask: number[];
  oppo_supports: number[];
  oppo_supports_mask: number[];
  self_combat_statuses: number[];
  self_combat_statuses_mask: number[];
  oppo_combat_statuses: number[];
  oppo_combat_statuses_mask: number[];
  self_char_entities: number[];
  self_char_entities_mask: number[];
  oppo_char_entities: number[];
  oppo_char_entities_mask: number[];
  action_mask: number[];
  mcts_policy: number[];
  reward: number;
  is_terminal: boolean;
  hp_after_5_turns: number;
  cards_playable_next: number[];
  oppo_hand_features: number[];
  kill_within_3: number[];
  reaction_next_attack: number;
  dice_effective_actions: number;
}

// ─── Helpers ───────────────────────────────────────────────────────────

const encoder = new NeuralStateEncoder();
const indexer = new ActionIndexer();

function f32ToArray(f: Float32Array): number[] {
  return Array.from(f);
}

function computeOppoHandFeatures(
  state: PureGameState,
  perspective: 0 | 1,
): number[] {
  const oppo = state.gameState.players[flip(perspective)];
  const result = new Float32Array(CARD_FEATURE_DIM);
  if (oppo.hands.length === 0) return f32ToArray(result);

  for (const hand of oppo.hands) {
    const id = (hand.definition.id % 10000) / 10000;
    result[0] += id;
  }
  result[0] /= oppo.hands.length;

  return f32ToArray(result);
}

function getActiveCharHp(state: PureGameState, perspective: 0 | 1): number {
  const player = state.gameState.players[perspective];
  const active = player.characters.find(
    (c) => c.id === player.activeCharacterId,
  );
  if (!active) return 0;
  const hp = typeof active.variables.health === "number" ? active.variables.health : 0;
  return hp / 10;
}

function countEffectiveActions(state: PureGameState, perspective: 0 | 1): number {
  if (state.gameState.currentTurn !== perspective) return 0;
  try {
    const actions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
      .filter((a) => a.validity === ActionValidity.VALID);
    const nonEnd = actions.filter((a) => a.action?.$case !== "declareEnd");
    return Math.min(nonEnd.length, 10) / 10;
  } catch {
    return 0;
  }
}

function buildMctsPolicy(
  debug: { candidates: Array<{ action: GameAction; visits: number }> },
  legalActions: readonly GameAction[],
  state: PureGameState,
  perspective: 0 | 1,
): number[] {
  const policy = new Float32Array(MAX_ACTION_SLOTS);
  let totalVisits = 0;
  for (const c of debug.candidates) totalVisits += c.visits;
  if (totalVisits === 0) {
    const mask = indexer.buildMask(legalActions, state.gameState, perspective);
    const count = mask.reduce((s, v) => s + v, 0);
    if (count > 0) {
      for (let i = 0; i < mask.length; i++) policy[i] = mask[i] / count;
    }
    return f32ToArray(policy);
  }
  for (const c of debug.candidates) {
    const idx = indexer.actionToIndex(c.action, state.gameState, perspective);
    if (idx >= 0 && idx < MAX_ACTION_SLOTS) {
      policy[idx] = c.visits / totalVisits;
    }
  }
  return f32ToArray(policy);
}

function computeReward(state: PureGameState, perspective: 0 | 1): number {
  const winner = state.gameState.winner;
  if (winner === null || winner === undefined) {
    const selfHp = totalTeamHp(state, perspective);
    const oppoHp = totalTeamHp(state, flip(perspective) as 0 | 1);
    if (selfHp === oppoHp) return 0;
    return selfHp > oppoHp ? 0.3 : -0.3;
  }
  return winner === perspective ? 1.0 : -1.0;
}

function totalTeamHp(state: PureGameState, who: 0 | 1): number {
  let total = 0;
  for (const c of state.gameState.players[who].characters) {
    const hp = typeof c.variables.health === "number" ? c.variables.health : 0;
    total += Math.max(0, hp);
  }
  return total;
}

// ─── Single Game Runner ────────────────────────────────────────────────

async function runOneGame(
  gameId: number,
  data: any,
  cfg: BootstrapConfig,
): Promise<{ steps: StepRecord[][]; winner: number | null; totalSteps: number }> {
  const config = {
    ...getDefaultGameConfig(),
    errorLevel: "toleratePreview" as const,
  };
  const engine = new PureGameEngine(data, config, {
    rerollDice: (_s: any, _w: any, dice: DiceType[], _r: any) =>
      dice.filter((d) => d !== DiceType.Omni),
  });

  const d1 = DECK_POOL[gameId % DECK_POOL.length];
  const d2 = DECK_POOL[(gameId + 1) % DECK_POOL.length];
  let state = engine.createInitialState(
    DEFAULT_ASSETS_MANAGER.decode(d1),
    DEFAULT_ASSETS_MANAGER.decode(d2),
  );

  const agents = [
    new ISMCTSAgent({
      iterations: cfg.iterations,
      determinizationCount: cfg.determinizations,
      maxDepth: cfg.maxDepth,
    }),
    new ISMCTSAgent({
      iterations: cfg.iterations,
      determinizationCount: cfg.determinizations,
      maxDepth: cfg.maxDepth,
    }),
  ];

  const episodes: StepRecord[][] = [[], []];
  let totalSteps = 0;
  let actionsThisTurn = 0;
  let lastTurnPlayer = -1;

  while (!state.isFinished && totalSteps < MAX_TOTAL_STEPS) {
    const currentRound = state.gameState.roundNumber;
    if (currentRound > MAX_ROUNDS) break;

    const who = state.gameState.currentTurn;
    if (who !== lastTurnPlayer) {
      actionsThisTurn = 0;
      lastTurnPlayer = who;
    }

    let legalActions: GameAction[];
    try {
      legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
        .filter((a) => a.validity === ActionValidity.VALID);
    } catch {
      break;
    }
    if (legalActions.length === 0) break;

    let chosenAction: GameAction;
    let mctsPolicy: number[];

    if (actionsThisTurn >= MAX_ACTIONS_PER_TURN) {
      const endAction = legalActions.find((a) => a.action?.$case === "declareEnd");
      if (endAction) {
        chosenAction = endAction;
        mctsPolicy = new Array(MAX_ACTION_SLOTS).fill(0);
        mctsPolicy[0] = 1.0;
      } else {
        chosenAction = legalActions[0];
        mctsPolicy = new Array(MAX_ACTION_SLOTS).fill(0);
      }
    } else {
      try {
        const { action, debug } = agents[who].getBestActionWithDebug(
          state, legalActions, who,
        );
        chosenAction = action;
        mctsPolicy = buildMctsPolicy(debug, legalActions, state, who);
      } catch {
        chosenAction = legalActions[0];
        mctsPolicy = new Array(MAX_ACTION_SLOTS).fill(0);
      }
    }

    const encoded = encoder.encode(state, who);
    const actionMask = indexer.buildMask(legalActions, state.gameState, who);

    const step: StepRecord = {
      global_features: f32ToArray(encoded.global_features),
      self_characters: f32ToArray(encoded.self_characters),
      oppo_characters: f32ToArray(encoded.oppo_characters),
      hand_cards: f32ToArray(encoded.hand_cards),
      hand_mask: f32ToArray(encoded.hand_cards_mask),
      summons: f32ToArray(encoded.self_summons),
      summons_mask: f32ToArray(encoded.self_summons_mask),
      self_supports: f32ToArray(encoded.self_supports),
      self_supports_mask: f32ToArray(encoded.self_supports_mask),
      oppo_supports: f32ToArray(encoded.oppo_supports),
      oppo_supports_mask: f32ToArray(encoded.oppo_supports_mask),
      self_combat_statuses: f32ToArray(encoded.self_combat_statuses),
      self_combat_statuses_mask: f32ToArray(encoded.self_combat_statuses_mask),
      oppo_combat_statuses: f32ToArray(encoded.oppo_combat_statuses),
      oppo_combat_statuses_mask: f32ToArray(encoded.oppo_combat_statuses_mask),
      self_char_entities: f32ToArray(encoded.self_char_entities),
      self_char_entities_mask: f32ToArray(encoded.self_char_entities_mask),
      oppo_char_entities: f32ToArray(encoded.oppo_char_entities),
      oppo_char_entities_mask: f32ToArray(encoded.oppo_char_entities_mask),
      action_mask: f32ToArray(actionMask),
      mcts_policy: mctsPolicy,
      reward: 0,
      is_terminal: false,
      hp_after_5_turns: 0,
      cards_playable_next: new Array(10).fill(0),
      oppo_hand_features: computeOppoHandFeatures(state, who),
      kill_within_3: new Array(6).fill(0),
      reaction_next_attack: 0,
      dice_effective_actions: countEffectiveActions(state, who),
    };

    episodes[who].push(step);

    try {
      state = await engine.execute(state, chosenAction);
    } catch {
      break;
    }

    actionsThisTurn++;
    totalSteps++;
  }

  for (const who of [0, 1] as const) {
    const reward = computeReward(state, who);
    const ep = episodes[who];
    if (ep.length > 0) {
      for (const step of ep) step.reward = 0;
      ep[ep.length - 1].reward = reward;
      ep[ep.length - 1].is_terminal = true;
    }

    for (let i = 0; i < ep.length; i++) {
      ep[i].hp_after_5_turns = getActiveCharHp(state, who);
    }
  }

  return {
    steps: episodes,
    winner: state.gameState.winner,
    totalSteps,
  };
}

// ─── Main ──────────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, def: string) => {
    const idx = args.indexOf(`--${name}`);
    return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
  };

  const numGames = parseInt(getArg("games", "100"), 10);
  const parallel = parseInt(getArg("parallel", "4"), 10);
  const outputDir = resolve(getArg("output", "./episodes"));
  const cfg: BootstrapConfig = {
    iterations: parseInt(getArg("iterations", "120"), 10),
    determinizations: parseInt(getArg("determinizations", "3"), 10),
    maxDepth: parseInt(getArg("maxDepth", "8"), 10),
  };

  if (!existsSync(outputDir)) mkdirSync(outputDir, { recursive: true });

  const data = getData(CURRENT_VERSION);
  console.log(
    `Bootstrap: ${numGames} games, ${parallel} parallel, ` +
    `ISMCTS(iter=${cfg.iterations}, det=${cfg.determinizations}, depth=${cfg.maxDepth}), ` +
    `output → ${outputDir}`,
  );

  let completedGames = 0;
  let totalStepsAll = 0;
  let totalEpisodes = 0;
  const allEpisodes: StepRecord[][] = [];
  const startTime = Date.now();

  for (let batch = 0; batch < numGames; batch += parallel) {
    const batchSize = Math.min(parallel, numGames - batch);
    const promises = [];
    for (let i = 0; i < batchSize; i++) {
      promises.push(runOneGame(batch + i, data, cfg));
    }

    const results = await Promise.all(promises);

    for (const result of results) {
      completedGames++;
      totalStepsAll += result.totalSteps;
      for (const ep of result.steps) {
        if (ep.length > 0) {
          allEpisodes.push(ep);
          totalEpisodes++;
        }
      }
    }

    if (completedGames % 10 === 0 || completedGames === numGames) {
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      const gps = (completedGames / ((Date.now() - startTime) / 1000)).toFixed(2);
      console.log(
        `  [${completedGames}/${numGames}] episodes=${totalEpisodes} ` +
        `steps=${totalStepsAll} elapsed=${elapsed}s (${gps} games/s)`,
      );
    }

    if (allEpisodes.length >= 200) {
      const chunkFile = resolve(outputDir, `bootstrap_${Date.now()}.jsonl`);
      const lines = allEpisodes.map((ep) => JSON.stringify(ep));
      writeFileSync(chunkFile, lines.join("\n") + "\n");
      console.log(`  flushed ${allEpisodes.length} episodes → ${chunkFile}`);
      allEpisodes.length = 0;
    }
  }

  if (allEpisodes.length > 0) {
    const chunkFile = resolve(outputDir, `bootstrap_${Date.now()}.jsonl`);
    const lines = allEpisodes.map((ep) => JSON.stringify(ep));
    writeFileSync(chunkFile, lines.join("\n") + "\n");
    console.log(`  flushed ${allEpisodes.length} episodes → ${chunkFile}`);
  }

  const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(
    `\nDone: ${completedGames} games, ${totalEpisodes} episodes, ` +
    `${totalStepsAll} total steps in ${totalElapsed}s`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

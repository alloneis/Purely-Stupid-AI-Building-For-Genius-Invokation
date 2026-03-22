/**
 * ArenaEngine — Pit two ONNX models against each other and measure win rate.
 *
 * Key design constraints:
 *   1. Dual-Batching: Model A and Model B use SEPARATE BatchedInferenceQueue
 *      instances. Requests are never mixed across sessions.
 *   2. MCTS Isolation: In quality mode, each player gets a distinct ISMCTSAgent
 *      with its own NeuralEvaluator. They never share MCTS trees.
 *   3. Side-Swapping: Half the games have A=P0/B=P1, the other half A=P1/B=P0
 *      to eliminate first-player advantage bias.
 */

import { ActionValidity, DiceType } from "@gi-tcg/typings";
import { flip } from "@gi-tcg/utils";
import { Game, type CreateInitialStateConfig } from "../../game";
import type { GameData } from "../../builder/registry";
import { RuleEngine } from "../rule_engine";
import { ISMCTSAgent } from "../ismcts_agent";
import type { GameAction, PureGameState, GameConfig, RuleEngineDecisionProvider } from "../types";
import { NeuralStateEncoder } from "./state_encoder";
import { ActionIndexer } from "./action_encoder";
import {
  BatchedInferenceQueue,
  NeuralEvaluator,
  type OnnxSession,
  type OnnxTensorFactory,
} from "./neural_evaluator";
import { MAX_ACTION_SLOTS } from "./constants";

// ─── Config & Result Types ─────────────────────────────────────────────

export interface ArenaConfig {
  gamesPerSide: number;
  temperature: number;
  maxRounds: number;
  maxActionsPerTurn: number;
  maxTotalSteps: number;
  mode: "fast" | "quality";
  mctsIterations: number;
  mctsDeterminizations: number;
  mctsMaxDepth: number;
}

const DEFAULT_ARENA_CONFIG: ArenaConfig = {
  gamesPerSide: 50,
  temperature: 0.5,
  maxRounds: 15,
  maxActionsPerTurn: 15,
  maxTotalSteps: 500,
  mode: "fast",
  mctsIterations: 50,
  mctsDeterminizations: 1,
  mctsMaxDepth: 6,
};

export interface ArenaGameStats {
  avgGameLength: number;
  avgSurvivingCharsWinner: number;
  avgTurnsToFirstKill: number;
  avgActionsPerTurn: number;
}

export interface ArenaResult {
  modelA: string;
  modelB: string;
  totalGames: number;
  winsA: number;
  winsB: number;
  draws: number;
  winRateA: number;
  winRateCI95: [number, number];
  eloDelta: number;
  isSignificant: boolean;
  gameStats: ArenaGameStats;
  perGame: ArenaGameRecord[];
}

export interface ArenaGameRecord {
  gameId: number;
  modelAPlayer: 0 | 1;
  winner: number | null;
  totalSteps: number;
  elapsedMs: number;
  turnsToFirstKill: number;
  survivingCharsWinner: number;
  actionsPerTurn: number;
}

// ─── Statistics Helpers ────────────────────────────────────────────────

function wilsonScoreCI(
  wins: number,
  total: number,
  z = 1.96,
): [number, number] {
  if (total === 0) return [0, 1];
  const p = wins / total;
  const z2 = z * z;
  const denom = 1 + z2 / total;
  const centre = p + z2 / (2 * total);
  const margin = z * Math.sqrt(p * (1 - p) / total + z2 / (4 * total * total));
  return [
    Math.max(0, (centre - margin) / denom),
    Math.min(1, (centre + margin) / denom),
  ];
}

function eloDelta(winRate: number): number {
  if (winRate <= 0.001) return -800;
  if (winRate >= 0.999) return 800;
  return -400 * Math.log10(1 / winRate - 1);
}

// ─── Game Helpers ──────────────────────────────────────────────────────

function totalTeamHp(state: PureGameState, who: 0 | 1): number {
  let total = 0;
  for (const c of state.gameState.players[who].characters) {
    const hp = typeof c.variables.health === "number" ? c.variables.health : 0;
    total += Math.max(0, hp);
  }
  return total;
}

function aliveCharCount(state: PureGameState, who: 0 | 1): number {
  let count = 0;
  for (const c of state.gameState.players[who].characters) {
    const hp = typeof c.variables.health === "number" ? c.variables.health : 0;
    if (hp > 0) count++;
  }
  return count;
}

function sampleFromPolicy(probs: Float32Array, temperature: number): number {
  if (temperature <= 0.01) {
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[best]) best = i;
    }
    return best;
  }

  if (Math.abs(temperature - 1.0) > 0.01) {
    const logits = new Float32Array(probs.length);
    let maxL = -Infinity;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > 0) {
        logits[i] = Math.log(probs[i] + 1e-10) / temperature;
        if (logits[i] > maxL) maxL = logits[i];
      } else {
        logits[i] = -100;
      }
    }
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
      logits[i] = Math.exp(logits[i] - maxL);
      sum += logits[i];
    }
    let r = Math.random() * sum;
    for (let i = 0; i < logits.length; i++) {
      r -= logits[i];
      if (r <= 0) return i;
    }
    return logits.length - 1;
  }

  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

// ─── ArenaEngine ───────────────────────────────────────────────────────

export class ArenaEngine {
  private readonly cfg: ArenaConfig;
  private readonly neuralA: NeuralEvaluator;
  private readonly neuralB: NeuralEvaluator;
  private readonly batchQueueA: BatchedInferenceQueue;
  private readonly batchQueueB: BatchedInferenceQueue;
  private readonly gameData: GameData;
  private readonly gameConfig: GameConfig;
  private readonly decisionProvider: RuleEngineDecisionProvider;
  private readonly indexer = new ActionIndexer();
  private readonly modelAName: string;
  private readonly modelBName: string;

  constructor(
    sessionA: OnnxSession,
    sessionB: OnnxSession,
    tfA: OnnxTensorFactory,
    tfB: OnnxTensorFactory,
    gameData: GameData,
    gameConfig: GameConfig,
    modelAName: string,
    modelBName: string,
    config?: Partial<ArenaConfig>,
    decisionProvider?: RuleEngineDecisionProvider,
  ) {
    this.cfg = { ...DEFAULT_ARENA_CONFIG, ...config };
    this.neuralA = new NeuralEvaluator({ session: sessionA, tensorFactory: tfA });
    this.neuralB = new NeuralEvaluator({ session: sessionB, tensorFactory: tfB });
    this.batchQueueA = new BatchedInferenceQueue(sessionA, tfA, 8, 1);
    this.batchQueueB = new BatchedInferenceQueue(sessionB, tfB, 8, 1);
    this.gameData = gameData;
    this.gameConfig = gameConfig;
    this.modelAName = modelAName;
    this.modelBName = modelBName;
    this.decisionProvider = decisionProvider ?? {
      rerollDice: (_s, _w, dice) => dice.filter((d: DiceType) => d !== DiceType.Omni),
    };
  }

  /**
   * Run the full arena: gamesPerSide with A=P0, gamesPerSide with A=P1.
   * Calls onProgress after each game completes.
   */
  async run(
    deckPairs: Array<[any, any]>,
    onProgress?: (completed: number, total: number, record: ArenaGameRecord) => void,
  ): Promise<ArenaResult> {
    const totalGames = this.cfg.gamesPerSide * 2;
    const records: ArenaGameRecord[] = [];

    for (let i = 0; i < this.cfg.gamesPerSide; i++) {
      const [deck1, deck2] = deckPairs[i % deckPairs.length];
      const record = await this.runOneGame(i, deck1, deck2, 0);
      records.push(record);
      onProgress?.(records.length, totalGames, record);
    }

    for (let i = 0; i < this.cfg.gamesPerSide; i++) {
      const [deck1, deck2] = deckPairs[i % deckPairs.length];
      const gameId = this.cfg.gamesPerSide + i;
      const record = await this.runOneGame(gameId, deck1, deck2, 1);
      records.push(record);
      onProgress?.(records.length, totalGames, record);
    }

    return this.computeResult(records, totalGames);
  }

  dispose(): void {
    this.batchQueueA.dispose();
    this.batchQueueB.dispose();
  }

  // ─── Single Game ───────────────────────────────────────────────

  private async runOneGame(
    gameId: number,
    deck1: any,
    deck2: any,
    modelAPlayer: 0 | 1,
  ): Promise<ArenaGameRecord> {
    const t0 = performance.now();
    let state = this.createInitialState(deck1, deck2);

    const neuralForPlayer: [NeuralEvaluator, NeuralEvaluator] = modelAPlayer === 0
      ? [this.neuralA, this.neuralB]
      : [this.neuralB, this.neuralA];
    const queueForPlayer: [BatchedInferenceQueue, BatchedInferenceQueue] = modelAPlayer === 0
      ? [this.batchQueueA, this.batchQueueB]
      : [this.batchQueueB, this.batchQueueA];

    let agents: [ISMCTSAgent, ISMCTSAgent] | null = null;
    if (this.cfg.mode === "quality") {
      agents = [
        new ISMCTSAgent({
          iterations: this.cfg.mctsIterations,
          determinizationCount: this.cfg.mctsDeterminizations,
          maxDepth: this.cfg.mctsMaxDepth,
          neuralEvaluator: neuralForPlayer[0],
        }),
        new ISMCTSAgent({
          iterations: this.cfg.mctsIterations,
          determinizationCount: this.cfg.mctsDeterminizations,
          maxDepth: this.cfg.mctsMaxDepth,
          neuralEvaluator: neuralForPlayer[1],
        }),
      ];
    }

    let totalSteps = 0;
    let actionsThisTurn = 0;
    let lastTurnPlayer = -1;
    let turnsToFirstKill = -1;
    let totalTurns = 0;
    let totalActions = 0;

    const initialChars = [
      state.gameState.players[0].characters.length,
      state.gameState.players[1].characters.length,
    ];

    while (!state.isFinished && totalSteps < this.cfg.maxTotalSteps) {
      if (state.gameState.roundNumber > this.cfg.maxRounds) break;

      const who = state.gameState.currentTurn;
      if (who !== lastTurnPlayer) {
        actionsThisTurn = 0;
        lastTurnPlayer = who;
        totalTurns++;
      }

      let legalActions: GameAction[];
      try {
        legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
          .filter((a) => a.validity === ActionValidity.VALID);
      } catch { break; }
      if (legalActions.length === 0) break;

      let chosenAction: GameAction;

      if (actionsThisTurn >= this.cfg.maxActionsPerTurn) {
        const endAction = legalActions.find((a) => a.action?.$case === "declareEnd");
        chosenAction = endAction ?? legalActions[0];
      } else if (this.cfg.mode === "quality" && agents) {
        try {
          const { action } = await agents[who].getBestActionNeuralAsync(
            state, legalActions, who,
          );
          chosenAction = action;
        } catch {
          chosenAction = legalActions[0];
        }
      } else {
        const neural = neuralForPlayer[who];
        const queue = queueForPlayer[who];
        const { encoded, actionMask } = neural.encodeForBatch(state, who);
        const inferResult = await queue.enqueue(encoded, actionMask);
        const probs = neural.inferResultToProbs(inferResult, legalActions, state, who);
        const chosenIdx = sampleFromPolicy(probs, this.cfg.temperature);
        const mapping = this.indexer.buildMapping(legalActions, state.gameState, who);
        chosenAction = mapping.indexToAction.get(chosenIdx) ?? legalActions[0];
      }

      try {
        state = this.executeAction(state, chosenAction);
      } catch { break; }

      if (turnsToFirstKill < 0) {
        for (const p of [0, 1] as const) {
          const alive = aliveCharCount(state, p);
          if (alive < initialChars[p]) {
            turnsToFirstKill = totalSteps;
            break;
          }
        }
      }

      actionsThisTurn++;
      totalActions++;
      totalSteps++;
    }

    const winner = state.gameState.winner;
    let winnerModelA: number | null = null;
    if (winner !== null) {
      winnerModelA = winner === modelAPlayer ? 0 : 1;
    }

    const survivingCharsWinner = winner !== null
      ? aliveCharCount(state, winner)
      : 0;

    return {
      gameId,
      modelAPlayer,
      winner: winnerModelA,
      totalSteps,
      elapsedMs: performance.now() - t0,
      turnsToFirstKill: turnsToFirstKill < 0 ? totalSteps : turnsToFirstKill,
      survivingCharsWinner,
      actionsPerTurn: totalTurns > 0 ? totalActions / totalTurns : 0,
    };
  }

  // ─── Result Computation ────────────────────────────────────────

  private computeResult(
    records: ArenaGameRecord[],
    totalGames: number,
  ): ArenaResult {
    let winsA = 0;
    let winsB = 0;
    let draws = 0;
    let totalLength = 0;
    let totalSurviving = 0;
    let totalFirstKill = 0;
    let totalActionsPerTurn = 0;
    let decisiveGames = 0;

    for (const r of records) {
      totalLength += r.totalSteps;
      totalFirstKill += r.turnsToFirstKill;
      totalActionsPerTurn += r.actionsPerTurn;

      if (r.winner === 0) {
        winsA++;
        totalSurviving += r.survivingCharsWinner;
        decisiveGames++;
      } else if (r.winner === 1) {
        winsB++;
        totalSurviving += r.survivingCharsWinner;
        decisiveGames++;
      } else {
        draws++;
      }
    }

    const decisive = winsA + winsB;
    const winRateA = decisive > 0 ? winsA / decisive : 0.5;
    const ci = wilsonScoreCI(winsA, decisive);
    const elo = eloDelta(winRateA);
    const isSignificant = ci[0] > 0.5;

    return {
      modelA: this.modelAName,
      modelB: this.modelBName,
      totalGames,
      winsA,
      winsB,
      draws,
      winRateA,
      winRateCI95: ci,
      eloDelta: elo,
      isSignificant,
      gameStats: {
        avgGameLength: totalGames > 0 ? totalLength / totalGames : 0,
        avgSurvivingCharsWinner: decisiveGames > 0 ? totalSurviving / decisiveGames : 0,
        avgTurnsToFirstKill: totalGames > 0 ? totalFirstKill / totalGames : 0,
        avgActionsPerTurn: totalGames > 0 ? totalActionsPerTurn / totalGames : 0,
      },
      perGame: records,
    };
  }

  // ─── Game Infrastructure ───────────────────────────────────────

  private createInitialState(deck1: any, deck2: any): PureGameState {
    const initialStateConfig: CreateInitialStateConfig = {
      decks: [deck1, deck2],
      data: this.gameData,
    };
    const gameState = RuleEngine.bootstrapToActionPhase(
      Game.createInitialState(initialStateConfig),
      { decisionProvider: this.decisionProvider },
    ).nextState;
    return {
      config: this.gameConfig,
      gameState,
      history: [],
      turn: 1,
      isFinished: false,
    };
  }

  private executeAction(state: PureGameState, action: GameAction): PureGameState {
    const { nextState } = RuleEngine.execute(state.gameState, action, {
      decisionProvider: this.decisionProvider,
    });
    return {
      ...state,
      history: [],
      gameState: nextState,
      turn: state.turn + 1,
      isFinished: nextState.phase === "gameEnd",
      winner: nextState.winner ?? undefined,
    };
  }
}

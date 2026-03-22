/**
 * NeuralSelfPlayEngine — High-throughput self-play for neural data generation.
 *
 * Two modes:
 *   "fast"    — ONNX policy sampling only, no MCTS. ~0.5s/game single, batched even faster.
 *   "quality" — ONNX + light IS-MCTS for refined policy. ~10s/game.
 *
 * Uses BatchedInferenceQueue with pre-allocated memory pools (zero GC pressure
 * in the hot loop). Multiple concurrent games share a single batched ONNX call.
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
  type InferResult,
  type OnnxSession,
  type OnnxTensorFactory,
} from "./neural_evaluator";
import { MAX_ACTION_SLOTS, CARD_FEATURE_DIM } from "./constants";

// ─── Config ────────────────────────────────────────────────────────────

export interface SelfPlayConfig {
  mode: "fast" | "quality";
  batchSize: number;
  temperature: number;
  maxRounds: number;
  maxActionsPerTurn: number;
  maxTotalSteps: number;
  mctsIterations: number;
  mctsDeterminizations: number;
  mctsMaxDepth: number;
}

const DEFAULT_CONFIG: SelfPlayConfig = {
  mode: "fast",
  batchSize: 8,
  temperature: 1.0,
  maxRounds: 15,
  maxActionsPerTurn: 15,
  maxTotalSteps: 500,
  mctsIterations: 50,
  mctsDeterminizations: 1,
  mctsMaxDepth: 6,
};

// ─── Step Record (matches train.py data contract) ──────────────────────

export interface StepRecord {
  global_features: number[];
  self_characters: number[];
  oppo_characters: number[];
  hand_cards: number[];
  hand_mask: number[];
  summons: number[];
  summons_mask: number[];
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

export interface EpisodeResult {
  gameId: number;
  episodes: StepRecord[][];
  winner: number | null;
  totalSteps: number;
  elapsedMs: number;
}

// ─── Helpers ───────────────────────────────────────────────────────────

function f32ToArray(f: Float32Array): number[] {
  return Array.from(f);
}

function totalTeamHp(state: PureGameState, who: 0 | 1): number {
  let total = 0;
  for (const c of state.gameState.players[who].characters) {
    const hp = typeof c.variables.health === "number" ? c.variables.health : 0;
    total += Math.max(0, hp);
  }
  return total;
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

function getActiveCharHp(state: PureGameState, perspective: 0 | 1): number {
  const player = state.gameState.players[perspective];
  const active = player.characters.find((c) => c.id === player.activeCharacterId);
  if (!active) return 0;
  const hp = typeof active.variables.health === "number" ? active.variables.health : 0;
  return hp / 10;
}

function computeOppoHandFeatures(state: PureGameState, perspective: 0 | 1): number[] {
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

// ─── Engine ────────────────────────────────────────────────────────────

export class NeuralSelfPlayEngine {
  private readonly cfg: SelfPlayConfig;
  private readonly neural: NeuralEvaluator;
  private readonly batchQueue: BatchedInferenceQueue;
  private readonly gameData: GameData;
  private readonly gameConfig: GameConfig;
  private readonly decisionProvider: RuleEngineDecisionProvider;
  private readonly encoder = new NeuralStateEncoder();
  private readonly indexer = new ActionIndexer();

  constructor(
    session: OnnxSession,
    tf: OnnxTensorFactory,
    gameData: GameData,
    gameConfig: GameConfig,
    config?: Partial<SelfPlayConfig>,
    decisionProvider?: RuleEngineDecisionProvider,
  ) {
    this.cfg = { ...DEFAULT_CONFIG, ...config };
    this.neural = new NeuralEvaluator({ session, tensorFactory: tf });
    this.batchQueue = new BatchedInferenceQueue(
      session, tf, this.cfg.batchSize, 1,
    );
    this.gameData = gameData;
    this.gameConfig = gameConfig;
    this.decisionProvider = decisionProvider ?? {
      rerollDice: (_s, _w, dice) => dice.filter((d: DiceType) => d !== DiceType.Omni),
    };
  }

  /**
   * Run a batch of games concurrently. All games that need an inference
   * at the same time share a single batched ONNX call.
   */
  async runBatch(
    gameIds: number[],
    deckPairs: Array<[string, string]>,
    decodeDeck: (code: string) => any,
  ): Promise<EpisodeResult[]> {
    const promises = gameIds.map((id, i) => {
      const [d1, d2] = deckPairs[i % deckPairs.length];
      return this.cfg.mode === "fast"
        ? this.runFastGame(id, decodeDeck(d1), decodeDeck(d2))
        : this.runQualityGame(id, decodeDeck(d1), decodeDeck(d2));
    });
    return Promise.all(promises);
  }

  dispose(): void {
    this.batchQueue.dispose();
  }

  // ─── Fast Mode ───────────────────────────────────────────────────

  private async runFastGame(
    gameId: number,
    deck1: any,
    deck2: any,
  ): Promise<EpisodeResult> {
    const t0 = performance.now();
    let state = this.createInitialState(deck1, deck2);
    const episodes: StepRecord[][] = [[], []];
    let totalSteps = 0;
    let actionsThisTurn = 0;
    let lastTurnPlayer = -1;

    while (!state.isFinished && totalSteps < this.cfg.maxTotalSteps) {
      if (state.gameState.roundNumber > this.cfg.maxRounds) break;

      const who = state.gameState.currentTurn;
      if (who !== lastTurnPlayer) {
        actionsThisTurn = 0;
        lastTurnPlayer = who;
      }

      let legalActions: GameAction[];
      try {
        legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
          .filter((a) => a.validity === ActionValidity.VALID);
      } catch { break; }
      if (legalActions.length === 0) break;

      let chosenAction: GameAction;
      let policyArray: number[];

      if (actionsThisTurn >= this.cfg.maxActionsPerTurn) {
        const endAction = legalActions.find((a) => a.action?.$case === "declareEnd");
        chosenAction = endAction ?? legalActions[0];
        policyArray = new Array(MAX_ACTION_SLOTS).fill(0);
      } else {
        const { encoded, actionMask } = this.neural.encodeForBatch(state, who);
        const inferResult = await this.batchQueue.enqueue(encoded, actionMask);
        const probs = this.neural.inferResultToProbs(inferResult, legalActions, state, who);

        policyArray = f32ToArray(probs);
        const chosenIdx = sampleFromPolicy(probs, this.cfg.temperature);
        const mapping = this.indexer.buildMapping(legalActions, state.gameState, who);
        chosenAction = mapping.indexToAction.get(chosenIdx) ?? legalActions[0];
      }

      const enc = this.encoder.encode(state, who);
      const mask = this.indexer.buildMask(legalActions, state.gameState, who);
      episodes[who].push(this.buildStep(enc, mask, policyArray, state, who));

      try {
        state = this.executeAction(state, chosenAction);
      } catch { break; }

      actionsThisTurn++;
      totalSteps++;
    }

    this.assignRewards(episodes, state);
    return {
      gameId,
      episodes,
      winner: state.gameState.winner,
      totalSteps,
      elapsedMs: performance.now() - t0,
    };
  }

  // ─── Quality Mode ────────────────────────────────────────────────

  private async runQualityGame(
    gameId: number,
    deck1: any,
    deck2: any,
  ): Promise<EpisodeResult> {
    const t0 = performance.now();
    let state = this.createInitialState(deck1, deck2);

    const agents = [
      new ISMCTSAgent({
        iterations: this.cfg.mctsIterations,
        determinizationCount: this.cfg.mctsDeterminizations,
        maxDepth: this.cfg.mctsMaxDepth,
        neuralEvaluator: this.neural,
      }),
      new ISMCTSAgent({
        iterations: this.cfg.mctsIterations,
        determinizationCount: this.cfg.mctsDeterminizations,
        maxDepth: this.cfg.mctsMaxDepth,
        neuralEvaluator: this.neural,
      }),
    ];

    const episodes: StepRecord[][] = [[], []];
    let totalSteps = 0;
    let actionsThisTurn = 0;
    let lastTurnPlayer = -1;

    while (!state.isFinished && totalSteps < this.cfg.maxTotalSteps) {
      if (state.gameState.roundNumber > this.cfg.maxRounds) break;

      const who = state.gameState.currentTurn;
      if (who !== lastTurnPlayer) {
        actionsThisTurn = 0;
        lastTurnPlayer = who;
      }

      let legalActions: GameAction[];
      try {
        legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
          .filter((a) => a.validity === ActionValidity.VALID);
      } catch { break; }
      if (legalActions.length === 0) break;

      let chosenAction: GameAction;
      let policyArray: number[];

      if (actionsThisTurn >= this.cfg.maxActionsPerTurn) {
        const endAction = legalActions.find((a) => a.action?.$case === "declareEnd");
        chosenAction = endAction ?? legalActions[0];
        policyArray = new Array(MAX_ACTION_SLOTS).fill(0);
      } else {
        try {
          const { action, debug } = await agents[who].getBestActionNeuralAsync(
            state, legalActions, who,
          );
          chosenAction = action;
          policyArray = this.buildMctsPolicy(debug, legalActions, state, who);
        } catch {
          chosenAction = legalActions[0];
          policyArray = new Array(MAX_ACTION_SLOTS).fill(0);
        }
      }

      const enc = this.encoder.encode(state, who);
      const mask = this.indexer.buildMask(legalActions, state.gameState, who);
      episodes[who].push(this.buildStep(enc, mask, policyArray, state, who));

      try {
        state = this.executeAction(state, chosenAction);
      } catch { break; }

      actionsThisTurn++;
      totalSteps++;
    }

    this.assignRewards(episodes, state);
    return {
      gameId,
      episodes,
      winner: state.gameState.winner,
      totalSteps,
      elapsedMs: performance.now() - t0,
    };
  }

  // ─── Shared Internals ────────────────────────────────────────────

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

  private buildStep(
    enc: ReturnType<NeuralStateEncoder["encode"]>,
    actionMask: Float32Array,
    policyArray: number[],
    state: PureGameState,
    who: 0 | 1,
  ): StepRecord {
    return {
      global_features: f32ToArray(enc.global_features),
      self_characters: f32ToArray(enc.self_characters),
      oppo_characters: f32ToArray(enc.oppo_characters),
      hand_cards: f32ToArray(enc.hand_cards),
      hand_mask: f32ToArray(enc.hand_cards_mask),
      summons: f32ToArray(enc.self_summons),
      summons_mask: f32ToArray(enc.self_summons_mask),
      action_mask: f32ToArray(actionMask),
      mcts_policy: policyArray,
      reward: 0,
      is_terminal: false,
      hp_after_5_turns: 0,
      cards_playable_next: new Array(10).fill(0),
      oppo_hand_features: computeOppoHandFeatures(state, who),
      kill_within_3: new Array(6).fill(0),
      reaction_next_attack: 0,
      dice_effective_actions: countEffectiveActions(state, who),
    };
  }

  private assignRewards(episodes: StepRecord[][], state: PureGameState): void {
    for (const who of [0, 1] as const) {
      const reward = computeReward(state, who);
      const ep = episodes[who];
      if (ep.length > 0) {
        ep[ep.length - 1].reward = reward;
        ep[ep.length - 1].is_terminal = true;
      }
      for (const step of ep) {
        step.hp_after_5_turns = getActiveCharHp(state, who);
      }
    }
  }

  private buildMctsPolicy(
    debug: { candidates: Array<{ action: GameAction; visits: number }> },
    legalActions: readonly GameAction[],
    state: PureGameState,
    perspective: 0 | 1,
  ): number[] {
    const policy = new Float32Array(MAX_ACTION_SLOTS);
    let totalVisits = 0;
    for (const c of debug.candidates) totalVisits += c.visits;
    if (totalVisits === 0) {
      const mask = this.indexer.buildMask(legalActions, state.gameState, perspective);
      const count = mask.reduce((s, v) => s + v, 0);
      if (count > 0) {
        for (let i = 0; i < mask.length; i++) policy[i] = mask[i] / count;
      }
      return f32ToArray(policy);
    }
    for (const c of debug.candidates) {
      const idx = this.indexer.actionToIndex(c.action, state.gameState, perspective);
      if (idx >= 0 && idx < MAX_ACTION_SLOTS) {
        policy[idx] = c.visits / totalVisits;
      }
    }
    return f32ToArray(policy);
  }
}

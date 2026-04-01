/**
 * NeuralEvaluator — Drop-in replacement for DynamicHeuristicEvaluator.
 *
 * Wraps an ONNX inference session and provides:
 *   - evaluate(state, perspective) → scalar value  [-1, 1]
 *   - getPolicy(state, legalActions, perspective) → Float32Array probabilities
 *   - BatchedInferenceQueue for GC-free, zero-alloc batched ONNX calls
 *
 * The inference session is injected via the OnnxSession interface so the same
 * class works in Node (onnxruntime-node), Web Worker (onnxruntime-web), or tests.
 */

import { ActionValidity } from "@gi-tcg/typings";
import type { GameAction, PureGameState } from "../types";
import { RuleEngine } from "../rule_engine";
import { NeuralStateEncoder, type EncodedState } from "./state_encoder";
import { ActionIndexer } from "./action_encoder";
import {
  MAX_ACTION_SLOTS,
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
} from "./constants";

// ─── Portable ONNX Session Interface ───────────────────────────────────

/**
 * Minimal interface matching onnxruntime-web's InferenceSession.
 * Callers provide the concrete implementation.
 */
export interface OnnxSession {
  run(
    feeds: Record<string, OnnxTensor>,
    outputNames?: readonly string[],
  ): Promise<Record<string, OnnxTensor>>;
}

export interface OnnxTensor {
  readonly data: Float32Array | Int32Array | BigInt64Array;
  readonly dims: readonly number[];
}

export interface OnnxTensorFactory {
  create(data: Float32Array, dims: readonly number[]): OnnxTensor;
}

// ─── Config ────────────────────────────────────────────────────────────

export interface NeuralEvaluatorOptions {
  session: OnnxSession;
  tensorFactory: OnnxTensorFactory;
  /** Terminal state value (returned when game is over). Default: 10_000 */
  terminalValue?: number;
  /** Policy temperature for softmax conversion. Default: 1.0 */
  policyTemperature?: number;
}

// ─── ONNX Output Names (must match model.py _OUTPUT_NAMES) ────────────

const OUTPUT_VALUE = "value";
const OUTPUT_LOG_POLICY = "log_policy";

const NEEDED_OUTPUTS = [OUTPUT_VALUE, OUTPUT_LOG_POLICY] as const;

// ─── Pre-computed per-sample sizes for each ONNX input ─────────────────

const CHAR_FLAT = MAX_CHARACTERS * CHARACTER_FEATURE_DIM;
const HAND_FLAT = MAX_HAND_CARDS * CARD_FEATURE_DIM;
const SUMMON_FLAT = MAX_SUMMONS * ENTITY_FEATURE_DIM;
const SUPPORT_FLAT = MAX_SUPPORTS * ENTITY_FEATURE_DIM;
const COMBAT_STATUS_FLAT = MAX_COMBAT_STATUSES * ENTITY_FEATURE_DIM;
const CHAR_ENT_FLAT = MAX_CHARACTERS * MAX_CHARACTER_ENTITIES * ENTITY_FEATURE_DIM;
const CHAR_ENT_MASK_FLAT = MAX_CHARACTERS * MAX_CHARACTER_ENTITIES;

interface InputSizes {
  global_features: number;
  self_characters: number;
  oppo_characters: number;
  hand_cards: number;
  hand_mask: number;
  summons: number;
  summons_mask: number;
  self_supports: number;
  self_supports_mask: number;
  oppo_supports: number;
  oppo_supports_mask: number;
  self_combat_statuses: number;
  self_combat_statuses_mask: number;
  oppo_combat_statuses: number;
  oppo_combat_statuses_mask: number;
  self_char_entities: number;
  self_char_entities_mask: number;
  oppo_char_entities: number;
  oppo_char_entities_mask: number;
  action_mask: number;
}

const SAMPLE_SIZES: InputSizes = {
  global_features: GLOBAL_FEATURE_DIM,
  self_characters: CHAR_FLAT,
  oppo_characters: CHAR_FLAT,
  hand_cards: HAND_FLAT,
  hand_mask: MAX_HAND_CARDS,
  summons: SUMMON_FLAT,
  summons_mask: MAX_SUMMONS,
  self_supports: SUPPORT_FLAT,
  self_supports_mask: MAX_SUPPORTS,
  oppo_supports: SUPPORT_FLAT,
  oppo_supports_mask: MAX_SUPPORTS,
  self_combat_statuses: COMBAT_STATUS_FLAT,
  self_combat_statuses_mask: MAX_COMBAT_STATUSES,
  oppo_combat_statuses: COMBAT_STATUS_FLAT,
  oppo_combat_statuses_mask: MAX_COMBAT_STATUSES,
  self_char_entities: CHAR_ENT_FLAT,
  self_char_entities_mask: CHAR_ENT_MASK_FLAT,
  oppo_char_entities: CHAR_ENT_FLAT,
  oppo_char_entities_mask: CHAR_ENT_MASK_FLAT,
  action_mask: MAX_ACTION_SLOTS,
};

const INPUT_NAMES = Object.keys(SAMPLE_SIZES) as (keyof InputSizes)[];

// ─── BatchedInferenceQueue ─────────────────────────────────────────────

export interface InferResult {
  value: number;
  logPolicy: Float32Array;
}

interface PendingRequest {
  resolve: (result: InferResult) => void;
  reject: (error: Error) => void;
}

/**
 * Coalesces multiple inference requests into a single batched ONNX call.
 *
 * Memory pools are pre-allocated once at construction for the max batch size.
 * Each enqueue() call copies encoded data into the pool via .set() at the
 * correct offset — zero allocations in the hot loop.
 */
export class BatchedInferenceQueue {
  private readonly maxBatch: number;
  private readonly flushIntervalMs: number;
  private readonly session: OnnxSession;
  private readonly tf: OnnxTensorFactory;

  private readonly pools: Record<keyof InputSizes, Float32Array>;
  private readonly pending: PendingRequest[] = [];
  private count = 0;
  private flushTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(
    session: OnnxSession,
    tf: OnnxTensorFactory,
    maxBatch = 16,
    flushIntervalMs = 1,
  ) {
    this.session = session;
    this.tf = tf;
    this.maxBatch = maxBatch;
    this.flushIntervalMs = flushIntervalMs;

    this.pools = {} as Record<keyof InputSizes, Float32Array>;
    for (const key of INPUT_NAMES) {
      this.pools[key] = new Float32Array(maxBatch * SAMPLE_SIZES[key]);
    }
  }

  /**
   * Enqueue one inference request. Data is copied into the pre-allocated pool.
   * Returns a Promise that resolves when the batched ONNX call completes.
   */
  enqueue(
    encoded: EncodedState,
    actionMask: Float32Array,
  ): Promise<InferResult> {
    return new Promise<InferResult>((resolve, reject) => {
      const idx = this.count;

      this.pools.global_features.set(encoded.global_features, idx * SAMPLE_SIZES.global_features);
      this.pools.self_characters.set(encoded.self_characters, idx * SAMPLE_SIZES.self_characters);
      this.pools.oppo_characters.set(encoded.oppo_characters, idx * SAMPLE_SIZES.oppo_characters);
      this.pools.hand_cards.set(encoded.hand_cards, idx * SAMPLE_SIZES.hand_cards);
      this.pools.hand_mask.set(encoded.hand_cards_mask, idx * SAMPLE_SIZES.hand_mask);
      this.pools.summons.set(encoded.self_summons, idx * SAMPLE_SIZES.summons);
      this.pools.summons_mask.set(encoded.self_summons_mask, idx * SAMPLE_SIZES.summons_mask);
      this.pools.self_supports.set(encoded.self_supports, idx * SAMPLE_SIZES.self_supports);
      this.pools.self_supports_mask.set(encoded.self_supports_mask, idx * SAMPLE_SIZES.self_supports_mask);
      this.pools.oppo_supports.set(encoded.oppo_supports, idx * SAMPLE_SIZES.oppo_supports);
      this.pools.oppo_supports_mask.set(encoded.oppo_supports_mask, idx * SAMPLE_SIZES.oppo_supports_mask);
      this.pools.self_combat_statuses.set(encoded.self_combat_statuses, idx * SAMPLE_SIZES.self_combat_statuses);
      this.pools.self_combat_statuses_mask.set(encoded.self_combat_statuses_mask, idx * SAMPLE_SIZES.self_combat_statuses_mask);
      this.pools.oppo_combat_statuses.set(encoded.oppo_combat_statuses, idx * SAMPLE_SIZES.oppo_combat_statuses);
      this.pools.oppo_combat_statuses_mask.set(encoded.oppo_combat_statuses_mask, idx * SAMPLE_SIZES.oppo_combat_statuses_mask);
      this.pools.self_char_entities.set(encoded.self_char_entities, idx * SAMPLE_SIZES.self_char_entities);
      this.pools.self_char_entities_mask.set(encoded.self_char_entities_mask, idx * SAMPLE_SIZES.self_char_entities_mask);
      this.pools.oppo_char_entities.set(encoded.oppo_char_entities, idx * SAMPLE_SIZES.oppo_char_entities);
      this.pools.oppo_char_entities_mask.set(encoded.oppo_char_entities_mask, idx * SAMPLE_SIZES.oppo_char_entities_mask);
      this.pools.action_mask.set(actionMask, idx * SAMPLE_SIZES.action_mask);

      this.pending.push({ resolve, reject });
      this.count++;

      if (this.count >= this.maxBatch) {
        this.flush();
      } else if (!this.flushTimer) {
        this.flushTimer = setTimeout(() => this.flush(), this.flushIntervalMs);
      }
    });
  }

  /** Force-flush all pending requests into one ONNX call. */
  flush(): void {
    if (this.count === 0) return;
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }

    const N = this.count;
    const batch = this.pending.splice(0, N);
    this.count = 0;

    const feeds: Record<string, OnnxTensor> = {};
    for (const key of INPUT_NAMES) {
      const perSample = SAMPLE_SIZES[key];
      const slice = this.pools[key].subarray(0, N * perSample);
      const dims = this.dimsForKey(key, N);
      feeds[key] = this.tf.create(slice, dims);
    }

    this.session.run(feeds, NEEDED_OUTPUTS).then(
      (outputs) => {
        const values = outputs[OUTPUT_VALUE].data as Float32Array;
        const logPolicies = outputs[OUTPUT_LOG_POLICY].data as Float32Array;
        for (let i = 0; i < batch.length; i++) {
          batch[i].resolve({
            value: values[i],
            logPolicy: logPolicies.subarray(
              i * MAX_ACTION_SLOTS,
              (i + 1) * MAX_ACTION_SLOTS,
            ),
          });
        }
      },
      (err) => {
        const error = err instanceof Error ? err : new Error(String(err));
        for (const req of batch) req.reject(error);
      },
    );
  }

  private dimsForKey(key: keyof InputSizes, N: number): readonly number[] {
    switch (key) {
      case "global_features":
        return [N, GLOBAL_FEATURE_DIM];
      case "self_characters":
      case "oppo_characters":
        return [N, MAX_CHARACTERS, CHARACTER_FEATURE_DIM];
      case "hand_cards":
        return [N, MAX_HAND_CARDS, CARD_FEATURE_DIM];
      case "hand_mask":
        return [N, MAX_HAND_CARDS];
      case "summons":
        return [N, MAX_SUMMONS, ENTITY_FEATURE_DIM];
      case "summons_mask":
        return [N, MAX_SUMMONS];
      case "self_supports":
      case "oppo_supports":
        return [N, MAX_SUPPORTS, ENTITY_FEATURE_DIM];
      case "self_supports_mask":
      case "oppo_supports_mask":
        return [N, MAX_SUPPORTS];
      case "self_combat_statuses":
      case "oppo_combat_statuses":
        return [N, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM];
      case "self_combat_statuses_mask":
      case "oppo_combat_statuses_mask":
        return [N, MAX_COMBAT_STATUSES];
      case "self_char_entities":
      case "oppo_char_entities":
        return [N, MAX_CHARACTERS * MAX_CHARACTER_ENTITIES, ENTITY_FEATURE_DIM];
      case "self_char_entities_mask":
      case "oppo_char_entities_mask":
        return [N, MAX_CHARACTERS * MAX_CHARACTER_ENTITIES];
      case "action_mask":
        return [N, MAX_ACTION_SLOTS];
    }
  }

  dispose(): void {
    if (this.flushTimer) {
      clearTimeout(this.flushTimer);
      this.flushTimer = null;
    }
  }
}

// ─── NeuralEvaluator ───────────────────────────────────────────────────

export class NeuralEvaluator {
  readonly session: OnnxSession;
  readonly tf: OnnxTensorFactory;
  private readonly encoder = new NeuralStateEncoder();
  private readonly indexer = new ActionIndexer();
  private readonly terminalValue: number;
  private readonly policyTemp: number;

  constructor(options: NeuralEvaluatorOptions) {
    this.session = options.session;
    this.tf = options.tensorFactory;
    this.terminalValue = options.terminalValue ?? 10_000;
    this.policyTemp = options.policyTemperature ?? 1.0;
  }

  /** Encode a state into raw tensors (for use with BatchedInferenceQueue). */
  encodeForBatch(
    state: PureGameState,
    perspective: 0 | 1,
  ): { encoded: EncodedState; actionMask: Float32Array } {
    const encoded = this.encoder.encode(state, perspective);
    const legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
      .filter((a) => a.validity === ActionValidity.VALID);
    const actionMask = this.indexer.buildMask(legalActions, state.gameState, perspective);
    return { encoded, actionMask };
  }

  /**
   * Evaluate a game state from the given perspective.
   * Returns a scalar value in [-1, 1] (from the neural net),
   * or +/-terminalValue for terminal states.
   *
   * Compatible with DynamicHeuristicEvaluator.evaluate() signature.
   */
  evaluate(
    state: PureGameState,
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): number {
    const winner = state.gameState.winner;
    if (state.isFinished || state.gameState.phase === "gameEnd" || winner !== null) {
      if (winner === null) return 0;
      return winner === perspective ? this.terminalValue : -this.terminalValue;
    }
    throw new Error(
      "NeuralEvaluator.evaluate() is synchronous but ONNX inference is async. " +
      "Use evaluateAsync() instead.",
    );
  }

  async evaluateAsync(
    state: PureGameState,
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): Promise<number> {
    const winner = state.gameState.winner;
    if (state.isFinished || state.gameState.phase === "gameEnd" || winner !== null) {
      if (winner === null) return 0;
      return winner === perspective ? this.terminalValue : -this.terminalValue;
    }
    const result = await this.infer(state, perspective);
    return result.value;
  }

  async getPolicy(
    state: PureGameState,
    legalActions: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): Promise<Float32Array> {
    const result = await this.infer(state, perspective);
    return this.logPolicyToProbs(result.logPolicy, legalActions, state, perspective);
  }

  async evaluateWithPolicy(
    state: PureGameState,
    legalActions: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): Promise<{ value: number; policy: Float32Array }> {
    const winner = state.gameState.winner;
    if (state.isFinished || state.gameState.phase === "gameEnd" || winner !== null) {
      const termValue = winner === null ? 0 : winner === perspective ? this.terminalValue : -this.terminalValue;
      return { value: termValue, policy: new Float32Array(MAX_ACTION_SLOTS) };
    }
    const result = await this.infer(state, perspective);
    const policy = this.logPolicyToProbs(result.logPolicy, legalActions, state, perspective);
    return { value: result.value, policy };
  }

  /** Convert raw InferResult (from batch queue) into masked probabilities. */
  inferResultToProbs(
    result: InferResult,
    legalActions: readonly GameAction[],
    state: PureGameState,
    perspective: 0 | 1,
  ): Float32Array {
    return this.logPolicyToProbs(result.logPolicy, legalActions, state, perspective);
  }

  // ─── Internal ──────────────────────────────────────────────────────

  private logPolicyToProbs(
    logPolicy: Float32Array,
    legalActions: readonly GameAction[],
    state: PureGameState,
    perspective: 0 | 1,
  ): Float32Array {
    const { mask } = this.indexer.buildMapping(legalActions, state.gameState, perspective);
    const probs = new Float32Array(MAX_ACTION_SLOTS);
    let maxLogit = -Infinity;
    for (let i = 0; i < MAX_ACTION_SLOTS; i++) {
      if (mask[i] > 0) {
        const logit = logPolicy[i] / this.policyTemp;
        if (logit > maxLogit) maxLogit = logit;
      }
    }
    let sumExp = 0;
    for (let i = 0; i < MAX_ACTION_SLOTS; i++) {
      if (mask[i] > 0) {
        const e = Math.exp(Math.max(-30, Math.min(30, logPolicy[i] / this.policyTemp - maxLogit)));
        probs[i] = e;
        sumExp += e;
      }
    }
    if (sumExp > 0) {
      for (let i = 0; i < MAX_ACTION_SLOTS; i++) probs[i] /= sumExp;
    }
    return probs;
  }

  private async infer(
    state: PureGameState,
    perspective: 0 | 1,
  ): Promise<InferResult> {
    const encoded = this.encoder.encode(state, perspective);
    const legalActions = RuleEngine.getPossibleActions(state.gameState, { fastMode: true })
      .filter((a) => a.validity === ActionValidity.VALID);
    const actionMask = this.indexer.buildMask(legalActions, state.gameState, perspective);

    const CHAR_ENT_SLOTS = MAX_CHARACTERS * MAX_CHARACTER_ENTITIES;
    const feeds: Record<string, OnnxTensor> = {
      global_features: this.tf.create(encoded.global_features, [1, GLOBAL_FEATURE_DIM]),
      self_characters: this.tf.create(encoded.self_characters, [1, MAX_CHARACTERS, CHARACTER_FEATURE_DIM]),
      oppo_characters: this.tf.create(encoded.oppo_characters, [1, MAX_CHARACTERS, CHARACTER_FEATURE_DIM]),
      hand_cards: this.tf.create(encoded.hand_cards, [1, MAX_HAND_CARDS, CARD_FEATURE_DIM]),
      hand_mask: this.tf.create(encoded.hand_cards_mask, [1, MAX_HAND_CARDS]),
      summons: this.tf.create(encoded.self_summons, [1, MAX_SUMMONS, ENTITY_FEATURE_DIM]),
      summons_mask: this.tf.create(encoded.self_summons_mask, [1, MAX_SUMMONS]),
      self_supports: this.tf.create(encoded.self_supports, [1, MAX_SUPPORTS, ENTITY_FEATURE_DIM]),
      self_supports_mask: this.tf.create(encoded.self_supports_mask, [1, MAX_SUPPORTS]),
      oppo_supports: this.tf.create(encoded.oppo_supports, [1, MAX_SUPPORTS, ENTITY_FEATURE_DIM]),
      oppo_supports_mask: this.tf.create(encoded.oppo_supports_mask, [1, MAX_SUPPORTS]),
      self_combat_statuses: this.tf.create(encoded.self_combat_statuses, [1, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM]),
      self_combat_statuses_mask: this.tf.create(encoded.self_combat_statuses_mask, [1, MAX_COMBAT_STATUSES]),
      oppo_combat_statuses: this.tf.create(encoded.oppo_combat_statuses, [1, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM]),
      oppo_combat_statuses_mask: this.tf.create(encoded.oppo_combat_statuses_mask, [1, MAX_COMBAT_STATUSES]),
      self_char_entities: this.tf.create(encoded.self_char_entities, [1, CHAR_ENT_SLOTS, ENTITY_FEATURE_DIM]),
      self_char_entities_mask: this.tf.create(encoded.self_char_entities_mask, [1, CHAR_ENT_SLOTS]),
      oppo_char_entities: this.tf.create(encoded.oppo_char_entities, [1, CHAR_ENT_SLOTS, ENTITY_FEATURE_DIM]),
      oppo_char_entities_mask: this.tf.create(encoded.oppo_char_entities_mask, [1, CHAR_ENT_SLOTS]),
      action_mask: this.tf.create(actionMask, [1, MAX_ACTION_SLOTS]),
    };

    const outputs = await this.session.run(feeds, NEEDED_OUTPUTS);
    return {
      value: (outputs[OUTPUT_VALUE].data as Float32Array)[0],
      logPolicy: outputs[OUTPUT_LOG_POLICY].data as Float32Array,
    };
  }
}

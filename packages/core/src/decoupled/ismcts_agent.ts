import { ActionValidity, DiceType } from "@gi-tcg/typings";
import { flip } from "@gi-tcg/utils";
import type { GameState } from "../base/state";
import { elementOfCharacter } from "../utils";
import { ActionPruner, type ActionPrunerOptions } from "./action_pruner";
import {
  DynamicHeuristicEvaluator,
  type DynamicHeuristicEvaluatorOptions,
} from "./dynamic_heuristic_evaluator";
import type { NeuralEvaluator } from "./neural/neural_evaluator";
import { ActionIndexer } from "./neural/action_encoder";
import { RuleEngine } from "./rule_engine";
import type { GameAction, PureGameState } from "./types";

export interface ISMCTSAgentOptions {
  iterations?: number;
  determinizationCount?: number;
  maxDepth?: number;
  exploration?: number;
  minimaxBlend?: number;
  rng?: () => number;
  evaluator?: DynamicHeuristicEvaluator;
  evaluatorOptions?: DynamicHeuristicEvaluatorOptions;
  prunerOptions?: ActionPrunerOptions;
  /** When provided, uses neural net for root policy priors (PUCT) and leaf evaluation. */
  neuralEvaluator?: NeuralEvaluator;
}

export interface ISMCTSActionStat {
  action: GameAction;
  visits: number;
  meanValue: number;
}

export interface ISMCTSDecisionDebug {
  selected: ISMCTSActionStat;
  candidates: ISMCTSActionStat[];
  top3: ISMCTSActionStat[];
  determinizationCount: number;
  simulationsPerDeterminization: number;
  processLogs: string[];
}

interface NodeStats {
  action: GameAction;
  visits: number;
  valueSum: number;
}

interface SearchNode {
  parent: SearchNode | null;
  actionFromParent: GameAction | null;
  children: SearchNode[];
  visits: number;
  valueSum: number;
  minimaxValue: number;
}

interface SelectionResult {
  child: SearchNode;
  nextState: PureGameState;
}

interface TrajectoryEntry {
  node: SearchNode;
  playerToAct: 0 | 1 | null;
}

const DEFAULT_OPTIONS: Required<
  Pick<
    ISMCTSAgentOptions,
    "iterations" | "determinizationCount" | "maxDepth" | "exploration" | "minimaxBlend"
  >
> = {
  iterations: 420,
  determinizationCount: 5,
  maxDepth: 12,
  exploration: Math.SQRT2,
  minimaxBlend: 0.35,
};

const DETERMINIZATION_DICE_POOL: readonly DiceType[] = [
  DiceType.Cryo,
  DiceType.Hydro,
  DiceType.Pyro,
  DiceType.Electro,
  DiceType.Anemo,
  DiceType.Geo,
  DiceType.Dendro,
];

function isTerminalState(state: PureGameState): boolean {
  return (
    state.isFinished ||
    state.gameState.phase === "gameEnd" ||
    state.gameState.winner !== null
  );
}

function currentPlayerOf(state: PureGameState): 0 | 1 | null {
  return isTerminalState(state) ? null : state.gameState.currentTurn;
}

export class ISMCTSAgent {
  private readonly iterations: number;
  private readonly determinizationCount: number;
  private readonly maxDepth: number;
  private readonly exploration: number;
  private readonly minimaxBlend: number;
  private readonly rng: () => number;
  private readonly evaluator: DynamicHeuristicEvaluator;
  private readonly pruner: ActionPruner;
  private readonly neural: NeuralEvaluator | null;

  constructor(options: ISMCTSAgentOptions = {}) {
    this.iterations = options.iterations ?? DEFAULT_OPTIONS.iterations;
    this.determinizationCount =
      options.determinizationCount ?? DEFAULT_OPTIONS.determinizationCount;
    this.maxDepth = options.maxDepth ?? DEFAULT_OPTIONS.maxDepth;
    this.exploration = options.exploration ?? DEFAULT_OPTIONS.exploration;
    this.minimaxBlend = options.minimaxBlend ?? DEFAULT_OPTIONS.minimaxBlend;
    this.rng = options.rng ?? Math.random;
    this.evaluator =
      options.evaluator ?? new DynamicHeuristicEvaluator(options.evaluatorOptions);
    this.pruner = new ActionPruner({
      evaluator: this.evaluator,
      ...options.prunerOptions,
    });
    this.neural = options.neuralEvaluator ?? null;
  }

  getBestAction(
    state: PureGameState,
    legalActions?: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): GameAction {
    return this.getBestActionWithDebug(state, legalActions, perspective).action;
  }

  /**
   * Neural-enhanced async version. Pre-computes policy + value via ONNX,
   * then runs MCTS using neural priors for PUCT and neural value for leaf eval.
   * Falls back to heuristic-only getBestActionWithDebug if no neural evaluator.
   */
  async getBestActionNeuralAsync(
    state: PureGameState,
    legalActions?: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): Promise<{ action: GameAction; debug: ISMCTSDecisionDebug }> {
    if (!this.neural) {
      return this.getBestActionWithDebug(state, legalActions, perspective);
    }

    const baseActions =
      legalActions?.filter((a) => a.validity === ActionValidity.VALID) ??
      RuleEngine.getPossibleActions(state.gameState, { fastMode: true }).filter(
        (a) => a.validity === ActionValidity.VALID,
      );

    if (baseActions.length === 0) {
      throw new Error("ISMCTSAgent: no legal action");
    }
    if (baseActions.length === 1) {
      const stat: ISMCTSActionStat = { action: baseActions[0], visits: 0, meanValue: 0 };
      return {
        action: baseActions[0],
        debug: {
          selected: stat, candidates: [stat], top3: [stat],
          determinizationCount: 0, simulationsPerDeterminization: 0,
          processLogs: ["single action, skipping MCTS"],
        },
      };
    }

    const { value: neuralValue, policy: neuralPolicy } =
      await this.neural.evaluateWithPolicy(state, baseActions, perspective);

    this.cachedNeuralPolicy = neuralPolicy;
    this.cachedNeuralValue = neuralValue;

    try {
      return this.getBestActionWithDebug(state, baseActions, perspective);
    } finally {
      this.cachedNeuralPolicy = null;
      this.cachedNeuralValue = null;
    }
  }

  private cachedNeuralPolicy: Float32Array | null = null;
  private cachedNeuralValue: number | null = null;

  getBestActionWithDebug(
    state: PureGameState,
    legalActions?: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): { action: GameAction; debug: ISMCTSDecisionDebug } {
    const processLogs: string[] = [];
    const pushLog = (message: string) => {
      if (processLogs.length < 80) {
        processLogs.push(message);
      }
    };

    const baseActions =
      legalActions?.filter((a) => a.validity === ActionValidity.VALID) ??
      RuleEngine.getPossibleActions(state.gameState, { fastMode: true }).filter(
        (a) => a.validity === ActionValidity.VALID,
      );
    pushLog(
      `validActions=${baseActions.length}, perspective=${perspective}, turn=${state.gameState.currentTurn}`,
    );

    if (baseActions.length === 0) {
      throw new Error("ISMCTSAgent: no legal action");
    }

    const prunedRootCandidates = this.pruner.prune(state, baseActions, perspective);
    const rootCandidates = this.buildRootCandidates(
      state,
      baseActions,
      prunedRootCandidates,
      state.gameState.currentTurn,
    );
    pushLog(
      `rootCandidates=${rootCandidates.length}/${baseActions.length}${
        prunedRootCandidates.length === 0
          ? " (pruner empty -> fallback all)"
          : rootCandidates.length > prunedRootCandidates.length
            ? " (soft-prune widened)"
            : ""
      }`,
    );
    if (rootCandidates.length <= 1) {
      const selected = rootCandidates[0];
      const selectedStat: ISMCTSActionStat = {
        action: selected,
        visits: 0,
        meanValue: 0,
      };
      pushLog(`single candidate -> ${this.describeAction(selected)}`);
      return {
        action: selected,
        debug: {
          selected: selectedStat,
          candidates: [selectedStat],
          top3: [selectedStat],
          determinizationCount: this.determinizationCount,
          simulationsPerDeterminization: 0,
          processLogs,
        },
      };
    }
    const rootActionKeys = new Set(rootCandidates.map((a) => this.actionKey(a)));

    const aggregate = new Map<string, NodeStats>();
    const simulationPerRoot = Math.max(
      1,
      Math.floor(this.iterations / Math.max(1, this.determinizationCount)),
    );

    for (let i = 0; i < this.determinizationCount; i++) {
      const sampledRootState = this.determinize(state, perspective);
      const root = this.createNode(null, null);

      for (let j = 0; j < simulationPerRoot; j++) {
        this.runSimulation(root, perspective, sampledRootState, rootActionKeys);
      }

      for (const child of root.children) {
        const action = child.actionFromParent;
        if (!action) {
          continue;
        }
        const key = this.actionKey(action);
        const prev = aggregate.get(key);
        if (prev) {
          prev.visits += child.visits;
          prev.valueSum += child.valueSum;
          continue;
        }
        aggregate.set(key, {
          action,
          visits: child.visits,
          valueSum: child.valueSum,
        });
      }

      const detTop = [...root.children]
        .map((child) => {
          const action = child.actionFromParent;
          if (!action) {
            return null;
          }
          return {
            action,
            visits: child.visits,
            meanValue: child.valueSum / Math.max(1, child.visits),
          } satisfies ISMCTSActionStat;
        })
        .filter((entry): entry is ISMCTSActionStat => entry !== null)
        .sort((lhs, rhs) => {
          if (rhs.visits !== lhs.visits) {
            return rhs.visits - lhs.visits;
          }
          return rhs.meanValue - lhs.meanValue;
        })
        .slice(0, 3);
      if (detTop.length > 0) {
        pushLog(
          `det ${i + 1}/${this.determinizationCount}: ${detTop
            .map((entry, idx) => `#${idx + 1} ${this.formatStat(entry)}`)
            .join("; ")}`,
        );
      } else {
        pushLog(`det ${i + 1}/${this.determinizationCount}: no explored child`);
      }
    }

    let best: NodeStats | null = null;
    for (const stats of aggregate.values()) {
      if (!best) {
        best = stats;
        continue;
      }
      if (stats.visits > best.visits) {
        best = stats;
        continue;
      }
      if (stats.visits === best.visits) {
        const lhs = stats.valueSum / Math.max(1, stats.visits);
        const rhs = best.valueSum / Math.max(1, best.visits);
        if (lhs > rhs) {
          best = stats;
        }
      }
    }

    const rankedCandidates =
      aggregate.size > 0
        ? [...aggregate.values()]
            .map<ISMCTSActionStat>((stats) => ({
              action: stats.action,
              visits: stats.visits,
              meanValue: stats.valueSum / Math.max(1, stats.visits),
            }))
            .sort((lhs, rhs) => {
              if (rhs.visits !== lhs.visits) {
                return rhs.visits - lhs.visits;
              }
              return rhs.meanValue - lhs.meanValue;
            })
        : rootCandidates.map<ISMCTSActionStat>((action) => ({
            action,
            visits: 0,
            meanValue: 0,
          }));
    if (aggregate.size === 0) {
      pushLog("aggregate is empty, fallback to root candidates");
    }

    const selected =
      best !== null
        ? {
            action: best.action,
            visits: best.visits,
            meanValue: best.valueSum / Math.max(1, best.visits),
          }
        : rankedCandidates[0];
    const top3 = rankedCandidates.slice(0, 3);
    if (top3.length > 0) {
      pushLog(
        `final top3: ${top3
          .map((entry, idx) => `#${idx + 1} ${this.formatStat(entry)}`)
          .join("; ")}`,
      );
    }
    pushLog(`selected: ${this.formatStat(selected)}`);

    return {
      action: selected.action,
      debug: {
        selected,
        candidates: rankedCandidates,
        top3,
        determinizationCount: this.determinizationCount,
        simulationsPerDeterminization: simulationPerRoot,
        processLogs,
      },
    };
  }

  private runSimulation(
    root: SearchNode,
    rootPlayer: 0 | 1,
    currentUniverseState: PureGameState,
    rootActionKeys: ReadonlySet<string>,
  ): void {
    let node = root;
    let state = currentUniverseState;
    let depth = 0;

    const trajectory: TrajectoryEntry[] = [
      {
        node,
        playerToAct: currentPlayerOf(state),
      },
    ];

    while (depth < this.maxDepth && !isTerminalState(state)) {
      const untriedActions = this.getLegalUntriedActions(
        node,
        state,
        node === root ? rootActionKeys : undefined,
      );

      if (untriedActions.length > 0) {
        const expanded = this.expandNode(node, state, untriedActions);
        if (expanded) {
          node = expanded.child;
          state = expanded.nextState;
          depth += 1;
          trajectory.push({
            node,
            playerToAct: currentPlayerOf(state),
          });
        }
        break;
      }

      const selected = this.selectChild(node, state, rootPlayer);
      if (!selected) {
        break;
      }

      node = selected.child;
      state = selected.nextState;
      depth += 1;
      trajectory.push({
        node,
        playerToAct: currentPlayerOf(state),
      });
    }

    const leafValue = this.rollout(state, rootPlayer, depth);

    for (let i = trajectory.length - 1; i >= 0; i--) {
      const entry = trajectory[i];
      const current = entry.node;
      const firstVisit = current.visits === 0;
      current.visits += 1;
      current.valueSum += leafValue;

      if (firstVisit) {
        current.minimaxValue = leafValue;
        continue;
      }

      if (entry.playerToAct === null || entry.playerToAct === rootPlayer) {
        current.minimaxValue = Math.max(current.minimaxValue, leafValue);
      } else {
        current.minimaxValue = Math.min(current.minimaxValue, leafValue);
      }
    }
  }

  private selectChild(
    node: SearchNode,
    state: PureGameState,
    rootPlayer: 0 | 1,
  ): SelectionResult | null {
    if (node.children.length === 0 || isTerminalState(state)) {
      return null;
    }

    const currentPlayer = state.gameState.currentTurn;
    const parentVisits = Math.max(1, node.visits);
    const sqrtParentVisits = Math.sqrt(parentVisits);

    let bestResult: SelectionResult | null = null;
    let bestScore = Number.NEGATIVE_INFINITY;
    const scoredChildren = node.children
      .map((child) => {
        const action = child.actionFromParent;
        if (!action) {
          return null;
        }
        return {
          child,
          action,
          priorScore: this.pruner.quickActionGain(action, state.gameState, currentPlayer),
        };
      })
      .filter(
        (
          entry,
        ): entry is {
          child: SearchNode;
          action: GameAction;
          priorScore: number;
        } => entry !== null,
      );
    if (scoredChildren.length === 0) {
      return null;
    }

    let priorProbabilities: number[];
    if (this.cachedNeuralPolicy && node.parent === null) {
      // At the root node during neural-enhanced search, use the neural policy
      // directly as PUCT priors instead of heuristic softmax.
      priorProbabilities = scoredChildren.map((entry) => {
        const idx = this.actionKeyToSlotIndex(entry.action, state.gameState, currentPlayer);
        return idx >= 0 ? (this.cachedNeuralPolicy![idx] || 1e-6) : 1e-6;
      });
      const probSum = priorProbabilities.reduce((s, v) => s + v, 0);
      if (probSum > 0) {
        priorProbabilities = priorProbabilities.map((p) => p / probSum);
      }
    } else {
      priorProbabilities = this.toSoftmaxProbabilities(
        scoredChildren.map((entry) => entry.priorScore),
        1.25,
      );
    }

    for (let i = 0; i < scoredChildren.length; i++) {
      const { child, action } = scoredChildren[i];
      const nextState = this.transition(state, action);
      if (!nextState) {
        continue;
      }

      const mean = child.visits === 0 ? 0 : child.valueSum / child.visits;
      const minimax = child.visits === 0 ? mean : child.minimaxValue;
      const blended = this.minimaxBlend * minimax + (1 - this.minimaxBlend) * mean;
      const exploitation = currentPlayer === rootPlayer ? blended : -blended;
      const prior = priorProbabilities[i] ?? 0;
      const exploration =
        this.exploration * prior * (sqrtParentVisits / Math.max(1, child.visits));
      const score = exploitation + exploration;

      if (score > bestScore) {
        bestScore = score;
        bestResult = {
          child,
          nextState,
        };
      }
    }

    return bestResult;
  }

  private expandNode(
    node: SearchNode,
    state: PureGameState,
    legalUntriedActions: readonly GameAction[],
  ): SelectionResult | null {
    const who = state.gameState.currentTurn;
    const candidates = [...legalUntriedActions];

    while (candidates.length > 0) {
      const index = this.pickActionIndexBySoftmax(candidates, state.gameState, who, 1.1);
      const [action] = candidates.splice(index, 1);
      const nextState = this.transition(state, action);
      if (!nextState) {
        continue;
      }
      const child = this.createNode(node, action);
      node.children.push(child);
      return {
        child,
        nextState,
      };
    }

    return null;
  }

  private rollout(initial: PureGameState, rootPlayer: 0 | 1, depth: number): number {
    let state = initial;
    let currentDepth = depth;

    while (currentDepth < this.maxDepth && !isTerminalState(state)) {
      const legal = RuleEngine.getPossibleActions(state.gameState, { fastMode: true }).filter(
        (action) => action.validity === ActionValidity.VALID,
      );
      if (legal.length === 0) {
        break;
      }

      const who = state.gameState.currentTurn;
      const candidates = [...legal];
      let nextState: PureGameState | null = null;
      while (candidates.length > 0) {
        const selectedIndex = this.pickActionIndexBySoftmax(
          candidates,
          state.gameState,
          who,
          0.95,
        );
        const [selectedAction] = candidates.splice(selectedIndex, 1);
        nextState = this.transition(state, selectedAction);
        if (nextState) {
          break;
        }
      }
      if (!nextState) {
        break;
      }
      state = nextState;

      currentDepth += 1;
    }

    const heuristicValue = this.evaluator.evaluate(state, rootPlayer);
    if (this.cachedNeuralValue !== null) {
      // Blend: 70% heuristic (which sees the actual rollout leaf state)
      // + 30% neural root value (provides learned "intuition").
      // The neural value is in [-1, 1], scale it to match heuristic magnitude.
      const scaledNeural = this.cachedNeuralValue * Math.abs(heuristicValue || 1);
      return 0.7 * heuristicValue + 0.3 * scaledNeural;
    }
    return heuristicValue;
  }

  private createNode(
    parent: SearchNode | null,
    actionFromParent: GameAction | null,
  ): SearchNode {
    return {
      parent,
      actionFromParent,
      children: [],
      visits: 0,
      valueSum: 0,
      minimaxValue: 0,
    };
  }

  private getLegalUntriedActions(
    node: SearchNode,
    state: PureGameState,
    rootActionKeys?: ReadonlySet<string>,
  ): GameAction[] {
    const legal = RuleEngine.getPossibleActions(state.gameState, { fastMode: true }).filter(
      (action) => action.validity === ActionValidity.VALID,
    );

    const filtered =
      rootActionKeys && node.parent === null
        ? legal.filter((action) => rootActionKeys.has(this.actionKey(action)))
        : legal;

    const triedActionKeys = new Set(
      node.children
        .map((child) => child.actionFromParent)
        .filter((action): action is GameAction => action !== null)
        .map((action) => this.actionKey(action)),
    );

    return filtered.filter((action) => !triedActionKeys.has(this.actionKey(action)));
  }

  private transition(state: PureGameState, action: GameAction): PureGameState | null {
    try {
      const nextGameState = RuleEngine.execute(state.gameState, action).nextState;
      return {
        ...state,
        gameState: nextGameState,
        history: [],
        turn: state.turn + 1,
        isFinished: nextGameState.phase === "gameEnd",
        winner: nextGameState.winner ?? undefined,
      };
    } catch {
      return null;
    }
  }

  private determinize(state: PureGameState, perspective: 0 | 1): PureGameState {
    const opponent = flip(perspective);
    const target = state.gameState.players[opponent];

    const cards = [...target.hands, ...target.pile];
    const activeCharacter = target.characters.find((c) => c.id === target.activeCharacterId);
    const activeElement = activeCharacter ? elementOfCharacter(activeCharacter.definition) : DiceType.Omni;
    const preferredElement = activeElement === DiceType.Void ? DiceType.Omni : activeElement;

    if (cards.length <= 1 && target.dice.length === 0) {
      return state;
    }

    const shuffledCards = this.shuffle(cards);
    const handCount = target.hands.length;
    const randomizedDice = target.dice.map(() => {
      const randomValue = this.rng();
      if (randomValue < 0.3) {
        return DiceType.Omni;
      }
      if (randomValue < 0.7) {
        return preferredElement;
      }
      return DETERMINIZATION_DICE_POOL[
        Math.floor(this.rng() * DETERMINIZATION_DICE_POOL.length)
      ];
    });

    const opponentPlayer = {
      ...target,
      hands: shuffledCards.slice(0, handCount),
      pile: shuffledCards.slice(handCount),
      dice: randomizedDice,
    };

    const players = [...state.gameState.players] as [
      GameState["players"][0],
      GameState["players"][1],
    ];
    players[opponent] = opponentPlayer;

    return {
      ...state,
      history: [],
      gameState: {
        ...state.gameState,
        players,
      },
    };
  }

  private shuffle<T>(items: readonly T[]): T[] {
    const next = [...items];
    for (let i = next.length - 1; i > 0; i--) {
      const j = Math.floor(this.rng() * (i + 1));
      const temp = next[i];
      next[i] = next[j];
      next[j] = temp;
    }
    return next;
  }

  private describeAction(action: GameAction): string {
    const payload = action.action;
    if (!payload) {
      return "unknown";
    }
    switch (payload.$case) {
      case "useSkill":
        return `useSkill(${payload.value.skillDefinitionId})`;
      case "playCard":
        return `playCard(${payload.value.cardDefinitionId})`;
      case "switchActive":
        return `switchActive(${payload.value.characterDefinitionId})`;
      case "elementalTuning":
        return `elementalTuning(${payload.value.removedCardId})`;
      case "declareEnd":
        return "declareEnd";
    }
  }

  private formatStat(stat: ISMCTSActionStat): string {
    return `${this.describeAction(stat.action)} (visits=${stat.visits}, mean=${stat.meanValue.toFixed(3)})`;
  }

  private buildRootCandidates(
    state: PureGameState,
    baseActions: readonly GameAction[],
    prunedRootCandidates: readonly GameAction[],
    actor: 0 | 1,
  ): GameAction[] {
    if (prunedRootCandidates.length === 0 || baseActions.length <= 4) {
      return [...baseActions];
    }

    const targetCount = Math.min(
      baseActions.length,
      Math.max(prunedRootCandidates.length, Math.ceil(baseActions.length * 0.6)),
    );
    if (prunedRootCandidates.length >= targetCount) {
      return [...prunedRootCandidates];
    }

    const chosen = [...prunedRootCandidates];
    const chosenKeys = new Set(chosen.map((action) => this.actionKey(action)));
    const remainder = baseActions.filter((action) => !chosenKeys.has(this.actionKey(action)));
    const rankedRemainder = remainder
      .map((action) => ({
        action,
        score: this.pruner.quickActionGain(action, state.gameState, actor),
      }))
      .sort((lhs, rhs) => rhs.score - lhs.score);

    for (const item of rankedRemainder) {
      if (chosen.length >= targetCount) {
        break;
      }
      chosen.push(item.action);
    }
    return chosen;
  }

  private toSoftmaxProbabilities(scores: readonly number[], temperature: number): number[] {
    if (scores.length === 0) {
      return [];
    }
    const safeTemperature = Math.max(0.05, temperature);
    const normalized = scores.map((score) =>
      Number.isFinite(score) ? score / safeTemperature : -30,
    );
    const maxScore = Math.max(...normalized);
    const exps = normalized.map((score) => Math.exp(Math.max(-30, Math.min(30, score - maxScore))));
    const sumExp = exps.reduce((sum, value) => sum + value, 0);
    if (!Number.isFinite(sumExp) || sumExp <= 0) {
      return scores.map(() => 1 / scores.length);
    }
    return exps.map((value) => value / sumExp);
  }

  private pickActionIndexBySoftmax(
    actions: readonly GameAction[],
    gameState: GameState,
    actor: 0 | 1,
    temperature: number,
  ): number {
    if (actions.length <= 1) {
      return 0;
    }
    const probabilities = this.toSoftmaxProbabilities(
      actions.map((action) => this.pruner.quickActionGain(action, gameState, actor)),
      temperature,
    );
    let roll = this.rng();
    for (let i = 0; i < probabilities.length; i++) {
      roll -= probabilities[i];
      if (roll <= 0) {
        return i;
      }
    }
    return probabilities.length - 1;
  }

  private actionKey(action: GameAction): string {
    return JSON.stringify({
      action: action.action,
      cost: action.requiredCost,
      dice: action.autoSelectedDice,
      fast: action.isFast,
    });
  }

  private readonly _actionIndexer = new ActionIndexer();

  private actionKeyToSlotIndex(
    action: GameAction,
    gameState: GameState,
    perspective: 0 | 1,
  ): number {
    return this._actionIndexer.actionToIndex(action, gameState, perspective);
  }
}

import { ActionValidity, Aura, DiceType, type Action } from "@gi-tcg/typings";
import { flip } from "@gi-tcg/utils";
import type { GameState } from "../base/state";
import { elementOfCharacter } from "../utils";
import { RuleEngine } from "./rule_engine";
import { DynamicHeuristicEvaluator } from "./dynamic_heuristic_evaluator";
import type { GameAction, PureGameState } from "./types";

export interface ActionPrunerOptions {
  thresholdUv?: number;
  maxCounterActions?: number;
  evaluator?: DynamicHeuristicEvaluator;
}

export interface PrunedActionScore {
  action: GameAction;
  score: number;
  selfGain: number;
  counterGain: number;
  overkillPenalty: number;
}

const DEFAULT_THRESHOLD_UV = 3;
const DEFAULT_COUNTER_LIMIT = 12;

function countResourceCost(action: Action): { dice: number; energy: number } {
  let dice = 0;
  let energy = 0;
  for (const req of action.requiredCost) {
    if (req.type === DiceType.Energy) {
      energy += req.count;
      continue;
    }
    if (req.type === DiceType.Legend) {
      continue;
    }
    dice += req.count;
  }
  return { dice, energy };
}

function maxCharacterHp(player: GameState["players"][number]): number {
  let max = 0;
  for (const character of player.characters) {
    const hp =
      typeof character.variables.health === "number"
        ? character.variables.health
        : 0;
    if (hp > max) {
      max = hp;
    }
  }
  return max;
}

export class ActionPruner {
  private readonly thresholdUv: number;
  private readonly maxCounterActions: number;
  private readonly evaluator: DynamicHeuristicEvaluator;

  constructor(options: ActionPrunerOptions = {}) {
    this.thresholdUv = options.thresholdUv ?? DEFAULT_THRESHOLD_UV;
    this.maxCounterActions = options.maxCounterActions ?? DEFAULT_COUNTER_LIMIT;
    this.evaluator = options.evaluator ?? new DynamicHeuristicEvaluator();
  }

  prune(
    state: PureGameState,
    actions: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): GameAction[] {
    const ranked = this.rankActions(state, actions, perspective);
    if (ranked.length === 0) {
      return [];
    }

    const best = ranked[0].score;
    const kept = ranked
      .filter((item) => item.score >= best - this.thresholdUv)
      .map((item) => item.action);

    if (kept.length > 0) {
      return kept;
    }
    return [ranked[0].action];
  }

  rankActions(
    state: PureGameState,
    actions: readonly GameAction[],
    perspective: 0 | 1 = state.gameState.currentTurn,
  ): PrunedActionScore[] {
    if (actions.length === 0) {
      return [];
    }
    const baseline = this.evaluator.evaluate(state, perspective);
    const result: PrunedActionScore[] = [];

    for (const action of actions) {
      if (action.validity !== ActionValidity.VALID) {
        continue;
      }

      let nextGameState: GameState;
      try {
        nextGameState = RuleEngine.execute(state.gameState, action).nextState;
      } catch {
        continue;
      }

      const nextState = this.toPureState(state, nextGameState);
      const selfGain = this.evaluator.evaluate(nextState, perspective) - baseline;
      const counterGain = this.estimateBestCounterGain(nextGameState, perspective);
      const overkillPenalty = this.estimateOverkillPenalty(state.gameState, action, perspective);
      const score = selfGain - counterGain - overkillPenalty;

      result.push({
        action,
        score,
        selfGain,
        counterGain,
        overkillPenalty,
      });
    }

    result.sort((a, b) => b.score - a.score);
    return result;
  }

  private estimateBestCounterGain(
    stateAfterSelf: GameState,
    perspective: 0 | 1,
  ): number {
    const opponent = flip(perspective);
    if (stateAfterSelf.phase === "gameEnd" || stateAfterSelf.winner !== null) {
      return 0;
    }

    if (stateAfterSelf.currentTurn !== opponent) {
      return (
        stateAfterSelf.players[opponent].dice.length * 0.08 +
        stateAfterSelf.players[opponent].hands.length * 0.06
      );
    }

    const counters = RuleEngine.getPossibleActions(stateAfterSelf, { fastMode: true })
      .filter((a) => a.validity === ActionValidity.VALID)
      .slice(0, this.maxCounterActions);

    let best = 0;
    for (const counter of counters) {
      const score = this.quickActionGain(counter, stateAfterSelf, opponent);
      if (score > best) {
        best = score;
      }
    }
    return best;
  }

  public quickActionGain(
    action: GameAction,
    state: GameState,
    actor: 0 | 1,
  ): number {
    const payload = action.action;
    if (!payload) {
      return Number.NEGATIVE_INFINITY;
    }
    const { dice, energy } = countResourceCost(action);
    const actorPlayer = state.players[actor];
    const defender = state.players[flip(actor)];
    const activeSelf = actorPlayer.characters.find((c) => c.id === actorPlayer.activeCharacterId);
    const activeEnemy = defender.characters.find((c) => c.id === defender.activeCharacterId);
    const enemyAura =
      typeof activeEnemy?.variables.aura === "number"
        ? (activeEnemy.variables.aura as Aura)
        : Aura.None;

    let gain = 0;
    switch (payload.$case) {
      case "useSkill": {
        gain = 3.2;
        const targetId = payload.value.mainDamageTargetId;
        if (typeof targetId === "number") {
          const hp = this.findCharacterHp(state, targetId);
          if (hp > 0 && hp <= 3) {
            gain += 1.8;
          }
        }
        if (activeSelf && enemyAura !== Aura.None) {
          const activeElement = elementOfCharacter(activeSelf.definition);
          const reaction = RuleEngine.getTriggeredReaction(activeElement, enemyAura);
          if (reaction !== null) {
            // Reaction-ready skills get extra rollout attention.
            gain += 1.1;
          }
        }
        break;
      }
      case "playCard": {
        gain = payload.value.willBeEffectless ? -6.0 : 1.2;
        break;
      }
      case "switchActive": {
        gain = 0.4;
        const nextActive = actorPlayer.characters.find(
          (c) => c.id === payload.value.characterId,
        );
        const nextEnergy =
          typeof nextActive?.variables.energy === "number" ? nextActive.variables.energy : 0;
        const nextMaxEnergy =
          typeof nextActive?.variables.maxEnergy === "number"
            ? Math.max(1, nextActive.variables.maxEnergy)
            : 3;
        if (nextEnergy >= nextMaxEnergy) {
          gain += 1.4;
        }
        if (enemyAura !== Aura.None) {
          const nextElement = nextActive ? elementOfCharacter(nextActive.definition) : DiceType.Void;
          if (RuleEngine.canTriggerReaction(nextElement, enemyAura)) {
            // Encourage switch lines that prepare immediate reaction pressure.
            gain += 1.8;
          }
        }
        const fromHp =
          typeof activeSelf?.variables.health === "number" ? activeSelf.variables.health : 10;
        if (fromHp <= 2) {
          gain += 1.2;
        }
        break;
      }
      case "elementalTuning":
        gain = 0.35;
        break;
      case "declareEnd":
        gain = actorPlayer.dice.length === 0 ? 1.4 : -2.0;
        break;
      default:
        gain = 0;
        break;
    }

    const tempo = action.isFast ? 0.3 : 0;
    const pressure = Math.min(1.2, maxCharacterHp(defender) * 0.06);
    const affordability = actorPlayer.dice.length >= dice ? 0.25 : -1.0;
    const costPenalty = dice * 0.42 + energy * 1.0;

    return gain + tempo + pressure + affordability - costPenalty;
  }

  private estimateOverkillPenalty(
    state: GameState,
    action: GameAction,
    perspective: 0 | 1,
  ): number {
    const payload = action.action;
    if (!payload) {
      return 0;
    }

    let targetId: number | undefined;
    switch (payload.$case) {
      case "useSkill":
        targetId = payload.value.mainDamageTargetId ?? payload.value.targetIds[0];
        break;
      case "playCard":
        targetId = payload.value.targetIds[0];
        break;
      default:
        return 0;
    }

    if (typeof targetId !== "number") {
      return 0;
    }

    const targetHp = this.findCharacterHp(state, targetId);
    if (targetHp <= 0) {
      return 0;
    }

    const estimatedDamage = this.estimateActionDamage(action, perspective);
    return this.evaluator.estimateOverkillPenalty(estimatedDamage, targetHp);
  }

  private estimateActionDamage(action: GameAction, _perspective: 0 | 1): number {
    const { dice, energy } = countResourceCost(action);
    const payload = action.action;
    if (!payload) {
      return 0;
    }
    switch (payload.$case) {
      case "useSkill":
        return 2 + dice * 0.5 + energy * 1.2;
      case "playCard":
        return payload.value.willBeEffectless ? 0 : 1.5 + dice * 0.35;
      case "elementalTuning":
      case "switchActive":
      case "declareEnd":
      default:
        return 0;
    }
  }

  private toPureState(state: PureGameState, nextGameState: GameState): PureGameState {
    return {
      ...state,
      gameState: nextGameState,
      history: [],
      turn: state.turn + 1,
      isFinished: nextGameState.phase === "gameEnd",
      winner: nextGameState.winner ?? undefined,
    };
  }

  private findCharacterHp(state: GameState, entityId: number): number {
    for (const player of state.players) {
      for (const character of player.characters) {
        if (character.id !== entityId) {
          continue;
        }
        return typeof character.variables.health === "number"
          ? character.variables.health
          : 0;
      }
    }
    return 0;
  }
}

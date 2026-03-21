import { Aura } from "@gi-tcg/typings";
import { flip } from "@gi-tcg/utils";
import type { CharacterState, PlayerState } from "../base/state";
import type { PureGameState } from "./types";

export interface DynamicHeuristicWeights {
  wHp: number;
  wRes: number;
  wEn: number;
  wAura: number;
  wOverkill: number;
  wSummon: number;
  wSupport: number;
  wEquip: number;
  resourceFragilityBoost: number;
  terminalValue: number;
}

export interface DynamicHeuristicEvaluatorOptions {
  weights?: Partial<DynamicHeuristicWeights>;
}

const DEFAULT_WEIGHTS: DynamicHeuristicWeights = {
  wHp: 1.0,
  wRes: 1.0,
  wEn: 1.2,
  wAura: 0.6,
  wOverkill: 1.0,
  wSummon: 1.1,
  wSupport: 0.7,
  wEquip: 1.0,
  resourceFragilityBoost: 1.2,
  terminalValue: 10_000,
};

interface PlayerSnapshot {
  totalHp: number;
  resourceUv: number;
  auraCount: number;
  weightedEnergy: number;
  teamSurvival: number;
  boardValue: number;
}

function readNumber(input: unknown, fallback = 0): number {
  return typeof input === "number" && Number.isFinite(input) ? input : fallback;
}

function readAlive(character: CharacterState): boolean {
  const alive = character.variables.alive;
  if (typeof alive === "boolean") {
    return alive;
  }
  if (typeof alive === "number") {
    return alive > 0;
  }
  return readNumber(character.variables.health) > 0;
}

export class DynamicHeuristicEvaluator {
  private readonly weights: DynamicHeuristicWeights;

  constructor(options: DynamicHeuristicEvaluatorOptions = {}) {
    this.weights = { ...DEFAULT_WEIGHTS, ...options.weights };
  }

  evaluate(state: PureGameState, perspective: 0 | 1 = state.gameState.currentTurn): number {
    const winner = state.gameState.winner;
    if (state.isFinished || state.gameState.phase === "gameEnd" || winner !== null) {
      if (winner === null) return 0;
      return winner === perspective ? this.weights.terminalValue : -this.weights.terminalValue;
    }

    const opponent = flip(perspective);
    const self = this.snapshotPlayer(state.gameState.players[perspective], state.gameState.players[opponent]);
    const oppo = this.snapshotPlayer(state.gameState.players[opponent], state.gameState.players[perspective]);

    const dynamicSelfResourceWeight =
      this.weights.wRes + (1 - self.teamSurvival) * this.weights.resourceFragilityBoost;
    const dynamicOppoResourceWeight =
      this.weights.wRes + (1 - oppo.teamSurvival) * this.weights.resourceFragilityBoost;

    const hpTerm = (self.totalHp - oppo.totalHp) * this.weights.wHp;
    const resourceTerm =
      self.resourceUv * dynamicSelfResourceWeight -
      oppo.resourceUv * dynamicOppoResourceWeight;
    const auraGain = (oppo.auraCount - self.auraCount) * this.weights.wAura;
    const energyTerm = self.weightedEnergy - oppo.weightedEnergy;
    const boardTerm = (self.boardValue - oppo.boardValue) * 0.45;

    return hpTerm + resourceTerm + auraGain + energyTerm + boardTerm;
  }

  private snapshotPlayer(player: PlayerState, opponent: PlayerState): PlayerSnapshot {
    let totalHp = 0;
    let auraCount = 0;
    let weightedEnergy = 0;
    let survivalSum = 0;
    let aliveCount = 0;
    let boardValue = 0;

    boardValue += player.summons.length * this.weights.wSummon;
    for (const summon of player.summons) {
      const usageLike =
        readNumber(summon.variables.usage) + readNumber(summon.variables.usages);
      boardValue += usageLike * 0.5;
    }

    for (const support of player.supports) {
      const supportCharges =
        readNumber(support.variables.usage) + readNumber(support.variables.usages);
      const supportActiveValue = supportCharges > 0 ? 1 + Math.min(2, supportCharges) * 0.25 : 0.35;
      boardValue += this.weights.wSupport * supportActiveValue;
    }

    for (const status of player.combatStatuses) {
      const shield = readNumber(status.variables.shield);
      totalHp += shield;
      boardValue += shield > 0 ? 0.5 : 0.2;
    }

    for (const character of player.characters) {
      if (!readAlive(character)) continue;

      let hp = Math.max(0, readNumber(character.variables.health));

      for (const entity of character.entities) {
        const type = entity.definition.type;
        const tags = entity.definition.tags;
        const shield = readNumber(entity.variables.shield);
        const usageLike =
          readNumber(entity.variables.usage) + readNumber(entity.variables.usages);
        hp += shield;

        if (type === "equipment" || tags.includes("weapon") || tags.includes("artifact")) {
          boardValue += this.weights.wEquip + Math.min(1, usageLike) * 0.25;
        } else {
          boardValue += usageLike > 0 ? 0.35 : 0.2;
        }
      }

      totalHp += hp;

      const aura = readNumber(character.variables.aura, Aura.None);
      if (aura !== Aura.None) auraCount += 1;

      const survival = this.estimateSurvival(character, opponent);
      const energy = Math.max(0, readNumber(character.variables.energy));
      const equipBonus = character.entities.filter((e) => e.definition.type === "equipment").length;
      weightedEnergy += energy * this.weights.wEn * survival * survival + equipBonus * survival;

      survivalSum += survival;
      aliveCount += 1;
    }

    const teamSurvival = aliveCount === 0 ? 0 : survivalSum / aliveCount;

    return {
      totalHp,
      resourceUv: player.dice.length + player.hands.length,
      auraCount,
      weightedEnergy,
      teamSurvival,
      boardValue,
    };
  }

  estimateSurvival(character: CharacterState, opponentState: PlayerState): number {
    const hp = Math.max(0, readNumber(character.variables.health));
    if (hp <= 0 || !readAlive(character)) return 0;
    const maxHp = Math.max(1, readNumber(character.variables.maxHealth, hp));
    const hpRatio = hp / maxHp;
    const pressure = this.estimateOpponentPressure(opponentState);
    const margin = hp - pressure;
    const logistic = 1 / (1 + Math.exp(-0.85 * margin));
    const survival = 0.55 * hpRatio + 0.45 * logistic;
    return Math.max(0, Math.min(1, survival));
  }

  estimateOverkillPenalty(estimatedDamage: number, targetRemainingHp: number): number {
    const overkill = estimatedDamage - targetRemainingHp;
    return overkill <= 0 ? 0 : overkill * this.weights.wOverkill;
  }

  private estimateOpponentPressure(opponent: PlayerState): number {
    const active = opponent.characters.find((c) => c.id === opponent.activeCharacterId);
    const activeEnergy = Math.max(0, readNumber(active?.variables.energy));
    const maxEnergy = Math.max(1, readNumber(active?.variables.maxEnergy, 3));
    const burstFactor = activeEnergy / maxEnergy;
    const dicePressure = Math.min(4, Math.floor(opponent.dice.length / 3));
    const handPressure = Math.min(2, Math.floor(opponent.hands.length / 4));
    return 2 + burstFactor * 2 + dicePressure + handPressure;
  }
}

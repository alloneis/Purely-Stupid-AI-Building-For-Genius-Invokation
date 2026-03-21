import { checkDice, chooseDiceValue, flip } from "@gi-tcg/utils";
import {
  ActionValidity,
  Aura,
  DamageType,
  DiceType,
  Reaction,
  type Action,
  type LunarReaction,
} from "@gi-tcg/typings";
import {
  ActionEventArg,
  EventArg,
  defineSkillInfo,
  DisposeEventArg,
  GenericModifyActionEventArg,
  ModifyRollEventArg,
  PlayCardEventArg,
  PlayerEventArg,
  UseSkillEventArg,
  ZeroHealthEventArg,
  type ActionInfo,
  type EventAndRequest,
  type InitiativeSkillEventArg,
} from "../base/skill";
import { getReaction, type NontrivialDamageType } from "../base/reaction";
import type {
  AnyState,
  CharacterState,
  EntityState,
  ErrorLevel,
  GameState,
} from "../base/state";
import { GiTcgCoreInternalError, GiTcgDataError } from "../error";
import { exposeAction } from "../io";
import { type StatePatch, StateMutator } from "../mutator";
import { type ActionInfoWithModification, ActionPreviewer } from "../preview";
import { SkillExecutor } from "../skill_executor";
import {
  applyAttachmentModifications,
  applyAutoSelectedDiceToAction,
  checkImmune,
  elementOfCharacter,
  getActiveCharacterIndex,
  getEntityArea,
  getEntityById,
  initiativeSkillsOfPlayer,
  isChargedPlunging,
  isSkillDisabled,
  playSkillOfCard,
  sortDice,
} from "../utils";
import type {
  RuleEngineDecisionProvider,
  RuleEngineExecuteResult,
} from "./types";

export interface RuleEngineExecuteOption {
  readonly decisionProvider?: RuleEngineDecisionProvider;
  /**
   * When true, caller already applied action-phase prelude logic
   * (set canCharged + onBeforeAction), so RuleEngine must not repeat it.
   */
  readonly skipActionPhasePrelude?: boolean;
}

export interface RuleEngineBootstrapOption {
  readonly decisionProvider?: RuleEngineDecisionProvider;
}

export interface RuleEngineActionQueryOption {
  readonly fastMode?: boolean;
}

/**
 * Pure in-memory rule kernel.
 *
 * Important constraints:
 * - No IO/RPC inside this class.
 * - Any request* event means upstream must split it into an explicit action or
 *   inject a decision provider.
 */
export class RuleEngine {
  static getTriggeredReaction(
    activeElement: DiceType,
    targetAura: Aura,
    enabledLunarReactions: readonly LunarReaction[] = [],
  ): Reaction | null {
    const damageType = this.elementToDamageType(activeElement);
    if (damageType === null) {
      return null;
    }
    return getReaction({
      type: damageType,
      targetAura,
      enabledLunarReactions,
    }).reaction;
  }

  static canTriggerReaction(
    activeElement: DiceType,
    targetAura: Aura,
    enabledLunarReactions: readonly LunarReaction[] = [],
  ): boolean {
    return this.getTriggeredReaction(activeElement, targetAura, enabledLunarReactions) !== null;
  }

  static bootstrapToActionPhase(
    initialState: GameState,
    opt: RuleEngineBootstrapOption = {},
  ): RuleEngineExecuteResult {
    const mutator = new StateMutator(initialState, {
      silent: true,
      patchDelivery: "deferred",
      onNotify: () => {},
      onPause: async () => {},
    });
    this.advanceToActionPhase(mutator, opt.decisionProvider);
    return {
      nextState: mutator.state,
      patches: mutator.drainStagedPatches(),
    };
  }

  private static elementToDamageType(
    activeElement: DiceType,
  ): NontrivialDamageType | null {
    switch (activeElement) {
      case DiceType.Cryo:
        return DamageType.Cryo;
      case DiceType.Hydro:
        return DamageType.Hydro;
      case DiceType.Pyro:
        return DamageType.Pyro;
      case DiceType.Electro:
        return DamageType.Electro;
      case DiceType.Anemo:
        return DamageType.Anemo;
      case DiceType.Geo:
        return DamageType.Geo;
      case DiceType.Dendro:
        return DamageType.Dendro;
      default:
        return null;
    }
  }

  static execute(
    initialState: GameState,
    action: Action,
    opt: RuleEngineExecuteOption = {},
  ): RuleEngineExecuteResult {
    const mutator = new StateMutator(initialState, {
      silent: true,
      patchDelivery: "deferred",
      onNotify: () => {},
      onPause: async () => {},
    });
    this.advanceToActionPhase(mutator, opt.decisionProvider);
    if (mutator.state.phase === "gameEnd") {
      return {
        nextState: mutator.state,
        patches: mutator.drainStagedPatches(),
      };
    }
    const pendingEvents: EventAndRequest[] = [];
    this.applyInitialAction(
      mutator,
      action,
      pendingEvents,
      opt.decisionProvider,
      opt.skipActionPhasePrelude ?? false,
    );
    this.drainEvents(mutator, pendingEvents, opt.decisionProvider);
    this.advanceToActionPhase(mutator, opt.decisionProvider);
    return {
      nextState: mutator.state,
      patches: mutator.drainStagedPatches(),
    };
  }

  static getPossibleActions(
    state: GameState,
    opt: RuleEngineActionQueryOption = {},
  ): Action[] {
    const fastMode = opt.fastMode ?? true;
    if (state.phase !== "action") {
      return [];
    }
    const who = state.currentTurn;
    const player = state.players[who];
    if (!player.characters.some((ch) => ch.id === player.activeCharacterId)) {
      return [];
    }
    const activeCh = player.characters[getActiveCharacterIndex(player)];
    const result: ActionInfo[] = [];

    const skillDisabled = isSkillDisabled(activeCh);
    for (const { caller, skill } of initiativeSkillsOfPlayer(player)) {
      const { charged, plunging } = isChargedPlunging(skill, player);
      const skillInfo = defineSkillInfo({
        caller,
        definition: skill,
        charged,
        plunging,
      });
      const actionInfoBase = {
        type: "useSkill" as const,
        who,
        skill: skillInfo,
        targets: [],
        fast: skill.initiativeSkillConfig.shouldFast,
        cost: skill.initiativeSkillConfig.requiredCost,
        autoSelectedDice: [],
      };
      if (skillDisabled) {
        result.push({
          ...actionInfoBase,
          validity: ActionValidity.DISABLED,
        });
        continue;
      }
      const allTargets = skill.initiativeSkillConfig.getTarget(state, skillInfo);
      if (allTargets.length === 0) {
        result.push({
          ...actionInfoBase,
          validity: ActionValidity.NO_TARGET,
        });
        continue;
      }
      for (const arg of allTargets) {
        if (!skill.filter(state, skillInfo, arg)) {
          result.push({
            ...actionInfoBase,
            validity: ActionValidity.CONDITION_NOT_MET,
          });
          continue;
        }
        result.push({
          ...actionInfoBase,
          targets: arg.targets,
          validity: ActionValidity.VALID,
        });
      }
    }

    const defaultTunedToType = elementOfCharacter(activeCh.definition);
    for (const card of player.hands) {
      let allTargets: InitiativeSkillEventArg[];
      const {
        disableTuning,
        willBeEffectless,
        shouldFast,
        requiredCost,
        tunedToType = defaultTunedToType,
      } = applyAttachmentModifications(state, card);
      const skillDef = playSkillOfCard(card.definition);
      if (skillDef) {
        const skillInfo = defineSkillInfo({
          caller: card,
          definition: skillDef,
        });
        const actionInfoBase = {
          type: "playCard" as const,
          who,
          skill: skillInfo,
          cost: requiredCost,
          fast: shouldFast,
          autoSelectedDice: [],
          targets: [],
          willBeEffectless,
        };
        if (
          card.definition.type === "support" &&
          player.supports.length === state.config.maxSupportsCount
        ) {
          allTargets = player.supports.map((st) => ({ targets: [st] }));
        } else {
          allTargets = skillDef.initiativeSkillConfig.getTarget(state, skillInfo);
        }
        if (allTargets.length === 0) {
          result.push({
            ...actionInfoBase,
            validity: ActionValidity.NO_TARGET,
          });
        } else {
          for (const arg of allTargets) {
            if (!skillDef.filter(state, skillInfo, arg)) {
              result.push({
                ...actionInfoBase,
                validity: ActionValidity.CONDITION_NOT_MET,
              });
              continue;
            }
            result.push({
              ...actionInfoBase,
              targets: arg.targets,
              validity: ActionValidity.VALID,
            });
          }
        }
      } else {
        console?.warn(`Card ${card.definition.id} has no play skill defined.`);
      }
      result.push({
        type: "elementalTuning",
        card,
        who,
        result: tunedToType,
        fast: true,
        cost: new Map([[DiceType.Void, 1]]),
        autoSelectedDice: [],
        validity: disableTuning ? ActionValidity.DISABLED : ActionValidity.VALID,
      });
    }

    result.push(
      ...player.characters
        .filter((ch) => ch.variables.alive && ch.id !== activeCh.id)
        .map((ch) => ({
          type: "switchActive" as const,
          who,
          from: activeCh,
          to: ch,
          fromReaction: false,
          fast: false,
          cost: new Map([[DiceType.Void, 1]]),
          autoSelectedDice: [],
          validity: ActionValidity.VALID,
        })),
    );

    result.push({
      type: "declareEnd",
      who,
      fast: false,
      cost: new Map(),
      autoSelectedDice: [],
      validity: ActionValidity.VALID,
    });

    const playerConfig = {
      alwaysOmni: false,
      allowTuningAnyDice: false,
    };
    if (fastMode) {
      return result
        .map((a) =>
          applyAutoSelectedDiceToAction(
            a as unknown as ActionInfoWithModification,
            player,
            playerConfig,
          ),
        )
        .map(exposeAction);
    }
    const skipError = (
      ["toleratePreview", "skipPhase"] as ErrorLevel[]
    ).includes(state.config.errorLevel);
    const previewer = new ActionPreviewer(state, who, skipError);
    return result
      .map((a) => previewer.modifyAndPreviewSync(a))
      .map((a) =>
        applyAutoSelectedDiceToAction(
          a as ActionInfoWithModification,
          player,
          playerConfig,
        ),
      )
      .map(exposeAction);
  }

  private static advanceToActionPhase(
    mutator: StateMutator,
    decisionProvider?: RuleEngineDecisionProvider,
  ): void {
    while (mutator.state.phase !== "action" && mutator.state.phase !== "gameEnd") {
      switch (mutator.state.phase) {
        case "initHands":
          this.runInitHands(mutator, decisionProvider);
          break;
        case "initActives":
          this.runInitActives(mutator, decisionProvider);
          break;
        case "roll":
          this.runRollPhase(mutator, decisionProvider);
          break;
        case "end":
          this.runEndPhase(mutator, decisionProvider);
          break;
        default:
          throw new GiTcgCoreInternalError(
            `RuleEngine bootstrap does not support phase ${mutator.state.phase}`,
          );
      }
    }
  }

  private static runInitHands(
    mutator: StateMutator,
    decisionProvider?: RuleEngineDecisionProvider,
  ): void {
    for (const who of [0, 1] as const) {
      const events = mutator.drawCardsPlain(
        who,
        mutator.state.config.initialHandsCount,
      );
      this.drainEvents(mutator, [...events], decisionProvider);
    }
    mutator.mutate({
      type: "changePhase",
      newPhase: "initActives",
    });
  }

  private static runInitActives(
    mutator: StateMutator,
    decisionProvider?: RuleEngineDecisionProvider,
  ): void {
    const chosen: [CharacterState, CharacterState] = [
      this.chooseInitialActive(mutator, 0, decisionProvider),
      this.chooseInitialActive(mutator, 1, decisionProvider),
    ];
    mutator.postChooseActive(chosen[0], chosen[1]);
    for (const who of [0, 1] as const) {
      const events = mutator.switchActive(who, chosen[who], { fast: null });
      this.drainEvents(mutator, [...events], decisionProvider);
    }
    this.drainEvents(
      mutator,
      [["onBattleBegin", new EventArg(mutator.state)]],
      decisionProvider,
    );
    mutator.mutate({
      type: "changePhase",
      newPhase: "roll",
    });
    mutator.mutate({
      type: "stepRound",
    });
  }

  private static runRollPhase(
    mutator: StateMutator,
    decisionProvider?: RuleEngineDecisionProvider,
  ): void {
    this.drainEvents(
      mutator,
      [["onRoundBegin", new EventArg(mutator.state)]],
      decisionProvider,
    );
    const rollParams: Array<{ fixed: readonly DiceType[]; rerollCount: number }> = [];
    for (const who of [0, 1] as const) {
      const rollModifier = new ModifyRollEventArg(mutator.state, who);
      this.drainEvents(
        mutator,
        [["modifyRoll", rollModifier]],
        decisionProvider,
      );
      rollParams[who] = {
        fixed: rollModifier._fixedDice,
        rerollCount: 1 + rollModifier._extraRerollCount,
      };
    }
    for (const who of [0, 1] as const) {
      const { fixed, rerollCount } = rollParams[who];
      const player = mutator.state.players[who];
      const initDice = sortDice(player, [
        ...fixed,
        ...mutator.randomDice(
          Math.max(0, mutator.state.config.initialDiceCount - fixed.length),
        ),
      ]);
      mutator.mutate({
        type: "resetDice",
        who,
        value: initDice,
        reason: "roll",
      });
      this.applyRerolls(mutator, who, rerollCount, decisionProvider);
    }
    mutator.mutate({
      type: "changePhase",
      newPhase: "action",
    });
    this.drainEvents(
      mutator,
      [["onActionPhase", new EventArg(mutator.state)]],
      decisionProvider,
    );
  }

  private static runEndPhase(
    mutator: StateMutator,
    decisionProvider?: RuleEngineDecisionProvider,
  ): void {
    this.drainEvents(
      mutator,
      [["onEndPhase", new EventArg(mutator.state)]],
      decisionProvider,
    );
    for (const who of [mutator.state.currentTurn, flip(mutator.state.currentTurn)]) {
      const events = mutator.drawCardsPlain(who, 2);
      this.drainEvents(mutator, [...events], decisionProvider);
    }
    this.drainEvents(
      mutator,
      [["onRoundEnd", new EventArg(mutator.state)]],
      decisionProvider,
    );
    for (const who of [0, 1] as const) {
      for (const flagName of [
        "hasDefeated",
        "canPlunging",
        "declaredEnd",
      ] as const) {
        mutator.mutate({
          type: "setPlayerFlag",
          who,
          flagName,
          value: false,
        });
      }
      mutator.mutate({
        type: "clearRoundSkillLog",
        who,
      });
    }
    mutator.mutate({
      type: "stepRound",
    });
    if (mutator.state.roundNumber >= mutator.state.config.maxRoundsCount) {
      mutator.mutate({
        type: "changePhase",
        newPhase: "gameEnd",
      });
      return;
    }
    mutator.mutate({
      type: "changePhase",
      newPhase: "roll",
    });
  }

  private static applyRerolls(
    mutator: StateMutator,
    who: 0 | 1,
    rerollCount: number,
    decisionProvider?: RuleEngineDecisionProvider,
  ): void {
    for (let i = 0; i < rerollCount; i++) {
      const currentDice = [...mutator.state.players[who].dice];
      const rerollCountLeft = rerollCount - i;
      const diceToReroll =
        decisionProvider?.rerollDice?.(
          mutator.state,
          who,
          currentDice,
          rerollCountLeft,
        ) ?? [];
      if (diceToReroll.length === 0) {
        return;
      }
      const remainingDice = [...currentDice];
      for (const dice of diceToReroll) {
        const index = remainingDice.indexOf(dice);
        if (index < 0) {
          throw new GiTcgDataError(
            `Invalid reroll dice for player ${who}: ${dice}`,
          );
        }
        remainingDice.splice(index, 1);
      }
      mutator.mutate({
        type: "resetDice",
        who,
        value: sortDice(mutator.state.players[who], [
          ...remainingDice,
          ...mutator.randomDice(diceToReroll.length),
        ]),
        reason: "roll",
      });
    }
  }

  private static chooseInitialActive(
    mutator: StateMutator,
    who: 0 | 1,
    decisionProvider?: RuleEngineDecisionProvider,
  ): CharacterState {
    const player = mutator.state.players[who];
    const candidates = player.characters.filter(
      (ch) => ch.variables.alive && ch.id !== player.activeCharacterId,
    );
    if (candidates.length === 0) {
      throw new GiTcgDataError(
        `No available initial active character for player ${who}`,
      );
    }
    const pickedId =
      decisionProvider?.chooseActive?.(
        mutator.state,
        who,
        candidates.map((c) => c.id),
      ) ?? candidates[0].id;
    const target = candidates.find((c) => c.id === pickedId);
    if (!target) {
      throw new GiTcgDataError(
        `Invalid initial active decision for player ${who}: ${pickedId}`,
      );
    }
    return target;
  }

  private static applyInitialAction(
    mutator: StateMutator,
    action: Action,
    pendingEvents: EventAndRequest[],
    decisionProvider?: RuleEngineDecisionProvider,
    skipActionPhasePrelude = false,
  ): void {
    if (action.validity !== ActionValidity.VALID) {
      throw new GiTcgDataError(
        `RuleEngine only accepts valid actions, got ${ActionValidity[action.validity]}`,
      );
    }
    const actionPayload = action.action;
    if (!actionPayload) {
      throw new GiTcgDataError("Action payload is empty");
    }

    const who = mutator.state.currentTurn;
    if (!skipActionPhasePrelude) {
      mutator.mutate({
        type: "setPlayerFlag",
        who,
        flagName: "canCharged",
        value: mutator.state.players[who].dice.length % 2 === 0,
      });

      // Keep action-phase semantics: onBeforeAction must be settled before cost/action execution.
      this.drainEvents(
        mutator,
        [["onBeforeAction", new PlayerEventArg(mutator.state, who)]],
        decisionProvider,
      );
    }

    const selectedDice = action.autoSelectedDice
      .filter((d) => d !== 0)
      .map((d) => d as DiceType);

    switch (actionPayload.$case) {
      case "switchActive": {
        const { characterId } = actionPayload.value;
        const to = getEntityById(mutator.state, characterId);
        if (!to || to.definition.type !== "character") {
          throw new GiTcgDataError(`Invalid switch target id ${characterId}`);
        }
        const toCharacter = to as CharacterState;
        const player = mutator.state.players[who];
        const from = player.characters[getActiveCharacterIndex(player)];
        if (!toCharacter.variables.alive || toCharacter.id === from.id) {
          throw new GiTcgDataError(`Invalid switch target id ${characterId}`);
        }
        const details = this.applyActionModifiers(
          mutator,
          {
            type: "switchActive",
            who,
            from,
            to: toCharacter,
            fromReaction: false,
            cost: new Map([[DiceType.Void, 1]]),
            fast: false,
            validity: ActionValidity.VALID,
            autoSelectedDice: selectedDice,
          },
          decisionProvider,
        );
        this.consumeActionCost(
          mutator,
          who,
          details.cost,
          selectedDice,
          player.characters[getActiveCharacterIndex(player)],
        );
        pendingEvents.push(
          ...mutator.switchActive(who, toCharacter, { fast: details.fast }),
        );
        const actionInfo: ActionInfo = {
          type: "switchActive",
          who,
          from,
          to: toCharacter,
          fromReaction: false,
          ...details,
        };
        pendingEvents.push(["onAction", new ActionEventArg(mutator.state, actionInfo)]);
        if (!details.fast) {
          mutator.mutate({
            type: "switchTurn",
          });
        }
        break;
      }
      case "declareEnd": {
        const details = this.applyActionModifiers(
          mutator,
          {
            type: "declareEnd",
            who,
            cost: new Map(),
            fast: false,
            validity: ActionValidity.VALID,
            autoSelectedDice: selectedDice,
          },
          decisionProvider,
        );
        this.consumeActionCost(
          mutator,
          who,
          details.cost,
          selectedDice,
          mutator.state.players[who].characters[
            getActiveCharacterIndex(mutator.state.players[who])
          ],
        );
        mutator.mutate({
          type: "setPlayerFlag",
          who,
          flagName: "declaredEnd",
          value: true,
        });
        const actionInfo: ActionInfo = {
          type: "declareEnd",
          who,
          ...details,
        };
        pendingEvents.push(["onAction", new ActionEventArg(mutator.state, actionInfo)]);
        if (!details.fast) {
          mutator.mutate({
            type: "switchTurn",
          });
        }
        break;
      }
      case "useSkill": {
        const { skillDefinitionId, targetIds, mainDamageTargetId } =
          actionPayload.value;
        const player = mutator.state.players[who];
        const active = player.characters[getActiveCharacterIndex(player)];
        if (isSkillDisabled(active)) {
          throw new GiTcgDataError(
            `Skill ${skillDefinitionId} is disabled on current active character`,
          );
        }
        const skillAndCaller = initiativeSkillsOfPlayer(player).find(
          ({ skill }) => skill.id === skillDefinitionId,
        );
        if (!skillAndCaller) {
          throw new GiTcgDataError(`Skill ${skillDefinitionId} not available`);
        }
        const { caller, skill } = skillAndCaller;
        const { charged, plunging } = isChargedPlunging(skill, player);
        const skillInfo = defineSkillInfo({
          caller,
          definition: skill,
          charged,
          plunging,
        });
        const targets = this.resolveTargets(mutator.state, targetIds);
        const details = this.applyActionModifiers(
          mutator,
          {
            type: "useSkill",
            who,
            skill: skillInfo,
            targets,
            mainDamageTargetId: mainDamageTargetId ?? undefined,
            cost: new Map(skill.initiativeSkillConfig.requiredCost),
            fast: skill.initiativeSkillConfig.shouldFast,
            validity: ActionValidity.VALID,
            autoSelectedDice: selectedDice,
          },
          decisionProvider,
        );
        this.consumeActionCost(
          mutator,
          who,
          details.cost,
          selectedDice,
          mutator.state.players[who].characters[
            getActiveCharacterIndex(mutator.state.players[who])
          ],
        );
        const callerArea = getEntityArea(mutator.state, caller.id);
        this.drainEvents(
          mutator,
          [["onBeforeUseSkill", new UseSkillEventArg(mutator.state, callerArea, skillInfo)]],
          decisionProvider,
        );
        pendingEvents.push(
          ...SkillExecutor.executeSkillSync(mutator, skillInfo, { targets }),
        );
        pendingEvents.push([
          "onUseSkill",
          new UseSkillEventArg(mutator.state, callerArea, skillInfo),
        ]);
        const actionInfo: ActionInfo = {
          type: "useSkill",
          who,
          skill: skillInfo,
          targets,
          mainDamageTargetId: mainDamageTargetId ?? undefined,
          ...details,
        };
        pendingEvents.push(["onAction", new ActionEventArg(mutator.state, actionInfo)]);
        if (!details.fast) {
          mutator.mutate({
            type: "switchTurn",
          });
        }
        break;
      }
      case "playCard": {
        const { cardId, cardDefinitionId, targetIds, willBeEffectless } =
          actionPayload.value;
        const player = mutator.state.players[who];
        const card = player.hands.find((h) => h.id === cardId);
        if (!card) {
          throw new GiTcgDataError(`Card ${cardId} not in hands`);
        }
        if (card.definition.id !== cardDefinitionId) {
          throw new GiTcgDataError(
            `Card definition mismatch: expected ${cardDefinitionId}, got ${card.definition.id}`,
          );
        }
        const skillDef = playSkillOfCard(card.definition);
        if (!skillDef) {
          throw new GiTcgDataError(`Card ${cardDefinitionId} has no play skill`);
        }
        const skillInfo = defineSkillInfo({
          caller: card,
          definition: skillDef,
        });
        const targets = this.resolveTargets(mutator.state, targetIds);
        const {
          requiredCost,
          shouldFast,
        } = applyAttachmentModifications(mutator.state, card);
        const details = this.applyActionModifiers(
          mutator,
          {
            type: "playCard",
            who,
            skill: skillInfo,
            targets,
            willBeEffectless,
            cost: new Map(requiredCost),
            fast: shouldFast,
            validity: ActionValidity.VALID,
            autoSelectedDice: selectedDice,
          },
          decisionProvider,
        );
        this.consumeActionCost(
          mutator,
          who,
          details.cost,
          selectedDice,
          mutator.state.players[who].characters[
            getActiveCharacterIndex(mutator.state.players[who])
          ],
        );
        const actionInfo: ActionInfo = {
          type: "playCard",
          who,
          skill: skillInfo,
          targets,
          willBeEffectless,
          ...details,
        };
        if (card.definition.tags.includes("legend")) {
          mutator.mutate({
            type: "setPlayerFlag",
            who,
            flagName: "legendUsed",
            value: true,
          });
        }
        this.drainEvents(
          mutator,
          [["onBeforePlayCard", new PlayCardEventArg(mutator.state, actionInfo)]],
          decisionProvider,
        );
        if (willBeEffectless) {
          mutator.mutate({
            type: "removeEntity",
            from: { who, type: "hands", cardId: card.id },
            oldState: card,
            reason: "eventCardPlayNoEffect",
          });
        } else {
          pendingEvents.push(
            ...SkillExecutor.executeSkillSync(mutator, skillInfo, { targets }),
          );
        }
        pendingEvents.push(["onPlayCard", new PlayCardEventArg(mutator.state, actionInfo)]);
        pendingEvents.push(["onAction", new ActionEventArg(mutator.state, actionInfo)]);
        if (!details.fast) {
          mutator.mutate({
            type: "switchTurn",
          });
        }
        break;
      }
      case "elementalTuning": {
        const { removedCardId, targetDice } = actionPayload.value;
        const player = mutator.state.players[who];
        const card = player.hands.find((h) => h.id === removedCardId);
        if (!card) {
          throw new GiTcgDataError(`Card ${removedCardId} not in hands`);
        }
        const details = this.applyActionModifiers(
          mutator,
          {
            type: "elementalTuning",
            who,
            card,
            result: targetDice as DiceType,
            cost: new Map([[DiceType.Void, 1]]),
            fast: true,
            validity: ActionValidity.VALID,
            autoSelectedDice: selectedDice,
          },
          decisionProvider,
        );
        this.consumeActionCost(
          mutator,
          who,
          details.cost,
          selectedDice,
          mutator.state.players[who].characters[
            getActiveCharacterIndex(mutator.state.players[who])
          ],
        );
        const actionInfo: ActionInfo = {
          type: "elementalTuning",
          who,
          card,
          result: targetDice as DiceType,
          ...details,
        };
        const tuneCardEventArg = new DisposeEventArg(
          mutator.state,
          card,
          "elementalTuning",
          { who, type: "hands", cardId: card.id },
          null,
        );
        mutator.mutate({
          type: "removeEntity",
          from: { who, type: "hands", cardId: card.id },
          oldState: card,
          reason: "elementalTuning",
        });
        const updatedPlayer = mutator.state.players[who];
        mutator.mutate({
          type: "resetDice",
          who,
          value: sortDice(updatedPlayer, [...updatedPlayer.dice, targetDice as DiceType]),
          reason: "elementalTuning",
          conversionTargetHint: targetDice as DiceType,
        });
        pendingEvents.push(["onDispose", tuneCardEventArg]);
        pendingEvents.push(["onAction", new ActionEventArg(mutator.state, actionInfo)]);
        if (!details.fast) {
          mutator.mutate({
            type: "switchTurn",
          });
        }
        break;
      }
      default: {
        throw new GiTcgCoreInternalError("RuleEngine action is not supported");
      }
    }

    if (
      mutator.state.players[0].declaredEnd &&
      mutator.state.players[1].declaredEnd
    ) {
      mutator.mutate({
        type: "changePhase",
        newPhase: "end",
      });
    }
  }

  private static drainEvents(
    mutator: StateMutator,
    pendingEvents: EventAndRequest[],
    decisionProvider?: RuleEngineDecisionProvider,
  ) {
    while (pendingEvents.length > 0) {
      const event = pendingEvents.shift();
      if (!event) {
        continue;
      }
      const preprocessEvents = this.preprocessEvent(mutator, event);
      if (preprocessEvents.length > 0) {
        pendingEvents.unshift(...preprocessEvents);
      }
      const emitted = SkillExecutor.executeEventSync(mutator, event);
      pendingEvents.push(...emitted);
      const forcedSwitchEvents = this.resolveForcedSwitches(
        mutator,
        decisionProvider,
      );
      if (forcedSwitchEvents.length > 0) {
        pendingEvents.push(...forcedSwitchEvents);
      }
    }
  }

  private static consumeActionCost(
    mutator: StateMutator,
    who: 0 | 1,
    cost: ReadonlyMap<DiceType, number>,
    preferredDice: readonly DiceType[],
    activeCharacter: CharacterState,
  ): void {
    const player = mutator.state.players[who];
    let selected = [...preferredDice];
    if (!checkDice(cost, selected)) {
      selected = chooseDiceValue(cost, player.dice);
    }
    if (!checkDice(cost, selected)) {
      throw new GiTcgDataError("Selected dice doesn't meet requirement");
    }
    const operatingDice = [...player.dice];
    for (const type of selected) {
      const index = operatingDice.indexOf(type);
      if (index === -1) {
        throw new GiTcgDataError(`Selected dice ${type} not in player's pool`);
      }
      operatingDice.splice(index, 1);
    }
    mutator.mutate({
      type: "resetDice",
      who,
      value: operatingDice,
      reason: "consume",
    });

    const requiredEnergy = cost.get(DiceType.Energy) ?? 0;
    if (requiredEnergy <= 0) {
      return;
    }
    const currentEnergy = activeCharacter.variables.energy;
    if (currentEnergy < requiredEnergy) {
      throw new GiTcgDataError("Active character does not have enough energy");
    }
    mutator.mutate({
      type: "modifyEntityVar",
      state: activeCharacter,
      varName: "energy",
      value: currentEnergy - requiredEnergy,
      direction: "decrease",
    });
  }

  private static applyActionModifiers(
    mutator: StateMutator,
    baseAction: ActionInfo,
    decisionProvider?: RuleEngineDecisionProvider,
  ): Pick<ActionInfo, "cost" | "fast" | "validity" | "autoSelectedDice"> {
    const eventArg = new GenericModifyActionEventArg(mutator.state, baseAction);
    this.drainEvents(
      mutator,
      [
        ["modifyAction0", eventArg],
        ["modifyAction1", eventArg],
        ["modifyAction2", eventArg],
        ["modifyAction3", eventArg],
      ],
      decisionProvider,
    );
    const modified = eventArg.action;
    return {
      cost: new Map(modified.cost),
      fast: modified.fast,
      validity: modified.validity,
      autoSelectedDice: modified.autoSelectedDice
        .filter((d) => d !== 0)
        .map((d) => d as DiceType),
    };
  }

  private static resolveTargets(
    state: GameState,
    targetIds: readonly number[],
  ): AnyState[] {
    return targetIds.map((id) => {
      const target = getEntityById(state, id);
      if (!target) {
        throw new GiTcgDataError(`Unknown target id ${id}`);
      }
      return target as AnyState;
    });
  }

  private static preprocessEvent(
    mutator: StateMutator,
    event: EventAndRequest,
  ): EventAndRequest[] {
    const [name, arg] = event;
    if (name !== "onDamageOrHeal" || !arg.isDamageTypeDamage()) {
      return [];
    }
    if (!arg.damageInfo.causeDefeated) {
      return [];
    }

    const zeroHealthEventArg = new ZeroHealthEventArg(
      arg.onTimeState,
      arg.damageInfo,
      arg.option,
    );
    if (checkImmune(mutator.state, zeroHealthEventArg)) {
      return [["modifyZeroHealth", zeroHealthEventArg]];
    }

    const character = getEntityById(mutator.state, arg.target.id) as CharacterState;
    if (character?.variables.alive) {
      const { who } = getEntityArea(mutator.state, character.id);
      mutator.mutate({
        type: "modifyEntityVar",
        state: character,
        varName: "alive",
        value: 0,
        direction: "decrease",
      });
      const energyVarName =
        character.definition.specialEnergy?.variableName ?? "energy";
      mutator.mutate({
        type: "modifyEntityVar",
        state: character,
        varName: energyVarName,
        value: 0,
        direction: "decrease",
      });
      mutator.mutate({
        type: "modifyEntityVar",
        state: character,
        varName: "aura",
        value: Aura.None,
        direction: null,
      });
      mutator.mutate({
        type: "setPlayerFlag",
        who,
        flagName: "hasDefeated",
        value: true,
      });
    }

    const failedPlayers = ([0, 1] as const).filter((who) => {
      const p = mutator.state.players[who];
      return p.characters.every((ch) => !ch.variables.alive);
    });
    if (failedPlayers.length === 2) {
      mutator.mutate({
        type: "changePhase",
        newPhase: "gameEnd",
      });
      return [];
    }
    if (failedPlayers.length === 1) {
      const who = failedPlayers[0];
      mutator.mutate({
        type: "changePhase",
        newPhase: "gameEnd",
      });
      mutator.mutate({
        type: "setWinner",
        winner: flip(who),
      });
    }
    return [];
  }

  private static resolveForcedSwitches(
    mutator: StateMutator,
    decisionProvider?: RuleEngineDecisionProvider,
  ): EventAndRequest[] {
    if (
      mutator.state.phase === "gameEnd" ||
      mutator.state.phase === "initHands" ||
      mutator.state.phase === "initActives"
    ) {
      return [];
    }

    const chosen: [CharacterState | null, CharacterState | null] = [null, null];
    const switchEventsByWho: [EventAndRequest[], EventAndRequest[]] = [[], []];

    for (const who of [0, 1] as const) {
      const player = mutator.state.players[who];
      if (!player.characters.some((ch) => ch.id === player.activeCharacterId)) {
        continue;
      }
      const active = player.characters[getActiveCharacterIndex(player)];
      if (active.variables.alive) {
        continue;
      }
      const candidates = player.characters.filter(
        (ch) => ch.variables.alive && ch.id !== active.id,
      );
      if (candidates.length === 0) {
        continue;
      }
      const pickedId =
        decisionProvider?.chooseActive?.(
          mutator.state,
          who,
          candidates.map((c) => c.id),
        ) ?? candidates[0].id;
      const target = candidates.find((c) => c.id === pickedId);
      if (!target) {
        throw new GiTcgDataError(
          `Invalid chooseActive decision for player ${who}: ${pickedId}`,
        );
      }
      chosen[who] = target;
      switchEventsByWho[who].push(...mutator.switchActive(who, target));
    }

    if (!chosen[0] && !chosen[1]) {
      return [];
    }
    mutator.postChooseActive(chosen[0], chosen[1]);
    const currentTurn = mutator.state.currentTurn;
    return [
      ...switchEventsByWho[currentTurn],
      ...switchEventsByWho[flip(currentTurn)],
    ];
  }

  static extractStagedPatches(mutator: StateMutator): StatePatch[] {
    return mutator.drainStagedPatches();
  }
}

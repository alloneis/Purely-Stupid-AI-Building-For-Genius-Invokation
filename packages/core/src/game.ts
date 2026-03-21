import { checkDice, flip } from "@gi-tcg/utils";
import {
  DiceType,
  type ExposedMutation,
  type RpcMethod,
  PbPlayerStatus,
  type RpcRequestPayloadOf,
  type RpcResponsePayloadOf,
  createRpcRequest,
  type PbExposedMutation,
  unFlattenOneof,
  ActionValidity,
  type Action,
  type Deck,
} from "@gi-tcg/typings";
import {
  getDefaultGameConfig,
  getVersionBehavior,
  StateSymbol,
  stringifyState,
  type EntityState,
  type AnyState,
  type CharacterState,
  type EntityDefinition,
  type ExtensionState,
  type GameConfig,
  type GameState,
  type PlayerState,
  type VersionBehavior,
} from "./base/state";
import { CURRENT_VERSION, type Version } from "./base/version";
import type { Mutation } from "./base/mutation";
import {
  type IoErrorHandler,
  type PauseHandler,
  exposeMutation,
  exposeState,
  verifyRpcResponse,
} from "./io";
import {
  getActiveCharacterIndex,
  findReplaceAction,
  shuffle,
  sortDice,
  type Writable,
} from "./utils";
import type { GameData } from "./builder/registry";
import {
  type EventAndRequest,
  EventArg,
  ModifyRollEventArg,
  PlayerEventArg,
  type SkillInfo,
} from "./base/skill";
import { executeQueryOnState } from "./query";
import {
  GiTcgCoreInternalError,
  GiTcgDataError,
  GiTcgError,
  GiTcgIoError,
} from "./error";
import { DetailLogType, DetailLogger } from "./log";
import { type GeneralSkillArg, SkillExecutor } from "./skill_executor";
import {
  type InternalNotifyOption,
  type InternalPauseOption,
  type MutatorConfig,
  StateMutator,
} from "./mutator";
import { Player } from "./player";
import type { CharacterDefinition } from "./base/character";
import { RuleEngine } from "./decoupled/rule_engine";

export interface DeckConfig extends Deck {
  noShuffle?: boolean;
}

/** A helper class to iterate id when constructing initial state */
class IdIter {
  static readonly INITIAL_ID = -500000;
  #id = IdIter.INITIAL_ID;
  get id() {
    return this.#id--;
  }
}

/** Initialize player state from deck */
function initPlayerState(
  who: 0 | 1,
  data: GameData,
  deck: DeckConfig,
  idIter: IdIter,
): PlayerState {
  let initialPile: readonly EntityDefinition[] = deck.cards.map((id) => {
    const def = data.entities.get(id);
    if (typeof def === "undefined") {
      throw new GiTcgDataError(`Unknown card id ${id}`);
    }
    return def;
  });
  const characterDefs: readonly CharacterDefinition[] = deck.characters.map(
    (id) => {
      const def = data.characters.get(id);
      if (typeof def === "undefined") {
        throw new GiTcgDataError(`Unknown character id ${id}`);
      }
      return def;
    },
  );
  if (!deck.noShuffle) {
    initialPile = shuffle(initialPile);
  }
  // Put legend cards at the front.
  const compFn = (def: EntityDefinition) => {
    if (def.tags.includes("legend")) {
      return 0;
    } else {
      return 1;
    }
  };
  initialPile = initialPile.toSortedBy(compFn);
  const characters: CharacterState[] = [];
  const pile: EntityState[] = [];
  for (const definition of characterDefs) {
    characters.push({
      [StateSymbol]: "character",
      id: idIter.id,
      definition,
      variables: Object.fromEntries(
        Object.entries(definition.varConfigs).map(
          ([name, { initialValue }]) => [name, initialValue],
        ),
      ) as any,
      entities: [],
    });
  }
  for (const definition of initialPile) {
    pile.push({
      [StateSymbol]: "entity",
      id: idIter.id,
      definition,
      variables: Object.fromEntries(
        Object.entries(definition.varConfigs).map(
          ([name, { initialValue }]) => [name, initialValue],
        ),
      ),
      attachments: [],
    });
  }
  return {
    [StateSymbol]: "player",
    who,
    activeCharacterId: 0,
    characters,
    initialPile,
    pile,
    hands: [],
    dice: [],
    combatStatuses: [],
    summons: [],
    supports: [],
    declaredEnd: false,
    canCharged: false,
    canPlunging: false,
    hasDefeated: false,
    legendUsed: false,
    skipNextTurn: false,
    roundSkillLog: new Map(),
    removedEntities: [],
  };
}

export interface CreateInitialStateConfig
  extends Writable<Partial<GameConfig>> {
  versionBehavior?: VersionBehavior | Version;
  decks: readonly [DeckConfig, DeckConfig];
  data: GameData;
}

export class Game {
  private readonly logger: DetailLogger;

  private _terminated = false;
  private finishResolvers: PromiseWithResolvers<0 | 1 | null> | null = null;

  private readonly mutator: StateMutator;
  private readonly mutatorConfig: MutatorConfig;
  readonly players: readonly [Player, Player];

  public onPause: PauseHandler | null = null;
  public onIoError: IoErrorHandler | null = null;

  constructor(initialState: GameState) {
    this.logger = new DetailLogger();
    this.mutatorConfig = {
      logger: this.logger,
      onNotify: (opt) => this.mutatorNotifyHandler(opt),
      onPause: (opt) => this.mutatorPauseHandler(opt),
      howToChooseActive: (who, chs) => this.rpcChooseActive(who, chs),
      howToReroll: (who) => this.rpcReroll(who),
      howToSwitchHands: (who) => this.rpcSwitchHands(who),
      howToSelectCard: (who, cards) => this.rpcSelectCard(who, cards),
    };
    this.mutator = new StateMutator(initialState, this.mutatorConfig);
    this.players = [new Player(), new Player()];
    // this.initPlayerCards(0);
    // this.initPlayerCards(1);
  }

  static createInitialState(opt: CreateInitialStateConfig): GameState {
    const {
      decks,
      data,
      versionBehavior = CURRENT_VERSION,
      ...partialConfig
    } = opt;
    const extensions = data.extensions
      .values()
      .map<ExtensionState>((v) => ({
        [StateSymbol]: "extension",
        definition: v,
        state: v.initialState,
      }))
      .toArray();
    const config = mergeGameConfigWithDefault(partialConfig);
    const behavior =
      typeof versionBehavior === "string"
        ? getVersionBehavior(versionBehavior)
        : versionBehavior;
    const idIter = new IdIter();
    const state: GameState = {
      [StateSymbol]: "game",
      data,
      config,
      versionBehavior: behavior,
      players: [
        initPlayerState(0, data, decks[0], idIter),
        initPlayerState(1, data, decks[1], idIter),
      ],
      iterators: {
        random: config.randomSeed,
        id: idIter.id,
      },
      phase: "initHands",
      currentTurn: 0,
      roundNumber: 0,
      winner: null,
      extensions,
      delayingEventArgs: [],
    };
    return state;
  }

  get config() {
    return this.state.config;
  }

  get detailLog() {
    return this.logger.getLogs();
  }

  get state() {
    return this.mutator.state;
  }

  mutate(mutation: Mutation) {
    if (!this._terminated) {
      this.mutator.mutate(mutation);
    }
  }

  query(who: 0 | 1, query: string): AnyState[] {
    return executeQueryOnState(this.state, who, query);
  }

  // private lastNotifiedState: [string, string] = ["", ""];
  private notifyOneImpl(who: 0 | 1, opt: InternalNotifyOption) {
    const playerIo = this.players[who].io;
    const stateMutations = opt.stateMutations
      .map((m) => exposeMutation(who, m))
      .filter((em): em is ExposedMutation => !!em);
    const state = exposeState(who, opt.state);
    state.player[0].status = this.players[0].status;
    state.player[1].status = this.players[1].status;
    const mutation: PbExposedMutation[] = [
      ...stateMutations,
      ...opt.exposedMutations,
    ].map((m) => ({
      mutation: unFlattenOneof<PbExposedMutation["mutation"]>(m),
    }));
    playerIo.notify({ mutation, state });
  }
  private notifyOne(who: 0 | 1, mutation?: ExposedMutation, state?: GameState) {
    this.notifyOneImpl(who, {
      state: state ?? this.state,
      canResume: false,
      stateMutations: [],
      exposedMutations: mutation ? [mutation] : [],
    });
  }

  private async mutatorNotifyHandler(opt: InternalNotifyOption) {
    if (this._terminated) {
      return;
    }
    for (const i of [0, 1] as const) {
      this.notifyOneImpl(i, opt);
    }
  }
  private async mutatorPauseHandler(opt: InternalPauseOption) {
    if (this._terminated) {
      return;
    }
    const { state, canResume, stateMutations } = opt;
    await this.onPause?.(state, [...stateMutations], canResume);
    if (state.phase === "gameEnd") {
      this.gotWinner(state.winner);
    }
  }

  private async tryPhase(fn: (this: Game) => Promise<void>) {
    try {
      await fn.call(this);
    } catch (e) {
      if (e instanceof GiTcgError && this.config.errorLevel === "skipPhase") {
        // skip.
      } else {
        throw e;
      }
    }
  }

  async start(): Promise<0 | 1 | null> {
    if (this.finishResolvers !== null) {
      throw new GiTcgCoreInternalError(
        `Game already started. Please use a new Game instance instead of start multiple time.`,
      );
    }
    this.finishResolvers = Promise.withResolvers();
    this.logger.clearLogs();
    (async () => {
      try {
        await this.mutator.notifyAndPause({ force: true, canResume: true });
        while (!this._terminated) {
          switch (this.state.phase) {
            case "initHands":
              await this.tryPhase(this.initHands);
              break;
            case "initActives":
              await this.tryPhase(this.initActives);
              break;
            case "roll":
              await this.tryPhase(this.rollPhase);
              break;
            case "action":
              await this.tryPhase(this.actionPhase);
              break;
            case "end":
              await this.tryPhase(this.endPhase);
              break;
            default:
              break;
          }
          this.mutate({ type: "clearRemovedEntities" });
          await this.mutator.notifyAndPause({ canResume: true });
        }
      } catch (e) {
        if (e instanceof GiTcgIoError) {
          this.onIoError?.(e);
          await this.gotWinner(flip(e.who));
        } else if (e instanceof GiTcgError) {
          this.finishResolvers?.reject(e);
        } else {
          let message = String(e);
          if (e instanceof Error) {
            message = e.message;
            if (e.stack) {
              message += "\n" + e.stack;
            }
          }
          this.finishResolvers?.reject(
            new GiTcgCoreInternalError(`Unexpected error: ` + message),
          );
        }
      }
    })();
    return this.finishResolvers.promise;
  }

  giveUp(who: 0 | 1) {
    this.gotWinner(flip(who));
  }

  /** Winner is decided, switch to `gameEnd` phase. */
  private async gotWinner(winner: 0 | 1 | null) {
    if (this.state.phase !== "gameEnd") {
      this.mutate({
        type: "changePhase",
        newPhase: "gameEnd",
      });
      if (winner !== null) {
        this.mutate({
          type: "setWinner",
          winner,
        });
      }
      await this.mutator.notifyAndPause();
    }
    this.finishResolvers?.resolve(winner);
    if (!this._terminated) {
      this._terminated = true;
      Object.freeze(this);
    }
  }
  /** Force terminate the game and stop further mutations. */
  terminate() {
    this.finishResolvers?.reject(
      new GiTcgCoreInternalError("User call terminate."),
    );
    if (!this._terminated) {
      this._terminated = true;
      Object.freeze(this);
    }
  }

  private setPlayerStatus(who: 0 | 1, method: RpcMethod | null) {
    let status;
    switch (method) {
      case "action":
        status = PbPlayerStatus.ACTING;
        break;
      case "chooseActive":
        status = PbPlayerStatus.CHOOSING_ACTIVE;
        break;
      case "selectCard":
        status = PbPlayerStatus.SELECTING_CARDS;
        break;
      case "rerollDice":
        status = PbPlayerStatus.REROLLING;
        break;
      case "switchHands":
        status = PbPlayerStatus.SWITCHING_HANDS;
        break;
      default:
        status = PbPlayerStatus.UNSPECIFIED;
    }
    // @ts-expect-error writing private props
    this.players[who]._status = status;
    this.mutator.notify({
      mutations: [
        {
          $case: "playerStatusChange",
          who,
          status,
        },
      ],
    });
  }

  private async rpc<M extends RpcMethod>(
    who: 0 | 1,
    method: M,
    request: RpcRequestPayloadOf<M>,
  ): Promise<RpcResponsePayloadOf<M>> {
    if (this._terminated) {
      throw new GiTcgCoreInternalError(`Game has been terminated`);
    }
    try {
      this.setPlayerStatus(who, method);
      const resp = await this.players[who].io.rpc(
        createRpcRequest(method, request as any),
      );
      const { value } = resp.response!;
      verifyRpcResponse(method, value);
      return value;
    } catch (e) {
      if (e instanceof Error) {
        throw new GiTcgIoError(who, e.message, { cause: e?.cause });
      } else {
        throw new GiTcgIoError(who, String(e));
      }
    } finally {
      this.setPlayerStatus(who, null);
    }
  }

  private async initHands() {
    using l = this.mutator.subLog(DetailLogType.Phase, `In initHands phase:`);
    for (const who of [0, 1] as const) {
      const events = this.mutator.drawCardsPlain(
        who,
        this.config.initialHandsCount,
      );
      await this.handleEvents(events);
    }
    await this.mutator.notifyAndPause();
    const events = (
      await Promise.all([
        this.mutator.switchHands(0),
        this.mutator.switchHands(1),
      ])
    ).flat(1);
    await this.handleEvents(events);
    this.mutate({
      type: "changePhase",
      newPhase: "initActives",
    });
  }

  private async initActives() {
    using l = this.mutator.subLog(DetailLogType.Phase, `In initActive phase:`);
    const [a0, a1] = await Promise.all([
      this.mutator.chooseActive(0),
      this.mutator.chooseActive(1),
    ]);
    this.mutator.postChooseActive(a0, a1);
    await this.switchActive(0, a0, null);
    await this.switchActive(1, a1, null);
    await this.handleEvent("onBattleBegin", new EventArg(this.state));
    this.mutate({
      type: "changePhase",
      newPhase: "roll",
    });
    this.mutate({
      type: "stepRound",
    });
  }

  private async rpcChooseActive(
    who: 0 | 1,
    candidateIds: number[],
  ): Promise<number> {
    const { activeCharacterId } = await this.rpc(who, "chooseActive", {
      candidateIds,
    });
    if (!candidateIds.includes(activeCharacterId)) {
      throw new GiTcgIoError(
        who,
        `Invalid active character id ${activeCharacterId}`,
      );
    }
    this.notifyOne;
    return activeCharacterId;
  }

  private async rollPhase() {
    using l = this.mutator.subLog(
      DetailLogType.Phase,
      `In roll phase (round ${this.state.roundNumber}):`,
    );
    await this.handleEvent("onRoundBegin", new EventArg(this.state));
    // onRoll event
    interface RollParams {
      fixed: readonly DiceType[];
      count: number;
    }
    const rollParams: RollParams[] = [];
    for (const who of [0, 1] as const) {
      const rollModifier = new ModifyRollEventArg(this.state, who);
      await this.handleEvent("modifyRoll", rollModifier);
      rollParams.push({
        fixed: rollModifier._fixedDice,
        count: 1 + rollModifier._extraRerollCount,
      });
    }

    await Promise.all(
      ([0, 1] as const).map(async (who) => {
        const { fixed, count } = rollParams[who];
        const initDice = sortDice(this.state.players[who], [
          ...fixed,
          ...this.mutator.randomDice(
            Math.max(0, this.config.initialDiceCount - fixed.length),
            this.players[who].config.alwaysOmni,
          ),
        ]);
        this.mutate({
          type: "resetDice",
          who,
          value: initDice,
          reason: "roll",
        });
        this.notifyOne(who);
        await this.mutator.reroll(who, count);
      }),
    );
    this.mutate({
      type: "changePhase",
      newPhase: "action",
    });
    await this.handleEvent("onActionPhase", new EventArg(this.state));
  }
  private async actionPhase() {
    const who = this.state.currentTurn;
    const player = () => this.state.players[who];
    this.mutate({
      type: "setPlayerFlag",
      who,
      flagName: "canCharged",
      value: player().dice.length % 2 === 0,
    });
    if (player().declaredEnd) {
      this.mutate({
        type: "switchTurn",
      });
    } else if (player().skipNextTurn) {
      this.mutate({
        type: "setPlayerFlag",
        who,
        flagName: "skipNextTurn",
        value: false,
      });
      this.mutate({
        type: "switchTurn",
      });
    } else {
      using l = this.mutator.subLog(
        DetailLogType.Phase,
        `In action phase (round ${this.state.roundNumber}, turn ${this.state.currentTurn}):`,
      );
      await this.handleEvent(
        "onBeforeAction",
        new PlayerEventArg(this.state, who),
      );
      const replaceActionEventArg = new PlayerEventArg(this.state, who);
      const replacedSkill = findReplaceAction(
        this.state,
        replaceActionEventArg,
      );
      if (replacedSkill) {
        this.mutator.log(
          DetailLogType.Other,
          `Found replaced action from ${stringifyState(replacedSkill.caller)}`,
        );
        await this.executeSkill(replacedSkill, replaceActionEventArg);
        this.mutate({
          type: "switchTurn",
        });
      } else {
        const actions = this.availableActions();
        const { chosenActionIndex, usedDice } = await this.rpc(who, "action", {
          action: actions,
        });
        if (chosenActionIndex < 0 || chosenActionIndex >= actions.length) {
          throw new GiTcgIoError(who, `User chosen index out of range`);
        }
        const action = actions[chosenActionIndex];
        if (action.validity !== ActionValidity.VALID) {
          throw new GiTcgIoError(
            who,
            `User-selected action is invalid: ${
              ActionValidity[action.validity]
            }`,
          );
        }

        const selectedDiceForCheck = usedDice as unknown as DiceType[];
        const requiredCost = new Map<DiceType, number>();
        for (const req of action.requiredCost) {
          requiredCost.set(req.type as DiceType, req.count);
        }
        if (!checkDice(requiredCost, selectedDiceForCheck)) {
          throw new GiTcgIoError(
            who,
            `Selected dice doesn't meet requirement:\nRequired: ${JSON.stringify(
              Object.fromEntries(requiredCost.entries()),
            )}, selected: ${JSON.stringify(usedDice)}`,
          );
        }
        if (
          !this.players[who].config.allowTuningAnyDice &&
          action.action?.$case === "elementalTuning" &&
          (selectedDiceForCheck[0] === DiceType.Omni ||
            selectedDiceForCheck[0] === action.action.value.targetDice)
        ) {
          throw new GiTcgIoError(
            who,
            `Elemental tuning cannot use omni dice or active character's element`,
          );
        }

        const actionWithDice = {
          ...action,
          autoSelectedDice: usedDice,
        };
        const { nextState, patches } = RuleEngine.execute(
          this.state,
          actionWithDice,
          {
            skipActionPhasePrelude: true,
          },
        );
        this.mutator.resetState(
          nextState,
          {
            stateMutations: [],
            exposedMutations: patches,
          },
          {
            deferNotify: true,
          },
        );
      }
    }
    if (
      this.state.players[0].declaredEnd &&
      this.state.players[1].declaredEnd
    ) {
      this.mutate({
        type: "changePhase",
        newPhase: "end",
      });
    }
  }
  private async endPhase() {
    using l = this.mutator.subLog(
      DetailLogType.Phase,
      `In end phase (round ${this.state.roundNumber}, turn ${this.state.currentTurn}):`,
    );
    await this.handleEvent("onEndPhase", new EventArg(this.state));
    for (const who of [this.state.currentTurn, flip(this.state.currentTurn)]) {
      const events = this.mutator.drawCardsPlain(who, 2);
      await this.handleEvents(events);
    }
    await this.handleEvent("onRoundEnd", new EventArg(this.state));
    for (const who of [0, 1] as const) {
      for (const flagName of [
        "hasDefeated",
        "canPlunging",
        "declaredEnd",
      ] as const) {
        this.mutate({
          type: "setPlayerFlag",
          who,
          flagName,
          value: false,
        });
      }
      this.mutate({
        type: "clearRoundSkillLog",
        who,
      });
    }
    this.mutate({
      type: "stepRound",
    });
    if (this.state.roundNumber >= this.config.maxRoundsCount) {
      this.gotWinner(null);
    } else {
      this.mutate({
        type: "changePhase",
        newPhase: "roll",
      });
    }
  }

  static getPossibleActions(state: GameState): Action[] {
    return RuleEngine.getPossibleActions(state, {
      fastMode: false,
    });
  }

  availableActions(): Action[] {
    return Game.getPossibleActions(this.state);
  }
  private async rpcSwitchHands(who: 0 | 1) {
    const { removedHandIds } = await this.rpc(who, "switchHands", {});
    this.notifyOne(who, {
      $case: "switchHandsDone",
      who,
      count: removedHandIds.length,
    });
    this.notifyOne(flip(who), {
      $case: "switchHandsDone",
      who,
      count: removedHandIds.length,
    });
    return removedHandIds;
  }

  private async rpcReroll(who: 0 | 1) {
    const { diceToReroll } = await this.rpc(who, "rerollDice", {});
    this.notifyOne(who, {
      $case: "rerollDone",
      who,
      count: diceToReroll.length,
    });
    this.notifyOne(flip(who), {
      $case: "rerollDone",
      who,
      count: 0,
    });
    return diceToReroll;
  }

  private async rpcSelectCard(who: 0 | 1, candidateDefinitionIds: number[]) {
    const { selectedDefinitionId } = await this.rpc(who, "selectCard", {
      candidateDefinitionIds,
    });
    if (!candidateDefinitionIds.includes(selectedDefinitionId)) {
      throw new GiTcgIoError(who, `Selected card not in candidates`);
    }
    this.notifyOne(who, {
      $case: "selectCardDone",
      who,
      selectedDefinitionId,
    });
    this.notifyOne(flip(who), {
      $case: "selectCardDone",
      who,
      selectedDefinitionId: 0,
    });
    return selectedDefinitionId;
  }

  private async switchActive(
    who: 0 | 1,
    to: CharacterState,
    fast: boolean | null,
  ) {
    const events = this.mutator.switchActive(who, to, { fast });
    await this.handleEvents(events);
  }

  private async executeSkill(skillInfo: SkillInfo, arg: GeneralSkillArg) {
    await SkillExecutor.executeSkill(this.mutator, skillInfo, arg);
  }
  private async handleEvent(...args: EventAndRequest) {
    await SkillExecutor.handleEvent(this.mutator, ...args);
  }
  private async handleEvents(events: EventAndRequest[]) {
    await SkillExecutor.handleEvents(this.mutator, events);
  }
}

export function mergeGameConfigWithDefault(
  config?: Partial<GameConfig>,
): GameConfig {
  config = JSON.parse(JSON.stringify(config ?? {}));
  return {
    ...getDefaultGameConfig(),
    ...config,
  };
}

import { ActionValidity } from "@gi-tcg/typings";
import { Game, type CreateInitialStateConfig } from "../game";
import type { GameData } from "../builder/registry";
import { RuleEngine } from "./rule_engine";
import type {
  GameAction,
  GameConfig,
  PureExecuteOption,
  PureGameState,
  PureLogicEngine,
  RuleEngineDecisionProvider,
} from "./types";

/**
 * Decoupled pure engine facade.
 * It delegates state transition to RuleEngine and keeps high-level history metadata.
 */
export class PureGameEngine implements PureLogicEngine {
  constructor(
    private readonly gameData: GameData,
    private readonly config: GameConfig,
    private readonly defaultDecisionProvider?: RuleEngineDecisionProvider,
  ) {}

  async execute(
    state: PureGameState,
    action: GameAction,
    opt?: PureExecuteOption,
  ): Promise<PureGameState> {
    const timestamp = Date.now();
    const disableHistory = opt?.disableHistory ?? true;
    const { nextState } = RuleEngine.execute(state.gameState, action, {
      ...opt,
      decisionProvider:
        opt?.decisionProvider ?? this.defaultDecisionProvider,
    });
    const afterSnapshot: PureGameState = {
      ...state,
      history: [],
      gameState: nextState,
      turn: state.turn + 1,
      isFinished: nextState.phase === "gameEnd",
      winner: nextState.winner ?? undefined,
    };

    if (disableHistory) {
      return {
        ...afterSnapshot,
        history: [],
      };
    }

    return {
      ...afterSnapshot,
      history: [
        ...state.history,
        {
          turn: state.turn,
          action,
          stateBefore: null,
          stateAfter: null,
          timestamp,
        },
      ],
    };
  }

  getPossibleActions(_state: PureGameState): GameAction[] {
    return RuleEngine.getPossibleActions(_state.gameState, {
      fastMode: true,
    });
  }

  isValidAction(_state: PureGameState, action: GameAction): boolean {
    return action.validity === ActionValidity.VALID;
  }

  createInitialState(deck1: any, deck2: any): PureGameState {
    const initialStateConfig: CreateInitialStateConfig = {
      decks: [deck1, deck2],
      data: this.gameData,
    };
    const gameState = RuleEngine.bootstrapToActionPhase(
      Game.createInitialState(initialStateConfig),
      {
        decisionProvider: this.defaultDecisionProvider,
      },
    ).nextState;
    return {
      config: this.config,
      gameState,
      history: [],
      turn: 1,
      isFinished: false,
    };
  }
}

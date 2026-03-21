import type { GameState, GameConfig } from '../base/state';
import type { Action } from '@gi-tcg/typings';
import { DiceType } from '@gi-tcg/typings';
import type { StatePatch } from '../mutator';

// 重新导出 GameConfig
export type { GameConfig };

/**
 * 扩展的游戏配置接口（如果需要额外的配置项）
 */
export interface ExtendedGameConfig extends GameConfig {
  enableDetailedLogging?: boolean;
}

/**
 * 纯游戏状态接口 - 不包含任何 DOM 元素、Canvas 纹理或计时器
 * 这是一个完整的游戏状态快照，可以被序列化/反序列化
 */
export interface PureGameState {
  /** 游戏配置 */
  config: GameConfig;

  /** 当前游戏状态 */
  gameState: GameState;

  /** 游戏历史记录（用于回放） */
  history: GameStateLogEntry[];

  /** 当前回合数 */
  turn: number;

  /** 游戏是否结束 */
  isFinished: boolean;

  /** 获胜者（如果游戏结束） */
  winner?: number;
}

/**
 * 游戏动作 - 玩家可以执行的所有操作
 */
export type GameAction = Action;

export interface RuleEngineDecisionProvider {
  chooseActive?: (
    state: GameState,
    who: 0 | 1,
    candidates: readonly number[],
  ) => number;
  rerollDice?: (
    state: GameState,
    who: 0 | 1,
    dice: readonly DiceType[],
    rerollCountLeft: number,
  ) => readonly DiceType[];
}

export interface RuleEngineExecuteResult {
  nextState: GameState;
  patches: StatePatch[];
}

export interface PureExecuteOption {
  decisionProvider?: RuleEngineDecisionProvider;
  disableHistory?: boolean;
}

/**
 * 纯逻辑引擎接口
 * 引擎的任务：给定一个状态 + 一个动作 = 返回新状态
 */
export interface PureLogicEngine {
  /**
   * 执行动作，返回新的游戏状态
   * @param state 当前游戏状态
   * @param action 要执行的动作
   * @returns 新的游戏状态
   */
  execute(
    state: PureGameState,
    action: GameAction,
    opt?: PureExecuteOption,
  ): Promise<PureGameState>;

  /**
   * 获取当前状态下所有可能的动作
   * @param state 当前游戏状态
   * @returns 可能的动作列表
   */
  getPossibleActions(state: PureGameState): GameAction[];

  /**
   * 验证动作是否有效
   * @param state 当前游戏状态
   * @param action 要验证的动作
   * @returns 是否有效
   */
  isValidAction(state: PureGameState, action: GameAction): boolean;
}

/**
 * 玩家控制器接口 - 解耦的关键
 * 不管是真人（网页点击）、AI（算法）还是网络玩家，都通过这个接口
 */
export interface PlayerController {
  /**
   * 获取玩家ID
   */
  getPlayerId(): number;

  /**
   * 在当前状态下选择一个动作
   * @param state 当前游戏状态
   * @returns 选择的动作
   */
  selectAction(state: PureGameState): Promise<GameAction>;

  /**
   * 通知玩家状态变化（用于观察者模式）
   * @param state 新的游戏状态
   */
  notifyStateChange(state: PureGameState): void;

  /**
   * 通知玩家游戏事件（可选，用于动画或音效）
   * @param event 游戏事件
   */
  notifyEvent?(event: GameEvent): void;
}

/**
 * 游戏事件 - 用于通知UI的纯数据事件
 */
export interface GameEvent {
  type: 'action_executed' | 'turn_started' | 'game_ended' | 'character_damaged' | 'skill_used';
  playerId?: number;
  data: any; // 纯数据，没有DOM引用
  timestamp: number;
}

/**
 * 视觉层观察者接口
 * UI 只观察状态变化，不直接操作游戏逻辑
 */
export interface GameViewObserver {
  /**
   * 当游戏状态发生变化时调用
   * @param oldState 旧状态
   * @param newState 新状态
   * @param events 本次变化产生的事件
   */
  onStateChanged(oldState: PureGameState, newState: PureGameState, events: GameEvent[]): void;

  /**
   * 当需要用户输入时调用
   * @param playerId 需要输入的玩家
   * @param state 当前状态
   */
  onPlayerInputRequired(playerId: number, state: PureGameState): void;
}


/**
 * 游戏历史记录条目
 */
export interface GameStateLogEntry {
  turn: number;
  action: GameAction;
  stateBefore: PureGameState | null;
  stateAfter: PureGameState | null;
  timestamp: number;
}

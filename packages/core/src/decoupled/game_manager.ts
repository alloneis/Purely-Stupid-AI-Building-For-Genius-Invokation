import type {
  PureGameState,
  GameAction,
  GameEvent,
  PlayerController,
  GameViewObserver,
  GameConfig
} from './types';
import { PureGameEngine } from './pure_engine';
import type { GameData } from '../builder/registry';

/**
 * 游戏管理器 - 协调纯逻辑引擎、玩家控制器和视图观察者
 * 这是解耦架构的核心协调器
 */
export class GameManager {
  private engine: PureGameEngine;
  private players: PlayerController[];
  private observers: GameViewObserver[] = [];
  private currentState: PureGameState;
  private isRunning: boolean = false;

  constructor(gameData: GameData, players: PlayerController[], config: GameConfig, initialDecks: [any, any]) {
    this.engine = new PureGameEngine(gameData, config);
    this.players = players;

    // 初始化游戏状态
    this.currentState = this.engine.createInitialState(initialDecks[0], initialDecks[1]);
  }

  /**
   * 添加视图观察者
   */
  addObserver(observer: GameViewObserver): void {
    this.observers.push(observer);
  }

  /**
   * 移除视图观察者
   */
  removeObserver(observer: GameViewObserver): void {
    const index = this.observers.indexOf(observer);
    if (index > -1) {
      this.observers.splice(index, 1);
    }
  }

  /**
   * 启动游戏循环
   */
  async startGame(): Promise<void> {
    if (this.isRunning) {
      return;
    }

    this.isRunning = true;
    this.notifyObservers(this.currentState, this.currentState, []);

    try {
      while (this.isRunning && !this.currentState.isFinished) {
        await this.playTurn();
      }

      // 游戏结束
      this.handleGameEnd();

    } catch (error) {
      console.error('Game loop error:', error);
      this.isRunning = false;
    }
  }

  /**
   * 停止游戏
   */
  stopGame(): void {
    this.isRunning = false;
  }

  /**
   * 获取当前游戏状态
   */
  getCurrentState(): PureGameState {
    return { ...this.currentState };
  }

  /**
   * 执行单个动作（用于调试或手动控制）
   */
  async executeAction(action: GameAction): Promise<PureGameState> {
    if (!this.engine.isValidAction(this.currentState, action)) {
      throw new Error('Invalid action');
    }

    const oldState = this.currentState;
    this.currentState = await this.engine.execute(this.currentState, action);

    // 生成事件
    const events = this.generateEvents(oldState, this.currentState, action);

    // 通知观察者
    this.notifyObservers(oldState, this.currentState, events);

    return this.currentState;
  }

  /**
   * 重新开始游戏
   */
  restartGame(initialDecks: [any, any]): void {
    this.stopGame();
    this.currentState = this.engine.createInitialState(initialDecks[0], initialDecks[1]);
    this.startGame();
  }

  /**
   * 执行一个完整的回合
   */
  private async playTurn(): Promise<void> {
    const currentPlayerId = this.getCurrentPlayerId();
    const player = this.players.find(p => p.getPlayerId() === currentPlayerId);

    if (!player) {
      throw new Error(`Player ${currentPlayerId} not found`);
    }

    // 通知UI需要玩家输入
    this.notifyPlayerInputRequired(currentPlayerId);

    try {
      // 等待玩家选择动作
      const action = await player.selectAction(this.currentState);

      // 执行动作
      await this.executeAction(action);

    } catch (error) {
      console.error(`Player ${currentPlayerId} action failed:`, error);
      // 可以在这里实现跳过回合或默认动作的逻辑
    }
  }

  /**
   * 获取当前应该行动的玩家ID
   */
  private getCurrentPlayerId(): number {
    // 这里需要根据游戏状态确定当前玩家
    // 简化实现：交替玩家
    return this.currentState.turn % this.players.length;
  }

  /**
   * 处理游戏结束
   */
  private handleGameEnd(): void {
    const winner = this.currentState.winner;
    console.log(`Game ended. Winner: Player ${winner}`);

    // 通知所有观察者游戏结束
    const gameEndEvent: GameEvent = {
      type: 'game_ended',
      data: { winner },
      timestamp: Date.now()
    };

    this.notifyObservers(this.currentState, this.currentState, [gameEndEvent]);
  }

  /**
   * 生成游戏事件（用于UI动画）
   */
  private generateEvents(oldState: PureGameState, newState: PureGameState, action: GameAction): GameEvent[] {
    const events: GameEvent[] = [
      {
        type: 'action_executed',
        data: { action },
        timestamp: Date.now()
      }
    ];

    // 可以在这里添加更多的游戏事件检测逻辑
    // 例如：角色受伤、技能使用、回合开始等

    return events;
  }

  /**
   * 通知所有观察者状态变化
   */
  private notifyObservers(oldState: PureGameState, newState: PureGameState, events: GameEvent[]): void {
    for (const observer of this.observers) {
      try {
        observer.onStateChanged(oldState, newState, events);
      } catch (error) {
        console.error('Observer notification failed:', error);
      }
    }
  }

  /**
   * 通知需要玩家输入
   */
  private notifyPlayerInputRequired(playerId: number): void {
    for (const observer of this.observers) {
      try {
        observer.onPlayerInputRequired(playerId, this.currentState);
      } catch (error) {
        console.error('Observer input notification failed:', error);
      }
    }
  }

  /**
   * 通知所有玩家状态变化
   */
  private notifyPlayersStateChange(): void {
    for (const player of this.players) {
      try {
        player.notifyStateChange(this.currentState);
      } catch (error) {
        console.error(`Player ${player.getPlayerId()} notification failed:`, error);
      }
    }
  }
}
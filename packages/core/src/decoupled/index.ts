/**
 * 解耦游戏架构
 *
 * 这个模块实现了完整的解耦架构：
 * 1. 纯数据接口 - 不包含任何DOM元素、Canvas纹理或计时器
 * 2. 纯逻辑引擎 - 状态 + 动作 = 新状态
 * 3. 抽象控制器接口 - Player接口，适配真人/AI/网络玩家
 * 4. 视觉与逻辑分离 - UI只是观察者，监听状态变化播放动画
 */

// 类型定义
export type {
  PureGameState,
  GameAction,
  GameEvent,
  PlayerController,
  GameViewObserver,
  GameConfig,
  GameStateLogEntry,
  PureLogicEngine,
  PureExecuteOption,
  RuleEngineDecisionProvider,
  RuleEngineExecuteResult,
} from './types';

// 纯逻辑引擎
export { PureGameEngine } from './pure_engine';
export { RuleEngine, type RuleEngineExecuteOption } from "./rule_engine";
export {
  DynamicHeuristicEvaluator,
  type DynamicHeuristicEvaluatorOptions,
  type DynamicHeuristicWeights,
} from "./dynamic_heuristic_evaluator";
export {
  ActionPruner,
  type ActionPrunerOptions,
  type PrunedActionScore,
} from "./action_pruner";
export { ISMCTSAgent, type ISMCTSAgentOptions } from "./ismcts_agent";

export {
  TRAINED_RPC_ACTION_HEURISTIC_WEIGHTS,
  TRAINED_RPC_ACTION_HEURISTIC_METADATA,
} from "./trained_rpc_action_weights";

// Neural network encoders, evaluator, and self-play
export {
  NeuralStateEncoder,
  type EncodedState,
  ActionIndexer,
  type ActionMapping,
  describeActionIndex,
  NeuralEvaluator,
  BatchedInferenceQueue,
  type NeuralEvaluatorOptions,
  type OnnxSession,
  type OnnxTensor,
  type OnnxTensorFactory,
  type InferResult,
  NeuralSelfPlayEngine,
  type SelfPlayConfig,
  type StepRecord,
  type EpisodeResult,
  ArenaEngine,
  type ArenaConfig,
  type ArenaResult,
  type ArenaGameStats,
  type ArenaGameRecord,
  MAX_CHARACTERS,
  MAX_HAND_CARDS,
  MAX_SUMMONS,
  MAX_SUPPORTS,
  MAX_COMBAT_STATUSES,
  MAX_CHARACTER_ENTITIES,
  MAX_ACTION_SLOTS,
  GLOBAL_FEATURE_DIM,
  CHARACTER_FEATURE_DIM,
  CARD_FEATURE_DIM,
  ENTITY_FEATURE_DIM,
} from "./neural";

// 游戏管理器
export { GameManager } from './game_manager';


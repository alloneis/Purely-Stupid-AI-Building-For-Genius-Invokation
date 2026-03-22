export {
  NeuralStateEncoder,
  type EncodedState,
} from "./state_encoder";

export {
  ActionIndexer,
  type ActionMapping,
  describeActionIndex,
} from "./action_encoder";

export {
  NeuralEvaluator,
  BatchedInferenceQueue,
  type NeuralEvaluatorOptions,
  type OnnxSession,
  type OnnxTensor,
  type OnnxTensorFactory,
  type InferResult,
} from "./neural_evaluator";

export {
  NeuralSelfPlayEngine,
  type SelfPlayConfig,
  type StepRecord,
  type EpisodeResult,
} from "./self_play_engine";

export {
  ArenaEngine,
  type ArenaConfig,
  type ArenaResult,
  type ArenaGameStats,
  type ArenaGameRecord,
} from "./arena_engine";

export {
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
} from "./constants";

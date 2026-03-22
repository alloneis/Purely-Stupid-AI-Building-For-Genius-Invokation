/// <reference lib="webworker" />

import * as Core from "@gi-tcg/core";
import getData from "@gi-tcg/data";
import type { Action as RpcAction } from "@gi-tcg/typings";
import * as ort from "onnxruntime-web";

type IsmctsWorkerRequest = {
  id: number;
  type: "bestAction" | "loadModel";
  serializedState?: unknown;
  legalActions?: RpcAction[];
  who?: 0 | 1;
  shardId?: number;
  searchConfig?: {
    iterations?: number;
    determinizationCount?: number;
    maxDepth?: number;
    exploration?: number;
    minimaxBlend?: number;
  };
  modelUrl?: string;
};

type IsmctsWorkerResponse =
  | {
      id: number;
      ok: true;
      bestAction: RpcAction | null;
      reason?: string;
      candidates?: Array<{
        action: RpcAction;
        visits: number;
        meanValue: number;
      }>;
      top3?: Array<{
        action: RpcAction;
        visits: number;
        meanValue: number;
      }>;
      processLogs?: string[];
      shardId?: number;
    }
  | {
      id: number;
      ok: false;
      error: string;
    };

const workerScope = self as DedicatedWorkerGlobalScope;

const ISMCTSAgentCtor = (Core as any).ISMCTSAgent as
  | (new (options?: any) => {
      getBestAction: (...args: any[]) => RpcAction;
      getBestActionWithDebug?: (...args: any[]) => {
        action: RpcAction;
        debug: {
          selected: { action: RpcAction; visits: number; meanValue: number };
          candidates: Array<{
            action: RpcAction;
            visits: number;
            meanValue: number;
          }>;
          top3: Array<{
            action: RpcAction;
            visits: number;
            meanValue: number;
          }>;
          determinizationCount: number;
          simulationsPerDeterminization: number;
          processLogs: string[];
        };
      };
    })
  | undefined;
const deserializeGameStateLog = (Core as any).deserializeGameStateLog as
  | ((data: any, serialized: unknown) => Array<{ state: any }>)
  | undefined;
const coreVersion = (Core as any).CURRENT_VERSION;
const gameData = getData(coreVersion);
const DEFAULT_SEARCH_CONFIG = {
  iterations: 1500,
  determinizationCount: 5,
  maxDepth: 25,
} as const;
const agentCache = new Map<string, InstanceType<NonNullable<typeof ISMCTSAgentCtor>>>();

// ─── Neural ONNX Infrastructure ──────────────────────────────────────

const NeuralEvaluatorCtor = (Core as any).NeuralEvaluator as
  | (new (options: any) => any)
  | undefined;

let onnxSession: ort.InferenceSession | null = null;
let neuralEvaluator: any = null;

const tensorFactory = {
  create(data: Float32Array, dims: readonly number[]): ort.Tensor {
    return new ort.Tensor("float32", data, dims as number[]);
  },
};

async function loadOnnxModel(url: string): Promise<void> {
  ort.env.wasm.numThreads = 1;
  onnxSession = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  if (NeuralEvaluatorCtor) {
    neuralEvaluator = new NeuralEvaluatorCtor({
      session: onnxSession,
      tensorFactory,
    });
  }
  agentCache.clear();
}

function toPureStateSnapshot(state: any): any {
  return {
    config: state.config,
    gameState: state,
    history: [],
    turn: state.roundNumber,
    isFinished: state.phase === "gameEnd",
    winner: state.winner ?? undefined,
  };
}

function post(message: IsmctsWorkerResponse) {
  workerScope.postMessage(message);
}

function describeAction(action: RpcAction | null | undefined): string {
  if (!action?.action) {
    return "unknown";
  }
  const payload = action.action;
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

function formatReason(debug: {
  selected: { action: RpcAction; visits: number; meanValue: number };
  top3: Array<{ action: RpcAction; visits: number; meanValue: number }>;
  determinizationCount: number;
  simulationsPerDeterminization: number;
}): string {
  const top = debug.top3
    .map(
      (candidate, index) =>
        `#${index + 1} ${describeAction(candidate.action)} (visits=${candidate.visits}, mean=${candidate.meanValue.toFixed(3)})`,
    )
    .join("; ");
  return `ISMCTS selected ${describeAction(debug.selected.action)}; selected visits=${debug.selected.visits}, mean=${debug.selected.meanValue.toFixed(3)}, det=${debug.determinizationCount}, simPerDet=${debug.simulationsPerDeterminization}. ${top}`;
}

function compactActionForResponse(action: RpcAction): RpcAction {
  return {
    action: action.action,
    requiredCost: action.requiredCost,
    autoSelectedDice: action.autoSelectedDice,
    validity: action.validity,
    isFast: action.isFast,
    preview: [],
  };
}

function getAgent(searchConfig?: IsmctsWorkerRequest["searchConfig"]) {
  if (!ISMCTSAgentCtor) {
    return null;
  }
  const normalized = {
    ...DEFAULT_SEARCH_CONFIG,
    ...searchConfig,
  };
  const neuralSuffix = neuralEvaluator ? ":neural" : "";
  const key = JSON.stringify(normalized) + neuralSuffix;
  const cached = agentCache.get(key);
  if (cached) {
    return cached;
  }
  const agentOpts: any = { ...normalized };
  if (neuralEvaluator) {
    agentOpts.neuralEvaluator = neuralEvaluator;
  }
  const agent = new ISMCTSAgentCtor(agentOpts);
  agentCache.set(key, agent);
  return agent;
}

async function handleBestAction(request: IsmctsWorkerRequest): Promise<void> {
  const ismcts = getAgent(request.searchConfig);
  if (!ismcts || !deserializeGameStateLog) {
    throw new Error("ISMCTS worker is unavailable");
  }

  const serializedState =
    typeof request.serializedState === "string"
      ? JSON.parse(request.serializedState)
      : request.serializedState;
  const deserialized = deserializeGameStateLog(gameData, serializedState);
  const restoredState = deserialized[0]?.state;
  if (!restoredState) {
    throw new Error("Failed to restore game state in worker");
  }

  const pureState = toPureStateSnapshot(restoredState);
  let bestAction: RpcAction | null = null;
  let reason: string | undefined;
  let candidates:
    | Array<{ action: RpcAction; visits: number; meanValue: number }>
    | undefined;
  let top3:
    | Array<{ action: RpcAction; visits: number; meanValue: number }>
    | undefined;
  let processLogs: string[] | undefined;

  if (neuralEvaluator && typeof ismcts.getBestActionNeuralAsync === "function") {
    const result = await ismcts.getBestActionNeuralAsync(
      pureState,
      request.legalActions,
      request.who,
    );
    bestAction = result.action;
    reason = formatReason(result.debug);
    candidates = result.debug.candidates.map((entry: any) => ({
      action: compactActionForResponse(entry.action),
      visits: entry.visits,
      meanValue: entry.meanValue,
    }));
    top3 = result.debug.top3.map((entry: any) => ({
      action: compactActionForResponse(entry.action),
      visits: entry.visits,
      meanValue: entry.meanValue,
    }));
    processLogs = result.debug.processLogs;
  } else if (typeof ismcts.getBestActionWithDebug === "function") {
    const result = ismcts.getBestActionWithDebug(
      pureState,
      request.legalActions,
      request.who,
    );
    bestAction = result.action;
    reason = formatReason(result.debug);
    candidates = result.debug.candidates.map((entry: any) => ({
      action: compactActionForResponse(entry.action),
      visits: entry.visits,
      meanValue: entry.meanValue,
    }));
    top3 = result.debug.top3.map((entry: any) => ({
      action: compactActionForResponse(entry.action),
      visits: entry.visits,
      meanValue: entry.meanValue,
    }));
    processLogs = result.debug.processLogs;
  } else {
    bestAction = ismcts.getBestAction(
      pureState,
      request.legalActions,
      request.who,
    );
    reason = `ISMCTS selected ${describeAction(bestAction)}`;
    candidates = [];
    top3 = [];
    processLogs = [];
  }

  post({
    id: request.id,
    ok: true,
    bestAction: bestAction ? compactActionForResponse(bestAction) : null,
    reason,
    candidates,
    top3,
    processLogs,
    shardId: request.shardId,
  });
}

workerScope.onmessage = (event: MessageEvent<IsmctsWorkerRequest>) => {
  const request = event.data;

  if (request.type === "loadModel") {
    if (!request.modelUrl) {
      post({ id: request.id, ok: false, error: "No modelUrl provided" });
      return;
    }
    loadOnnxModel(request.modelUrl)
      .then(() => {
        post({
          id: request.id,
          ok: true,
          bestAction: null,
          reason: `Neural model loaded from ${request.modelUrl}`,
          shardId: request.shardId,
        });
      })
      .catch((err) => {
        post({
          id: request.id,
          ok: false,
          error: `Failed to load ONNX model: ${err instanceof Error ? err.message : String(err)}`,
        });
      });
    return;
  }

  if (request.type === "bestAction") {
    handleBestAction(request).catch((error) => {
      post({
        id: request.id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    });
    return;
  }
};

export {};

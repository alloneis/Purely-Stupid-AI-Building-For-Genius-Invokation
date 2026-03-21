import * as Core from "@gi-tcg/core";
import type { Action as RpcAction } from "@gi-tcg/typings";

const DEFAULT_TIMEOUT_MS = 12_000;
const DEFAULT_SEARCH_CONFIG: Required<
  Pick<
    IsmctsSearchConfig,
    "iterations" | "determinizationCount" | "maxDepth" | "exploration" | "minimaxBlend"
  >
> = {
  iterations: 420,
  determinizationCount: 5,
  maxDepth: 12,
  exploration: Math.SQRT2,
  minimaxBlend: 0.35,
};

export interface IsmctsActionStat {
  action: RpcAction;
  visits: number;
  meanValue: number;
}

export interface IsmctsSearchConfig {
  iterations?: number;
  determinizationCount?: number;
  maxDepth?: number;
  exploration?: number;
  minimaxBlend?: number;
}

export interface IsmctsDecision {
  bestAction: RpcAction | null;
  reason?: string;
  top3?: IsmctsActionStat[];
  processLogs?: string[];
}

export interface AIWorkerWrapperOptions {
  poolSize?: number;
  parallelShards?: number;
  timeoutMs?: number;
  searchConfig?: IsmctsSearchConfig;
}

type IsmctsWorkerRequest = {
  id: number;
  type: "bestAction";
  serializedState: unknown;
  legalActions: RpcAction[];
  who: 0 | 1;
  shardId?: number;
  searchConfig?: IsmctsSearchConfig;
};

type IsmctsWorkerResponse =
  | {
      id: number;
      ok: true;
      bestAction: RpcAction | null;
      reason?: string;
      candidates?: IsmctsActionStat[];
      top3?: IsmctsActionStat[];
      processLogs?: string[];
      shardId?: number;
    }
  | {
      id: number;
      ok: false;
      error: string;
    };

type PendingTask = {
  workerIndex: number;
  resolve: (result: Extract<IsmctsWorkerResponse, { ok: true }>) => void;
  reject: (error: Error) => void;
  timeoutId: ReturnType<typeof setTimeout>;
};

type WorkerSlot = {
  worker: Worker;
  inFlight: number;
};

function defaultPoolSize() {
  const cpu = typeof navigator !== "undefined" ? navigator.hardwareConcurrency ?? 2 : 2;
  return Math.max(1, Math.min(4, cpu));
}

function clampInt(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) {
    return min;
  }
  const n = Math.floor(value);
  return Math.max(min, Math.min(max, n));
}

function serializeStateForWorker(state: any): unknown {
  const serialize = (Core as any).serializeGameStateLog as
    | ((entries: readonly { state: any; canResume: boolean }[]) => unknown)
    | undefined;
  if (!serialize) {
    throw new Error("serializeGameStateLog is unavailable");
  }
  const serialized = serialize([
    {
      state,
      canResume: false,
    },
  ]);
  return JSON.stringify(serialized, (_key, value) => {
    if (typeof value === "function") {
      return undefined;
    }
    return value;
  });
}

function compactActionForWorker(action: RpcAction): RpcAction {
  return {
    action: action.action,
    requiredCost: action.requiredCost,
    autoSelectedDice: action.autoSelectedDice,
    validity: action.validity,
    isFast: action.isFast,
    preview: [],
  };
}

function actionKey(action: RpcAction): string {
  return JSON.stringify({
    action: action.action,
    cost: action.requiredCost,
    dice: action.autoSelectedDice,
    fast: action.isFast,
  });
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

export class AIWorkerWrapper {
  private readonly workers: WorkerSlot[] = [];
  private readonly pending = new Map<number, PendingTask>();
  private readonly timeoutMs: number;
  private readonly parallelShards: number;
  private readonly searchConfig: Required<IsmctsSearchConfig>;
  private seq = 1;
  private terminated = false;

  constructor(options: AIWorkerWrapperOptions = {}) {
    if (typeof Worker === "undefined") {
      throw new Error("Web Worker is unavailable in this environment");
    }
    const poolSize = clampInt(options.poolSize ?? defaultPoolSize(), 1, 16);
    this.parallelShards = clampInt(options.parallelShards ?? poolSize, 1, poolSize);
    this.timeoutMs = clampInt(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, 1000, 120_000);
    this.searchConfig = {
      ...DEFAULT_SEARCH_CONFIG,
      ...options.searchConfig,
    };

    for (let i = 0; i < poolSize; i++) {
      this.workers.push({
        worker: this.createWorker(i),
        inFlight: 0,
      });
    }
  }

  getBestAction(
    state: any,
    legalActions: readonly RpcAction[],
    who: 0 | 1,
  ): Promise<IsmctsDecision> {
    if (this.terminated) {
      return Promise.reject(new Error("AI worker wrapper terminated"));
    }
    const serializedState = serializeStateForWorker(state);
    const compactActions = legalActions.map(compactActionForWorker);
    if (compactActions.length === 0) {
      return Promise.reject(new Error("No legal actions for AI worker"));
    }

    const shardCount = Math.max(1, Math.min(this.parallelShards, this.workers.length));
    if (shardCount <= 1) {
      const workerIndex = this.pickLeastBusyWorker();
      return this.dispatch(workerIndex, {
        id: this.nextId(),
        type: "bestAction",
        serializedState,
        legalActions: compactActions,
        who,
        searchConfig: this.searchConfig,
        shardId: 1,
      }).then((single) => this.toDecisionFromSingle(single));
    }

    return this.dispatchSharded(
      serializedState,
      compactActions,
      who,
      shardCount,
      this.searchConfig.iterations,
    );
  }

  terminate(): void {
    if (this.terminated) {
      return;
    }
    this.terminated = true;
    for (const [id, task] of this.pending) {
      clearTimeout(task.timeoutId);
      task.reject(new Error("AI worker wrapper terminated"));
      this.pending.delete(id);
    }
    for (const slot of this.workers) {
      slot.worker.terminate();
      slot.inFlight = 0;
    }
  }

  private createWorker(workerIndex: number): Worker {
    const worker = new Worker(new URL("../workers/ismcts.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (event: MessageEvent<IsmctsWorkerResponse>) => {
      this.handleWorkerMessage(workerIndex, event.data);
    };
    worker.onerror = () => {
      this.handleWorkerCrash(workerIndex);
    };
    return worker;
  }

  private handleWorkerMessage(workerIndex: number, message: IsmctsWorkerResponse) {
    const task = this.pending.get(message.id);
    if (!task) {
      return;
    }
    this.pending.delete(message.id);
    this.workers[workerIndex].inFlight = Math.max(0, this.workers[workerIndex].inFlight - 1);
    clearTimeout(task.timeoutId);

    if (message.ok) {
      task.resolve(message);
      return;
    }
    task.reject(new Error(message.error));
  }

  private handleWorkerCrash(workerIndex: number) {
    for (const [id, task] of this.pending) {
      if (task.workerIndex !== workerIndex) {
        continue;
      }
      clearTimeout(task.timeoutId);
      task.reject(new Error(`AI worker ${workerIndex} crashed`));
      this.pending.delete(id);
    }
    const oldWorker = this.workers[workerIndex].worker;
    oldWorker.terminate();
    this.workers[workerIndex] = {
      worker: this.createWorker(workerIndex),
      inFlight: 0,
    };
  }

  private nextId(): number {
    const id = this.seq;
    this.seq += 1;
    return id;
  }

  private pickLeastBusyWorker(excluded: ReadonlySet<number> = new Set()): number {
    let picked = -1;
    let bestLoad = Number.POSITIVE_INFINITY;
    for (let i = 0; i < this.workers.length; i++) {
      if (excluded.has(i)) {
        continue;
      }
      const load = this.workers[i].inFlight;
      if (load < bestLoad) {
        bestLoad = load;
        picked = i;
      }
    }
    if (picked >= 0) {
      return picked;
    }
    return 0;
  }

  private dispatch(
    workerIndex: number,
    request: IsmctsWorkerRequest,
  ): Promise<Extract<IsmctsWorkerResponse, { ok: true }>> {
    if (this.terminated) {
      return Promise.reject(new Error("AI worker wrapper terminated"));
    }
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pending.delete(request.id);
        this.workers[workerIndex].inFlight = Math.max(0, this.workers[workerIndex].inFlight - 1);
        reject(new Error(`AI worker timeout (${this.timeoutMs}ms)`));
      }, this.timeoutMs);

      this.pending.set(request.id, {
        workerIndex,
        resolve,
        reject,
        timeoutId,
      });
      this.workers[workerIndex].inFlight += 1;
      this.workers[workerIndex].worker.postMessage(request);
    });
  }

  private async dispatchSharded(
    serializedState: unknown,
    legalActions: readonly RpcAction[],
    who: 0 | 1,
    shardCount: number,
    totalIterations: number,
  ): Promise<IsmctsDecision> {
    const splitIterations = this.splitIterations(totalIterations, shardCount);
    const usedWorkers = new Set<number>();
    const tasks: Array<Promise<Extract<IsmctsWorkerResponse, { ok: true }>>> = [];

    for (let shardIndex = 0; shardIndex < shardCount; shardIndex++) {
      const workerIndex = this.pickLeastBusyWorker(usedWorkers);
      usedWorkers.add(workerIndex);
      tasks.push(
        this.dispatch(workerIndex, {
          id: this.nextId(),
          type: "bestAction",
          serializedState,
          legalActions: [...legalActions],
          who,
          shardId: shardIndex + 1,
          searchConfig: {
            ...this.searchConfig,
            iterations: splitIterations[shardIndex],
          },
        }),
      );
    }

    const settled = await Promise.allSettled(tasks);
    const successful = settled
      .filter(
        (item): item is PromiseFulfilledResult<Extract<IsmctsWorkerResponse, { ok: true }>> =>
          item.status === "fulfilled",
      )
      .map((item) => item.value);

    if (successful.length === 0) {
      const firstRejected = settled.find((item) => item.status === "rejected");
      throw new Error(
        firstRejected?.status === "rejected"
          ? firstRejected.reason instanceof Error
            ? firstRejected.reason.message
            : String(firstRejected.reason)
          : "All AI worker shards failed",
      );
    }

    return this.mergeShardResults(successful, shardCount);
  }

  private splitIterations(total: number, shardCount: number): number[] {
    const safeTotal = Math.max(shardCount, Math.floor(total));
    const base = Math.floor(safeTotal / shardCount);
    const remain = safeTotal % shardCount;
    return Array.from({ length: shardCount }, (_, index) => base + (index < remain ? 1 : 0));
  }

  private toDecisionFromSingle(
    response: Extract<IsmctsWorkerResponse, { ok: true }>,
  ): IsmctsDecision {
    return {
      bestAction: response.bestAction,
      reason: response.reason,
      top3: response.top3,
      processLogs: response.processLogs,
    };
  }

  private mergeShardResults(
    responses: Array<Extract<IsmctsWorkerResponse, { ok: true }>>,
    shardCount: number,
  ): IsmctsDecision {
    const aggregate = new Map<
      string,
      {
        action: RpcAction;
        visits: number;
        valueSum: number;
      }
    >();

    for (const response of responses) {
      const source = response.candidates ?? response.top3 ?? [];
      for (const stat of source) {
        const key = actionKey(stat.action);
        const visits = Math.max(0, stat.visits);
        const value = Number.isFinite(stat.meanValue) ? stat.meanValue : 0;
        const prev = aggregate.get(key);
        if (prev) {
          prev.visits += visits;
          prev.valueSum += value * Math.max(1, visits);
          continue;
        }
        aggregate.set(key, {
          action: stat.action,
          visits,
          valueSum: value * Math.max(1, visits),
        });
      }
      if (source.length === 0 && response.bestAction) {
        const key = actionKey(response.bestAction);
        if (!aggregate.has(key)) {
          aggregate.set(key, {
            action: response.bestAction,
            visits: 0,
            valueSum: 0,
          });
        }
      }
    }

    const ranked = [...aggregate.values()]
      .map<IsmctsActionStat>((entry) => ({
        action: entry.action,
        visits: entry.visits,
        meanValue: entry.valueSum / Math.max(1, entry.visits),
      }))
      .sort((lhs, rhs) => {
        if (rhs.visits !== lhs.visits) {
          return rhs.visits - lhs.visits;
        }
        return rhs.meanValue - lhs.meanValue;
      });

    const top3 = ranked.slice(0, 3);
    const bestAction = top3[0]?.action ?? responses[0].bestAction;
    const reasonTop = top3
      .map(
        (entry, index) =>
          `#${index + 1} ${describeAction(entry.action)} (visits=${entry.visits}, mean=${entry.meanValue.toFixed(3)})`,
      )
      .join("; ");
    const reason = `ISMCTS shard scheduler selected ${describeAction(bestAction)} from ${responses.length}/${shardCount} shards. ${reasonTop}`;
    const processLogs = responses
      .flatMap((response, index) =>
        (response.processLogs ?? [])
          .slice(0, 8)
          .map((line) => `[S${response.shardId ?? index + 1}] ${line}`),
      )
      .slice(0, 96);

    return {
      bestAction,
      reason,
      top3,
      processLogs,
    };
  }
}

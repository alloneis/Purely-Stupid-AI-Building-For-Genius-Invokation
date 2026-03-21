

import { A } from "@solidjs/router";
import { createSignal, onCleanup, Show } from "solid-js";
import * as Core from "@gi-tcg/core";
import {
  CURRENT_VERSION,
  Game,
  type CancellablePlayerIO,
  type GameStateLogEntry,
  serializeGameStateLog,
} from "@gi-tcg/core";
import getData from "@gi-tcg/data";
import { DEFAULT_ASSETS_MANAGER } from "@gi-tcg/assets-manager";
import { createClient } from "@gi-tcg/web-ui-core";
import "@gi-tcg/web-ui-core/style.css";
import { Layout } from "../layouts/Layout";
import {
  AI_ISMCTS_DETERMINIZATION_COUNT,
  AI_ISMCTS_ITERATIONS,
  AI_ISMCTS_MAX_DEPTH,
  AI_WORKER_PARALLEL_SHARDS,
  AI_WORKER_POOL_SIZE,
  AI_WORKER_TIMEOUT_MS,
} from "../config";
import { AIWorkerWrapper } from "../ai/game_manager";
import {
  ActionValidity,
  createRpcResponse,
  DiceType,
  type Action as RpcAction,
  type Notification,
  type PbGameState,
  type RpcRequest,
  type RpcResponse,
} from "@gi-tcg/typings";

interface IsmctsDecision {
  bestAction: RpcAction | null;
  reason?: string;
  top3?: Array<{
    action: RpcAction;
    visits: number;
    meanValue: number;
  }>;
  processLogs?: string[];
}

interface IsmctsWorkerClient {
  getBestAction(
    state: any,
    legalActions: readonly RpcAction[],
    who: 0 | 1,
  ): Promise<IsmctsDecision>;
  terminate(): void;
}

const HUMAN_WHO: 0 = 0;
const AI_WHO: 1 = 1;
const AI_THINKING_MS = 320;

const INIT_HUMAN_DECK =
  "FZDByRUNGRCB0WoNFlGgWpEPE0AB9TAPFGCB9kgWGIERCoEQDLFADcQQDPFgacYWDJAA";
const INIT_AI_DECK =
  "FdHxNj8TAWDQxFkMFkAhyWIYCYDA45wOCUDQ5J0PGEAh9IIPGWDh9p4YC6FgirYRCxAA";

const ELEMENT_TAG_TO_DICE: Record<string, DiceType> = {
  cryo: DiceType.Cryo,
  hydro: DiceType.Hydro,
  pyro: DiceType.Pyro,
  electro: DiceType.Electro,
  anemo: DiceType.Anemo,
  geo: DiceType.Geo,
  dendro: DiceType.Dendro,
};

const delay = (ms: number) =>
  new Promise<void>((resolve) => setTimeout(resolve, ms));

const isTerminateError = (error: unknown) =>
  error instanceof Error && error.message.includes("User call terminate");

function normalizeDice(dice: readonly number[]): number[] {
  return dice.filter((d) => d > 0);
}

function arrayEqualNumber(a: readonly number[] = [], b: readonly number[] = []): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function isSameAction(a1: RpcAction, a2: RpcAction): boolean {
  const p1 = a1.action;
  const p2 = a2.action;
  if (!p1 || !p2 || p1.$case !== p2.$case) {
    return false;
  }

  switch (p1.$case) {
    case "useSkill": {
      const l = p1.value;
      const r = (p2 as Extract<typeof p2, { $case: "useSkill" }>).value;
      return (
        l.skillDefinitionId === r.skillDefinitionId &&
        arrayEqualNumber(l.targetIds, r.targetIds) &&
        (l.mainDamageTargetId ?? null) === (r.mainDamageTargetId ?? null)
      );
    }
    case "playCard": {
      const l = p1.value;
      const r = (p2 as Extract<typeof p2, { $case: "playCard" }>).value;
      return l.cardId === r.cardId && arrayEqualNumber(l.targetIds, r.targetIds);
    }
    case "switchActive": {
      const l = p1.value;
      const r = (p2 as Extract<typeof p2, { $case: "switchActive" }>).value;
      return l.characterId === r.characterId;
    }
    case "elementalTuning": {
      const l = p1.value;
      const r = (p2 as Extract<typeof p2, { $case: "elementalTuning" }>).value;
      return l.removedCardId === r.removedCardId && l.targetDice === r.targetDice;
    }
    case "declareEnd":
      return true;
    default:
      return false;
  }
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

function detectElementDiceFromCharacter(character: any): DiceType | null {
  const tags = character?.definition?.tags;
  if (!Array.isArray(tags)) {
    return null;
  }
  for (const tag of tags) {
    if (typeof tag !== "string") {
      continue;
    }
    const mapped = ELEMENT_TAG_TO_DICE[tag];
    if (mapped !== undefined) {
      return mapped;
    }
  }
  return null;
}

function logIsmctsDecision(logDebug: (message: string) => void, decision: IsmctsDecision) {
  if (decision.top3 && decision.top3.length > 0) {
    logDebug("ISMCTS TOP3:");
    for (const [index, item] of decision.top3.entries()) {
      logDebug(
        `  #${index + 1} ${describeAction(item.action)} (visits=${item.visits}, mean=${item.meanValue.toFixed(3)})`,
      );
    }
  }
  if (decision.processLogs && decision.processLogs.length > 0) {
    const shown = decision.processLogs.slice(0, 16);
    logDebug(`ISMCTS process logs (${shown.length}/${decision.processLogs.length}):`);
    for (const line of shown) {
      logDebug(`  ${line}`);
    }
    if (decision.processLogs.length > shown.length) {
      logDebug(`  ... ${decision.processLogs.length - shown.length} more lines`);
    }
  }
}

function estimateCardDiceCost(card: any): number {
  const playSkillOfCard = (Core as any).playSkillOfCard as
    | ((definition: any) => any)
    | undefined;
  if (!playSkillOfCard) {
    return 2;
  }
  const playSkill = playSkillOfCard(card.definition);
  const requiredCost: Map<number, number> | undefined =
    playSkill?.initiativeSkillConfig?.requiredCost;
  if (!requiredCost || typeof requiredCost.entries !== "function") {
    return 2;
  }

  let total = 0;
  for (const [type, count] of requiredCost.entries()) {
    if (type === DiceType.Energy || type === DiceType.Legend) {
      continue;
    }
    total += count;
  }
  return total;
}

function chooseInitialActiveHeuristically(
  state: any,
  who: 0 | 1,
  candidateIds: readonly number[],
): number | null {
  const player = state?.players?.[who];
  const characters: any[] = Array.isArray(player?.characters)
    ? player.characters
    : [];
  const candidates = characters.filter((c) => candidateIds.includes(c.id));
  if (candidates.length === 0) {
    return candidateIds[0] ?? null;
  }

  const scored = candidates.map((character) => {
    const hp = Number(character?.variables?.health ?? 0);
    const maxHp = Number(character?.variables?.maxHealth ?? hp);
    const energy = Number(character?.variables?.energy ?? 0);
    const maxEnergy = Number(character?.variables?.maxEnergy ?? 0);
    const score = hp + maxHp * 0.25 + energy * 0.4 + maxEnergy * 0.1;
    return {
      id: character.id as number,
      score,
    };
  });

  scored.sort((a, b) => b.score - a.score);
  return scored[0]?.id ?? candidateIds[0] ?? null;
}

function chooseSwitchHandsHeuristically(
  state: any,
  who: 0 | 1,
  fallbackHandIds: readonly number[],
): number[] {
  const player = state?.players?.[who];
  const hands: any[] = Array.isArray(player?.hands) ? player.hands : [];
  if (hands.length === 0) {
    return [...fallbackHandIds].slice(0, Math.floor(fallbackHandIds.length / 2));
  }

  const activeCharacter = (player?.characters ?? []).find(
    (character: any) => character?.id === player?.activeCharacterId,
  );
  const activeDefinitionId = activeCharacter?.definition?.id;

  const scored = hands.map((card) => {
    const tags: string[] = Array.isArray(card?.definition?.tags)
      ? card.definition.tags
      : [];
    const type = card?.definition?.type;
    const cost = estimateCardDiceCost(card);
    const ownerCharacterId =
      card?.definition?.characterId ??
      card?.definition?.associatedCharacterId ??
      card?.definition?.talentCharacterId;

    let keepScore = 0;
    if (tags.includes("legend")) {
      keepScore += 100;
    }
    if (type === "support") {
      keepScore += 18;
    }
    if (tags.includes("talent")) {
      keepScore += 12;
    }

    if (cost <= 1) {
      keepScore += 30;
    } else if (cost === 2) {
      keepScore += 8;
    } else if (cost >= 3) {
      keepScore -= 25;
    }

    if (typeof ownerCharacterId === "number") {
      if (ownerCharacterId === activeDefinitionId) {
        keepScore += 20;
      } else {
        keepScore -= 8;
      }
    }

    return {
      id: card.id as number,
      keepScore,
    };
  });

  scored.sort((a, b) => a.keepScore - b.keepScore);
  const maxRemove = Math.floor(scored.length / 2);
  const removed = scored.filter((item) => item.keepScore < 0).slice(0, maxRemove);

  if (removed.length === 0 && scored.length > 0 && scored[0].keepScore <= 5) {
    return [scored[0].id];
  }
  return removed.map((item) => item.id);
}

function chooseRerollDiceHeuristically(
  state: any,
  who: 0 | 1,
  dice: readonly number[],
): number[] {
  const player = state?.players?.[who];
  const activeCharacter = (player?.characters ?? []).find(
    (character: any) => character?.id === player?.activeCharacterId,
  );
  const activeElement = detectElementDiceFromCharacter(activeCharacter);

  return dice.filter((d) => {
    if (d === DiceType.Omni) {
      return false;
    }
    if (activeElement !== null && d === activeElement) {
      return false;
    }
    return true;
  });
}

function createIsmctsWorkerClient(logDebug: (message: string) => void): IsmctsWorkerClient | null {
  if (typeof Worker === "undefined") {
    return null;
  }

  try {
    const wrapper = new AIWorkerWrapper({
      poolSize: AI_WORKER_POOL_SIZE,
      parallelShards: AI_WORKER_PARALLEL_SHARDS,
      timeoutMs: AI_WORKER_TIMEOUT_MS,
      searchConfig: {
        iterations: AI_ISMCTS_ITERATIONS,
        determinizationCount: AI_ISMCTS_DETERMINIZATION_COUNT,
        maxDepth: AI_ISMCTS_MAX_DEPTH,
      },
    });
    logDebug(
      `AI scheduler ready: pool=${AI_WORKER_POOL_SIZE}, shards=${AI_WORKER_PARALLEL_SHARDS}, iter=${AI_ISMCTS_ITERATIONS}, det=${AI_ISMCTS_DETERMINIZATION_COUNT}, depth=${AI_ISMCTS_MAX_DEPTH}`,
    );
    return wrapper;
  } catch (error) {
    logDebug(
      `AI scheduler init failed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return null;
  }
}

function fallbackActionDecision(
  actions: readonly RpcAction[],
): { chosenActionIndex: number; usedDice: number[] } {
  const chosenActionIndex = actions.findIndex(
    (action) => action.validity === ActionValidity.VALID,
  );
  if (chosenActionIndex < 0) {
    throw new Error("AI fallback failed: no valid action");
  }
  const chosen = actions[chosenActionIndex];
  return {
    chosenActionIndex,
    usedDice: normalizeDice(chosen.autoSelectedDice),
  };
}

function createSimpleAiIo(
  who: 0 | 1,
  getGameState: () => any | null,
  ismctsWorkerClient: IsmctsWorkerClient | null,
  logDebug: (message: string) => void,
): CancellablePlayerIO {
  let latestState: PbGameState | null = null;

  return {
    notify: (notification: Notification) => {
      if (notification.state) {
        latestState = notification.state;
      }
    },
    cancelRpc: () => {
      // no-op for this simple AI
    },
    rpc: async (request: RpcRequest): Promise<RpcResponse> => {
      await delay(AI_THINKING_MS);
      const payload = request.request;
      if (!payload) {
        throw new Error("Invalid AI request: missing payload");
      }

      const gameState = getGameState();

      switch (payload.$case) {
        case "chooseActive": {
          const activeCharacterId =
            chooseInitialActiveHeuristically(
              gameState,
              who,
              payload.value.candidateIds,
            ) ?? payload.value.candidateIds[0];
          if (typeof activeCharacterId !== "number") {
            throw new Error("AI chooseActive failed: no candidate ids");
          }
          logDebug(`AI chooseActive -> ${activeCharacterId}`);
          return createRpcResponse("chooseActive", { activeCharacterId });
        }
        case "switchHands": {
          const fallbackHandIds = (latestState?.player[who].handCard ?? []).map(
            (card) => card.id,
          );
          const removedHandIds = chooseSwitchHandsHeuristically(
            gameState,
            who,
            fallbackHandIds,
          );
          logDebug(`AI switchHands -> removed ${removedHandIds.length} cards`);
          return createRpcResponse("switchHands", { removedHandIds });
        }
        case "rerollDice": {
          const dice = latestState?.player[who].dice ?? [];
          const diceToReroll = chooseRerollDiceHeuristically(gameState, who, dice);
          logDebug(`AI rerollDice -> reroll ${diceToReroll.length} dice`);
          return createRpcResponse("rerollDice", { diceToReroll });
        }
        case "selectCard": {
          const selectedDefinitionId = payload.value.candidateDefinitionIds[0] ?? 0;
          logDebug(`AI selectCard -> ${selectedDefinitionId}`);
          return createRpcResponse("selectCard", { selectedDefinitionId });
        }
        case "action": {
          const rpcActions = payload.value.action;
          const legalActions = rpcActions.filter(
            (action) => action.validity === ActionValidity.VALID,
          );
          logDebug(`AI action request: ${legalActions.length} legal actions`);
          if (legalActions.length === 0) {
            throw new Error("AI action failed: no legal action");
          }

          if (ismctsWorkerClient && gameState) {
            try {
              const decision = await ismctsWorkerClient.getBestAction(
                gameState,
                legalActions,
                who,
              );
              logIsmctsDecision(logDebug, decision);
              const bestAction = decision.bestAction;
              if (bestAction) {
                const chosenActionIndex = rpcActions.findIndex(
                  (action) =>
                    action.validity === ActionValidity.VALID &&
                    isSameAction(action, bestAction),
                );
                if (chosenActionIndex >= 0) {
                  const chosen = rpcActions[chosenActionIndex];
                  const usedDice = normalizeDice(
                    chosen.autoSelectedDice.length > 0
                      ? chosen.autoSelectedDice
                      : bestAction.autoSelectedDice,
                  );
                  logDebug(
                    `AI chose ${describeAction(chosen)} with [${usedDice.join(",")}]. ${decision.reason ?? "No reason provided."}`,
                  );
                  return createRpcResponse("action", {
                    chosenActionIndex,
                    usedDice,
                  });
                }
                logDebug(
                  `ISMCTS mapping failed for ${describeAction(bestAction)}, fallback to first valid action`,
                );
              } else {
                logDebug("ISMCTS returned no action, fallback to first valid action");
              }
            } catch (error) {
              logDebug(
                `ISMCTS failed (${error instanceof Error ? error.message : String(error)}), fallback to first valid action`,
              );
            }
          } else {
            logDebug("ISMCTS unavailable, fallback to first valid action");
          }

          const decision = fallbackActionDecision(rpcActions);
          const { chosenActionIndex, usedDice } = decision;
          logDebug(
            `Fallback action ${describeAction(rpcActions[chosenActionIndex])} with [${usedDice.join(",")}]`,
          );
          return createRpcResponse("action", { chosenActionIndex, usedDice });
        }
      }
    },
  };
}

function exportBattleLog(logEntries: readonly GameStateLogEntry[]) {
  if (logEntries.length === 0) {
    return;
  }
  const logs = serializeGameStateLog(logEntries);
  const blob = new Blob([JSON.stringify({ ...logs, gv: CURRENT_VERSION })], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "aibattle-log.json";
  link.click();
  URL.revokeObjectURL(url);
}

function exportDebugLog(lines: readonly string[]) {
  if (lines.length === 0) {
    return;
  }
  const blob = new Blob([lines.join("\n")], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "aibattle-debug.log";
  link.click();
  URL.revokeObjectURL(url);
}

export default function AIBattle() {
  const [humanIo, Chessboard] = createClient(HUMAN_WHO, {
    onGiveUp: () => currentGame()?.giveUp(HUMAN_WHO),
  });

  const [humanDeckCode, setHumanDeckCode] = createSignal(INIT_HUMAN_DECK);
  const [aiDeckCode, setAiDeckCode] = createSignal(INIT_AI_DECK);
  const [currentGame, setCurrentGame] = createSignal<Game | null>(null);
  const [running, setRunning] = createSignal(false);
  const [statusText, setStatusText] = createSignal("Idle");
  const [winner, setWinner] = createSignal<0 | 1 | null | undefined>(undefined);
  const [errorMessage, setErrorMessage] = createSignal<string | null>(null);
  const [stateLog, setStateLog] = createSignal<GameStateLogEntry[]>([]);
  const [debugLogs, setDebugLogs] = createSignal<string[]>([]);

  const appendDebugLog = (message: string) => {
    const line = `[${new Date().toLocaleTimeString("zh-CN", { hour12: false })}] ${message}`;
    setDebugLogs((logs) => [...logs.slice(-399), line]);
  };

  const ismctsWorkerClient = createIsmctsWorkerClient(appendDebugLog);

  const stopGame = () => {
    const game = currentGame();
    if (game) {
      humanIo.cancelRpc();
      game.terminate();
      appendDebugLog("Game terminated");
    }
    setCurrentGame(null);
    setRunning(false);
  };

  const startGame = () => {
    stopGame();
    setWinner(undefined);
    setErrorMessage(null);
    setStateLog([]);
    setDebugLogs([]);
    appendDebugLog("Starting new AI battle");

    try {
      const humanDeck = DEFAULT_ASSETS_MANAGER.decode(humanDeckCode().trim());
      const aiDeck = DEFAULT_ASSETS_MANAGER.decode(aiDeckCode().trim());

      const state = Game.createInitialState({
        decks: [humanDeck, aiDeck],
        data: getData(CURRENT_VERSION),
        versionBehavior: CURRENT_VERSION,
      });

      let runtimeGame: Game | null = null;
      const aiIo = createSimpleAiIo(
        AI_WHO,
        () => runtimeGame?.state ?? null,
        ismctsWorkerClient,
        appendDebugLog,
      );
      const game = new Game(state);
      runtimeGame = game;
      game.players[HUMAN_WHO].io = humanIo;
      game.players[AI_WHO].io = aiIo;
      game.players[AI_WHO].config = {
        alwaysOmni: true,
        allowTuningAnyDice: true,
      };
      game.onPause = async (state, _mutations, canResume) => {
        setStateLog((logs) => [...logs, { state, canResume }]);
        appendDebugLog(
          `Pause snapshot: round=${state.roundNumber}, turn=${state.currentTurn}, phase=${state.phase}, resumable=${canResume}`,
        );
      };
      game.onIoError = (error) => {
        setErrorMessage(error.message);
        appendDebugLog(`IO error: ${error.message}`);
      };

      setCurrentGame(game);
      setRunning(true);
      setStatusText("Running (ISMCTS)");

      game
        .start()
        .then((battleWinner) => {
          setWinner(battleWinner);
          setRunning(false);
          if (battleWinner === null) {
            setStatusText("Finished: draw");
          } else {
            setStatusText(
              battleWinner === HUMAN_WHO ? "Finished: you win" : "Finished: AI wins",
            );
          }
          appendDebugLog(
            `Game finished: winner=${battleWinner === null ? "draw" : battleWinner}`,
          );
        })
        .catch((error) => {
          if (isTerminateError(error)) {
            return;
          }
          setRunning(false);
          setStatusText("Failed");
          setErrorMessage(error instanceof Error ? error.message : String(error));
          appendDebugLog(
            `Game failed: ${error instanceof Error ? error.message : String(error)}`,
          );
        });
    } catch (error) {
      setRunning(false);
      setStatusText("Failed to start");
      setErrorMessage(error instanceof Error ? error.message : String(error));
      appendDebugLog(
        `Game start failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  };

  onCleanup(() => {
    ismctsWorkerClient?.terminate();
    stopGame();
  });

  return (
    <Layout>
      <div class="container mx-auto h-full flex flex-col gap-4 min-h-0">
        <div class="flex flex-wrap items-center gap-3">
          <h2 class="text-2xl font-bold">AI Battle (Local)</h2>
          <A href="/" class="btn btn-soft-primary">
            Back Home
          </A>
          <button class="btn btn-solid-green" onClick={startGame}>
            {running() ? "Restart" : "Start"}
          </button>
          <button
            class="btn btn-soft-primary"
            onClick={() => exportBattleLog(stateLog())}
            disabled={stateLog().length === 0}
          >
            Export Log
          </button>
          <button
            class="btn btn-soft-primary"
            onClick={() => exportDebugLog(debugLogs())}
            disabled={debugLogs().length === 0}
          >
            Export Debug Log
          </button>
          <span class="text-sm text-gray-500">{statusText()}</span>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <label class="flex flex-col gap-1">
            <span class="font-medium">Your deck code</span>
            <input
              class="input input-solid"
              value={humanDeckCode()}
              onInput={(e) => setHumanDeckCode(e.currentTarget.value)}
              spellcheck={false}
            />
          </label>
          <label class="flex flex-col gap-1">
            <span class="font-medium">AI deck code</span>
            <input
              class="input input-solid"
              value={aiDeckCode()}
              onInput={(e) => setAiDeckCode(e.currentTarget.value)}
              spellcheck={false}
            />
          </label>
        </div>

        <div class="flex flex-wrap gap-4 items-center">
          <span class="text-sm text-gray-600">AI mode: ISMCTS (Worker)</span>
          <Show when={winner() !== undefined}>
            <span class="text-sm text-gray-600">
              Winner: {winner() === null ? "draw" : winner() === HUMAN_WHO ? "you" : "AI"}
            </span>
          </Show>
        </div>

        <Show when={errorMessage()}>
          {(message) => <div class="alert alert-outline-error">{message()}</div>}
        </Show>

        <div class="min-h-[7rem] max-h-52 overflow-auto rounded border border-gray-200 bg-gray-50 p-3">
          <div class="mb-2 flex items-center justify-between">
            <span class="font-medium">Debug Log (Human-readable)</span>
            <button
              class="btn btn-soft-primary btn-xs"
              onClick={() => setDebugLogs([])}
              disabled={debugLogs().length === 0}
            >
              Clear
            </button>
          </div>
          <pre class="text-xs whitespace-pre-wrap break-words">
            {debugLogs().length > 0 ? debugLogs().join("\n") : "No debug log yet."}
          </pre>
        </div>

        <div class="flex-grow min-h-0">
          <Chessboard autoHeight class="h-full" />
        </div>
      </div>
    </Layout>
  );
}

#!/usr/bin/env bun

import { DEFAULT_ASSETS_MANAGER } from "@gi-tcg/assets-manager";
import getData from "@gi-tcg/data";
import { ActionValidity, DiceType } from "@gi-tcg/typings";
import {
 CURRENT_VERSION,
 PureGameEngine,
 type GameData,
} from "@gi-tcg/core";
import { getDefaultGameConfig } from "./src/base/state";

const DECK_1 =
  "FZDByRUNGRCB0WoNFlGgWpEPE0AB9TAPFGCB9kgWGIERCoEQDLFADcQQDPFgacYWDJAA";
const DECK_2 =
  "FdHxNj8TAWDQxFkMFkAhyWIYCYDA45wOCUDQ5J0PGEAh9IIPGWDh9p4YC6FgirYRCxAA";

async function runRandomAI() {
  const data = getData(CURRENT_VERSION);
  const config = {
    ...getDefaultGameConfig(),
    errorLevel: "toleratePreview" as const,
  };
  let rerollCount = 0;
  const engine = new PureGameEngine(data, config, {
    rerollDice: (_state, _who, dice, _rerollCountLeft) => {
      const chosen = dice.filter((d) => d !== DiceType.Omni);
      if (chosen.length > 0) {
        rerollCount++;
      }
      return chosen;
    },
  });
  let state = engine.createInitialState(
    DEFAULT_ASSETS_MANAGER.decode(DECK_1),
    DEFAULT_ASSETS_MANAGER.decode(DECK_2),
  );

  console.log("=== headless random simulation start ===");
  const startTime = Date.now();
  let stepCount = 0;

  while (!state.isFinished && stepCount < 10000) {
    const actions = engine
      .getPossibleActions(state)
      .filter((a) => a.validity === ActionValidity.VALID);
    if (actions.length === 0) {
      console.log("No legal action. Simulation aborted.");
      break;
    }
    const candidates = [...actions];
    let executed = false;
    while (candidates.length > 0 && !executed) {
      const index = Math.floor(Math.random() * candidates.length);
      const [randomAction] = candidates.splice(index, 1);
      try {
        state = await engine.execute(state, randomAction);
        executed = true;
      } catch (e) {
        // Fast action generation is intentionally approximate for throughput tests.
        // If one action fails in sync execution, try another legal candidate.
        void e;
      }
    }
    if (!executed) {
      console.log("No executable action under sync constraints. Simulation aborted.");
      break;
    }
    stepCount++;
  }

  const costTime = Date.now() - startTime;
  const speed = costTime > 0 ? Math.floor(stepCount / (costTime / 1000)) : stepCount;
  console.log("=== simulation finished ===");
  console.log(`steps: ${stepCount}`);
  console.log(`time: ${costTime} ms`);
  console.log(`speed: ${speed} steps/s`);
  console.log(`winner: ${state.winner ?? "draw/none"}`);
  console.log(`rerolls: ${rerollCount}`);
}

void runRandomAI();

import { test, expect } from "bun:test";
import { shuffle, sortDice } from "../utils";
import { DiceType } from "@gi-tcg/typings";
import { StateSymbol, type PlayerState } from "../base/state";

test("sort dice", () => {
  const dice = [
    DiceType.Omni,
    DiceType.Electro,
    DiceType.Electro,
    DiceType.Dendro,
    DiceType.Pyro,
    DiceType.Pyro,
    DiceType.Cryo,
    DiceType.Hydro,
    DiceType.Anemo,
  ];
  const shuffled = shuffle(dice);
  // 草和雷是出战角色的骰子（有效骰）
  const playerState: PlayerState = {
    [StateSymbol]: "player",
    who: 0,
    activeCharacterId: -1,
    characters: [
      {
        [StateSymbol]: "character",
        id: -1,
        entities: [],
        variables: {} as never,
        definition: {
          __definition: "characters",
          id: 1601,
          skills: [],
          tags: ["dendro"],
          type: "character",
          varConfigs: {} as never,
          version: {
            from: "official",
            value: {
              predicate: "until",
              version: "v3.3.0",
            },
          },
          associatedNightsoulsBlessing: null,
          specialEnergy: null,
          enabledLunarReactions: [],
        },
      },
      {
        [StateSymbol]: "character",
        id: -2,
        entities: [],
        variables: {} as never,
        definition: {
          __definition: "characters",
          id: 1401,
          skills: [],
          tags: ["electro"],
          type: "character",
          varConfigs: {} as never,
          version: {
            from: "official",
            value: {
              predicate: "until",
              version: "v3.3.0",
            },
          },
          associatedNightsoulsBlessing: null,
          specialEnergy: null,
          enabledLunarReactions: [],
        },
      },
    ],
    hands: [],
    pile: [],
    initialPile: [],
    dice: shuffled,
    summons: [],
    supports: [],
    combatStatuses: [],
    canCharged: false,
    canPlunging: false,
    declaredEnd: false,
    skipNextTurn: false,
    hasDefeated: false,
    legendUsed: false,
    roundSkillLog: new Map(),
    removedEntities: [],
  };
  const sorted = sortDice(playerState, shuffled);
  expect(sorted).toEqual(dice);
});

import { Aura } from "@gi-tcg/typings";
import type { SkillDefinition } from "./skill";
import type { EntityDefinition, VariableConfig } from "./entity";
import type { WithVersionInfo } from "./version";
import type { LunarReaction } from "@gi-tcg/typings";

export type ElementTag =
  | "cryo"
  | "hydro"
  | "pyro"
  | "electro"
  | "anemo"
  | "geo"
  | "dendro";

export const WEAPON_TAGS = [
  "sword",
  "claymore",
  "pole",
  "catalyst",
  "bow",
  "otherWeapon",
] as const;

export type WeaponTag = (typeof WEAPON_TAGS)[number];

export const NATION_TAGS = [
  "mondstadt",
  "liyue",
  "inazuma",
  "sumeru",
  "fontaine",
  "natlan",
  "nodkrai",
  "fatui",
  "eremite",
  "monster",
  "hilichurl",
  "sacread",
  "calamity",
] as const;

export type NationTag = (typeof NATION_TAGS)[number];

// 虽然荒的英文是 Ousia，但是在代码里使用 pneuma 表示荒
export type ArkheTag =
  | "pneuma" // 荒
  | "ousia"; // 芒

export type CharacterTag =
  | ElementTag
  | WeaponTag
  | NationTag
  | ArkheTag;

export interface CharacterDefinition extends WithVersionInfo {
  readonly __definition: "characters";
  readonly type: "character";
  readonly id: number;
  readonly tags: readonly CharacterTag[];
  readonly varConfigs: CharacterVariableConfigs;
  readonly skills: readonly SkillDefinition[];
  readonly associatedNightsoulsBlessing: EntityDefinition | null;
  readonly enabledLunarReactions: readonly LunarReaction[];
  readonly specialEnergy: SpecialEnergyConfig | null;
}

export interface SpecialEnergyConfig {
  readonly variableName: string;
  readonly slotSize: number;
}

export interface CharacterVariableConfigs {
  readonly health: VariableConfig;
  readonly energy: VariableConfig;
  readonly maxHealth: VariableConfig;
  readonly maxEnergy: VariableConfig;
  readonly aura: VariableConfig<Aura>;
  readonly alive: VariableConfig<0 | 1>;
  readonly [x: string]: VariableConfig;
}

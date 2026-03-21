export { character } from "./character";
export { skill, ListenTo } from "./skill";
export { card } from "./card";
export { summon, status, combatStatus } from "./entity";
export { attachment } from "./attachment";
export { extension } from "./extension";
export {
  Registry,
  type GameData,
  type OnResolvedCallback,
  type VersionResolver,
  type IRegistrationScope,
} from "./registry";
export type {
  CardHandle,
  CharacterHandle,
  CombatStatusHandle,
  EntityHandle,
  EquipmentHandle,
  SkillHandle,
  StatusHandle,
  SummonHandle,
  SupportHandle,
  PassiveSkillHandle,
  ExtensionHandle,
} from "./type";
export { DiceType, DamageType, Aura, Reaction } from "@gi-tcg/typings";
export type {
  PlainCharacterState as CharacterState,
  PlainEntityState as EntityState,
} from "./context/utils";
export type { CharacterDefinition, EntityDefinition } from "../base/state";
export {
  type CustomEvent,
  createCustomEvent as customEvent,
} from "../base/custom_event";

export { originalDiceCostSizeOfCard as originalDiceCostOfCard } from "../utils";
export { flip, pair } from "@gi-tcg/utils";

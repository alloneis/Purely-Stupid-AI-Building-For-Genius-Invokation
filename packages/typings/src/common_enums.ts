import {
  DiceType as PbDiceType,
  DiceRequirementType as PbDiceRequirementType,
  DiceRequirement as PbDiceRequirement,
  DamageType as PbDamageType,
  AuraType as PbAuraType,
  ReactionType as PbReactionType,
} from "./gen/enums";

export const DiceType = {
  Void: PbDiceRequirementType.VOID,
  Cryo: PbDiceType.CRYO,
  Hydro: PbDiceType.HYDRO,
  Pyro: PbDiceType.PYRO,
  Electro: PbDiceType.ELECTRO,
  Anemo: PbDiceType.ANEMO,
  Geo: PbDiceType.GEO,
  Dendro: PbDiceType.DENDRO,
  Omni: PbDiceType.OMNI,
  Aligned: PbDiceRequirementType.ALIGNED,
  Energy: PbDiceRequirementType.ENERGY,
  Legend: PbDiceRequirementType.LEGEND,
} as const;
export type DiceType = (typeof DiceType)[keyof typeof DiceType];

export type DiceRequirement = Map<DiceType, number>;
export type ReadonlyDiceRequirement = ReadonlyMap<DiceType, number>;

export const DamageType = {
  Physical: PbDamageType.PHYSICAL,
  Cryo: PbDamageType.CRYO,
  Hydro: PbDamageType.HYDRO,
  Pyro: PbDamageType.PYRO,
  Electro: PbDamageType.ELECTRO,
  Anemo: PbDamageType.ANEMO,
  Geo: PbDamageType.GEO,
  Dendro: PbDamageType.DENDRO,
  Piercing: PbDamageType.PIERCING,
  Heal: PbDamageType.HEAL,
} as const;
export type DamageType = (typeof DamageType)[keyof typeof DamageType];

export const Aura = {
  None: PbAuraType.NONE,
  Cryo: PbAuraType.CRYO,
  Hydro: PbAuraType.HYDRO,
  Pyro: PbAuraType.PYRO,
  Electro: PbAuraType.ELECTRO,
  Dendro: PbAuraType.DENDRO,
  CryoDendro: PbAuraType.CRYO_DENDRO,
} as const;
export type Aura = (typeof Aura)[keyof typeof Aura];

export const Reaction = {
  Melt: PbReactionType.MELT,
  Vaporize: PbReactionType.VAPORIZE,
  Overloaded: PbReactionType.OVERLOADED,
  Superconduct: PbReactionType.SUPERCONDUCT,
  ElectroCharged: PbReactionType.ELECTRO_CHARGED,
  Frozen: PbReactionType.FROZEN,
  SwirlCryo: PbReactionType.SWIRL_CRYO,
  SwirlHydro: PbReactionType.SWIRL_HYDRO,
  SwirlPyro: PbReactionType.SWIRL_PYRO,
  SwirlElectro: PbReactionType.SWIRL_ELECTRO,
  CrystallizeCryo: PbReactionType.CRYSTALLIZE_CRYO,
  CrystallizeHydro: PbReactionType.CRYSTALLIZE_HYDRO,
  CrystallizePyro: PbReactionType.CRYSTALLIZE_PYRO,
  CrystallizeElectro: PbReactionType.CRYSTALLIZE_ELECTRO,
  Burning: PbReactionType.BURNING,
  Bloom: PbReactionType.BLOOM,
  Quicken: PbReactionType.QUICKEN,
  LunarElectroCharged: PbReactionType.LUNAR_ELECTRO_CHARGED,
  LunarBloom: PbReactionType.LUNAR_BLOOM,
  LunarCrystallizeHydro: PbReactionType.LUNAR_CRYSTALLIZE_HYDRO,
} as const;
export type Reaction = (typeof Reaction)[keyof typeof Reaction];
export type LunarReaction =
  | typeof Reaction.LunarElectroCharged
  | typeof Reaction.LunarBloom
  | typeof Reaction.LunarCrystallizeHydro;

export const CHARACTER_TAG_SHIELD = 1 << 0;
export const CHARACTER_TAG_BARRIER = 1 << 1;
export const CHARACTER_TAG_DISABLE_SKILL = 1 << 2;
export const CHARACTER_TAG_NIGHTSOULS_BLESSING = 1 << 3;
export const CHARACTER_TAG_BOND_OF_LIFE = 1 << 4;

export const CARD_TAG_ABYSS = 1 << 0;
export const CARD_TAG_CONDUCTIVE = 1 << 1;

export {
  PbDiceType,
  PbDiceRequirementType,
  PbDiceRequirement,
  PbDamageType,
  PbAuraType,
  PbReactionType,
};

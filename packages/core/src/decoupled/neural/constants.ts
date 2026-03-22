import { Aura, DiceType } from "@gi-tcg/typings";
import type { ElementTag, WeaponTag } from "../../base/character";
import type { EntityType } from "../../base/entity";

export const MAX_CHARACTERS = 3;
export const MAX_HAND_CARDS = 10;
export const MAX_SUMMONS = 4;
export const MAX_SUPPORTS = 4;
export const MAX_COMBAT_STATUSES = 10;
export const MAX_CHARACTER_ENTITIES = 8;

export const ELEMENT_TAGS: readonly ElementTag[] = [
  "cryo",
  "hydro",
  "pyro",
  "electro",
  "anemo",
  "geo",
  "dendro",
] as const;

export const WEAPON_TAGS: readonly WeaponTag[] = [
  "sword",
  "claymore",
  "pole",
  "catalyst",
  "bow",
  "otherWeapon",
] as const;

export const AURA_VALUES: readonly Aura[] = [
  Aura.None,
  Aura.Cryo,
  Aura.Hydro,
  Aura.Pyro,
  Aura.Electro,
  Aura.Dendro,
  Aura.CryoDendro,
] as const;

export const DICE_TYPES: readonly DiceType[] = [
  DiceType.Omni,
  DiceType.Cryo,
  DiceType.Hydro,
  DiceType.Pyro,
  DiceType.Electro,
  DiceType.Anemo,
  DiceType.Geo,
  DiceType.Dendro,
] as const;

export const ENTITY_TYPES: readonly EntityType[] = [
  "eventCard",
  "status",
  "combatStatus",
  "equipment",
  "support",
  "summon",
] as const;

export const ELEMENT_DIM = ELEMENT_TAGS.length;
export const WEAPON_DIM = WEAPON_TAGS.length;
export const AURA_DIM = AURA_VALUES.length;
export const DICE_DIM = DICE_TYPES.length;
export const ENTITY_TYPE_DIM = ENTITY_TYPES.length;

export const GLOBAL_FEATURE_DIM =
  1 +  // roundNumber (normalized)
  1 +  // currentTurn (0 or 1, relative to perspective)
  1 +  // phase one-hot placeholder (action=1, else=0)
  DICE_DIM + // self dice counts
  DICE_DIM + // oppo dice counts
  1 +  // self hand count (normalized)
  1 +  // oppo hand count (normalized)
  1 +  // self declaredEnd
  1 +  // oppo declaredEnd
  1 +  // self hasDefeated
  1 +  // oppo hasDefeated
  1 +  // self legendUsed
  1;   // oppo legendUsed

export const CHARACTER_FEATURE_DIM =
  1 +  // hp (normalized by maxHp)
  1 +  // hp raw / 10
  1 +  // maxHp / 10
  1 +  // energy (normalized by maxEnergy)
  1 +  // energy raw / 5
  1 +  // maxEnergy / 5
  AURA_DIM + // aura one-hot
  1 +  // alive
  1 +  // isActive
  ELEMENT_DIM + // element one-hot
  WEAPON_DIM +  // weapon one-hot
  1 +  // total shield on this character
  1 +  // equipment count
  1 +  // status count
  1;   // entity count total

export const CARD_FEATURE_DIM =
  1 +  // definitionId (raw float: id%10000/10000; Python model hashes→nn.Embedding)
  ENTITY_TYPE_DIM + // type one-hot
  1 +  // dice cost total (normalized)
  1 +  // isFast (from action if available)
  1 +  // is equipment
  1 +  // is support
  1 +  // is event
  1 +  // is talent
  1 +  // is legend
  1 +  // is food
  1;   // willBeEffectless (0 if unknown)

export const ENTITY_FEATURE_DIM =
  1 +  // definitionId (raw float: id%10000/10000; Python model hashes→nn.Embedding)
  ENTITY_TYPE_DIM + // type one-hot
  1 +  // usage
  1 +  // duration
  1 +  // shield
  1 +  // usagePerRound (first one found)
  1 +  // is equipment tag
  1 +  // is weapon tag
  1 +  // is artifact tag
  1 +  // is shield tag
  1;   // visibleVar value (if any)

export const MAX_ACTION_SLOTS = 64;

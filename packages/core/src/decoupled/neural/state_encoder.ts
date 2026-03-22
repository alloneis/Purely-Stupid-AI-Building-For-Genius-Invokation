import { Aura, DiceType } from "@gi-tcg/typings";
import { flip } from "@gi-tcg/utils";
import type {
  CharacterState,
  EntityState,
  GameState,
  PlayerState,
} from "../../base/state";
import type { ElementTag, WeaponTag, CharacterTag } from "../../base/character";
import type { EntityType } from "../../base/entity";
import type { PureGameState } from "../types";
import {
  MAX_CHARACTERS,
  MAX_HAND_CARDS,
  MAX_SUMMONS,
  MAX_SUPPORTS,
  MAX_COMBAT_STATUSES,
  MAX_CHARACTER_ENTITIES,
  ELEMENT_TAGS,
  WEAPON_TAGS,
  AURA_VALUES,
  DICE_TYPES,
  ENTITY_TYPES,
  ELEMENT_DIM,
  WEAPON_DIM,
  AURA_DIM,
  DICE_DIM,
  ENTITY_TYPE_DIM,
  GLOBAL_FEATURE_DIM,
  CHARACTER_FEATURE_DIM,
  CARD_FEATURE_DIM,
  ENTITY_FEATURE_DIM,
} from "./constants";

export interface EncodedState {
  global_features: Float32Array;

  self_characters: Float32Array;
  oppo_characters: Float32Array;

  hand_cards: Float32Array;
  hand_cards_mask: Float32Array;

  self_summons: Float32Array;
  self_summons_mask: Float32Array;

  oppo_summons: Float32Array;
  oppo_summons_mask: Float32Array;

  self_supports: Float32Array;
  self_supports_mask: Float32Array;

  oppo_supports: Float32Array;
  oppo_supports_mask: Float32Array;

  self_combat_statuses: Float32Array;
  self_combat_statuses_mask: Float32Array;

  oppo_combat_statuses: Float32Array;
  oppo_combat_statuses_mask: Float32Array;

  self_char_entities: Float32Array;
  self_char_entities_mask: Float32Array;

  oppo_char_entities: Float32Array;
  oppo_char_entities_mask: Float32Array;
}

function readNum(input: unknown, fallback = 0): number {
  return typeof input === "number" && Number.isFinite(input) ? input : fallback;
}

function oneHotElement(tags: readonly CharacterTag[]): Float32Array {
  const out = new Float32Array(ELEMENT_DIM);
  for (let i = 0; i < ELEMENT_TAGS.length; i++) {
    if (tags.includes(ELEMENT_TAGS[i])) {
      out[i] = 1;
    }
  }
  return out;
}

function oneHotWeapon(tags: readonly CharacterTag[]): Float32Array {
  const out = new Float32Array(WEAPON_DIM);
  for (let i = 0; i < WEAPON_TAGS.length; i++) {
    if (tags.includes(WEAPON_TAGS[i] as CharacterTag)) {
      out[i] = 1;
    }
  }
  return out;
}

function oneHotAura(aura: Aura): Float32Array {
  const out = new Float32Array(AURA_DIM);
  const idx = AURA_VALUES.indexOf(aura);
  if (idx >= 0) {
    out[idx] = 1;
  }
  return out;
}

function oneHotEntityType(type: EntityType): Float32Array {
  const out = new Float32Array(ENTITY_TYPE_DIM);
  const idx = ENTITY_TYPES.indexOf(type);
  if (idx >= 0) {
    out[idx] = 1;
  }
  return out;
}

function diceCounts(dice: readonly DiceType[]): Float32Array {
  const out = new Float32Array(DICE_DIM);
  for (const d of dice) {
    const idx = DICE_TYPES.indexOf(d);
    if (idx >= 0) {
      out[idx] += 1;
    }
  }
  for (let i = 0; i < out.length; i++) {
    out[i] /= 8;
  }
  return out;
}

function encodeCharacter(
  character: CharacterState,
  activeId: number,
): Float32Array {
  const f = new Float32Array(CHARACTER_FEATURE_DIM);
  let offset = 0;

  const hp = Math.max(0, readNum(character.variables.health));
  const maxHp = Math.max(1, readNum(character.variables.maxHealth, 10));
  const energy = Math.max(0, readNum(character.variables.energy));
  const maxEnergy = Math.max(1, readNum(character.variables.maxEnergy, 3));
  const aura = readNum(character.variables.aura, Aura.None) as Aura;
  const alive =
    typeof character.variables.alive === "number"
      ? character.variables.alive
      : hp > 0
        ? 1
        : 0;

  f[offset++] = hp / maxHp;
  f[offset++] = hp / 10;
  f[offset++] = maxHp / 10;
  f[offset++] = energy / maxEnergy;
  f[offset++] = energy / 5;
  f[offset++] = maxEnergy / 5;

  const auraOh = oneHotAura(aura);
  f.set(auraOh, offset);
  offset += AURA_DIM;

  f[offset++] = alive ? 1 : 0;
  f[offset++] = character.id === activeId ? 1 : 0;

  const elemOh = oneHotElement(character.definition.tags);
  f.set(elemOh, offset);
  offset += ELEMENT_DIM;

  const weapOh = oneHotWeapon(character.definition.tags);
  f.set(weapOh, offset);
  offset += WEAPON_DIM;

  let totalShield = 0;
  let equipCount = 0;
  let statusCount = 0;
  for (const entity of character.entities) {
    totalShield += readNum(entity.variables.shield);
    if (
      entity.definition.type === "equipment" ||
      entity.definition.tags.includes("weapon") ||
      entity.definition.tags.includes("artifact")
    ) {
      equipCount++;
    } else {
      statusCount++;
    }
  }
  f[offset++] = totalShield / 5;
  f[offset++] = equipCount / 3;
  f[offset++] = statusCount / 5;
  f[offset++] = character.entities.length / 8;

  return f;
}

function encodeEntity(entity: EntityState): Float32Array {
  const f = new Float32Array(ENTITY_FEATURE_DIM);
  let offset = 0;

  f[offset++] = (entity.definition.id % 10000) / 10000;

  const typeOh = oneHotEntityType(entity.definition.type);
  f.set(typeOh, offset);
  offset += ENTITY_TYPE_DIM;

  f[offset++] = readNum(entity.variables.usage) / 5;
  f[offset++] = readNum(entity.variables.duration) / 5;
  f[offset++] = readNum(entity.variables.shield) / 5;

  let usagePerRound = 0;
  if (typeof (entity.variables as any).usagePerRound === "number") {
    usagePerRound = (entity.variables as any).usagePerRound;
  }
  f[offset++] = usagePerRound / 3;

  const tags = entity.definition.tags;
  f[offset++] = tags.includes("weapon" as any) || entity.definition.type === "equipment" ? 1 : 0;
  f[offset++] = tags.includes("weapon" as any) ? 1 : 0;
  f[offset++] = tags.includes("artifact" as any) ? 1 : 0;
  f[offset++] = tags.includes("shield" as any) ? 1 : 0;

  const visVarName = entity.definition.visibleVarName;
  if (visVarName && typeof (entity.variables as any)[visVarName] === "number") {
    f[offset++] = (entity.variables as any)[visVarName] / 5;
  } else {
    f[offset++] = 0;
  }

  return f;
}

function encodeHandCard(entity: EntityState): Float32Array {
  const f = new Float32Array(CARD_FEATURE_DIM);
  let offset = 0;

  f[offset++] = (entity.definition.id % 10000) / 10000;

  const typeOh = oneHotEntityType(entity.definition.type);
  f.set(typeOh, offset);
  offset += ENTITY_TYPE_DIM;

  let totalCost = 0;
  if (entity.definition.skills.length > 0) {
    const skill = entity.definition.skills[0];
    const config = skill.initiativeSkillConfig;
    if (config) {
      for (const [type, count] of config.requiredCost) {
        if (type !== DiceType.Energy && type !== DiceType.Legend) {
          totalCost += count;
        }
      }
    }
  }
  f[offset++] = totalCost / 5;

  const firstSkill = entity.definition.skills[0];
  f[offset++] = firstSkill?.initiativeSkillConfig?.shouldFast ? 1 : 0;

  const tags = entity.definition.tags;
  const type = entity.definition.type;
  f[offset++] = type === "equipment" ? 1 : 0;
  f[offset++] = type === "support" ? 1 : 0;
  f[offset++] = type === "eventCard" ? 1 : 0;
  f[offset++] = tags.includes("talent" as any) ? 1 : 0;
  f[offset++] = tags.includes("legend" as any) ? 1 : 0;
  f[offset++] = tags.includes("food" as any) ? 1 : 0;

  f[offset++] = 0;

  return f;
}

function padMatrix(
  items: Float32Array[],
  maxSlots: number,
  featureDim: number,
): { data: Float32Array; mask: Float32Array } {
  const data = new Float32Array(maxSlots * featureDim);
  const mask = new Float32Array(maxSlots);
  const count = Math.min(items.length, maxSlots);
  for (let i = 0; i < count; i++) {
    data.set(items[i].subarray(0, featureDim), i * featureDim);
    mask[i] = 1;
  }
  return { data, mask };
}

function encodePlayerCharacters(
  player: PlayerState,
): Float32Array {
  const charFeatures: Float32Array[] = [];
  for (let i = 0; i < MAX_CHARACTERS; i++) {
    if (i < player.characters.length) {
      charFeatures.push(
        encodeCharacter(player.characters[i], player.activeCharacterId),
      );
    } else {
      charFeatures.push(new Float32Array(CHARACTER_FEATURE_DIM));
    }
  }
  const flat = new Float32Array(MAX_CHARACTERS * CHARACTER_FEATURE_DIM);
  for (let i = 0; i < MAX_CHARACTERS; i++) {
    flat.set(charFeatures[i], i * CHARACTER_FEATURE_DIM);
  }
  return flat;
}

function encodeCharEntities(
  player: PlayerState,
): { data: Float32Array; mask: Float32Array } {
  const totalSlots = MAX_CHARACTERS * MAX_CHARACTER_ENTITIES;
  const data = new Float32Array(totalSlots * ENTITY_FEATURE_DIM);
  const mask = new Float32Array(totalSlots);

  for (let ci = 0; ci < MAX_CHARACTERS && ci < player.characters.length; ci++) {
    const character = player.characters[ci];
    const baseSlot = ci * MAX_CHARACTER_ENTITIES;
    const count = Math.min(character.entities.length, MAX_CHARACTER_ENTITIES);
    for (let ei = 0; ei < count; ei++) {
      const encoded = encodeEntity(character.entities[ei]);
      data.set(encoded, (baseSlot + ei) * ENTITY_FEATURE_DIM);
      mask[baseSlot + ei] = 1;
    }
  }

  return { data, mask };
}

export class NeuralStateEncoder {
  encode(state: PureGameState, perspective: 0 | 1): EncodedState {
    const gs = state.gameState;
    const selfPlayer = gs.players[perspective];
    const oppoPlayer = gs.players[flip(perspective)];

    const global_features = this.encodeGlobal(gs, perspective, selfPlayer, oppoPlayer);

    const self_characters = encodePlayerCharacters(selfPlayer);
    const oppo_characters = encodePlayerCharacters(oppoPlayer);

    const handEncoded = selfPlayer.hands.map(encodeHandCard);
    const { data: hand_cards, mask: hand_cards_mask } = padMatrix(
      handEncoded,
      MAX_HAND_CARDS,
      CARD_FEATURE_DIM,
    );

    const selfSumEncoded = selfPlayer.summons.map(encodeEntity);
    const { data: self_summons, mask: self_summons_mask } = padMatrix(
      selfSumEncoded,
      MAX_SUMMONS,
      ENTITY_FEATURE_DIM,
    );

    const oppoSumEncoded = oppoPlayer.summons.map(encodeEntity);
    const { data: oppo_summons, mask: oppo_summons_mask } = padMatrix(
      oppoSumEncoded,
      MAX_SUMMONS,
      ENTITY_FEATURE_DIM,
    );

    const selfSupEncoded = selfPlayer.supports.map(encodeEntity);
    const { data: self_supports, mask: self_supports_mask } = padMatrix(
      selfSupEncoded,
      MAX_SUPPORTS,
      ENTITY_FEATURE_DIM,
    );

    const oppoSupEncoded = oppoPlayer.supports.map(encodeEntity);
    const { data: oppo_supports, mask: oppo_supports_mask } = padMatrix(
      oppoSupEncoded,
      MAX_SUPPORTS,
      ENTITY_FEATURE_DIM,
    );

    const selfCsEncoded = selfPlayer.combatStatuses.map(encodeEntity);
    const { data: self_combat_statuses, mask: self_combat_statuses_mask } =
      padMatrix(selfCsEncoded, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM);

    const oppoCsEncoded = oppoPlayer.combatStatuses.map(encodeEntity);
    const { data: oppo_combat_statuses, mask: oppo_combat_statuses_mask } =
      padMatrix(oppoCsEncoded, MAX_COMBAT_STATUSES, ENTITY_FEATURE_DIM);

    const { data: self_char_entities, mask: self_char_entities_mask } =
      encodeCharEntities(selfPlayer);
    const { data: oppo_char_entities, mask: oppo_char_entities_mask } =
      encodeCharEntities(oppoPlayer);

    return {
      global_features,
      self_characters,
      oppo_characters,
      hand_cards,
      hand_cards_mask,
      self_summons,
      self_summons_mask,
      oppo_summons,
      oppo_summons_mask,
      self_supports,
      self_supports_mask,
      oppo_supports,
      oppo_supports_mask,
      self_combat_statuses,
      self_combat_statuses_mask,
      oppo_combat_statuses,
      oppo_combat_statuses_mask,
      self_char_entities,
      self_char_entities_mask,
      oppo_char_entities,
      oppo_char_entities_mask,
    };
  }

  private encodeGlobal(
    gs: GameState,
    perspective: 0 | 1,
    selfPlayer: PlayerState,
    oppoPlayer: PlayerState,
  ): Float32Array {
    const f = new Float32Array(GLOBAL_FEATURE_DIM);
    let offset = 0;

    f[offset++] = gs.roundNumber / 15;
    f[offset++] = gs.currentTurn === perspective ? 1 : 0;
    f[offset++] = gs.phase === "action" ? 1 : 0;

    const selfDice = diceCounts(selfPlayer.dice);
    f.set(selfDice, offset);
    offset += DICE_DIM;

    const oppoDice = diceCounts(oppoPlayer.dice);
    f.set(oppoDice, offset);
    offset += DICE_DIM;

    f[offset++] = selfPlayer.hands.length / 10;
    f[offset++] = oppoPlayer.hands.length / 10;
    f[offset++] = selfPlayer.declaredEnd ? 1 : 0;
    f[offset++] = oppoPlayer.declaredEnd ? 1 : 0;
    f[offset++] = selfPlayer.hasDefeated ? 1 : 0;
    f[offset++] = oppoPlayer.hasDefeated ? 1 : 0;
    f[offset++] = selfPlayer.legendUsed ? 1 : 0;
    f[offset++] = oppoPlayer.legendUsed ? 1 : 0;

    return f;
  }
}

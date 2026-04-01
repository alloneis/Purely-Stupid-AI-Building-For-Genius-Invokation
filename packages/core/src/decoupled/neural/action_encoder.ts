import { flip } from "@gi-tcg/utils";
import type { Action } from "@gi-tcg/typings";
import type {
  CharacterState,
  GameState,
  PlayerState,
} from "../../base/state";
import { MAX_ACTION_SLOTS } from "./constants";

/**
 * Fixed action‑space layout (128 slots):
 *
 *   [0]        declareEnd
 *   [1..3]     switchActive to self character position 0 / 1 / 2
 *   [4..18]    useSkill: charPos(0‑2) × skillSlot(0‑4)
 *              skillSlot: 0=normal, 1=elemental, 2=burst, 3=technique, 4=equipment
 *   [19..88]   playCard: handPos(0‑9) × targetVariant(0‑6)
 *              variant: 0=noTarget, 1‑3=selfChar, 4‑6=oppoChar
 *   [89..98]   elementalTuning: handPos(0‑9)
 *   [99..127]  reserved / padding
 */

const IDX_DECLARE_END = 0;
const IDX_SWITCH_BASE = 1;
const IDX_SKILL_BASE = 4;
const SKILL_SLOTS_PER_CHAR = 5;
const IDX_CARD_BASE = 19;
const TARGET_VARIANTS = 7;
const IDX_TUNING_BASE = 89;

export interface ActionMapping {
  index: number;
  action: Action;
}

export class ActionIndexer {
  /**
   * Build a deterministic index for a single Action.
   * Returns -1 if the action cannot be mapped (should not happen
   * for well-formed legal actions).
   */
  actionToIndex(
    action: Action,
    state: GameState,
    perspective: 0 | 1,
  ): number {
    const payload = action.action;
    if (!payload) return -1;

    const player = state.players[perspective];
    const opponent = state.players[flip(perspective)];

    switch (payload.$case) {
      case "declareEnd":
        return IDX_DECLARE_END;

      case "switchActive": {
        const charPos = findCharacterPosition(
          player,
          payload.value.characterId,
        );
        if (charPos < 0) return -1;
        return IDX_SWITCH_BASE + charPos;
      }

      case "useSkill": {
        const result = findSkillSlot(
          player,
          payload.value.skillDefinitionId,
        );
        if (!result) return -1;
        return (
          IDX_SKILL_BASE +
          result.charPos * SKILL_SLOTS_PER_CHAR +
          result.skillPos
        );
      }

      case "playCard": {
        const handPos = findHandPosition(player, payload.value.cardId);
        if (handPos < 0) return -1;
        const variant = getTargetVariant(
          payload.value.targetIds,
          player.characters,
          opponent.characters,
        );
        return IDX_CARD_BASE + handPos * TARGET_VARIANTS + variant;
      }

      case "elementalTuning": {
        const handPos = findHandPosition(
          player,
          payload.value.removedCardId,
        );
        if (handPos < 0) return -1;
        return IDX_TUNING_BASE + handPos;
      }
    }

    return -1;
  }

  /**
   * Given a list of legal actions, return the full mapping array
   * (index → action) and the Float32 mask. When two actions
   * collide on the same index, the first one wins.
   */
  buildMapping(
    legalActions: readonly Action[],
    state: GameState,
    perspective: 0 | 1,
  ): {
    mask: Float32Array;
    mappings: ActionMapping[];
    indexToAction: Map<number, Action>;
  } {
    const mask = new Float32Array(MAX_ACTION_SLOTS);
    const mappings: ActionMapping[] = [];
    const indexToAction = new Map<number, Action>();

    for (const action of legalActions) {
      const idx = this.actionToIndex(action, state, perspective);
      if (idx < 0 || idx >= MAX_ACTION_SLOTS) continue;
      mappings.push({ index: idx, action });
      if (!indexToAction.has(idx)) {
        indexToAction.set(idx, action);
        mask[idx] = 1;
      }
    }

    return { mask, mappings, indexToAction };
  }

  /**
   * Convenience: produce only the 1D mask for ONNX.
   */
  buildMask(
    legalActions: readonly Action[],
    state: GameState,
    perspective: 0 | 1,
  ): Float32Array {
    return this.buildMapping(legalActions, state, perspective).mask;
  }
}

// --------------- helpers ---------------

function findCharacterPosition(
  player: PlayerState,
  characterId: number,
): number {
  for (let i = 0; i < player.characters.length; i++) {
    if (player.characters[i].id === characterId) return i;
  }
  return -1;
}

function findHandPosition(player: PlayerState, cardId: number): number {
  for (let i = 0; i < player.hands.length; i++) {
    if (player.hands[i].id === cardId) return i;
  }
  return -1;
}

interface SkillSlot {
  charPos: number;
  skillPos: number;
}

function findSkillSlot(
  player: PlayerState,
  skillDefId: number,
): SkillSlot | null {
  for (let ci = 0; ci < player.characters.length; ci++) {
    const char = player.characters[ci];

    for (const skill of char.definition.skills) {
      if (skill.initiativeSkillConfig !== null && skill.id === skillDefId) {
        let slot: number;
        switch (skill.initiativeSkillConfig.skillType) {
          case "normal":    slot = 0; break;
          case "elemental": slot = 1; break;
          case "burst":     slot = 2; break;
          case "technique": slot = 3; break;
          default:          slot = 4; break;
        }
        return { charPos: ci, skillPos: slot };
      }
    }

    for (const entity of char.entities) {
      for (const skill of entity.definition.skills) {
        if (skill.initiativeSkillConfig !== null && skill.id === skillDefId) {
          return { charPos: ci, skillPos: 4 };
        }
      }
    }
  }
  return null;
}

function getTargetVariant(
  targetIds: number[],
  selfChars: readonly CharacterState[],
  oppoChars: readonly CharacterState[],
): number {
  if (targetIds.length === 0) return 0;

  const firstTarget = targetIds[0];

  for (let i = 0; i < selfChars.length; i++) {
    if (selfChars[i].id === firstTarget) return 1 + i; // self char 0-2
  }
  for (let i = 0; i < oppoChars.length; i++) {
    if (oppoChars[i].id === firstTarget) return 4 + i; // oppo char 0-2
  }

  return 0; // non-character target -> no-target slot
}

/**
 * Decode an action index back to a human-readable description.
 */
const SKILL_TYPE_NAMES = ["normal", "elemental", "burst", "technique", "equipment"];
const TARGET_VARIANT_NAMES = [
  "noTarget", "selfChar0", "selfChar1", "selfChar2",
  "oppoChar0", "oppoChar1", "oppoChar2",
];

export function describeActionIndex(index: number): string {
  if (index === IDX_DECLARE_END) return "declareEnd";

  if (index >= IDX_SWITCH_BASE && index < IDX_SKILL_BASE) {
    return `switchActive(charPos=${index - IDX_SWITCH_BASE})`;
  }

  if (index >= IDX_SKILL_BASE && index < IDX_CARD_BASE) {
    const offset = index - IDX_SKILL_BASE;
    const charPos = Math.floor(offset / SKILL_SLOTS_PER_CHAR);
    const skillSlot = offset % SKILL_SLOTS_PER_CHAR;
    return `useSkill(charPos=${charPos}, ${SKILL_TYPE_NAMES[skillSlot] ?? `slot${skillSlot}`})`;
  }

  if (index >= IDX_CARD_BASE && index < IDX_TUNING_BASE) {
    const offset = index - IDX_CARD_BASE;
    const handPos = Math.floor(offset / TARGET_VARIANTS);
    const variant = offset % TARGET_VARIANTS;
    return `playCard(handPos=${handPos}, ${TARGET_VARIANT_NAMES[variant] ?? `v${variant}`})`;
  }

  if (index >= IDX_TUNING_BASE && index < IDX_TUNING_BASE + 10) {
    return `elementalTuning(handPos=${index - IDX_TUNING_BASE})`;
  }

  return `reserved(${index})`;
}

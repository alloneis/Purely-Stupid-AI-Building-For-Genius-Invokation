/** This file provides a wrapper of getEntityById/getEntityArea with getRaw */

import type {
  AttachmentState,
  CharacterState,
  EntityState,
  GameState,
  StateSymbol,
} from "../../base/state";
import { getRaw } from "./reactive";
import {
  getEntityArea as getEntityAreaOriginal,
  getEntityById as getEntityByIdOriginal,
} from "../../utils";
import type { ExEntityType } from "../type";

export function getEntityArea(state: GameState, id: number) {
  return getEntityAreaOriginal(getRaw(state), id);
}

export function getEntityById(state: GameState, id: number) {
  return getEntityByIdOriginal(getRaw(state), id);
}

export {
  elementOfCharacter,
  getActiveCharacterIndex,
  nationOfCharacter,
  weaponOfCharacter,
  allSkills,
  diceCostSizeOfCard,
  isCharacterInitiativeSkill,
  sortDice,
} from "../../utils";

export type PlainCharacterState = Omit<CharacterState, StateSymbol>;
export type PlainEntityState = Omit<EntityState, StateSymbol>;
export type PlainAttachmentState = Omit<AttachmentState, StateSymbol>;
export type PlainAnyState =
  | PlainCharacterState
  | PlainEntityState
  | PlainAttachmentState;
export type ExPlainEntityState<TypeT extends ExEntityType> =
  TypeT extends "character"
    ? PlainCharacterState
    : TypeT extends "attachment"
      ? PlainAttachmentState
      : PlainEntityState;

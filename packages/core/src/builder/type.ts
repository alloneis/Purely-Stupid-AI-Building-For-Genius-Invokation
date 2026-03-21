import type { DamageType } from "@gi-tcg/typings";
import type { CharacterTag } from "../base/character";
import type { EntityTag, EntityType } from "../base/entity";
import type {
  AttachmentState,
  CharacterState,
  EntityState,
} from "../base/state";

export type CharacterHandle = number & { readonly _char: unique symbol };
export type SkillHandle = number & { readonly _skill: unique symbol };
export type PassiveSkillHandle = number & {
  readonly _passiveSkill: unique symbol;
};
export type EntityHandle = number & { readonly _entity: unique symbol };
export type CardHandle = EntityHandle & { readonly _card: unique symbol };
export type StatusHandle = EntityHandle & { readonly _stat: unique symbol };
export type CombatStatusHandle = EntityHandle & {
  readonly _cStat: unique symbol;
};
export type SummonHandle = number & { readonly sm: unique symbol };
export type SupportHandle = EntityHandle &
  CardHandle & { readonly _support: unique symbol };
export type EquipmentHandle = EntityHandle &
  CardHandle & { readonly _equip: unique symbol };

export type AttachmentHandle = number & { readonly _attach: unique symbol };

export type ExtensionHandle<T = unknown> = number & {
  readonly _extSym: unique symbol;
  readonly type: T;
};

export type ExEntityType = "character" | EntityType | "attachment";

export type ExEntityState<TypeT extends ExEntityType> =
  TypeT extends "character"
    ? CharacterState
    : TypeT extends "attachment"
      ? AttachmentState
      : EntityState;

export type HandleT<T extends ExEntityType> = T extends "character"
  ? CharacterHandle
  : T extends "attachment"
    ? AttachmentHandle
    : T extends "eventCard"
      ? CardHandle
      : T extends "combatStatus"
        ? CombatStatusHandle
        : T extends "status"
          ? StatusHandle
          : T extends "equipment"
            ? EquipmentHandle
            : T extends "summon"
              ? SummonHandle
              : T extends "support"
                ? SupportHandle
                : T extends "passiveSkill"
                  ? SkillHandle
                  : never;

export type ExTag<TypeT extends ExEntityType> = TypeT extends "character"
  ? CharacterTag
  : TypeT extends EntityType
    ? EntityTag
    : never;

export type AppliableDamageType =
  | typeof DamageType.Cryo
  | typeof DamageType.Hydro
  | typeof DamageType.Pyro
  | typeof DamageType.Electro
  | typeof DamageType.Dendro
  | typeof DamageType.Geo;

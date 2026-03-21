import type { StateKind } from "../../base/state";
import type { ExEntityType } from "../type";

export const ReactiveStateSymbol: unique symbol = Symbol("ReactiveState");
export type ReactiveStateSymbol = typeof ReactiveStateSymbol;

export const RawStateSymbol: unique symbol = Symbol("ReactiveState/RawState");
export type RawStateSymbol = typeof RawStateSymbol;

export const LatestStateSymbol: unique symbol = Symbol("ReactiveState/LatestState");
export type LatestStateSymbol = typeof LatestStateSymbol;

export type EntityTypeToStateKind = {
  character: "character";
  status: "entity";
  equipment: "entity";
  combatStatus: "entity";
  summon: "entity";
  support: "entity";
  eventCard: "entity";
  extension: "extension";
  attachment: "attachment";
};

export abstract class ReactiveStateBase {
  abstract get [ReactiveStateSymbol](): StateKind;
  declare [RawStateSymbol]: object;
  abstract get [LatestStateSymbol](): object;
  cast<Ty extends ExEntityType>(): this & {
    readonly [ReactiveStateSymbol]: EntityTypeToStateKind[Ty];
  } {
    return this as any;
  }
  latest(): this[LatestStateSymbol] {
    return this[LatestStateSymbol];
  }
}

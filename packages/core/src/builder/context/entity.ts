import type { EntityState, EntityVariables } from "../../base/state";
import { GiTcgDataError } from "../../error";
import { type EntityArea, type EntityDefinition } from "../../base/entity";
import { diceCostSizeOfCard, getEntityArea, getEntityById } from "./utils";
import type { ContextMetaBase, SkillContext } from "./skill";
import {
  LatestStateSymbol,
  RawStateSymbol,
  ReactiveStateBase,
  ReactiveStateSymbol,
} from "./reactive_base";

class ReadonlyEntity<Meta extends ContextMetaBase> extends ReactiveStateBase {
  override get [ReactiveStateSymbol](): "entity" {
    return "entity";
  }
  declare [RawStateSymbol]: EntityState;
  override get [LatestStateSymbol](): EntityState {
    const state = getEntityById(
      this.skillContext.rawState,
      this.id,
    ) as EntityState;
    return state;
  }

  // 行动牌 area 可能会变动，不缓存
  // protected _area: EntityArea | undefined;
  constructor(
    protected readonly skillContext: SkillContext<Meta>,
    public readonly id: number,
  ) {
    super();
  }

  protected get state(): EntityState {
    return this[LatestStateSymbol];
  }
  get definition(): EntityDefinition {
    return this.state.definition;
  }
  get area(): EntityArea {
    return (getEntityArea(this.skillContext.rawState, this.id));
  }
  get who() {
    return this.area.who;
  }
  isMine() {
    return this.area.who === this.skillContext.callerArea.who;
  }
  getVariable<Name extends string>(
    name: Name,
  ): NonNullable<EntityVariables[Name]> {
    return this.state.variables[name];
  }

  /** 当前元素骰费用 */
  diceCost() {
    return diceCostSizeOfCard(this.skillContext.rawState, this.latest());
  }

  empowered() {
    // Empowerment: 206
    return this.state.attachments.some((att => att.definition.id === 206));
  }

  get master() {
    if (this.area.type !== "characters") {
      throw new GiTcgDataError("master expect a character area");
    }
    return this.skillContext.get<"character">(this.area.characterId);
  }
}

export class Entity<Meta extends ContextMetaBase> extends ReadonlyEntity<Meta> {
  setVariable(prop: string, value: number) {
    this.skillContext.setVariable(prop, value, this.state);
  }
  addVariable(prop: string, value: number) {
    this.skillContext.addVariable(prop, value, this.state);
  }
  addVariableWithMax(prop: string, value: number, maxLimit: number) {
    this.skillContext.addVariableWithMax(prop, value, maxLimit, this.state);
  }
  consumeUsage(count = 1) {
    this.skillContext.consumeUsage(count, this.state);
  }
  resetUsagePerRound() {
    this.skillContext.mutate({
      type: "resetVariables",
      scope: "usagePerRound",
      state: this.state,
    });
  }
  dispose() {
    this.skillContext.dispose(this.state);
  }
}

export type TypedEntity<Meta extends ContextMetaBase> =
  Meta["readonly"] extends true ? ReadonlyEntity<Meta> : Entity<Meta>;

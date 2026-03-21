import type {
  AttachmentState,
  EntityState,
  EntityVariables,
} from "../../base/state";
import { GiTcgDataError } from "../../error";
import { type EntityArea, type EntityDefinition } from "../../base/entity";
import { getEntityArea, getEntityById } from "./utils";
import type { ContextMetaBase, SkillContext } from "./skill";
import {
  LatestStateSymbol,
  RawStateSymbol,
  ReactiveStateBase,
  ReactiveStateSymbol,
} from "./reactive_base";
import type { AttachmentDefinition } from "../../base/attachment";
import type { RxEntityState } from "./reactive";

class ReadonlyAttachment<
  Meta extends ContextMetaBase,
> extends ReactiveStateBase {
  override get [ReactiveStateSymbol](): "attachment" {
    return "attachment";
  }
  declare [RawStateSymbol]: AttachmentState;
  override get [LatestStateSymbol](): AttachmentState {
    const state = getEntityById(
      this.skillContext.rawState,
      this.id,
    ) as AttachmentState;
    return state;
  }

  // protected _area: EntityArea | undefined;
  constructor(
    protected readonly skillContext: SkillContext<Meta>,
    public readonly id: number,
  ) {
    super();
  }

  protected get state(): AttachmentState {
    return this[LatestStateSymbol];
  }
  get definition(): AttachmentDefinition {
    return this.state.definition;
  }
  get area(): EntityArea {
    return getEntityArea(this.skillContext.rawState, this.id);
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

  get master(): RxEntityState<Meta, "eventCard" | "support" | "equipment"> {
    if (this.area.type !== "hands" && this.area.type !== "pile") {
      throw new GiTcgDataError("master expect a hands/pile area");
    }
    return this.skillContext.get<"eventCard" | "support" | "equipment">(
      this.area.cardId,
    );
  }
}

export class Attachment<
  Meta extends ContextMetaBase,
> extends ReadonlyAttachment<Meta> {
  setVariable(prop: string, value: number) {
    this.skillContext.setVariable(prop, value, this.state);
  }
  addVariable(prop: string, value: number) {
    this.skillContext.addVariable(prop, value, this.state);
  }
  addVariableWithMax(prop: string, value: number, maxLimit: number) {
    this.skillContext.addVariableWithMax(prop, value, maxLimit, this.state);
  }
  resetUsagePerRound() {
    this.skillContext.mutate({
      type: "resetVariables",
      scope: "usagePerRound",
      state: this.state,
    });
  }
  dispose(): never {
    throw new GiTcgDataError(
      "Attachment can not be disposed directly, for now",
    );
  }
}

export type TypedAttachment<Meta extends ContextMetaBase> =
  Meta["readonly"] extends true ? ReadonlyAttachment<Meta> : Attachment<Meta>;

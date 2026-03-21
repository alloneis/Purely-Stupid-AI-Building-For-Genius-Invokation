import type { ContextMetaBase, SkillContext } from "../builder/context/skill";
import type { GuessedTypeOfQuery } from "./types";
import { type QueryArgs, doSemanticQueryAction } from "./semantic";
import { getAllEntities } from "../utils";
import type { AnyState, GameState } from "../base/state";
import type {
  InitiativeSkillEventArg,
  DamageOrHealEventArg,
  DamageInfo,
  SwitchActiveEventArg,
  UseSkillEventArg,
} from "../base/skill";
import { GiTcgDataError } from "../error";
import { getRaw, type RxEntityState } from "../builder/context/reactive";

export function executeQuery<
  Meta extends ContextMetaBase,
  const Q extends string,
>(ctx: SkillContext<Meta>, q: Q): RxEntityState<Meta, GuessedTypeOfQuery<Q>>[] {
  const targetLength = (ctx.eventArg as any)?.targets?.length ?? 0;
  const state = getRaw(ctx.rawState);
  const allEntities = getAllEntities(state);
  const arg: QueryArgs = {
    state,
    allEntities,
    callerWho: ctx.callerArea.who,
    candidates: allEntities,
    externals: {
      self: () => ctx.self.latest(),
      master: () => {
        const area = ctx.self.area;
        if (area.type !== "characters") {
          throw new GiTcgDataError(`This caller do not have @master`);
        }
        return ctx.get(area.characterId).latest();
      },
      event: {
        skillCaller: () => (ctx.eventArg as UseSkillEventArg).skill.caller,
        switchTo: () => (ctx.eventArg as SwitchActiveEventArg).switchInfo.to,
      },
      damage: {
        target: () => (ctx.eventArg as DamageOrHealEventArg<DamageInfo>).target,
      },
      targets: Object.fromEntries(
        new Array(targetLength)
          .fill(0)
          .map((_, i) => [
            `${i}`,
            () => (ctx.eventArg as InitiativeSkillEventArg).targets[i],
          ]),
      ),
    },
  };
  const result = doSemanticQueryAction(q, arg)
  return result.map((st) => ctx.get<any>(st)) as RxEntityState<Meta, GuessedTypeOfQuery<Q>>[];
}

export function executeQueryOnState(
  state: GameState,
  who: 0 | 1,
  q: string,
): AnyState[] {
  const allEntities = getAllEntities(state);
  return doSemanticQueryAction(q, {
    state,
    allEntities,
    candidates: allEntities,
    callerWho: who,
    externals: {},
  });
}

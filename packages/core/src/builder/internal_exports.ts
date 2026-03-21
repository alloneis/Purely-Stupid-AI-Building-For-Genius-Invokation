// INTERNAL exports
// 为其他包提供一些内部接口，如 @gi-tcg/test, @gi-tcg/data-vscode-ext

export { builderWeakRefs } from "./registry";
export { CardBuilder } from "./card";
export {
  TriggeredSkillBuilder,
  InitiativeSkillBuilder,
  TechniqueBuilder,
} from "./skill";
export { EntityBuilder } from "./entity";
export { CharacterBuilder } from "./character";
export { ExtensionBuilder, EXTENSION_ID_OFFSET } from "./extension";
export { SkillContext } from "./context/skill";
export { EVENT_MAP } from "../base/skill";

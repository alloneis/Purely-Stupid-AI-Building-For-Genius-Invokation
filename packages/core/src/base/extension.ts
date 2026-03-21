import type { SkillDefinition } from "./skill";
import type { VersionInfo } from "./version";

export interface ExtensionDefinition {
  readonly __definition: "extensions";
  readonly type: "extension";
  readonly id: number;
  readonly version: VersionInfo;
  readonly description: string;
  readonly initialState: unknown;
  readonly skills: readonly SkillDefinition[];
}

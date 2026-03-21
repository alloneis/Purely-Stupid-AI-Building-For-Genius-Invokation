import type { VariableConfig } from "../base/entity";

export function createVariable<const T extends number>(initialValue: T, forceOverwrite = false): VariableConfig<T> {
  return {
    initialValue,
    recreateBehavior: {
      type: forceOverwrite ? "overwrite" : "default",
    },
  };
}

export function createVariableCanAppend(initialValue: number, appendLimit = Infinity, appendValue?: number): VariableConfig {
  appendValue ??= initialValue;
  return {
    initialValue,
    recreateBehavior: {
      type: "append",
      appendLimit,
      appendValue,
    }
  }
}

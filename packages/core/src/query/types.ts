type Split<S extends string> = string extends S
  ? readonly string[]
  : S extends `${infer A} ${infer B}`
    ? readonly [A, ...(B extends "" ? readonly [] : Split<B>)]
    : readonly [S];

type GuessTypeFromSplitted<S extends readonly string[]> = S extends readonly [
  infer First extends string,
  ...infer Rest extends readonly string[],
]
  ? First extends
      | "character"
      | "characters"
      | "active"
      | "prev"
      | "next"
      | "standby"
    ? "character"
    : First extends "combat"
      ? "combatStatus"
      : First extends "summon" | "summons"
        ? "summon"
        : First extends "support" | "supports"
          ? "support"
          : First extends "status" | "statuses"
            ? "status"
            : First extends "equipment" | "equipments"
              ? "equipment"
              : GuessTypeFromSplitted<Rest>
  : any;

export type GuessedTypeOfQuery<Q extends string> = GuessTypeFromSplitted<
  Split<Q>
>;

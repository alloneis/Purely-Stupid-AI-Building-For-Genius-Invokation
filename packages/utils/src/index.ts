export * from "./dice";

export function flip(who: 0 | 1): 0 | 1 {
  return (1 - who) as 0 | 1;
}

export const PAIR_SYMBOL: unique symbol = Symbol("pair");

export function pair<T>(value: T): [T, T] {
  const ret: [T, T] = [value, value];
  Object.defineProperty(ret, PAIR_SYMBOL, { value: true });
  return ret;
}

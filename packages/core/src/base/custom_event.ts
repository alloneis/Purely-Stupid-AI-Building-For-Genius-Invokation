import { NoReactiveSymbol } from "../builder/context/reactive";

class CustomEvent<T = unknown> {
  [NoReactiveSymbol] = true;
  _typeGuard!: T;
  name: string;
  constructor(name?: string) {
    this.name = name ?? "";
  }
}

export { type CustomEvent };

export function createCustomEvent<T = void>(name?: string) {
  return new CustomEvent<T>(name);
}

export function isCustomEvent(value: unknown): value is CustomEvent {
  return value instanceof CustomEvent;
}

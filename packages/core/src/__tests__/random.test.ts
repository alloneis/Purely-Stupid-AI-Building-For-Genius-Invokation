import { test, expect } from "bun:test";
import { nextRandom, randomSeed } from "../random";

test("random seed", () => {
  const x = randomSeed();
  expect(x).toBeInteger();
  expect(x).toBeGreaterThanOrEqual(0);
  expect(x).toBeLessThan(2147483647);
})

test("random generator", () => {
  const seed = 3553;
  let x = seed;
  for (let i = 0; i < 5; i ++) {
    x = nextRandom(x);
  }
  expect(x).toBe(314840640);
})

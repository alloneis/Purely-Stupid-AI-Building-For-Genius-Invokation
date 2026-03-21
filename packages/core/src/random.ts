/**
 * This file contains a simple implementation of "minstd" Linear Congruential Generator.
 */

const A = 48271; // "minstd"
const C = 0;
const M = 2147483647; // 2^31 - 1

/**
 * Random integer in [0, 2147483647)
 */
export function randomSeed() {
  return Math.floor(Math.random() * M);
}

export function nextRandom(x: number) {
  return (A * x + C) % M;
}


import { SERVER_HOST, WEB_CLIENT_BASE_PATH } from "@gi-tcg/config";

export const BACKEND_BASE_URL = `${SERVER_HOST || location.origin}${WEB_CLIENT_BASE_PATH}api`;

export const GITHUB_AUTH_REDIRECT_URL = `${BACKEND_BASE_URL}/auth/github/callback`;

function parsePositiveIntEnv(
  key: string,
  fallback: number,
  min: number,
  max: number,
): number {
  const raw = import.meta.env[key];
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return fallback;
  }
  return Math.max(min, Math.min(max, Math.floor(parsed)));
}

const DEFAULT_AI_POOL_SIZE = Math.max(
  1,
  Math.min(4, (typeof navigator !== "undefined" ? navigator.hardwareConcurrency ?? 4 : 4) - 1),
);

export const AI_WORKER_POOL_SIZE = parsePositiveIntEnv(
  "VITE_AI_WORKER_POOL_SIZE",
  DEFAULT_AI_POOL_SIZE,
  1,
  16,
);
export const AI_WORKER_PARALLEL_SHARDS = parsePositiveIntEnv(
  "VITE_AI_WORKER_PARALLEL_SHARDS",
  AI_WORKER_POOL_SIZE,
  1,
  AI_WORKER_POOL_SIZE,
);
export const AI_WORKER_TIMEOUT_MS = parsePositiveIntEnv(
  "VITE_AI_WORKER_TIMEOUT_MS",
  12_000,
  1000,
  120_000,
);
export const AI_ISMCTS_ITERATIONS = parsePositiveIntEnv(
  "VITE_AI_ISMCTS_ITERATIONS",
  420,
  50,
  5000,
);
export const AI_ISMCTS_DETERMINIZATION_COUNT = parsePositiveIntEnv(
  "VITE_AI_ISMCTS_DETERMINIZATION_COUNT",
  5,
  1,
  32,
);
export const AI_ISMCTS_MAX_DEPTH = parsePositiveIntEnv(
  "VITE_AI_ISMCTS_MAX_DEPTH",
  12,
  2,
  80,
);

# `@gi-tcg/web-client`

## AI Worker Scheduler Env

You can tune local AI scheduling and search budget through Vite env variables:

- `VITE_AI_WORKER_POOL_SIZE` (default: auto, up to 4)
- `VITE_AI_WORKER_PARALLEL_SHARDS` (default: same as pool size)
- `VITE_AI_WORKER_TIMEOUT_MS` (default: `12000`)
- `VITE_AI_ISMCTS_ITERATIONS` (default: `420`)
- `VITE_AI_ISMCTS_DETERMINIZATION_COUNT` (default: `5`)
- `VITE_AI_ISMCTS_MAX_DEPTH` (default: `12`)


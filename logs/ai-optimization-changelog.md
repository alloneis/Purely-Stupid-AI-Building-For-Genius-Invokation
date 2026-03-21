# AI Optimization Changelog

Date: 2026-03-07
Source: `optimizetask`

## Contradictions filtered out

1. `dynamic_heuristic_evaluator` had opposite directions:
   - "Use board value for long-term investment"
   - "Delete board value/boardTerm entirely"
   - Practical decision: keep board value, but damp and de-bias it.
2. Search budget proposal (`iterations=1500`, `maxDepth=25`) conflicts with worker timeout and UI responsiveness.
   - Practical decision: moderate increase only.
3. "Remove all pruning" conflicts with runtime stability in browser worker.
   - Practical decision: switch to soft widening + weighted exploration, not full hard cut or full brute force.
4. Full reaction-aware attention (`canReact`, exact element chain) lacked stable helper pipeline in current code path.
   - Practical decision: use reliable priors from existing `quickActionGain` and add PUCT-style selection.

## Implemented practical optimizations

### 1) `packages/core/src/decoupled/ismcts_agent.ts`
- Increased default search budget moderately:
  - `iterations`: `240 -> 420`
  - `maxDepth`: `8 -> 12`
- Root candidate selection changed from hard prune to soft widening:
  - Keep pruned core set.
  - Add high-priority fallback actions from full legal set until a target ratio is reached.
- Child selection upgraded to PUCT-style exploration:
  - Exploration term is now prior-weighted by softmax probabilities from `quickActionGain`.
- Expansion no longer does hard prune-first random pick:
  - Uses weighted softmax sampling over untried legal actions.
- Rollout changed from deterministic top-pick to weighted softmax sampling:
  - Reduces "single-track" simulation bias.
  - Keeps low-probability alternatives alive.

### 2) `packages/core/src/decoupled/action_pruner.ts`
- Rebalanced `quickActionGain` to reduce reckless card spam:
  - `useSkill` reward increased and low-HP finish bonus widened (`<=3`).
  - `playCard` now strongly penalizes `willBeEffectless`.
  - `declareEnd` now state-aware: better only when no dice left.
  - `switchActive` gets context bonuses:
    - switch-to burst-ready target
    - emergency switch when current active is low HP
- Increased dice/energy cost penalty to better reflect resource burn.

### 3) `packages/core/src/decoupled/dynamic_heuristic_evaluator.ts`
- Kept board evaluation, but reduced fake-value bias:
  - Lowered default board weights (`wSummon`, `wSupport`, `wEquip`).
  - Applied global damping to `boardTerm` in final evaluation.
  - Support/status/entity board scoring now rewards actionable charge/usage more than passive presence.

## Explicitly not implemented (by design)

- No full deletion of board evaluation terms.
- No direct jump to ultra-heavy search settings (`1500 x depth 25`).
- No fully brute-force "no pruning anywhere" mode.
- No unfinished reaction helper stubs (`canReact`, incomplete element plumbing) merged into production path.

## Validation

- Type checks run after changes:
  - `bun run --filter @gi-tcg/core check` (pass)
  - `bun run --filter @gi-tcg/web-client check` (pass)

## Follow-up: canReact infrastructure

Date: 2026-03-07

- Added lightweight reaction query API to `RuleEngine`:
  - `RuleEngine.getTriggeredReaction(activeElement, targetAura, enabledLunarReactions?)`
  - `RuleEngine.canTriggerReaction(activeElement, targetAura, enabledLunarReactions?)`
- Internally mapped `DiceType` element to elemental `DamageType`, then reused canonical
  reaction logic from `base/reaction.ts` (`getReaction`) to avoid duplicate reaction tables.
- Integrated this into `ActionPruner.quickActionGain` attention:
  - `useSkill`: extra gain when active element can react with opponent active aura.
  - `switchActive`: extra gain when switch target element can react with opponent active aura.

## Follow-up: Worker pool scheduler

Date: 2026-03-07

- Added async Promise-based scheduler wrapper: `packages/web-client/src/ai/game_manager.ts`
  - Request/response correlation by unique `id`.
  - Timeout and crash handling.
  - Configurable worker pool and shard fan-out.
  - Sharded result merge by aggregated candidate visits/value.
- Upgraded worker protocol in `packages/web-client/src/workers/ismcts.worker.ts`
  - Supports per-request `searchConfig`.
  - Supports `shardId`.
  - Returns `candidates` for shard-level merge.
  - Keeps existing debug payload (`top3`, `processLogs`, `reason`).
- Integrated scheduler into `packages/web-client/src/pages/AIBattle.tsx`
  - Replaced inline single-worker transport with `AIWorkerWrapper`.
  - Uses `await agent.getBestAction(...)` directly.
  - Keeps existing human-readable debug log output.
- Added env setup knobs in `packages/web-client/src/config.ts` and documented in
  `packages/web-client/README.md`.



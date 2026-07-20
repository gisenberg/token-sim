# Token Sim

Token Sim is a deterministic agent-workload simulator for comparing model latency, context growth, cache behavior, token usage, and spend.
It models complete multi-request workloads instead of treating a model's decode rate as the end-to-end application rate.

## What it models

- Network, prefill, hidden reasoning, visible decode, tool argument decode, tool execution, subagent, and compaction phases.
- Uncached input, cache writes, cache hits, visible output, hidden reasoning output, and tool output as separate ledger entries.
- Per-request pricing, including provider cache rates and long-context pricing thresholds.
- Parallel subagent wall-clock time while retaining additive token usage and cost.
- Hardware and decode-engine profiles, including measured Qwen 3.5 122B MTP-2 results.
- Exact configured visible output across a completed workload.

The selected prompt context is added to an 18K-token coding-agent system and tool-schema baseline.
Both are included in the context meter, prefill time, cache accounting, and billed input.

The forecast chart and the running cards are projections of the same event ledger.
This keeps chart totals, card totals, playback, and final observed metrics consistent.

## Throughput metrics

Configured decode is the model's visible-token decode rate during visible decode phases.
Observed decode divides visible tokens by time spent in those phases, so it should match the configured rate after completion.
End-to-end output divides visible tokens by the entire workload duration, including prefill, reasoning, tools, and network waits.

Cloud decode rates are labeled as reference estimates because providers generally do not publish a stable standard-service token rate for every model.
Local rates labeled as measured come from the linked benchmark source.

## Development

```bash
npm install
npm run dev
```

The app is served under `/token-sim/` in production builds.

## Verification

```bash
npm run lint
npm test
npm run build
npm audit
```

The simulation tests cover exact output, cache behavior, cache invalidation after compaction, cost calculation, long-context rates, MTP-2 duration, parallel subagents, and time interpolation.

## Data maintenance

Model and pricing data live in `src/data/catalog.js`.
Every catalog entry includes a source URL and an evidence label for throughput.
Provider pricing and model availability should be rechecked before updating the catalog's as-of date.

The pure workload engine lives in `src/simulation/engine.js`.
UI code should consume its plan and sample APIs instead of calculating cost or timing independently.

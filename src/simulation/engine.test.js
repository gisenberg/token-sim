import { describe, expect, it } from 'vitest'
import {
  buildSimulationPlan,
  evaluatePlan,
  getEventDuration,
  resolvePricing,
} from './engine.js'

const cloudModel = {
  id: 'cloud-test',
  name: 'Cloud Test',
  hardware: 'Test API',
  decodeRate: 100,
  prefillRate: 1_000,
  maxContext: 1_000_000,
  cachePrefillFactor: 0.1,
  reasoningContextFactor: 0,
  pricing: {
    input: 2,
    cachedInput: 0.2,
    output: 10,
    longContext: { threshold: 200_000, input: 4, cachedInput: 0.4, output: 15 },
  },
}

const directPlan = (overrides = {}) => buildSimulationPlan({
  model: cloudModel,
  systemTokens: 1_000,
  promptTokens: 9_000,
  outputTokens: 2_000,
  reasoningTokens: 500,
  toolSteps: [],
  caching: true,
  networkLatencyMs: 100,
  ...overrides,
})

describe('buildSimulationPlan', () => {
  it('runs a direct response and lands exactly on the selected visible output', () => {
    const plan = directPlan()
    expect(plan.totals.visibleOutput).toBe(2_000)
    expect(plan.totals.reasoningOutput).toBe(500)
    expect(plan.totals.output).toBe(2_500)
    expect(plan.totals.input).toBe(10_000)
    expect(plan.totals.uncachedInput).toBe(10_000)
    expect(plan.totals.cachedInput).toBe(0)
    expect(plan.totals.requests).toBe(1)
    expect(evaluatePlan(plan, plan.duration).metrics).toEqual(plan.totals)
  })

  it('never treats the first request as a cache hit', () => {
    const plan = directPlan()
    const firstPrefill = plan.events.find((event) => event.kind === 'prefill')
    expect(firstPrefill.delta.uncachedInput).toBe(10_000)
    expect(firstPrefill.delta.cachedInput).toBe(0)
  })

  it('reuses only the exact previous request prefix in a tool loop', () => {
    const plan = directPlan({
      outputTokens: 1_000,
      reasoningTokens: 0,
      toolSteps: [{ label: 'Read file', decodeTokens: 100, resultTokens: 500, execMs: 0 }],
    })
    const prefills = plan.events.filter((event) => event.kind === 'prefill')
    const firstVisible = plan.events.find((event) => event.kind === 'visible-decode')
    expect(prefills).toHaveLength(2)
    expect(prefills[0].delta.uncachedInput).toBe(10_000)
    expect(prefills[1].delta.cachedInput).toBe(10_000)
    expect(prefills[1].delta.uncachedInput).toBe(100 + 500 + firstVisible.delta.visibleOutput)
    expect(plan.totals.visibleOutput).toBe(1_000)
    expect(plan.totals.toolOutput).toBe(100)
  })

  it('charges input and output from the same per-request ledger', () => {
    const plan = directPlan({ reasoningTokens: 0 })
    const expected = 10_000 / 1_000_000 * 2 + 2_000 / 1_000_000 * 10
    expect(plan.totals.cost).toBeCloseTo(expected, 10)
  })

  it('applies long-context rates per request', () => {
    expect(resolvePricing(cloudModel.pricing, 200_000).input).toBe(2)
    expect(resolvePricing(cloudModel.pricing, 200_001)).toMatchObject({
      input: 4,
      cachedInput: 0.4,
      output: 15,
      longContextApplied: true,
    })
  })

  it('books Anthropic-style cache writes separately from cache hits', () => {
    const model = {
      ...cloudModel,
      pricing: { input: 5, cachedInput: 0.5, cacheWriteInput: 6.25, output: 25 },
    }
    const plan = directPlan({ model, reasoningTokens: 0 })
    expect(plan.totals.cacheWriteInput).toBe(10_000)
    expect(plan.totals.uncachedInput).toBe(0)
    expect(plan.totals.cachedInput).toBe(0)
    expect(plan.totals.cost).toBeCloseTo(0.1125, 10)
  })

  it('bills every request as fresh input when caching is disabled', () => {
    const plan = directPlan({
      caching: false,
      reasoningTokens: 0,
      outputTokens: 1_000,
      toolSteps: [{ label: 'Read file', decodeTokens: 100, resultTokens: 500, execMs: 0 }],
    })
    expect(plan.totals.cachedInput).toBe(0)
    expect(plan.totals.cacheWriteInput).toBe(0)
    expect(plan.totals.uncachedInput).toBe(plan.totals.input)
  })

  it('invalidates the reusable prefix after automatic compaction', () => {
    const model = { ...cloudModel, maxContext: 20_000 }
    const plan = directPlan({
      model,
      outputTokens: 1_000,
      reasoningTokens: 0,
      toolSteps: [
        { label: 'Large read', decodeTokens: 100, resultTokens: 7_000, execMs: 0 },
        { label: 'Second read', decodeTokens: 100, resultTokens: 7_000, execMs: 0 },
      ],
    })
    const compactionIndex = plan.events.findIndex((event) => event.kind === 'compaction')
    const nextPrefill = plan.events.slice(compactionIndex + 1).find((event) => event.kind === 'prefill')
    expect(plan.context.compactions).toBeGreaterThan(0)
    expect(nextPrefill.delta.cachedInput).toBe(0)
    expect(nextPrefill.delta.uncachedInput).toBe(nextPrefill.detail.requestInput)
    expect(plan.totals.visibleOutput).toBe(1_000)
  })

  it('changes duration with a decode profile without changing observed token totals', () => {
    const baseline = directPlan({ reasoningTokens: 0, model: { ...cloudModel, decodeRate: 21 } })
    const mtp2 = directPlan({ reasoningTokens: 0, model: { ...cloudModel, decodeRate: 49 } })
    expect(baseline.totals.visibleOutput).toBe(mtp2.totals.visibleOutput)
    expect(getEventDuration(baseline, 'visible-decode')).toBeCloseTo(2_000 / 21, 10)
    expect(getEventDuration(mtp2, 'visible-decode')).toBeCloseTo(2_000 / 49, 10)
    expect(mtp2.duration).toBeLessThan(baseline.duration)
  })

  it('keeps subagent requests parallel in time and additive in cost and tokens', () => {
    const delegate = { ...cloudModel, id: 'delegate', decodeRate: 200, pricing: { input: 1, cachedInput: 0.1, output: 2 } }
    const plan = directPlan({
      reasoningTokens: 0,
      subagentWaves: [{ label: 'Explore', count: 3, contextPerAgent: 2_000, outputPerAgent: 500, toolsPerAgent: 2 }],
      delegate,
    })
    const wave = plan.events.find((event) => event.kind === 'subagents')
    expect(wave.delta.input).toBe(6_000)
    expect(wave.delta.output).toBe(1_500)
    expect(wave.delta.visibleOutput).toBe(0)
    expect(wave.delta.requests).toBe(3)
    expect(wave.duration).toBeCloseTo(2 + 1 + 2.5, 10)
    expect(plan.totals.visibleOutput).toBe(2_000)
    expect(plan.totals.output).toBe(3_500)
  })
})

describe('evaluatePlan', () => {
  it('interpolates active event metrics and clamps at both ends', () => {
    const plan = directPlan({ reasoningTokens: 0 })
    expect(evaluatePlan(plan, -10).metrics.cost).toBe(0)
    const decode = plan.events.find((event) => event.kind === 'visible-decode')
    const middle = evaluatePlan(plan, decode.start + decode.duration / 2)
    expect(middle.metrics.visibleOutput).toBeCloseTo(1_000, 6)
    expect(evaluatePlan(plan, plan.duration + 10).metrics).toEqual(plan.totals)
  })
})

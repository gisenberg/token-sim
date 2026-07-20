export const DEFAULT_SYSTEM_TOKENS = 18_000
export const DEFAULT_NETWORK_LATENCY_MS = 120

const EMPTY_METRICS = Object.freeze({
  cost: 0,
  input: 0,
  uncachedInput: 0,
  cachedInput: 0,
  cacheWriteInput: 0,
  output: 0,
  visibleOutput: 0,
  reasoningOutput: 0,
  toolOutput: 0,
  requests: 0,
})

const addMetrics = (left, right) => Object.fromEntries(
  Object.keys(EMPTY_METRICS).map((key) => [key, (left[key] ?? 0) + (right[key] ?? 0)]),
)

const scaleMetrics = (metrics, factor) => Object.fromEntries(
  Object.keys(EMPTY_METRICS).map((key) => [key, (metrics[key] ?? 0) * factor]),
)

export const resolvePricing = (pricing, requestInputTokens) => {
  if (!pricing) return null
  if (pricing.longContext && requestInputTokens > pricing.longContext.threshold) {
    return { ...pricing, ...pricing.longContext, longContextApplied: true }
  }
  return { ...pricing, longContextApplied: false }
}

const inputCost = (rates, usage) => {
  if (!rates) return 0
  return (
    usage.uncachedInput * rates.input
    + usage.cachedInput * (rates.cachedInput ?? rates.input)
    + usage.cacheWriteInput * (rates.cacheWriteInput ?? rates.input)
  ) / 1_000_000
}

const outputCost = (rates, tokens) => rates ? (tokens * rates.output) / 1_000_000 : 0

const distribute = (total, weights) => {
  if (weights.length === 0) return []
  const weightTotal = weights.reduce((sum, value) => sum + value, 0)
  if (weightTotal <= 0) return weights.map((_, index) => index === weights.length - 1 ? total : 0)
  let assigned = 0
  return weights.map((weight, index) => {
    if (index === weights.length - 1) return total - assigned
    const value = Math.floor(total * weight / weightTotal)
    assigned += value
    return value
  })
}

const createBuilder = () => {
  const events = []
  let time = 0
  let totals = { ...EMPTY_METRICS }

  const add = (event) => {
    const duration = Math.max(0, Number.isFinite(event.duration) ? event.duration : 0)
    const delta = { ...EMPTY_METRICS, ...event.delta }
    const next = {
      ...event,
      start: time,
      end: time + duration,
      duration,
      before: totals,
      after: addMetrics(totals, delta),
      delta,
    }
    events.push(next)
    time = next.end
    totals = next.after
    return next
  }

  return {
    add,
    finish: () => ({ events, duration: time, totals }),
  }
}

const planSubagentWave = ({ builder, wave, delegate, mainModel, caching }) => {
  const model = delegate ?? mainModel
  const decodeRate = model.decodeRate
  const prefillRate = model.prefillRate
  const count = wave.count
  const input = count * wave.contextPerAgent
  const visible = count * wave.outputPerAgent
  const rates = resolvePricing(model.pricing, wave.contextPerAgent)
  const usage = { uncachedInput: input, cachedInput: 0, cacheWriteInput: 0 }
  const prefillDuration = wave.contextPerAgent / prefillRate
  const toolDuration = (wave.toolsPerAgent ?? 0) * 0.5
  const decodeDuration = wave.outputPerAgent / decodeRate
  const duration = prefillDuration + toolDuration + decodeDuration

  builder.add({
    kind: 'subagents',
    label: `${wave.label} (${count} parallel)`,
    duration,
    modelId: model.id,
    delta: {
      input,
      uncachedInput: input,
      output: visible,
      requests: count,
      cost: inputCost(rates, usage) + outputCost(rates, visible),
    },
    detail: { count, caching, delegateName: model.name },
  })

  return visible
}

export const buildSimulationPlan = ({
  model,
  outputTokens,
  promptTokens,
  reasoningTokens = 0,
  toolSteps = [],
  subagentWaves = [],
  delegate = null,
  caching = true,
  systemTokens = DEFAULT_SYSTEM_TOKENS,
  networkLatencyMs = DEFAULT_NETWORK_LATENCY_MS,
}) => {
  if (!model || model.decodeRate <= 0 || model.prefillRate <= 0) {
    throw new Error('A model with positive decode and prefill rates is required')
  }
  if (outputTokens < 0 || promptTokens < 0 || reasoningTokens < 0) {
    throw new Error('Token counts cannot be negative')
  }

  const builder = createBuilder()
  const isCloud = Boolean(model.pricing)
  const steps = toolSteps ?? []
  const rounds = steps.length + 1
  const visibleWeights = steps.length > 0
    ? [...steps.map(() => 1), Math.max(5, steps.length * 5)]
    : [1]
  const visiblePerRound = distribute(outputTokens, visibleWeights)
  const thinkingWeights = steps.length > 0
    ? [...steps.map((step) => step.thinkTokens ?? 1), Math.max(1, reasoningTokens)]
    : [1]
  const reasoningPerRound = distribute(reasoningTokens, thinkingWeights)

  let contextTokens = systemTokens + promptTokens
  let previousRequestInput = 0
  let peakContext = contextTokens
  let compactions = 0
  let requestIndex = 0
  let pendingSubagentResults = 0

  const maybeCompact = () => {
    if (contextTokens <= model.maxContext * 0.8) return
    const target = Math.max(systemTokens, Math.round(model.maxContext * 0.55))
    const removed = Math.max(0, contextTokens - target)
    if (removed === 0) return
    builder.add({
      kind: 'compaction',
      label: `Compact ${removed.toLocaleString()} tokens`,
      duration: removed / (model.prefillRate * 0.5),
      delta: {},
      detail: { removed, before: contextTokens, after: target },
    })
    contextTokens = target
    previousRequestInput = 0
    compactions += 1
  }

  const runRequest = (roundIndex) => {
    maybeCompact()
    const requestInput = contextTokens
    const reusable = caching ? Math.min(previousRequestInput, requestInput) : 0
    const cacheWrite = caching && requestIndex === 0 && model.pricing?.cacheWriteInput
      ? requestInput
      : 0
    const cachedInput = cacheWrite ? 0 : reusable
    const uncachedInput = Math.max(0, requestInput - cachedInput - cacheWrite)
    const usage = { cachedInput, uncachedInput, cacheWriteInput: cacheWrite }
    const rates = resolvePricing(model.pricing, requestInput)

    if (isCloud) {
      builder.add({
        kind: 'network',
        label: `Request ${requestIndex + 1}`,
        duration: networkLatencyMs / 1000,
        delta: { requests: 1 },
        detail: { requestInput },
      })
    }

    const cachePrefillFactor = model.cachePrefillFactor ?? 0.1
    const effectivePrefill = uncachedInput + cacheWrite + cachedInput * cachePrefillFactor
    builder.add({
      kind: 'prefill',
      label: requestIndex === 0 ? 'Initial prefill' : 'Incremental prefill',
      duration: effectivePrefill / model.prefillRate,
      delta: {
        input: requestInput,
        ...usage,
        cost: inputCost(rates, usage),
        requests: isCloud ? 0 : 1,
      },
      detail: { requestInput, rates, effectivePrefill },
    })

    previousRequestInput = requestInput
    requestIndex += 1

    const reasoning = reasoningPerRound[roundIndex] ?? 0
    if (reasoning > 0) {
      builder.add({
        kind: 'reasoning',
        label: 'Reasoning',
        duration: reasoning / model.decodeRate,
        delta: {
          output: reasoning,
          reasoningOutput: reasoning,
          cost: outputCost(rates, reasoning),
        },
        detail: { tokens: reasoning, rate: model.decodeRate },
      })
      contextTokens += Math.round(reasoning * (model.reasoningContextFactor ?? 0))
    }

    const visible = visiblePerRound[roundIndex] ?? 0
    if (visible > 0) {
      builder.add({
        kind: 'visible-decode',
        label: roundIndex === rounds - 1 ? 'Final response' : 'Assistant update',
        duration: visible / model.decodeRate,
        delta: {
          output: visible,
          visibleOutput: visible,
          cost: outputCost(rates, visible),
        },
        detail: { tokens: visible, rate: model.decodeRate },
      })
      contextTokens += visible
    }

    const step = steps[roundIndex]
    if (!step) return
    const toolTokens = step.decodeTokens ?? 0
    if (toolTokens > 0) {
      builder.add({
        kind: 'tool-decode',
        label: step.label,
        duration: toolTokens / model.decodeRate,
        delta: {
          output: toolTokens,
          toolOutput: toolTokens,
          cost: outputCost(rates, toolTokens),
        },
        detail: { tokens: toolTokens, parallel: step.parallel },
      })
      contextTokens += toolTokens
    }

    builder.add({
      kind: 'tool-exec',
      label: step.label,
      duration: ((step.execMs ?? 0) + (isCloud ? networkLatencyMs : 0)) / 1000,
      delta: {},
      detail: { resultTokens: step.resultTokens ?? 0, parallel: step.parallel },
    })
    contextTokens += step.resultTokens ?? 0
    peakContext = Math.max(peakContext, contextTokens)
  }

  for (let roundIndex = 0; roundIndex < rounds; roundIndex += 1) {
    if (roundIndex === rounds - 1 && subagentWaves.length > 0) {
      for (const wave of subagentWaves) {
        pendingSubagentResults += planSubagentWave({ builder, wave, delegate, mainModel: model, caching })
      }
      contextTokens += pendingSubagentResults
      peakContext = Math.max(peakContext, contextTokens)
    }
    runRequest(roundIndex)
  }

  peakContext = Math.max(peakContext, contextTokens)
  const result = builder.finish()
  return {
    ...result,
    context: {
      initial: systemTokens + promptTokens,
      final: contextTokens,
      peak: peakContext,
      max: model.maxContext,
      compactions,
      overflow: peakContext > model.maxContext,
    },
    assumptions: {
      caching,
      systemTokens,
      networkLatencyMs: isCloud ? networkLatencyMs : 0,
    },
  }
}

export const evaluatePlan = (plan, elapsed) => {
  if (!plan || plan.events.length === 0) {
    return { elapsed: 0, metrics: { ...EMPTY_METRICS }, event: null, progress: 0 }
  }
  const clamped = Math.max(0, Math.min(elapsed, plan.duration))
  let metrics = { ...EMPTY_METRICS }
  let active = plan.events[plan.events.length - 1]

  for (const event of plan.events) {
    if (clamped >= event.end) {
      metrics = event.after
      continue
    }
    active = event
    if (clamped <= event.start || event.duration === 0) {
      metrics = event.before
    } else {
      const factor = (clamped - event.start) / event.duration
      metrics = addMetrics(event.before, scaleMetrics(event.delta, factor))
    }
    break
  }

  return {
    elapsed: clamped,
    metrics,
    event: clamped >= plan.duration ? null : active,
    progress: plan.duration > 0 ? clamped / plan.duration : 1,
    complete: clamped >= plan.duration,
  }
}

export const samplePlan = (plan, samples = 80) => {
  if (!plan || plan.duration === 0) return []
  const times = new Set([0, plan.duration])
  for (const event of plan.events) {
    times.add(event.start)
    times.add(event.end)
  }
  for (let index = 1; index < samples; index += 1) {
    times.add((plan.duration * index) / samples)
  }
  return [...times]
    .sort((left, right) => left - right)
    .map((time) => ({ t: time, ...evaluatePlan(plan, time).metrics }))
}

export const getEventDuration = (plan, kind) => plan.events
  .filter((event) => event.kind === kind)
  .reduce((sum, event) => sum + event.duration, 0)

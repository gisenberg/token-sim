import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  EFFORT_LEVELS,
  EXPERIMENT_CATEGORIES,
  EXPERIMENTS,
  HARDWARE,
  MODELS,
  MODEL_BY_ID,
  OUTPUT_PRESETS,
  PROMPT_PRESETS,
  SOURCES,
  SUBAGENT_PRESETS,
  TOOL_PRESETS,
  flattenToolSteps,
  resolveModelProfile,
} from './data/catalog.js'
import {
  buildSimulationPlan,
  evaluatePlan,
  getEventDuration,
  samplePlan,
} from './simulation/engine.js'

const TIER_COLORS = { 'S+': '#f8d46a', S: '#f4c95d', A: '#5fe0a4', B: '#66b7ff', C: '#b69cff' }
const CHART_COLORS = ['#ff806d', '#c5e86c', '#f0c766', '#e4a4e8', '#77d7ba', '#ffad5c', '#d99ae8', '#b7d26c']
const EVENT_COLORS = {
  network: '#536171',
  prefill: '#e8b04e',
  reasoning: '#c497dc',
  'visible-decode': '#a8d66d',
  'tool-decode': '#f49b55',
  'tool-exec': '#65748a',
  subagents: '#77cbb2',
  compaction: '#f07878',
}

const DELEGATE_BY_HARDWARE = {
  'Anthropic API': 'anthropic-sonnet-5',
  'OpenAI API': 'openai-gpt-56-luna',
  'Google API': 'google-gemini-31-flash-lite',
}

const RESPONSE_PARTS = `I found the race in the connection pool and fixed the lifecycle around waiters.

The implementation now removes timed-out requests atomically, closes stale connections, and records queue latency before handing a connection to the caller.

\`\`\`ts
async acquire(signal?: AbortSignal): Promise<Connection> {
  const permit = await this.waiters.enqueue(signal)
  try {
    return await this.checkout(permit)
  } catch (error) {
    permit.release()
    throw error
  }
}
\`\`\`

The focused tests cover cancellation, timeout cleanup, concurrent release, and graceful shutdown. The full suite passes.`.split(/(\s+|(?=[{}()[\];,.]))/).filter(Boolean)

const formatTokens = (value, digits = 1) => {
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(digits)}B`
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(digits)}M`
  if (value >= 1_000) return `${(value / 1_000).toFixed(digits)}K`
  return Math.round(value).toLocaleString()
}

const formatMoney = (value) => {
  if (value === 0) return '$0.00'
  if (value < 0.001) return `$${value.toFixed(5)}`
  if (value < 0.1) return `$${value.toFixed(4)}`
  return `$${value.toFixed(2)}`
}

const formatDuration = (seconds) => {
  if (seconds < 1) return `${Math.round(seconds * 1_000)}ms`
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.round(seconds % 60)
  if (minutes < 60) return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`
  const hours = Math.floor(minutes / 60)
  const minuteRemainder = minutes % 60
  return minuteRemainder ? `${hours}h ${minuteRemainder}m` : `${hours}h`
}

const formatRate = (value) => `${value >= 100 ? Math.round(value) : value.toFixed(1)} tok/s`

const getRoute = () => {
  const hash = window.location.hash.slice(1)
  if (hash === 'about') return { page: 'about', experimentId: null }
  if (hash === 'custom') return { page: 'sim', experimentId: 'custom' }
  return { page: 'sim', experimentId: EXPERIMENTS.some((experiment) => experiment.id === hash) ? hash : 'cloud-current' }
}

const OutputPreview = ({ count, complete }) => {
  if (count <= 0) return <div className="stream-empty">Output will appear here</div>
  const previewSize = Math.min(260, count)
  const start = Math.max(0, count - previewSize)
  const chunks = []
  for (let index = start; index < count; index += 1) chunks.push(RESPONSE_PARTS[index % RESPONSE_PARTS.length])
  return (
    <pre className="output-preview">
      {start > 0 && <span className="output-omitted">... {start.toLocaleString()} earlier output tokens ...{`\n\n`}</span>}
      {chunks.join('')}
      {!complete && <span className="cursor" />}
    </pre>
  )
}

const Timeline = ({ plan, elapsed }) => (
  <div className="timeline-wrap" aria-label="Simulation phase timeline">
    <div className="timeline-track">
      {plan.events.map((event, index) => (
        <span
          key={`${event.kind}-${event.start}-${index}`}
          className="timeline-segment"
          title={`${event.label}: ${formatDuration(event.duration)}`}
          style={{
            width: `${plan.duration ? event.duration / plan.duration * 100 : 0}%`,
            background: EVENT_COLORS[event.kind],
          }}
        />
      ))}
      <span className="timeline-playhead" style={{ left: `${plan.duration ? Math.min(100, elapsed / plan.duration * 100) : 0}%` }} />
    </div>
    <div className="timeline-legend">
      {['prefill', 'reasoning', 'visible-decode', 'tool-exec'].map((kind) => (
        <span key={kind}><i style={{ background: EVENT_COLORS[kind] }} />{kind === 'visible-decode' ? 'visible decode' : kind.replace('-', ' ')}</span>
      ))}
    </div>
  </div>
)

const ContextMeter = ({ plan }) => {
  const peakPercent = Math.min(100, plan.context.peak / plan.context.max * 100)
  return (
    <div className="context-block">
      <div className="meter-label">
        <span>Peak context</span>
        <strong>{formatTokens(plan.context.peak)} / {formatTokens(plan.context.max)}</strong>
      </div>
      <div className={`context-meter ${plan.context.overflow ? 'is-overflow' : ''}`}>
        <span style={{ width: `${peakPercent}%` }} />
        <i style={{ left: '80%' }} title="Compaction threshold" />
      </div>
      <div className="meter-note">
        <span>{plan.context.compactions ? `${plan.context.compactions} compaction${plan.context.compactions === 1 ? '' : 's'}` : 'No compaction'}</span>
        <span>{formatTokens(Math.max(0, plan.context.max - plan.context.peak))} headroom</span>
      </div>
    </div>
  )
}

const MemorySummary = ({ model, plan }) => {
  const hardware = HARDWARE[model.hardware]
  if (!hardware?.memoryGB) return null
  const kvGB = model.kvPerTokKB * plan.context.peak / 1024 / 1024
  const total = model.weightGB + kvGB
  const percent = Math.min(100, total / hardware.memoryGB * 100)
  return (
    <div className="memory-summary">
      <div className="memory-ring" style={{ '--fill': `${percent * 3.6}deg` }}><span /></div>
      <div>
        <strong>{total.toFixed(1)} / {hardware.memoryGB} GB working set</strong>
        <small>{model.weightGB.toFixed(1)} GB weights + {kvGB.toFixed(1)} GB peak KV</small>
      </div>
    </div>
  )
}

const PricingSummary = ({ model, snapshot, plan }) => {
  if (!model.pricing) return null
  const metrics = snapshot.metrics
  const hasLongRate = plan.events.some((event) => event.detail?.rates?.longContextApplied)
  return (
    <div className="pricing-summary">
      <div className="price-total">
        <span>Observed spend</span>
        <strong>{formatMoney(metrics.cost)}</strong>
      </div>
      <div className="price-breakdown">
        <span>{formatTokens(metrics.uncachedInput)} fresh input</span>
        <span>{formatTokens(metrics.cachedInput)} cache hits</span>
        {metrics.cacheWriteInput > 0 && <span>{formatTokens(metrics.cacheWriteInput)} cache writes</span>}
        <span>{formatTokens(metrics.output)} billed output</span>
      </div>
      <div className="rate-card">
        ${model.pricing.input}/M input · ${model.pricing.cachedInput ?? model.pricing.input}/M cached · ${model.pricing.output}/M output
        {hasLongRate && <b> · long-context rates active</b>}
      </div>
    </div>
  )
}

const ModelCard = ({
  model,
  baseModel,
  plan,
  elapsed,
  running,
  profileId,
  onProfileChange,
  outputTokens,
  onRemove,
}) => {
  const localElapsed = Math.min(elapsed, plan.duration)
  const snapshot = evaluatePlan(plan, localElapsed)
  const visible = Math.floor(snapshot.metrics.visibleOutput)
  const isReady = elapsed === 0 && !running
  const status = isReady ? 'Ready' : snapshot.complete ? 'Complete' : snapshot.event?.label ?? 'Working'
  const endToEndRate = localElapsed > 0 ? visible / localElapsed : 0
  const decodeDuration = getEventDuration(plan, 'visible-decode')
  const observedDecodeRate = decodeDuration > 0 ? plan.totals.visibleOutput / decodeDuration : 0
  const recentTools = plan.events.filter((event) =>
    ['tool-decode', 'tool-exec', 'subagents', 'compaction'].includes(event.kind) && event.start <= localElapsed,
  ).slice(-4)
  const longRate = plan.events.some((event) => event.detail?.rates?.longContextApplied)

  return (
    <article className={`model-card ${snapshot.complete && !isReady ? 'is-complete' : ''}`}>
      <div className="card-topline" style={{ background: HARDWARE[model.hardware]?.color }} />
      <header className="model-header">
        <div>
          <div className="model-name-row">
            <span className="tier" style={{ background: TIER_COLORS[model.tier] }}>{model.tier}</span>
            <h2>{model.name}</h2>
          </div>
          <div className="model-meta">
            <span style={{ color: HARDWARE[model.hardware]?.color }}>{model.hardware}</span>
            <a href={model.source} target="_blank" rel="noreferrer">source</a>
          </div>
        </div>
        <span className={`status status-${snapshot.event?.kind ?? 'idle'}`}>{status}</span>
        {onRemove && <button className="remove-model" onClick={onRemove} aria-label={`Remove ${model.name}`}>×</button>}
      </header>

      {baseModel.profiles?.length > 0 && (
        <label className="profile-control">
          <span>Decode profile</span>
          <select name={`decode-profile-${baseModel.id}`} value={profileId ?? baseModel.profiles[0].id} onChange={(event) => onProfileChange(event.target.value)}>
            {baseModel.profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.label}</option>)}
          </select>
        </label>
      )}

      <div className="evidence-row">
        <span>{model.quant ?? 'Hosted model'}</span>
        <span className={`evidence evidence-${model.rateEvidence?.replaceAll(' ', '-')}`}>{model.rateEvidence}</span>
        {longRate && <span className="long-rate-chip">long pricing</span>}
      </div>

      <div className="metric-grid">
        <div><span>Configured decode</span><strong>{formatRate(model.decodeRate)}</strong></div>
        <div><span>Observed decode</span><strong>{visible > 0 ? formatRate(observedDecodeRate) : 'Waiting'}</strong></div>
        <div><span>End-to-end output</span><strong>{visible > 0 ? formatRate(endToEndRate) : 'Waiting'}</strong></div>
        <div><span>Simulated time</span><strong>{formatDuration(localElapsed)} / {formatDuration(plan.duration)}</strong></div>
      </div>

      <ContextMeter plan={plan} />
      <MemorySummary model={model} plan={plan} />
      <PricingSummary model={model} snapshot={snapshot} plan={plan} />

      <div className="output-progress-row">
        <span>Visible output</span>
        <strong>{visible.toLocaleString()} / {outputTokens.toLocaleString()}</strong>
      </div>
      <div className="output-progress"><span style={{ width: `${outputTokens ? visible / outputTokens * 100 : 100}%`, background: HARDWARE[model.hardware]?.color }} /></div>
      <Timeline plan={plan} elapsed={localElapsed} />

      <div className="activity-panel">
        {recentTools.length > 0 && (
          <div className="activity-log">
            {recentTools.map((event, index) => (
              <span key={`${event.start}-${index}`} className={event.start <= localElapsed && event.end >= localElapsed ? 'is-active' : ''}>
                <i style={{ background: EVENT_COLORS[event.kind] }} />{event.label}
              </span>
            ))}
          </div>
        )}
        <OutputPreview count={visible} complete={snapshot.complete} />
      </div>

      {model.note && <p className="model-note">{model.note}</p>}
    </article>
  )
}

const CHART_TABS = [
  { id: 'cost', label: 'Spend', field: 'cost', format: formatMoney },
  { id: 'input', label: 'Billed input', field: 'input', format: (value) => formatTokens(value) },
  { id: 'output', label: 'Billed output', field: 'output', format: (value) => formatTokens(value) },
  { id: 'visibleOutput', label: 'Visible output', field: 'visibleOutput', format: (value) => formatTokens(value) },
]

const MetricsChart = ({ plans, models, elapsed, running }) => {
  const hasCost = models.some((model) => model.pricing)
  const [tab, setTab] = useState(hasCost ? 'cost' : 'visibleOutput')
  const [hoverTime, setHoverTime] = useState(null)
  const svgRef = useRef(null)
  const effectiveTab = !hasCost && tab === 'cost' ? 'visibleOutput' : tab
  const activeTab = CHART_TABS.find((candidate) => candidate.id === effectiveTab) ?? CHART_TABS[0]
  const maxTime = Math.max(1, ...plans.map((plan) => plan.duration))
  const samples = useMemo(() => plans.map((plan) => samplePlan(plan)), [plans])

  const maxValue = Math.max(1e-9, ...plans.map((plan) => plan.totals[activeTab.field] ?? 0))
  const width = 860
  const height = 240
  const padding = { top: 18, right: 22, bottom: 32, left: 72 }
  const plotWidth = width - padding.left - padding.right
  const plotHeight = height - padding.top - padding.bottom
  const x = (time) => padding.left + time / maxTime * plotWidth
  const y = (value) => padding.top + plotHeight - value / maxValue * plotHeight
  const currentTime = hoverTime ?? (running || elapsed > 0 ? Math.min(elapsed, maxTime) : null)

  const handlePointer = (event) => {
    const bounds = svgRef.current?.getBoundingClientRect()
    if (!bounds) return
    const relative = (event.clientX - bounds.left) / bounds.width * width
    setHoverTime(Math.max(0, Math.min(maxTime, (relative - padding.left) / plotWidth * maxTime)))
  }

  return (
    <section className="metrics-panel">
      <div className="metrics-heading">
        <div>
          <span className="eyebrow">Deterministic forecast</span>
          <h2>Workload ledger</h2>
        </div>
        <div className="chart-tabs">
          {CHART_TABS.filter((candidate) => hasCost || candidate.id !== 'cost').map((candidate) => (
            <button key={candidate.id} className={candidate.id === effectiveTab ? 'active' : ''} onClick={() => setTab(candidate.id)}>{candidate.label}</button>
          ))}
        </div>
      </div>
      <svg
        ref={svgRef}
        className="metrics-chart"
        viewBox={`0 0 ${width} ${height}`}
        onMouseMove={handlePointer}
        onMouseLeave={() => setHoverTime(null)}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
          <g key={fraction}>
            <line x1={padding.left} x2={width - padding.right} y1={y(maxValue * fraction)} y2={y(maxValue * fraction)} className="grid-line" />
            <text x={padding.left - 10} y={y(maxValue * fraction) + 3} textAnchor="end" className="axis-label">{activeTab.format(maxValue * fraction)}</text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
          <text key={fraction} x={x(maxTime * fraction)} y={height - 8} textAnchor={fraction === 0 ? 'start' : fraction === 1 ? 'end' : 'middle'} className="axis-label">{formatDuration(maxTime * fraction)}</text>
        ))}
        {samples.map((points, index) => {
          const model = models[index]
          const color = CHART_COLORS[index % CHART_COLORS.length]
          const path = points.map((point, pointIndex) => `${pointIndex ? 'L' : 'M'}${x(point.t).toFixed(2)},${y(point[activeTab.field] ?? 0).toFixed(2)}`).join(' ')
          const final = points[points.length - 1]
          return (
            <g key={`${model.id}-${index}`}>
              <path d={path} className="series-line" style={{ stroke: color }} />
              {final.t < maxTime && <line x1={x(final.t)} x2={x(maxTime)} y1={y(final[activeTab.field] ?? 0)} y2={y(final[activeTab.field] ?? 0)} className="series-extension" style={{ stroke: color }} />}
              <circle cx={x(final.t)} cy={y(final[activeTab.field] ?? 0)} r="3" fill={color} />
            </g>
          )
        })}
        {currentTime != null && (
          <line x1={x(currentTime)} x2={x(currentTime)} y1={padding.top} y2={padding.top + plotHeight} className="chart-cursor" />
        )}
      </svg>
      <div className="chart-legend">
        {models.map((model, index) => {
          const atTime = evaluatePlan(plans[index], currentTime == null ? plans[index].duration : Math.min(currentTime, plans[index].duration))
          return (
            <div key={`${model.id}-${index}`}>
              <span><i style={{ background: CHART_COLORS[index % CHART_COLORS.length] }} />{model.name}</span>
              <strong>{activeTab.format(atTime.metrics[activeTab.field] ?? 0)}</strong>
              <small>{formatDuration(plans[index].duration)} total</small>
            </div>
          )
        })}
      </div>
      <p className="chart-note">Solid lines end when a model finishes. Dashed extensions show that cumulative usage stays flat while slower models continue.</p>
    </section>
  )
}

const AboutPage = () => (
  <section className="about-page">
    <span className="eyebrow">Methodology</span>
    <h1>What this simulator measures</h1>
    <p>This is a deterministic agent-workload simulator, not a character animation with an independent cost counter. One event ledger drives the card output, token totals, latency, context growth, and spend chart.</p>

    <h2>Throughput has two meanings</h2>
    <p><strong>Decode throughput</strong> measures generated tokens only while the model is decoding. It should converge exactly on the selected decode profile. <strong>End-to-end output throughput</strong> divides visible output by total wall time, including network latency, prefill, hidden reasoning, tool calls, tool execution, subagents, and compaction. It is intentionally lower.</p>
    <p>Cloud decode and prefill rates use clearly labeled reference estimates, and hosted requests assume 120ms of network latency. Reasoning presets are deterministic hidden-token budgets rather than provider guarantees.</p>

    <h2>Cost ledger</h2>
    <p>Each main-model request includes an 18K-token coding-agent system and tool-schema baseline in addition to the selected prompt context. The context meter and billed input include both.</p>
    <p>The first request is not treated as a cache hit. Later tool-loop requests can reuse only the exact prior request prefix. New assistant output and tool results are fresh input on the next request. Anthropic cache creation uses the published 1.25x write rate, cache hits use the published 0.1x rate, and provider-specific long-context prices are selected per request.</p>
    <p>The cache toggle assumes that repeated prefixes remain warm and eligible for the provider's cached-input rate. It does not estimate eviction or explicit cache-storage fees.</p>
    <p>Reasoning and tool-call tokens are output tokens even when they are not visible. Google explicitly includes thinking tokens in output pricing. The chart therefore distinguishes billed output from visible output.</p>

    <h2>Speculative decoding and MTP-2</h2>
    <p>MTP uses a model's native multi-token prediction head to draft tokens that the target model verifies. The DGX Spark Qwen profile records 49 tok/s for vLLM with MTP-2, FlashInfer, and INT4+FP8, compared with 21 tok/s for the measured mainline llama.cpp configuration. The full 2.3x improvement is a stack result, not an MTP-only claim.</p>

    <h2>Sources and confidence</h2>
    <p>Cloud model availability and prices link to current first-party provider documentation. Cloud decode rates are labeled as reference estimates because standard API throughput is not guaranteed. Local decode, prefill, memory, and quality data are labeled measured and link to the local-model-eval result set.</p>
    <ul className="source-list">
      <li><a href={SOURCES.openaiModels}>OpenAI model catalog and pricing</a></li>
      <li><a href={SOURCES.anthropicFable}>Anthropic Fable 5</a>, <a href={SOURCES.anthropicSonnet5}>Sonnet 5</a>, and <a href={SOURCES.anthropicOpus48}>Opus 4.8</a></li>
      <li><a href={SOURCES.googlePricing}>Google Gemini API pricing</a></li>
      <li><a href={SOURCES.vllmMtp}>vLLM MTP documentation</a></li>
      <li><a href={SOURCES.spark}>Measured DGX Spark results</a></li>
    </ul>
    <p className="as-of">Catalog and pricing reviewed July 20, 2026.</p>
  </section>
)

const ModelPicker = ({ onChoose, onClose }) => (
  <div className="modal-backdrop" onMouseDown={onClose}>
    <div className="model-picker" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="eyebrow">Custom comparison</span><h2>Add a model</h2></div><button onClick={onClose} aria-label="Close model picker">×</button></header>
      <div className="picker-body">
        {Object.keys(HARDWARE).map((hardware) => (
          <section key={hardware}>
            <h3>{hardware}</h3>
            {MODELS.filter((model) => model.hardware === hardware).map((model) => (
              <button key={model.id} onClick={() => onChoose(model.id)}>
                <span className="tier" style={{ background: TIER_COLORS[model.tier] }}>{model.tier}</span>
                <strong>{model.name}</strong>
                <small>{model.decodeRate} tok/s</small>
              </button>
            ))}
          </section>
        ))}
      </div>
    </div>
  </div>
)

function App() {
  const [route, setRoute] = useState(getRoute)
  const [customModels, setCustomModels] = useState([])
  const [pickerOpen, setPickerOpen] = useState(false)
  const [profileIds, setProfileIds] = useState({})
  const [outputTokens, setOutputTokens] = useState(4_000)
  const [promptTokens, setPromptTokens] = useState(25_000)
  const [toolPresetIndex, setToolPresetIndex] = useState(2)
  const [subagentPresetIndex, setSubagentPresetIndex] = useState(0)
  const [effortId, setEffortId] = useState('high')
  const [caching, setCaching] = useState(true)
  const [timeScale, setTimeScale] = useState(10)
  const [elapsed, setElapsed] = useState(0)
  const [running, setRunning] = useState(false)
  const [paused, setPaused] = useState(false)
  const lastFrameRef = useRef(null)

  useEffect(() => {
    const handleHash = () => {
      setRoute(getRoute())
      setRunning(false)
      setPaused(false)
      setElapsed(0)
      lastFrameRef.current = null
    }
    window.addEventListener('hashchange', handleHash)
    return () => window.removeEventListener('hashchange', handleHash)
  }, [])

  const isCustom = route.experimentId === 'custom'
  const experiment = isCustom
    ? { id: 'custom', name: 'Custom Comparison', description: 'Build a focused model lineup', columns: 3, models: customModels }
    : EXPERIMENTS.find((candidate) => candidate.id === route.experimentId) ?? EXPERIMENTS[0]
  const baseModels = useMemo(() => experiment.models.map((id) => MODEL_BY_ID[id]).filter(Boolean), [experiment.models])
  const resolvedModels = useMemo(() => baseModels.map((model, index) =>
    resolveModelProfile(model, profileIds[`${model.id}:${index}`]),
  ), [baseModels, profileIds])
  const toolSteps = useMemo(() => flattenToolSteps(TOOL_PRESETS[toolPresetIndex].steps), [toolPresetIndex])
  const effort = EFFORT_LEVELS.find((level) => level.id === effortId) ?? EFFORT_LEVELS[3]
  const waves = SUBAGENT_PRESETS[subagentPresetIndex].waves

  const plans = useMemo(() => resolvedModels.map((model) => {
    const delegateId = DELEGATE_BY_HARDWARE[model.hardware]
    const delegateBase = delegateId ? MODEL_BY_ID[delegateId] : null
    const delegate = delegateBase ? resolveModelProfile(delegateBase) : null
    return buildSimulationPlan({
      model,
      outputTokens,
      promptTokens,
      reasoningTokens: Math.round((model.reasoningTokens ?? 0) * effort.multiplier),
      toolSteps,
      subagentWaves: model.pricing ? waves : [],
      delegate,
      caching,
    })
  }), [resolvedModels, outputTokens, promptTokens, effort.multiplier, toolSteps, waves, caching])
  const maxDuration = Math.max(0, ...plans.map((plan) => plan.duration))

  const reset = useCallback(() => {
    setRunning(false)
    setPaused(false)
    setElapsed(0)
    lastFrameRef.current = null
  }, [])

  useEffect(() => {
    if (!running || paused || maxDuration <= 0) return undefined
    let frameId
    const tick = (now) => {
      if (lastFrameRef.current == null) lastFrameRef.current = now
      const delta = Math.max(0, (now - lastFrameRef.current) / 1_000) * timeScale
      lastFrameRef.current = now
      setElapsed((current) => {
        const next = Math.min(maxDuration, current + delta)
        if (next >= maxDuration) setRunning(false)
        return next
      })
      frameId = requestAnimationFrame(tick)
    }
    frameId = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frameId)
  }, [running, paused, maxDuration, timeScale])

  const navigate = (id) => {
    window.location.hash = id
  }

  const toggleRun = () => {
    if (maxDuration <= 0) return
    if (running) {
      setPaused((value) => !value)
      lastFrameRef.current = null
      return
    }
    if (elapsed >= maxDuration) setElapsed(0)
    lastFrameRef.current = null
    setPaused(false)
    setRunning(true)
  }

  const updateProfile = (model, index, profileId) => {
    reset()
    setProfileIds((current) => ({ ...current, [`${model.id}:${index}`]: profileId }))
  }

  const changeControl = (setter, value) => {
    reset()
    setter(value)
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span>TS</span><div><strong>Token Sim</strong><small>Agent economics lab</small></div></div>
        <button className={`custom-link ${isCustom ? 'active' : ''}`} onClick={() => navigate('custom')}><span className="custom-link-full">+ Build comparison</span><span className="custom-link-compact">+ Compare</span></button>
        {EXPERIMENT_CATEGORIES.map((category) => (
          <section className="nav-section" key={category.id}>
            <h2>{category.label}</h2>
            {EXPERIMENTS.filter((candidate) => candidate.category === category.id).map((candidate) => (
              <button key={candidate.id} className={route.experimentId === candidate.id ? 'active' : ''} onClick={() => navigate(candidate.id)}>
                <span>{candidate.name}</span><small>{candidate.models.length}</small>
              </button>
            ))}
          </section>
        ))}
        <button className={`about-link ${route.page === 'about' ? 'active' : ''}`} onClick={() => navigate('about')}>Methodology and sources</button>
      </aside>

      <main className="main-content">
        {route.page === 'about' ? <AboutPage /> : (
          <>
            <header className="page-header">
              <div><span className="eyebrow">Agent workload simulator</span><h1>{experiment.name}</h1><p>{experiment.description}</p></div>
              <div className="accuracy-badge"><i />One ledger · exact output</div>
            </header>

            <section className="control-panel">
              <label><span>Prompt context <b>{formatTokens(promptTokens)}</b></span><select name="starting-context" value={promptTokens} onChange={(event) => changeControl(setPromptTokens, Number(event.target.value))}>{PROMPT_PRESETS.map((preset) => <option key={preset.tokens} value={preset.tokens}>{preset.label} · {formatTokens(preset.tokens)}</option>)}</select><small>{PROMPT_PRESETS.find((preset) => preset.tokens === promptTokens)?.description} · 18K system added</small></label>
              <label><span>Visible output <b>{formatTokens(outputTokens)}</b></span><select name="visible-output" value={outputTokens} onChange={(event) => changeControl(setOutputTokens, Number(event.target.value))}>{OUTPUT_PRESETS.map((preset) => <option key={preset.tokens} value={preset.tokens}>{preset.label} · {formatTokens(preset.tokens)}</option>)}</select><small>Exact across all assistant updates</small></label>
              <label><span>Agent loop <b>{toolSteps.length + 1} requests</b></span><select name="agent-loop" value={toolPresetIndex} onChange={(event) => changeControl(setToolPresetIndex, Number(event.target.value))}>{TOOL_PRESETS.map((preset, index) => <option key={preset.label} value={index}>{preset.label}</option>)}</select><small>{TOOL_PRESETS[toolPresetIndex].description}</small></label>
              <label><span>Reasoning effort <b>{effort.label}</b></span><select name="reasoning-effort" value={effortId} onChange={(event) => changeControl(setEffortId, event.target.value)}>{EFFORT_LEVELS.map((level) => <option key={level.id} value={level.id}>{level.label}</option>)}</select><small>Changes billed hidden output, not decode rate</small></label>
              <label><span>Subagents <b>{waves.reduce((sum, wave) => sum + wave.count, 0) || 'none'}</b></span><select name="subagents" value={subagentPresetIndex} onChange={(event) => changeControl(setSubagentPresetIndex, Number(event.target.value))}>{SUBAGENT_PRESETS.map((preset, index) => <option key={preset.label} value={index}>{preset.label}</option>)}</select><small>{SUBAGENT_PRESETS[subagentPresetIndex].description}</small></label>
              <label className="cache-control"><span>Prompt caching <b>{caching ? 'on' : 'off'}</b></span><button className={caching ? 'active' : ''} onClick={() => changeControl(setCaching, !caching)}><i />{caching ? 'Warm repeated prefixes' : 'All input billed fresh'}</button><small>First request is never a cache hit</small></label>
            </section>

            <section className="transport-bar">
              <button className="play-button" onClick={toggleRun} disabled={plans.length === 0}>{running && !paused ? 'Pause' : elapsed > 0 && elapsed < maxDuration ? 'Resume' : 'Run simulation'}</button>
              <button className="reset-button" onClick={reset}>Reset</button>
              <label className="speed-control"><span>Playback</span><select name="playback-speed" value={timeScale} onChange={(event) => setTimeScale(Number(event.target.value))}>{[1, 5, 10, 25, 50, 100].map((value) => <option key={value} value={value}>{value}x</option>)}</select></label>
              <div className="master-progress"><span style={{ width: `${maxDuration ? Math.min(100, elapsed / maxDuration * 100) : 0}%` }} /></div>
              <strong>{formatDuration(Math.min(elapsed, maxDuration))} / {formatDuration(maxDuration)}</strong>
            </section>

            {plans.length > 0 && <MetricsChart plans={plans} models={resolvedModels} elapsed={elapsed} running={running && !paused} />}

            {isCustom && plans.length === 0 && (
              <button className="empty-comparison" onClick={() => setPickerOpen(true)}><span>+</span><strong>Add your first model</strong><small>Compare cloud and local configurations in one ledger</small></button>
            )}

            <section className="model-grid" style={{ '--columns': Math.min(experiment.columns, Math.max(1, resolvedModels.length)) }}>
              {resolvedModels.map((model, index) => {
                const baseModel = baseModels[index]
                return (
                  <ModelCard
                    key={`${baseModel.id}-${index}`}
                    model={model}
                    baseModel={baseModel}
                    plan={plans[index]}
                    elapsed={elapsed}
                    running={running && !paused}
                    profileId={profileIds[`${baseModel.id}:${index}`]}
                    onProfileChange={(profileId) => updateProfile(baseModel, index, profileId)}
                    outputTokens={outputTokens}
                    onRemove={isCustom ? () => { reset(); setCustomModels((current) => current.filter((_, modelIndex) => modelIndex !== index)) } : null}
                  />
                )
              })}
              {isCustom && <button className="add-model-card" onClick={() => setPickerOpen(true)}><span>+</span><strong>Add model</strong></button>}
            </section>
          </>
        )}
      </main>
      {pickerOpen && <ModelPicker onClose={() => setPickerOpen(false)} onChoose={(id) => { reset(); setCustomModels((current) => [...current, id]); setPickerOpen(false) }} />}
    </div>
  )
}

export default App

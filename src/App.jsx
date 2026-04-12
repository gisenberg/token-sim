import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import './App.css'

// prefillRate: tokens/sec during prompt processing (measured or estimated)
// 5090 estimates derived from TTFT @ ~176 token prompt + bandwidth ratio
// M4 Max estimates from 410/1792 bandwidth ratio vs 5090
// Spark measured: 627 tok/s @ 32K for Qwen 122B
const MODELS = [
  // ── RTX 5090 (1,792 GB/s bandwidth) ──────────────────
  { id: '5090-gemma26b-q6', name: 'Gemma 4 26B-A4B', quant: 'Q6_K', hardware: 'RTX 5090', tier: 'S', tokPerSec: 139, prefillRate: 2900, vram: '26.7 GB', maxCtx: '262K', quality: '17/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-gemma31b', name: 'Gemma 4 31B-IT', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'S', tokPerSec: 50, prefillRate: 1900, vram: '23.6 GB', maxCtx: '58K', quality: '17/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-qwen27b-opus', name: 'Qwen 3.5 27B Opus', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'A', tokPerSec: 60, prefillRate: 1900, vram: '20.5 GB', maxCtx: '262K', quality: '17/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-gemma26b-q4', name: 'Gemma 4 26B-A4B', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'A', tokPerSec: 150, prefillRate: 3000, vram: '21.2 GB', maxCtx: '262K', quality: '16/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-harmonic27b', name: 'Harmonic 27B', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'A', tokPerSec: 61, prefillRate: 1800, vram: '20.5 GB', maxCtx: '262K', quality: '31/31', thinking: true, thinkingBudget: 16384, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-qwopus27b', name: 'Qwopus 3.5 27B-v3', quant: 'Q6_K', hardware: 'RTX 5090', tier: 'A', tokPerSec: 50, prefillRate: 1800, vram: '25.5 GB', maxCtx: '262K', quality: '16/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-gemma31b-opus', name: 'Gemma 31B Opus-Dist.', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'B', tokPerSec: 51, prefillRate: 2000, vram: '23.6 GB', maxCtx: '58K', quality: '16/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-qwen35b-a3b', name: 'Qwen 3.5 35B-A3B', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'C', tokPerSec: 174, prefillRate: 2400, vram: '24.8 GB', maxCtx: '262K', quality: '11/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-qwen27b-base', name: 'Qwen 3.5 27B', quant: 'Q6_K (base)', hardware: 'RTX 5090', tier: 'C', tokPerSec: 50, prefillRate: 1700, vram: '25.6 GB', maxCtx: '32K', quality: '10/17', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },
  { id: '5090-gemma-e4b', name: 'Gemma 4 E4B', quant: 'Q8_0', hardware: 'RTX 5090', tier: 'F', tokPerSec: 131, prefillRate: 5000, vram: '12.5 GB', maxCtx: '32K', quality: '5/22', thinking: false, thinkingBudget: 0, color: '#f87171', hwColor: '#86efac' },

  // ── M4 Max (410 GB/s bandwidth) ───────────────────────
  { id: 'm4-gemma31b', name: 'Gemma 4 31B-IT', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'S', tokPerSec: 15, prefillRate: 390, vram: '~24.3 GB', maxCtx: '64K', quality: '17/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-gemma26b-q6', name: 'Gemma 4 26B-A4B', quant: 'Q6_K', hardware: 'M4 Max', tier: 'S', tokPerSec: 66, prefillRate: 980, vram: '~23 GB', maxCtx: '32K', quality: '15/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen27b-mlx', name: 'Qwen 27B Opus MLX', quant: '4-bit', hardware: 'M4 Max', tier: 'A', tokPerSec: 19, prefillRate: 500, vram: '~14 GB', maxCtx: '32K', quality: '13/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen27b-opus', name: 'Qwen 27B Opus', quant: 'Q4_K_M (planar3)', hardware: 'M4 Max', tier: 'A', tokPerSec: 16, prefillRate: 440, vram: '~16.5 GB', maxCtx: '128K', quality: '11/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-gemma26b-q4', name: 'Gemma 4 26B-A4B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'A', tokPerSec: 59, prefillRate: 1150, vram: '~16.5 GB', maxCtx: '64K', quality: '11/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen9b', name: 'Qwen 3.5 9B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'B', tokPerSec: 35, prefillRate: 1750, vram: '~5.5 GB', maxCtx: '32K', quality: '9/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-nemotron4b', name: 'Nemotron 3 Nano 4B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'B', tokPerSec: 66, prefillRate: 2900, vram: '~2.8 GB', maxCtx: '32K', quality: '7/17', thinking: true, thinkingBudget: 8192, color: '#38bdf8', hwColor: '#93c5fd' },

  // ── DGX Spark (273 GB/s bandwidth) ────────────────────
  { id: 'spark-qwen122b-ik', name: 'Qwen 3.5 122B-A10B', quant: 'Q4_K_M (ik-llama)', hardware: 'DGX Spark', tier: 'S', tokPerSec: 26, prefillRate: 627, vram: '71 GB', maxCtx: '128K', quality: '17/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen122b-unsloth', name: 'Qwen 3.5 122B-A10B', quant: 'Q4_K_M (mainline)', hardware: 'DGX Spark', tier: 'S', tokPerSec: 21, prefillRate: 600, vram: '72 GB', maxCtx: '32K', quality: '18/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-glm45', name: 'GLM-4.5-Air', quant: 'Q4_K_M', hardware: 'DGX Spark', tier: 'A', tokPerSec: 22, prefillRate: 627, vram: '70 GB', maxCtx: '32K', quality: '15/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen122b-reap', name: 'Qwen 122B REAP-20', quant: 'Q4_K_M (pruned)', hardware: 'DGX Spark', tier: 'A', tokPerSec: 29, prefillRate: 700, vram: '57 GB', maxCtx: '32K', quality: '14/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen122b-mainline', name: 'Qwen 122B-A10B', quant: 'Q4_K_M (bartowski)', hardware: 'DGX Spark', tier: 'A', tokPerSec: 26, prefillRate: 620, vram: '71 GB', maxCtx: '32K', quality: '13/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen3-coder', name: 'Qwen3-Coder-Next', quant: 'UD-Q4_K_M', hardware: 'DGX Spark', tier: 'B', tokPerSec: 50, prefillRate: 800, vram: '46 GB', maxCtx: '32K', quality: '14/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-nemotron120b', name: 'Nemotron-3 Super 120B', quant: 'Q4_K_M', hardware: 'DGX Spark', tier: 'B', tokPerSec: 20, prefillRate: 500, vram: '87 GB', maxCtx: '32K', quality: '11/17', thinking: true, thinkingBudget: 16384, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-minimax', name: 'MiniMax-M2.5', quant: 'UD-Q3_K_XL', hardware: 'DGX Spark', tier: 'C', tokPerSec: 30, prefillRate: 400, vram: '96 GB', maxCtx: '32K', quality: '5/15', thinking: true, thinkingBudget: 16384, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-mistral119b', name: 'Mistral-Small-4 119B', quant: 'Q4_K_M', hardware: 'DGX Spark', tier: 'D', tokPerSec: 9, prefillRate: 350, vram: '69 GB', maxCtx: '32K', quality: '7/17', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-gemma31b-dense', name: 'Gemma 4 31B-IT', quant: 'Q8_0 (dense)', hardware: 'DGX Spark', tier: 'F', tokPerSec: 7, prefillRate: 250, vram: '~58 GB', maxCtx: '32K', quality: 'N/A', thinking: false, thinkingBudget: 0, color: '#a78bfa', hwColor: '#c4b5fd' },
]

const TIER_ORDER = { S: 0, A: 1, B: 2, C: 3, D: 4, F: 5 }

const HARDWARE_GROUPS = ['RTX 5090', 'M4 Max', 'DGX Spark'].map(hw => ({
  label: hw,
  ids: MODELS
    .filter(m => m.hardware === hw)
    .sort((a, b) => TIER_ORDER[a.tier] - TIER_ORDER[b.tier])
    .map(m => m.id),
}))

const generateText = (tokenCount) => {
  const words = [
    'Artificial', 'intelligence', 'is', 'transforming', 'how', 'we', 'interact', 'with', 'technology',
    'From', 'natural', 'language', 'processing', 'to', 'computer', 'vision', 'AI', 'systems',
    'are', 'becoming', 'increasingly', 'sophisticated', 'Large', 'language', 'models', 'can',
    'now', 'understand', 'context', 'generate', 'creative', 'content', 'and', 'assist',
    'with', 'complex', 'problem-solving', 'tasks', 'The', 'rapid', 'advancement', 'in',
    'this', 'field', 'continues', 'to', 'accelerate', 'new', 'breakthroughs', 'happening',
    'regularly', 'machine', 'learning', 'neural', 'networks', 'deep', 'learning', 'algorithms',
    'process', 'data', 'efficiently', 'pattern', 'recognition', 'prediction', 'automation',
    'innovation', 'research', 'development', 'deployment', 'scalability', 'performance',
    'optimization', 'accuracy', 'precision', 'recall', 'training', 'inference', 'models',
    'parameters', 'weights', 'biases', 'gradients', 'backpropagation', 'forward', 'pass',
    'activation', 'functions', 'layers', 'nodes', 'connections', 'architecture', 'design',
    'implementation', 'testing', 'validation', 'evaluation', 'metrics', 'benchmarks',
    'comparison', 'analysis', 'insights', 'discoveries', 'applications', 'use-cases',
    'examples', 'demonstrations', 'tutorials', 'documentation', 'resources', 'tools',
    'frameworks', 'libraries', 'platforms', 'infrastructure', 'cloud', 'edge', 'devices',
    'mobile', 'web', 'desktop', 'embedded', 'IoT', 'robotics', 'autonomous', 'vehicles',
    'healthcare', 'finance', 'education', 'entertainment', 'gaming', 'social', 'media',
    'e-commerce', 'marketing', 'sales', 'customer', 'service', 'support', 'feedback',
    'improvement', 'iteration', 'evolution', 'future', 'possibilities', 'potential',
    'challenges', 'opportunities', 'trends', 'predictions', 'speculations', 'hypotheses'
  ]

  const tokens = []
  for (let i = 0; i < tokenCount; i++) {
    const word = words[i % words.length]
    const suffix = i === tokenCount - 1 ? '' : (Math.random() > 0.7 ? '  ' : ' ')
    tokens.push(word + suffix)
  }
  return tokens
}

const PROMPT_PRESETS = [
  { label: 'Quick question', tokens: 500, desc: 'Short prompt, no context' },
  { label: 'Single file edit', tokens: 2000, desc: '~1 file + instructions' },
  { label: 'Multi-file task', tokens: 8000, desc: '3-5 files + conversation' },
  { label: 'Large refactor', tokens: 24000, desc: '10+ files + history' },
  { label: 'Full codebase context', tokens: 64000, desc: 'Deep repo exploration' },
  { label: 'Max context window', tokens: 100000, desc: 'Pushing the limits' },
]

const TIER_COLORS = {
  S: '#fbbf24',
  A: '#34d399',
  B: '#60a5fa',
  C: '#a78bfa',
  D: '#f87171',
  F: '#6b7280',
}

const formatTime = (seconds) => {
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 10) return `${seconds.toFixed(1)}s`
  return `${Math.round(seconds)}s`
}

const TokenStream = ({ model, tokens, isRunning, isReset, tokenCount, promptTokens, onComplete, streamIndex }) => {
  const [displayedTokens, setDisplayedTokens] = useState([])
  const [phase, setPhase] = useState('idle')
  const [thinkingTokensGenerated, setThinkingTokensGenerated] = useState(0)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [prefillElapsed, setPrefillElapsed] = useState(0)
  const intervalRef = useRef(null)
  const timerRef = useRef(null)
  const prefillTimerRef = useRef(null)
  const startTimeRef = useRef(null)
  const decodeStartRef = useRef(null)
  const totalIndexRef = useRef(0)
  const contentRef = useRef(null)
  const rafRef = useRef(null)
  const hasStartedRef = useRef(false)

  const thinkingBudget = model.thinkingBudget
  const totalTokens = thinkingBudget + tokenCount

  const scrollToBottom = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      if (contentRef.current) {
        contentRef.current.scrollTop = contentRef.current.scrollHeight
      }
    })
  }, [])

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (timerRef.current) clearInterval(timerRef.current)
      if (prefillTimerRef.current) clearTimeout(prefillTimerRef.current)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  useEffect(() => {
    if (isReset) {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (timerRef.current) clearInterval(timerRef.current)
      if (prefillTimerRef.current) clearTimeout(prefillTimerRef.current)
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      setDisplayedTokens([])
      setPhase('idle')
      setThinkingTokensGenerated(0)
      setElapsedTime(0)
      setPrefillElapsed(0)
      totalIndexRef.current = 0
      hasStartedRef.current = false
      startTimeRef.current = null
      decodeStartRef.current = null
      return
    }

    if (isRunning && !hasStartedRef.current) {
      hasStartedRef.current = true
      startTimeRef.current = Date.now()
      setPhase('prefill')

      timerRef.current = setInterval(() => {
        if (startTimeRef.current) {
          setElapsedTime(((Date.now() - startTimeRef.current) / 1000).toFixed(1))
        }
      }, 100)

      const prefillMs = (promptTokens / model.prefillRate) * 1000
      const prefillStart = Date.now()
      const animatePrefill = () => {
        const elapsed = Date.now() - prefillStart
        const progress = Math.min(elapsed / prefillMs, 1)
        setPrefillElapsed(progress)
        if (progress < 1) {
          prefillTimerRef.current = setTimeout(animatePrefill, 16)
        }
      }
      animatePrefill()

      prefillTimerRef.current = setTimeout(() => {
        decodeStartRef.current = Date.now()

        if (thinkingBudget > 0) {
          setPhase('thinking')
        } else {
          setPhase('streaming')
        }

        const interval = 1000 / model.tokPerSec
        intervalRef.current = setInterval(() => {
          if (totalIndexRef.current < totalTokens) {
            if (totalIndexRef.current < thinkingBudget) {
              setThinkingTokensGenerated(totalIndexRef.current + 1)
              totalIndexRef.current++
            } else {
              if (totalIndexRef.current === thinkingBudget && thinkingBudget > 0) {
                setPhase('streaming')
              }
              const displayIndex = totalIndexRef.current - thinkingBudget
              if (displayIndex < tokenCount) {
                setDisplayedTokens(prev => [...prev, tokens[displayIndex]])
              }
              totalIndexRef.current++
            }
          } else {
            clearInterval(intervalRef.current)
            clearInterval(timerRef.current)
            if (startTimeRef.current) {
              setElapsedTime(((Date.now() - startTimeRef.current) / 1000).toFixed(1))
            }
            setPhase('complete')
            onComplete(streamIndex)
          }
        }, interval)
      }, prefillMs)
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRunning, isReset, model.tokPerSec, model.prefillRate, promptTokens, thinkingBudget, tokenCount, totalTokens, tokens, onComplete, streamIndex])

  useEffect(() => {
    if (displayedTokens.length > 0) {
      scrollToBottom()
    }
  }, [displayedTokens, scrollToBottom])

  const totalProgress = totalTokens > 0
    ? ((thinkingTokensGenerated + displayedTokens.length) / totalTokens) * 100
    : 0
  const decodeElapsed = decodeStartRef.current
    ? (Date.now() - decodeStartRef.current) / 1000
    : 0
  const rate = displayedTokens.length > 0 && decodeElapsed > 0
    ? (displayedTokens.length / decodeElapsed).toFixed(1)
    : null

  const statusLabel = {
    idle: 'Ready',
    prefill: 'Prefill',
    thinking: 'Thinking',
    streaming: 'Streaming',
    complete: 'Done',
  }[phase]

  const cardClass = [
    'stream-card',
    (phase !== 'idle' && phase !== 'complete') && 'is-running',
    phase === 'complete' && 'is-complete'
  ].filter(Boolean).join(' ')

  const thinkingLabel = model.thinking
    ? `${model.thinkingBudget.toLocaleString()}`
    : 'Off'

  return (
    <div className={cardClass}>
      <div className="card-accent" style={{ background: model.color }} />

      <div className="card-header">
        <div className="model-info">
          <div className="model-title-row">
            <span className="tier-badge" style={{ background: TIER_COLORS[model.tier] }}>{model.tier}</span>
            <span className="model-name">{model.name}</span>
          </div>
          <span className="model-quant">{model.quant}</span>
        </div>
        <span className={`card-status ${phase}`}>{statusLabel}</span>
      </div>

      <div className="hw-row">
        <span className="hw-badge" style={{ color: model.hwColor }}>{model.hardware}</span>
        <span className="hw-spec">{model.tokPerSec} tok/s</span>
        <span className="hw-spec">{model.prefillRate} pp/s</span>
        <span className="hw-spec">{model.maxCtx} ctx</span>
        <span className="hw-spec">{model.quality} pass</span>
      </div>

      <div className="stats-row">
        <div className="stat">
          <span className="stat-label">Output</span>
          <span className="stat-value">{displayedTokens.length.toLocaleString()} / {tokenCount.toLocaleString()}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Thinking</span>
          <span className={`stat-value ${!model.thinking ? 'stat-dim' : ''}`}>{thinkingLabel}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Time</span>
          <span className="stat-value">{elapsedTime}s</span>
        </div>
        {rate && (
          <div className="stat">
            <span className="stat-label">Actual</span>
            <span className="stat-value">{rate} tok/s</span>
          </div>
        )}
      </div>

      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${totalProgress}%`, background: model.color }}
        />
      </div>

      {phase === 'prefill' && (
        <div className="prefill-banner">
          <div className="prefill-bar">
            <div
              className="prefill-bar-fill"
              style={{ width: `${prefillElapsed * 100}%`, background: model.color }}
            />
          </div>
          <span className="prefill-label">Prefilling {promptTokens.toLocaleString()} prompt tokens @ {model.prefillRate} tok/s — {formatTime(promptTokens / model.prefillRate)}</span>
        </div>
      )}

      {phase === 'thinking' && (
        <div className="thinking-banner">
          <span className="thinking-spinner" />
          <div className="thinking-detail">
            <span>Thinking</span>
            <span className="thinking-count">{thinkingTokensGenerated.toLocaleString()} / {thinkingBudget.toLocaleString()}</span>
          </div>
        </div>
      )}

      <div ref={contentRef} className="stream-content">
        {displayedTokens.length === 0 && phase === 'idle' && (
          <div className="stream-empty">Waiting to start</div>
        )}
        {displayedTokens.length === 0 && phase === 'prefill' && (
          <div className="stream-empty">Processing prompt...</div>
        )}
        {displayedTokens.length === 0 && phase === 'thinking' && (
          <div className="stream-empty">Reasoning...</div>
        )}
        {displayedTokens.map((token, i) => (
          <span key={i} className="token">{token}</span>
        ))}
        {phase === 'streaming' && (
          <span className="cursor" />
        )}
      </div>
    </div>
  )
}

function App() {
  const [isRunning, setIsRunning] = useState(false)
  const [isReset, setIsReset] = useState(false)
  const [tokenCount, setTokenCount] = useState(1500)
  const [promptTokens, setPromptTokens] = useState(2000)
  const [selectedIds, setSelectedIds] = useState([
    '5090-gemma26b-q6',
    'm4-gemma26b-q6',
    'spark-qwen122b-ik',
    '5090-gemma31b',
    'm4-gemma31b',
    'spark-qwen122b-unsloth',
  ])
  const [completedStreams, setCompletedStreams] = useState(new Set())

  const selectedModels = selectedIds.map(id => MODELS.find(m => m.id === id))

  const maxThinkingBudget = Math.max(...selectedModels.map(m => m.thinkingBudget))
  const maxTotalTokens = tokenCount + maxThinkingBudget

  const handleModelChange = (slotIndex, modelId) => {
    setSelectedIds(prev => {
      const next = [...prev]
      next[slotIndex] = modelId
      return next
    })
  }

  const handlePreset = (hardwareLabel) => {
    const group = HARDWARE_GROUPS.find(g => g.label === hardwareLabel)
    if (group) {
      setSelectedIds(group.ids.slice(0, 6).concat(
        Array(Math.max(0, 6 - group.ids.length)).fill(group.ids[0])
      ))
    }
  }

  const handleComplete = useCallback((index) => {
    setCompletedStreams(prev => {
      const next = new Set(prev)
      next.add(index)
      return next
    })
  }, [])

  const handleStart = () => {
    setIsReset(false)
    setCompletedStreams(new Set())
    setIsRunning(true)
  }

  const handleReset = () => {
    setIsRunning(false)
    setIsReset(true)
    setCompletedStreams(new Set())
    setTimeout(() => setIsReset(false), 100)
  }

  const tokens = useMemo(() => generateText(maxTotalTokens), [maxTotalTokens])
  const allComplete = completedStreams.size >= selectedIds.length
  const controlsDisabled = isRunning && !allComplete

  return (
    <div className="app">
      <header className="app-header">
        <h1>Token Speed Simulator</h1>
        <p>Compare local model inference across hardware platforms</p>
      </header>

      <div className="controls">
        <div className="control-group">
          <label>
            Output tokens
            <span>{tokenCount.toLocaleString()}</span>
          </label>
          <input
            type="range"
            min="400"
            max="10000"
            step="100"
            value={tokenCount}
            onChange={(e) => setTokenCount(parseInt(e.target.value))}
            disabled={controlsDisabled}
            className="slider"
          />
          <div className="slider-bounds">
            <span>400</span>
            <span>10,000</span>
          </div>
        </div>

        <div className="control-group">
          <label>
            Prompt context
            <span>{promptTokens.toLocaleString()} tokens</span>
          </label>
          <select
            value={promptTokens}
            onChange={(e) => setPromptTokens(parseInt(e.target.value))}
            disabled={controlsDisabled}
            className="prompt-select"
          >
            {PROMPT_PRESETS.map(p => (
              <option key={p.tokens} value={p.tokens}>
                {p.label} — {p.tokens.toLocaleString()} tokens
              </option>
            ))}
          </select>
          <div className="prompt-desc">
            {PROMPT_PRESETS.find(p => p.tokens === promptTokens)?.desc}
          </div>
        </div>

        <div className="button-group">
          <button
            onClick={handleStart}
            disabled={controlsDisabled}
            className="btn-start"
          >
            {controlsDisabled ? 'Running...' : 'Start'}
          </button>
          <button onClick={handleReset} className="btn-reset">Reset</button>
        </div>

        <div className="preset-group">
          <span className="preset-label">Presets</span>
          {HARDWARE_GROUPS.map(group => (
            <button
              key={group.label}
              onClick={() => handlePreset(group.label)}
              disabled={controlsDisabled}
              className="btn-preset"
            >
              Top 6 {group.label}
            </button>
          ))}
        </div>
      </div>

      {[0, 3].map(rowStart => (
        <div key={rowStart} className="sim-group">
          <div className="sim-row">
            {selectedIds.slice(rowStart, rowStart + 3).map((id, i) => (
              <select
                key={rowStart + i}
                value={id}
                onChange={(e) => handleModelChange(rowStart + i, e.target.value)}
                disabled={controlsDisabled}
                className="model-select"
              >
                {HARDWARE_GROUPS.map(group => (
                  <optgroup key={group.label} label={group.label}>
                    {group.ids.map(mid => {
                      const m = MODELS.find(x => x.id === mid)
                      return (
                        <option key={mid} value={mid}>
                          [{m.tier}] {m.name} {m.quant} — {m.tokPerSec} tok/s
                        </option>
                      )
                    })}
                  </optgroup>
                ))}
              </select>
            ))}
          </div>
          <div className="sim-row">
            {selectedModels.slice(rowStart, rowStart + 3).map((model, i) => (
              <TokenStream
                key={model.id + '-' + (rowStart + i)}
                model={model}
                tokens={tokens}
                isRunning={isRunning}
                isReset={isReset}
                tokenCount={tokenCount}
                promptTokens={promptTokens}
                onComplete={handleComplete}
                streamIndex={rowStart + i}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

export default App

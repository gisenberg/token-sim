import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import './App.css'

// ── Citation URLs ──
const CITATIONS = {
  'RTX 5090': 'https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_5090.md',
  'M4 Max': 'https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_M4MAX.md',
  'DGX Spark': 'https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_SPARK.md',
}

// ── Model Data ──
const MODELS = [
  // RTX 5090 (1,792 GB/s)
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
  // M4 Max (410 GB/s)
  { id: 'm4-gemma31b', name: 'Gemma 4 31B-IT', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'S', tokPerSec: 15, prefillRate: 390, vram: '~24.3 GB', maxCtx: '64K', quality: '17/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-gemma26b-q6', name: 'Gemma 4 26B-A4B', quant: 'Q6_K', hardware: 'M4 Max', tier: 'S', tokPerSec: 66, prefillRate: 980, vram: '~23 GB', maxCtx: '32K', quality: '15/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen27b-mlx', name: 'Qwen 27B Opus MLX', quant: '4-bit', hardware: 'M4 Max', tier: 'A', tokPerSec: 19, prefillRate: 500, vram: '~14 GB', maxCtx: '32K', quality: '13/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen27b-opus', name: 'Qwen 27B Opus', quant: 'Q4_K_M (planar3)', hardware: 'M4 Max', tier: 'A', tokPerSec: 16, prefillRate: 440, vram: '~16.5 GB', maxCtx: '128K', quality: '11/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-gemma26b-q4', name: 'Gemma 4 26B-A4B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'A', tokPerSec: 59, prefillRate: 1150, vram: '~16.5 GB', maxCtx: '64K', quality: '11/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen9b', name: 'Qwen 3.5 9B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'B', tokPerSec: 35, prefillRate: 1750, vram: '~5.5 GB', maxCtx: '32K', quality: '9/17', thinking: false, thinkingBudget: 0, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-nemotron4b', name: 'Nemotron 3 Nano 4B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'B', tokPerSec: 66, prefillRate: 2900, vram: '~2.8 GB', maxCtx: '32K', quality: '7/17', thinking: true, thinkingBudget: 8192, color: '#38bdf8', hwColor: '#93c5fd' },
  // DGX Spark (273 GB/s)
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

// ── Experiment Presets ──
const EXPERIMENT_CATEGORIES = [
  { id: 'platform', label: 'Platform Shootouts' },
  { id: 'cross-platform', label: 'Cross-Platform' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'quality-speed', label: 'Quality vs Speed' },
  { id: 'thinking', label: 'Thinking Models' },
]

const EXPERIMENTS = [
  { id: '5090-best', category: 'platform', name: '5090 Best 6', desc: 'Top models on RTX 5090', columns: 3, models: ['5090-gemma26b-q6','5090-gemma31b','5090-qwen27b-opus','5090-gemma26b-q4','5090-harmonic27b','5090-qwopus27b'] },
  { id: 'm4-best', category: 'platform', name: 'M4 Max Best 6', desc: 'Top models on M4 Max — bandwidth-limited', columns: 3, models: ['m4-gemma31b','m4-gemma26b-q6','m4-qwen27b-mlx','m4-qwen27b-opus','m4-gemma26b-q4','m4-qwen9b'] },
  { id: 'spark-best', category: 'platform', name: 'Spark Best 6', desc: '128GB unlocks 100B+ models', columns: 3, models: ['spark-qwen122b-ik','spark-qwen122b-unsloth','spark-glm45','spark-qwen122b-reap','spark-qwen122b-mainline','spark-qwen3-coder'] },

  { id: 'gemma26b-q6-xplat', category: 'cross-platform', name: 'Gemma 26B Q6: 5090 vs M4', desc: 'Same MoE model — 2.1x speed gap maps to bandwidth', columns: 2, models: ['5090-gemma26b-q6','m4-gemma26b-q6'] },
  { id: 'gemma31b-3way', category: 'cross-platform', name: 'Gemma 31B: Three Platforms', desc: '50 vs 15 vs 7 tok/s — more memory != faster', columns: 3, models: ['5090-gemma31b','m4-gemma31b','spark-gemma31b-dense'] },
  { id: 'gemma26b-quant-xplat', category: 'cross-platform', name: 'Gemma 26B: Q6 vs Q4 x Platform', desc: 'Quantization impact varies by hardware', columns: 2, models: ['5090-gemma26b-q6','5090-gemma26b-q4','m4-gemma26b-q6','m4-gemma26b-q4'] },

  { id: 'moe-vs-dense-5090', category: 'architecture', name: 'MoE vs Dense on 5090', desc: 'MoE activates 3-4B params — 2-3x faster', columns: 2, models: ['5090-gemma26b-q6','5090-qwen35b-a3b','5090-gemma31b','5090-qwen27b-opus'] },
  { id: 'moe-scale', category: 'architecture', name: 'MoE Scale: 26B to 122B', desc: '3x active params + slower HW = double penalty', columns: 3, models: ['5090-gemma26b-q6','5090-qwen35b-a3b','spark-qwen122b-ik'] },
  { id: 'qwen122b-variants', category: 'architecture', name: 'Qwen 122B Variants', desc: 'REAP pruning saves 14GB and adds 3 tok/s', columns: 2, models: ['spark-qwen122b-ik','spark-qwen122b-unsloth','spark-qwen122b-reap','spark-qwen122b-mainline'] },
  { id: 'small-vs-big-m4', category: 'architecture', name: 'Small vs Big on M4', desc: '26B MoE matches 4B speed at 2x quality', columns: 2, models: ['m4-nemotron4b','m4-qwen9b','m4-gemma26b-q6','m4-gemma31b'] },

  { id: 's-tier-showdown', category: 'quality-speed', name: 'S-Tier Showdown', desc: 'Best of each platform — 7x speed range', columns: 3, models: ['5090-gemma26b-q6','5090-gemma31b','m4-gemma31b','m4-gemma26b-q6','spark-qwen122b-ik','spark-qwen122b-unsloth'] },
  { id: 'speed-vs-quality-5090', category: 'quality-speed', name: 'Speed vs Quality on 5090', desc: 'Gemma 26B is the sweet spot', columns: 2, models: ['5090-qwen35b-a3b','5090-gemma-e4b','5090-gemma26b-q6','5090-gemma31b'] },
  { id: 'f-tier', category: 'quality-speed', name: 'The F-Tier', desc: 'Speed without quality or quality without speed', columns: 3, models: ['5090-gemma-e4b','spark-gemma31b-dense','spark-mistral119b'] },

  { id: 'thinking-compared', category: 'thinking', name: 'Thinking Models Compared', desc: '16K thinking = 4.4 min before output', columns: 2, models: ['5090-harmonic27b','m4-nemotron4b','spark-nemotron120b','spark-minimax'] },
  { id: 'thinking-vs-not-5090', category: 'thinking', name: 'Thinking vs Direct on 5090', desc: 'Is reasoning worth 11x wall time?', columns: 3, models: ['5090-harmonic27b','5090-qwen27b-opus','5090-gemma26b-q6'] },
]

// ── Token Generation ──
const CODE_RESPONSE = `I'll fix the race condition in the connection pool and add proper cleanup.

The issue is in \`src/db/pool.ts\` — the \`acquire()\` method doesn't check whether a connection was already returned before handing it out again. Here's the diff:

\`\`\`diff
--- a/src/db/pool.ts
+++ b/src/db/pool.ts
@@ -23,8 +23,12 @@ export class ConnectionPool {
   private connections: Connection[] = [];
   private waiting: Array<(conn: Connection) => void> = [];
+  private inUse = new Set<string>();

   async acquire(): Promise<Connection> {
     const conn = this.connections.find(
-      (c) => c.state === "idle"
+      (c) => c.state === "idle" && !this.inUse.has(c.id)
     );
     if (conn) {
+      this.inUse.add(conn.id);
       conn.state = "active";
       return conn;
     }
@@ -45,6 +49,7 @@ export class ConnectionPool {
   release(conn: Connection): void {
     conn.state = "idle";
+    this.inUse.delete(conn.id);
     const next = this.waiting.shift();
     if (next) {
       next(conn);
\`\`\`

The root cause was that \`find()\` could return a connection that another async caller had already claimed but hadn't yet set to "active". The \`inUse\` set acts as a synchronous lock.

I also need to update the shutdown method to wait for in-flight connections:

\`\`\`typescript
async shutdown(timeoutMs = 5000): Promise<void> {
  // Stop accepting new requests
  this.accepting = false;

  // Wait for all active connections to be released
  const deadline = Date.now() + timeoutMs;
  while (this.inUse.size > 0 && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
  }

  if (this.inUse.size > 0) {
    console.warn(
      \`Forcing shutdown with \${this.inUse.size} active connections\`
    );
  }

  // Close all connections
  await Promise.allSettled(
    this.connections.map((c) => c.close())
  );
  this.connections = [];
  this.inUse.clear();
}
\`\`\`

And the test should verify concurrent access doesn't hand out the same connection:

\`\`\`typescript
test("concurrent acquire does not return same connection", async () => {
  const pool = new ConnectionPool({ maxSize: 2 });

  // Acquire two connections simultaneously
  const [a, b] = await Promise.all([
    pool.acquire(),
    pool.acquire(),
  ]);

  expect(a.id).not.toBe(b.id);
  expect(pool.activeCount).toBe(2);

  pool.release(a);
  pool.release(b);
  expect(pool.activeCount).toBe(0);
});

test("shutdown waits for active connections", async () => {
  const pool = new ConnectionPool({ maxSize: 3 });
  const conn = await pool.acquire();

  const shutdownPromise = pool.shutdown(1000);

  // Release after a short delay
  setTimeout(() => pool.release(conn), 100);

  await shutdownPromise;
  expect(pool.size).toBe(0);
});
\`\`\`

The changes are backward-compatible — existing callers don't need to change. The \`inUse\` tracking adds negligible overhead since it's a Set lookup (O(1)).
`.trim()

const tokenizeResponse = (text) => {
  const raw = text.split(/(?<=\s)|(?=\s)|(?<=[\`\{\}\(\)\[\];:,.<>+\-=!&|])|(?=[\`\{\}\(\)\[\];:,.<>+\-=!&|])/)
  return raw.filter(t => t.length > 0)
}

const RESPONSE_TOKENS = tokenizeResponse(CODE_RESPONSE)

const generateText = (tokenCount) => {
  const tokens = []
  for (let i = 0; i < tokenCount; i++) {
    tokens.push(RESPONSE_TOKENS[i % RESPONSE_TOKENS.length])
  }
  return tokens
}

// ── Markdown Renderer ──
const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')

const renderMarkdown = (text) => {
  const lines = text.split('\n')
  let html = ''
  let inCode = false
  let lang = ''

  for (const line of lines) {
    if (line.startsWith('```')) {
      if (inCode) {
        html += '</code></pre>'
        inCode = false
        lang = ''
      } else {
        lang = line.slice(3).trim()
        html += '<pre class="md-pre"><code>'
        inCode = true
      }
      continue
    }

    if (inCode) {
      const esc = escapeHtml(line)
      if (lang === 'diff') {
        if (line.startsWith('+')) html += `<span class="md-add">${esc}</span>\n`
        else if (line.startsWith('-')) html += `<span class="md-del">${esc}</span>\n`
        else if (line.startsWith('@@')) html += `<span class="md-hunk">${esc}</span>\n`
        else html += esc + '\n'
      } else {
        html += esc + '\n'
      }
    } else {
      if (line.trim() === '') { html += '<br/>'; continue }
      let p = escapeHtml(line)
      p = p.replace(/`([^`]+)`/g, '<code class="md-inline">$1</code>')
      p = p.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      html += `<span class="md-text">${p}</span>\n`
    }
  }
  if (inCode) html += '</code></pre>'
  return html
}

// ── Presets ──
const PROMPT_PRESETS = [
  { label: 'Quick question', tokens: 500, desc: 'Short prompt, no context' },
  { label: 'Single file edit', tokens: 2000, desc: '~1 file + instructions' },
  { label: 'Multi-file task', tokens: 8000, desc: '3-5 files + conversation' },
  { label: 'Large refactor', tokens: 24000, desc: '10+ files + history' },
  { label: 'Full codebase context', tokens: 64000, desc: 'Deep repo exploration' },
  { label: 'Max context window', tokens: 100000, desc: 'Pushing the limits' },
]

const OUTPUT_PRESETS = [
  { label: 'Short answer', tokens: 200, desc: 'Quick explanation or fix' },
  { label: 'Single function', tokens: 500, desc: 'One function + explanation' },
  { label: 'File implementation', tokens: 1500, desc: 'Full file with tests' },
  { label: 'Multi-file change', tokens: 4000, desc: 'Several files + refactor' },
  { label: 'Large generation', tokens: 8000, desc: 'Major feature implementation' },
  { label: 'Max output', tokens: 16000, desc: 'Pushing output limits' },
]

const TIER_COLORS = { S: '#fbbf24', A: '#34d399', B: '#60a5fa', C: '#a78bfa', D: '#f87171', F: '#6b7280' }

const formatTime = (seconds) => {
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 10) return `${seconds.toFixed(1)}s`
  return `${Math.round(seconds)}s`
}

// ── TokenStream Component ──
const TokenStream = ({ model, tokens, isRunning, isReset, tokenCount, promptTokens, onComplete, streamIndex }) => {
  const [displayedTokens, setDisplayedTokens] = useState([])
  const [phase, setPhase] = useState('idle')
  const [thinkingTokensGenerated, setThinkingTokensGenerated] = useState(0)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [prefillElapsed, setPrefillElapsed] = useState(0)
  const [compactElapsed, setCompactElapsed] = useState(0)
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
      if (contentRef.current) contentRef.current.scrollTop = contentRef.current.scrollHeight
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
      setCompactElapsed(0)
      totalIndexRef.current = 0
      hasStartedRef.current = false
      startTimeRef.current = null
      decodeStartRef.current = null
      return
    }

    if (isRunning && !hasStartedRef.current) {
      hasStartedRef.current = true
      startTimeRef.current = Date.now()

      timerRef.current = setInterval(() => {
        if (startTimeRef.current) setElapsedTime(((Date.now() - startTimeRef.current) / 1000).toFixed(1))
      }, 100)

      const maxCtxTokens = parseInt(model.maxCtx) * 1000
      const usedTokens = promptTokens + thinkingBudget + tokenCount
      const usageRatio = usedTokens / maxCtxTokens
      const needsCompact = usageRatio > 0.8
      const compactTokens = needsCompact ? Math.max(0, usedTokens - maxCtxTokens * 0.6) : 0
      const compactMs = needsCompact ? (compactTokens / (model.prefillRate * 0.5)) * 1000 : 0

      const startPrefill = () => {
        setPhase('prefill')
        const prefillMs = (promptTokens / model.prefillRate) * 1000
        const prefillStart = Date.now()
        const animatePrefill = () => {
          const elapsed = Date.now() - prefillStart
          const progress = Math.min(elapsed / prefillMs, 1)
          setPrefillElapsed(progress)
          if (progress < 1) prefillTimerRef.current = setTimeout(animatePrefill, 16)
        }
        animatePrefill()

        prefillTimerRef.current = setTimeout(() => {
          decodeStartRef.current = Date.now()
          setPhase(thinkingBudget > 0 ? 'thinking' : 'streaming')

          const interval = 1000 / model.tokPerSec
          intervalRef.current = setInterval(() => {
            if (totalIndexRef.current < totalTokens) {
              if (totalIndexRef.current < thinkingBudget) {
                setThinkingTokensGenerated(totalIndexRef.current + 1)
                totalIndexRef.current++
              } else {
                if (totalIndexRef.current === thinkingBudget && thinkingBudget > 0) setPhase('streaming')
                const di = totalIndexRef.current - thinkingBudget
                if (di < tokenCount) setDisplayedTokens(prev => [...prev, tokens[di]])
                totalIndexRef.current++
              }
            } else {
              clearInterval(intervalRef.current)
              clearInterval(timerRef.current)
              if (startTimeRef.current) setElapsedTime(((Date.now() - startTimeRef.current) / 1000).toFixed(1))
              setPhase('complete')
              onComplete(streamIndex)
            }
          }, interval)
        }, prefillMs)
      }

      if (needsCompact) {
        setPhase('compacting')
        const compactStart = Date.now()
        const animateCompact = () => {
          const elapsed = Date.now() - compactStart
          const progress = Math.min(elapsed / compactMs, 1)
          setCompactElapsed(progress)
          if (progress < 1) prefillTimerRef.current = setTimeout(animateCompact, 16)
        }
        animateCompact()
        prefillTimerRef.current = setTimeout(startPrefill, compactMs)
      } else {
        startPrefill()
      }
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRunning, isReset, model.tokPerSec, model.prefillRate, model.maxCtx, promptTokens, thinkingBudget, tokenCount, totalTokens, tokens, onComplete, streamIndex])

  useEffect(() => {
    if (displayedTokens.length > 0) scrollToBottom()
  }, [displayedTokens, scrollToBottom])

  const totalProgress = totalTokens > 0 ? ((thinkingTokensGenerated + displayedTokens.length) / totalTokens) * 100 : 0
  const decodeElapsed = decodeStartRef.current ? (Date.now() - decodeStartRef.current) / 1000 : 0
  const rate = displayedTokens.length > 0 && decodeElapsed > 0 ? (displayedTokens.length / decodeElapsed).toFixed(1) : null

  const statusLabel = { idle: 'Ready', compacting: 'Compacting', prefill: 'Prefill', thinking: 'Thinking', streaming: 'Streaming', complete: 'Done' }[phase]
  const cardClass = ['stream-card', (phase !== 'idle' && phase !== 'complete') && 'is-running', phase === 'complete' && 'is-complete'].filter(Boolean).join(' ')
  const thinkingLabel = model.thinking ? `${model.thinkingBudget.toLocaleString()}` : 'Off'

  // Context window
  const maxCtxTokens = parseInt(model.maxCtx) * 1000
  const usedTokens = promptTokens + thinkingBudget + tokenCount
  const overflows = usedTokens > maxCtxTokens
  const promptPct = Math.min((promptTokens / maxCtxTokens) * 100, 100)
  const thinkingPct = Math.min((thinkingBudget / maxCtxTokens) * 100, 100 - promptPct)
  const outputPct = Math.min((tokenCount / maxCtxTokens) * 100, Math.max(0, 100 - promptPct - thinkingPct))

  // Markdown
  const markdownHtml = displayedTokens.length > 0 ? renderMarkdown(displayedTokens.join('')) : ''

  return (
    <div className={cardClass}>
      <div className="card-accent" style={{ background: model.color }} />

      <div className="card-header">
        <div className="model-info">
          <div className="model-title-row">
            <span className="tier-badge" style={{ background: TIER_COLORS[model.tier] }}>{model.tier}</span>
            <span className="model-name">{model.name}</span>
          </div>
          <span className="model-quant">
            {model.quant}
            <a className="citation-link" href={CITATIONS[model.hardware]} target="_blank" rel="noopener noreferrer" title="View benchmark data">cite</a>
          </span>
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

      <div className="ctx-bar-wrapper">
        <div className="ctx-bar-labels">
          <span>Context: {usedTokens.toLocaleString()} / {maxCtxTokens.toLocaleString()}{overflows ? ' — overflow!' : ''}</span>
        </div>
        <div className={`ctx-bar ${overflows ? 'ctx-overflow' : ''}`}>
          <span className="ctx-bar-tick" />
          <span className="ctx-bar-tick-label">compact</span>
          <div className="ctx-bar-fill-area">
            <div className="ctx-seg ctx-prompt" style={{ width: `${promptPct}%` }} />
            {thinkingBudget > 0 && <div className="ctx-seg ctx-thinking" style={{ width: `${thinkingPct}%` }} />}
            <div className="ctx-seg ctx-output" style={{ width: `${outputPct}%` }} />
          </div>
        </div>
        <div className="ctx-bar-legend">
          <span className="ctx-legend-item"><span className="ctx-swatch ctx-prompt" />Prompt</span>
          {thinkingBudget > 0 && <span className="ctx-legend-item"><span className="ctx-swatch ctx-thinking" />Thinking</span>}
          <span className="ctx-legend-item"><span className="ctx-swatch ctx-output" />Output</span>
          <span className="ctx-legend-item ctx-free">{Math.max(0, maxCtxTokens - usedTokens).toLocaleString()} free</span>
        </div>
      </div>

      <div className="stats-row">
        <div className="stat"><span className="stat-label">Output</span><span className="stat-value">{displayedTokens.length.toLocaleString()} / {tokenCount.toLocaleString()}</span></div>
        <div className="stat"><span className="stat-label">Thinking</span><span className={`stat-value ${!model.thinking ? 'stat-dim' : ''}`}>{thinkingLabel}</span></div>
        <div className="stat"><span className="stat-label">Time</span><span className="stat-value">{elapsedTime}s</span></div>
        {rate && <div className="stat"><span className="stat-label">Actual</span><span className="stat-value">{rate} tok/s</span></div>}
      </div>

      <div className="progress-track"><div className="progress-fill" style={{ width: `${totalProgress}%`, background: model.color }} /></div>

      {phase === 'compacting' && (
        <div className="compact-banner"><span className="compact-spinner" /><div className="compact-detail"><span>Compacting context — summarizing prior turns</span><div className="compact-bar"><div className="compact-bar-fill" style={{ width: `${compactElapsed * 100}%` }} /></div></div></div>
      )}

      {phase === 'prefill' && (
        <div className="prefill-banner"><div className="prefill-bar"><div className="prefill-bar-fill" style={{ width: `${prefillElapsed * 100}%`, background: model.color }} /></div><span className="prefill-label">Prefilling {promptTokens.toLocaleString()} tokens @ {model.prefillRate} tok/s — {formatTime(promptTokens / model.prefillRate)}</span></div>
      )}

      {phase === 'thinking' && (
        <div className="thinking-banner"><span className="thinking-spinner" /><div className="thinking-detail"><span>Thinking</span><span className="thinking-count">{thinkingTokensGenerated.toLocaleString()} / {thinkingBudget.toLocaleString()}</span></div></div>
      )}

      <div ref={contentRef} className="stream-content">
        {displayedTokens.length === 0 && phase === 'idle' && <div className="stream-empty">Waiting to start</div>}
        {displayedTokens.length === 0 && phase === 'prefill' && <div className="stream-empty">Processing prompt...</div>}
        {displayedTokens.length === 0 && phase === 'compacting' && <div className="stream-empty">Compacting...</div>}
        {displayedTokens.length === 0 && phase === 'thinking' && <div className="stream-empty">Reasoning...</div>}
        {markdownHtml && <div className="md-content" dangerouslySetInnerHTML={{ __html: markdownHtml }} />}
        {phase === 'streaming' && <span className="cursor" />}
      </div>
    </div>
  )
}

// ── App ──
function App() {
  const [activeExperiment, setActiveExperiment] = useState('5090-best')
  const [isRunning, setIsRunning] = useState(false)
  const [isReset, setIsReset] = useState(false)
  const [tokenCount, setTokenCount] = useState(1500)
  const [promptTokens, setPromptTokens] = useState(2000)
  const [completedStreams, setCompletedStreams] = useState(new Set())

  const experiment = EXPERIMENTS.find(e => e.id === activeExperiment)
  const selectedModels = experiment.models.map(id => MODELS.find(m => m.id === id))
  const maxThinkingBudget = Math.max(...selectedModels.map(m => m.thinkingBudget))
  const maxTotalTokens = tokenCount + maxThinkingBudget

  const handleExperimentChange = (id) => {
    if (isRunning) return
    setActiveExperiment(id)
    setCompletedStreams(new Set())
  }

  const handleComplete = useCallback((index) => {
    setCompletedStreams(prev => { const next = new Set(prev); next.add(index); return next })
  }, [])

  const handleStart = () => { setIsReset(false); setCompletedStreams(new Set()); setIsRunning(true) }
  const handleReset = () => { setIsRunning(false); setIsReset(true); setCompletedStreams(new Set()); setTimeout(() => setIsReset(false), 100) }

  const tokens = useMemo(() => generateText(maxTotalTokens), [maxTotalTokens])
  const allComplete = completedStreams.size >= experiment.models.length
  const controlsDisabled = isRunning && !allComplete

  // Chunk models into rows
  const cols = experiment.columns
  const rows = []
  for (let i = 0; i < selectedModels.length; i += cols) {
    rows.push({ start: i, models: selectedModels.slice(i, i + cols) })
  }

  return (
    <div className="app-layout">
      <nav className="experiment-nav">
        <div className="nav-title">Experiments</div>
        {EXPERIMENT_CATEGORIES.map(cat => (
          <div key={cat.id} className="nav-category">
            <div className="nav-cat-label">{cat.label}</div>
            {EXPERIMENTS.filter(e => e.category === cat.id).map(exp => (
              <button
                key={exp.id}
                className={`nav-item ${activeExperiment === exp.id ? 'active' : ''}`}
                onClick={() => handleExperimentChange(exp.id)}
                disabled={controlsDisabled}
              >
                <span className="nav-item-name">{exp.name}</span>
                <span className="nav-item-count">{exp.models.length}</span>
              </button>
            ))}
          </div>
        ))}
      </nav>

      <main className="app-main">
        <header className="app-header">
          <div className="header-top">
            <h1>Token Speed Simulator</h1>
            <a className="repo-link" href="https://github.com/gisenberg/local-model-eval" target="_blank" rel="noopener noreferrer">local-model-eval</a>
          </div>
          <p>{experiment.desc}</p>
        </header>

        <div className="controls">
          <div className="control-group">
            <label>Output tokens<span>{tokenCount.toLocaleString()}</span></label>
            <select value={tokenCount} onChange={(e) => setTokenCount(parseInt(e.target.value))} disabled={controlsDisabled} className="prompt-select">
              {OUTPUT_PRESETS.map(p => <option key={p.tokens} value={p.tokens}>{p.label} — {p.tokens.toLocaleString()}</option>)}
            </select>
            <div className="prompt-desc">{OUTPUT_PRESETS.find(p => p.tokens === tokenCount)?.desc}</div>
          </div>
          <div className="control-group">
            <label>Prompt context<span>{promptTokens.toLocaleString()}</span></label>
            <select value={promptTokens} onChange={(e) => setPromptTokens(parseInt(e.target.value))} disabled={controlsDisabled} className="prompt-select">
              {PROMPT_PRESETS.map(p => <option key={p.tokens} value={p.tokens}>{p.label} — {p.tokens.toLocaleString()}</option>)}
            </select>
            <div className="prompt-desc">{PROMPT_PRESETS.find(p => p.tokens === promptTokens)?.desc}</div>
          </div>
          <div className="button-group">
            <button onClick={handleStart} disabled={controlsDisabled} className="btn-start">{controlsDisabled ? 'Running...' : 'Start'}</button>
            <button onClick={handleReset} className="btn-reset">Stop</button>
          </div>
        </div>

        {rows.map(({ start, models }) => (
          <div key={start} className="sim-row" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            {models.map((model, i) => (
              <TokenStream
                key={model.id + '-' + (start + i)}
                model={model}
                tokens={tokens}
                isRunning={isRunning}
                isReset={isReset}
                tokenCount={tokenCount}
                promptTokens={promptTokens}
                onComplete={handleComplete}
                streamIndex={start + i}
              />
            ))}
          </div>
        ))}
      </main>
    </div>
  )
}

export default App

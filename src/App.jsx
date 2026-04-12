import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import './App.css'

// ── Citation URLs ──
const CITATIONS = {
  'RTX 5090': 'https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_5090.md',
  'M4 Max': 'https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_M4MAX.md',
  'DGX Spark': 'https://github.com/gisenberg/local-model-eval/blob/main/results/MODEL_RANKINGS_SPARK.md',
  'Anthropic API': 'https://artificialanalysis.ai/providers/anthropic',
  'Google API': 'https://artificialanalysis.ai/providers/google',
  'OpenAI API': 'https://artificialanalysis.ai/providers/openai',
  'OpenRouter Free': 'https://github.com/gisenberg/local-model-eval/blob/main/results/API_BENCH_5090.md',
  'NVIDIA Free': 'https://github.com/gisenberg/local-model-eval/blob/main/results/API_BENCH_5090.md',
}

// weightGB: base VRAM (weights + compute buffers, no KV). Derived from measured VRAM@32K minus KV@32K.
// kvPerTokKB: incremental KV per token. Measured from context-size deltas where available.
// 5090 values: measured via turbo4 experiments. M4/Spark: estimated from weight sizes + arch.
const HW_MEM = { 'RTX 5090': 32, 'M4 Max': 30, 'DGX Spark': 120, 'Anthropic API': 0, 'Google API': 0, 'OpenAI API': 0, 'OpenRouter Free': 0, 'NVIDIA Free': 0 }

const MODELS = [
  // RTX 5090 — measured VRAM from experiments/**/all_results.json (turbo4 KV @ 32K)
  // gemma26b-q6: 25,636 MB measured. KV ~5.3 KB/tok (turbo4, 5 non-SWA layers, 2 KV heads)
  { id: '5090-gemma26b-q6', name: 'Gemma 4 26B-A4B', quant: 'Q6_K', hardware: 'RTX 5090', tier: 'S', tokPerSec: 139, prefillRate: 2900, weightGB: 25.6, kvPerTokKB: 5.3, maxCtx: '262K', quality: '17/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // gemma31b: 28,025 MB @ 262K measured. KV breakdown: 2.7 GB non-SWA + 0.7 GB SWA = 3.4 GB total
  { id: '5090-gemma31b', name: 'Gemma 4 31B-IT', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'S', tokPerSec: 46, prefillRate: 1900, weightGB: 24.6, kvPerTokKB: 13, maxCtx: '262K', quality: '17/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // qwen27b-opus: 19,565 MB measured. Dense 32L, ~50 KB/tok turbo4
  { id: '5090-qwen27b-opus', name: 'Qwen 3.5 27B Opus', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'A', tokPerSec: 60, prefillRate: 1900, weightGB: 17.6, kvPerTokKB: 50, maxCtx: '262K', quality: '17/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // gemma26b-q4: 20,046 MB measured. Same arch as q6, ~5.3 KB/tok turbo4
  { id: '5090-gemma26b-q4', name: 'Gemma 4 26B-A4B', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'A', tokPerSec: 150, prefillRate: 3000, weightGB: 19.4, kvPerTokKB: 5.3, maxCtx: '262K', quality: '16/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // harmonic27b q4km: 19,995 MB measured. Qwen 27B arch, ~50 KB/tok turbo4
  { id: '5090-harmonic27b', name: 'Harmonic 27B', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'A', tokPerSec: 61, prefillRate: 1800, weightGB: 18.0, kvPerTokKB: 50, maxCtx: '262K', quality: '31/31', thinking: true, thinkingBudget: 16384, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // qwopus27b q6k: 27,912 MB @ 196K measured. DeltaNet hybrid (16/32 attn layers), 17 KB/tok turbo4
  { id: '5090-qwopus27b', name: 'Qwopus 3.5 27B-v3', quant: 'Q6_K', hardware: 'RTX 5090', tier: 'A', tokPerSec: 50, prefillRate: 1800, weightGB: 24.7, kvPerTokKB: 17, maxCtx: '262K', quality: '16/17', thinking: false, thinkingBudget: 0, outputMul: 2.5, color: '#f87171', hwColor: '#86efac' },
  // gemma31b-opus: 23,199 MB measured @ 32K. Same arch as gemma31b, ~13 KB/tok turbo4
  { id: '5090-gemma31b-opus', name: 'Gemma 31B Opus-Dist.', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'B', tokPerSec: 51, prefillRate: 2000, weightGB: 22.3, kvPerTokKB: 13, maxCtx: '262K', quality: '16/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // qwen35b-a3b: 23,910 MB measured. DeltaNet hybrid, ~5 KB/tok turbo4
  { id: '5090-qwen35b-a3b', name: 'Qwen 3.5 35B-A3B', quant: 'Q4_K_M', hardware: 'RTX 5090', tier: 'C', tokPerSec: 174, prefillRate: 2400, weightGB: 23.2, kvPerTokKB: 5, maxCtx: '262K', quality: '11/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // qwen27b-base q6k: 24,606 MB measured. Dense 32L, ~50 KB/tok turbo4
  { id: '5090-qwen27b-base', name: 'Qwen 3.5 27B', quant: 'Q6_K (base)', hardware: 'RTX 5090', tier: 'C', tokPerSec: 50, prefillRate: 1700, weightGB: 23.0, kvPerTokKB: 50, maxCtx: '196K', quality: '10/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // gemma-e4b: 12,108 MB measured (turbo4). Small model, ~10 KB/tok turbo4
  { id: '5090-gemma-e4b', name: 'Gemma 4 E4B', quant: 'Q8_0', hardware: 'RTX 5090', tier: 'F', tokPerSec: 131, prefillRate: 5000, weightGB: 11.5, kvPerTokKB: 10, maxCtx: '256K', quality: '5/22', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#f87171', hwColor: '#86efac' },
  // M4 Max — no measured VRAM in benchmarks, estimated from weight files + ~2 GB compute buffer
  { id: 'm4-gemma31b', name: 'Gemma 4 31B-IT', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'S', tokPerSec: 15, prefillRate: 390, weightGB: 20.3, kvPerTokKB: 47, maxCtx: '128K', quality: '17/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-gemma26b-q6', name: 'Gemma 4 26B-A4B', quant: 'Q6_K', hardware: 'M4 Max', tier: 'S', tokPerSec: 66, prefillRate: 980, weightGB: 24.6, kvPerTokKB: 20, maxCtx: '64K', quality: '15/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen27b-mlx', name: 'Qwen 27B Opus MLX', quant: '4-bit', hardware: 'M4 Max', tier: 'A', tokPerSec: 19, prefillRate: 500, weightGB: 16.0, kvPerTokKB: 200, maxCtx: '64K', quality: '13/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen27b-opus', name: 'Qwen 27B Opus', quant: 'Q4_K_M (planar3)', hardware: 'M4 Max', tier: 'A', tokPerSec: 16, prefillRate: 440, weightGB: 18.5, kvPerTokKB: 80, maxCtx: '128K', quality: '11/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-gemma26b-q4', name: 'Gemma 4 26B-A4B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'A', tokPerSec: 59, prefillRate: 1150, weightGB: 18.5, kvPerTokKB: 20, maxCtx: '64K', quality: '11/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-qwen9b', name: 'Qwen 3.5 9B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'B', tokPerSec: 35, prefillRate: 1750, weightGB: 7.5, kvPerTokKB: 128, maxCtx: '128K', quality: '9/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  { id: 'm4-nemotron4b', name: 'Nemotron 3 Nano 4B', quant: 'Q4_K_M', hardware: 'M4 Max', tier: 'B', tokPerSec: 66, prefillRate: 2900, weightGB: 4.8, kvPerTokKB: 16, maxCtx: '128K', quality: '7/17', thinking: true, thinkingBudget: 8192, outputMul: 1, color: '#38bdf8', hwColor: '#93c5fd' },
  // DGX Spark — no measured VRAM, estimated from weight files. KV is cheap (MoE, ~24 KB/tok f16)
  { id: 'spark-qwen122b-ik', name: 'Qwen 3.5 122B-A10B', quant: 'Q4_K_M (ik-llama)', hardware: 'DGX Spark', tier: 'S', tokPerSec: 26, prefillRate: 627, weightGB: 73, kvPerTokKB: 24, maxCtx: '256K', quality: '17/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen122b-unsloth', name: 'Qwen 3.5 122B-A10B', quant: 'Q4_K_M (mainline)', hardware: 'DGX Spark', tier: 'S', tokPerSec: 21, prefillRate: 600, weightGB: 74, kvPerTokKB: 24, maxCtx: '256K', quality: '18/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-glm45', name: 'GLM-4.5-Air', quant: 'Q4_K_M', hardware: 'DGX Spark', tier: 'A', tokPerSec: 22, prefillRate: 627, weightGB: 72, kvPerTokKB: 24, maxCtx: '128K', quality: '15/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen122b-reap', name: 'Qwen 122B REAP-20', quant: 'Q4_K_M (pruned)', hardware: 'DGX Spark', tier: 'A', tokPerSec: 29, prefillRate: 700, weightGB: 59, kvPerTokKB: 24, maxCtx: '256K', quality: '14/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen122b-mainline', name: 'Qwen 122B-A10B', quant: 'Q4_K_M (bartowski)', hardware: 'DGX Spark', tier: 'A', tokPerSec: 26, prefillRate: 620, weightGB: 73, kvPerTokKB: 24, maxCtx: '256K', quality: '13/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-qwen3-coder', name: 'Qwen3-Coder-Next', quant: 'UD-Q4_K_M', hardware: 'DGX Spark', tier: 'B', tokPerSec: 50, prefillRate: 800, weightGB: 48, kvPerTokKB: 24, maxCtx: '262K', quality: '14/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-nemotron120b', name: 'Nemotron-3 Super 120B', quant: 'Q4_K_M', hardware: 'DGX Spark', tier: 'B', tokPerSec: 20, prefillRate: 500, weightGB: 89, kvPerTokKB: 24, maxCtx: '32K', quality: '11/17', thinking: true, thinkingBudget: 16384, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-minimax', name: 'MiniMax-M2.5', quant: 'UD-Q3_K_XL', hardware: 'DGX Spark', tier: 'C', tokPerSec: 30, prefillRate: 400, weightGB: 98, kvPerTokKB: 20, maxCtx: '32K', quality: '5/15', thinking: true, thinkingBudget: 16384, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-mistral119b', name: 'Mistral-Small-4 119B', quant: 'Q4_K_M', hardware: 'DGX Spark', tier: 'D', tokPerSec: 9, prefillRate: 350, weightGB: 71, kvPerTokKB: 24, maxCtx: '32K', quality: '7/17', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  { id: 'spark-gemma31b-dense', name: 'Gemma 4 31B-IT', quant: 'Q8_0 (dense)', hardware: 'DGX Spark', tier: 'F', tokPerSec: 7, prefillRate: 250, weightGB: 33, kvPerTokKB: 47, maxCtx: '262K', quality: 'N/A', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#a78bfa', hwColor: '#c4b5fd' },
  // Anthropic API — speeds from Artificial Analysis, TTFT-derived prefill rates
  // Thinking budgets: typical coding task amounts, not max capability
  // costIn/costOut: $ per 1M tokens (input/output)
  { id: 'cloud-opus46-1m', name: 'Claude Opus 4.6', quant: '1M context', hardware: 'Anthropic API', tier: 'S', tokPerSec: 48, prefillRate: 15000, weightGB: 0, kvPerTokKB: 0, maxCtx: '1000K', quality: 'frontier', thinking: true, thinkingBudget: 1500, outputMul: 1, costIn: 5, costOut: 25, color: '#d97706', hwColor: '#fbbf24' },
  { id: 'cloud-sonnet46', name: 'Claude Sonnet 4.6', quant: '200K context', hardware: 'Anthropic API', tier: 'S', tokPerSec: 66, prefillRate: 25000, weightGB: 0, kvPerTokKB: 0, maxCtx: '200K', quality: 'frontier', thinking: true, thinkingBudget: 1200, outputMul: 1, costIn: 3, costOut: 15, color: '#d97706', hwColor: '#fbbf24' },
  { id: 'cloud-haiku45', name: 'Claude Haiku 4.5', quant: '200K context', hardware: 'Anthropic API', tier: 'A', tokPerSec: 92, prefillRate: 60000, weightGB: 0, kvPerTokKB: 0, maxCtx: '200K', quality: 'good', thinking: false, thinkingBudget: 0, outputMul: 1, costIn: 1, costOut: 5, color: '#d97706', hwColor: '#fbbf24' },
  // Google API
  { id: 'cloud-gemini31pro', name: 'Gemini 3.1 Pro', quant: '1M context', hardware: 'Google API', tier: 'S', tokPerSec: 126, prefillRate: 20000, weightGB: 0, kvPerTokKB: 0, maxCtx: '1000K', quality: 'frontier', thinking: true, thinkingBudget: 1500, outputMul: 1, costIn: 2, costOut: 12, color: '#059669', hwColor: '#34d399' },
  { id: 'cloud-gemini25pro', name: 'Gemini 2.5 Pro', quant: '1M context', hardware: 'Google API', tier: 'S', tokPerSec: 122, prefillRate: 20000, weightGB: 0, kvPerTokKB: 0, maxCtx: '1000K', quality: 'frontier', thinking: true, thinkingBudget: 1500, outputMul: 1, costIn: 1.25, costOut: 10, color: '#059669', hwColor: '#34d399' },
  { id: 'cloud-gemini3flash', name: 'Gemini 3 Flash', quant: '1M context', hardware: 'Google API', tier: 'S', tokPerSec: 153, prefillRate: 60000, weightGB: 0, kvPerTokKB: 0, maxCtx: '1000K', quality: 'frontier', thinking: true, thinkingBudget: 800, outputMul: 1, costIn: 0.5, costOut: 3, color: '#059669', hwColor: '#34d399' },
  // OpenAI API
  { id: 'cloud-gpt41', name: 'GPT-4.1', quant: '1M context', hardware: 'OpenAI API', tier: 'A', tokPerSec: 100, prefillRate: 30000, weightGB: 0, kvPerTokKB: 0, maxCtx: '1000K', quality: 'good', thinking: false, thinkingBudget: 0, outputMul: 1, costIn: 2, costOut: 8, color: '#4f46e5', hwColor: '#818cf8' },
  { id: 'cloud-o3mini', name: 'o3-mini (high)', quant: '200K context', hardware: 'OpenAI API', tier: 'S', tokPerSec: 152, prefillRate: 30000, weightGB: 0, kvPerTokKB: 0, maxCtx: '200K', quality: 'frontier', thinking: true, thinkingBudget: 2000, outputMul: 1, costIn: 1.1, costOut: 4.4, color: '#4f46e5', hwColor: '#818cf8' },
  { id: 'cloud-gpt53-codex', name: 'GPT-5.3 Codex', quant: '400K context', hardware: 'OpenAI API', tier: 'S', tokPerSec: 71, prefillRate: 20000, weightGB: 0, kvPerTokKB: 0, maxCtx: '400K', quality: 'frontier', thinking: true, thinkingBudget: 2000, outputMul: 1, costIn: 1.75, costOut: 7, color: '#4f46e5', hwColor: '#818cf8' },
  { id: 'cloud-gpt54', name: 'GPT-5.4', quant: '1M context', hardware: 'OpenAI API', tier: 'S', tokPerSec: 83, prefillRate: 25000, weightGB: 0, kvPerTokKB: 0, maxCtx: '1000K', quality: 'frontier', thinking: true, thinkingBudget: 1500, outputMul: 1, costIn: 2.5, costOut: 15, costInHigh: 5, costOutHigh: 22.5, costInThreshold: 272000, color: '#4f46e5', hwColor: '#818cf8' },
  { id: 'cloud-gpt54-mini', name: 'GPT-5.4 mini', quant: '400K context', hardware: 'OpenAI API', tier: 'S', tokPerSec: 168, prefillRate: 40000, weightGB: 0, kvPerTokKB: 0, maxCtx: '400K', quality: 'frontier', thinking: true, thinkingBudget: 800, outputMul: 1, costIn: 0.4, costOut: 1.6, color: '#4f46e5', hwColor: '#818cf8' },
  { id: 'cloud-gpt54-nano', name: 'GPT-5.4 nano', quant: '400K context', hardware: 'OpenAI API', tier: 'A', tokPerSec: 184, prefillRate: 50000, weightGB: 0, kvPerTokKB: 0, maxCtx: '400K', quality: 'good', thinking: true, thinkingBudget: 500, outputMul: 1, costIn: 0.2, costOut: 1.25, color: '#4f46e5', hwColor: '#818cf8' },
  { id: 'cloud-gpt51-codex-mini', name: 'GPT-5.1 Codex mini', quant: '400K context', hardware: 'OpenAI API', tier: 'A', tokPerSec: 185, prefillRate: 45000, weightGB: 0, kvPerTokKB: 0, maxCtx: '400K', quality: 'good', thinking: true, thinkingBudget: 800, outputMul: 1, costIn: 1.25, costOut: 10, color: '#4f46e5', hwColor: '#818cf8' },
  // OpenRouter Free / NVIDIA Free — measured via api_bench.py, quality from 4-benchmark coding suite
  // GPT-OSS 120B: 16/22 (73%) — passes ExprEval+A*+String, 0/6 on LRU (TypeScript syntax contamination)
  { id: 'free-gpt-oss-120b', name: 'GPT-OSS 120B', quant: '131K context', hardware: 'OpenRouter Free', tier: 'B', tokPerSec: 34, prefillRate: 5000, weightGB: 0, kvPerTokKB: 0, maxCtx: '131K', quality: '16/22', thinking: false, thinkingBudget: 0, outputMul: 1, color: '#10b981', hwColor: '#6ee7b7' },
]

// ── Experiment Presets ──
const EXPERIMENT_CATEGORIES = [
  { id: 'cloud', label: 'Cloud API' },
  { id: 'free-tier', label: 'Free Tier' },
  { id: 'cloud-vs-local', label: 'Cloud vs Local' },
  { id: 'platform', label: 'Local Platforms' },
  { id: 'cross-platform', label: 'Cross-Platform' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'quality-speed', label: 'Quality vs Speed' },
  { id: 'thinking', label: 'Thinking Models' },
]

const EXPERIMENTS = [
  // Cloud API
  { id: 'anthropic-lineup', category: 'cloud', name: 'Anthropic Lineup', desc: 'Opus 4.6 vs Sonnet 4.6 vs Haiku 4.5', columns: 2, models: ['cloud-opus46-1m','cloud-sonnet46','cloud-haiku45'] },
  { id: 'google-lineup', category: 'cloud', name: 'Google Lineup', desc: 'Gemini 3.1 Pro vs 3 Flash vs 2.5 Pro', columns: 3, models: ['cloud-gemini31pro','cloud-gemini3flash','cloud-gemini25pro'] },
  { id: 'openai-lineup', category: 'cloud', name: 'OpenAI Lineup', desc: 'GPT-5.4 family + Codex models', columns: 3, models: ['cloud-gpt54','cloud-gpt54-mini','cloud-gpt54-nano','cloud-gpt53-codex','cloud-gpt51-codex-mini','cloud-o3mini'] },
  { id: 'cloud-all', category: 'cloud', name: 'Cloud Frontier', desc: 'Top model from each provider', columns: 3, models: ['cloud-opus46-1m','cloud-gemini31pro','cloud-gpt54','cloud-sonnet46','cloud-gemini3flash','cloud-gpt54-mini'] },
{ id: 'cloud-speed', category: 'cloud', name: 'Cloud Speed Demons', desc: 'Fastest output from each provider', columns: 3, models: ['cloud-haiku45','cloud-gemini3flash','cloud-gpt54-nano'] },
  // Free Tier
  { id: 'free-vs-local-s', category: 'free-tier', name: 'Free API vs Local S-Tier', desc: 'GPT-OSS 120B (free) vs best local models on RTX 5090', columns: 3, models: ['free-gpt-oss-120b','5090-gemma26b-q6','5090-gemma31b','5090-harmonic27b'] },
  { id: 'free-vs-local-c', category: 'free-tier', name: 'Free API vs Local C-Tier', desc: 'GPT-OSS 120B (free) vs Qwen 35B — same LRU gap, different models', columns: 2, models: ['free-gpt-oss-120b','5090-qwen35b-a3b'] },
  { id: 'free-vs-paid', category: 'free-tier', name: 'Free vs Paid Cloud', desc: 'Can free models compete with frontier APIs?', columns: 3, models: ['free-gpt-oss-120b','cloud-haiku45','cloud-gpt54-nano','cloud-sonnet46'] },
  // Cloud vs Local
  { id: 'cloud-vs-5090', category: 'cloud-vs-local', name: 'Cloud vs RTX 5090', desc: 'Opus, Gemini Pro, GPT-5.4 vs fastest local GPU', columns: 3, models: ['cloud-opus46-1m','cloud-gemini31pro','cloud-gpt54','5090-gemma26b-q6','5090-gemma26b-q4','5090-qwen35b-a3b'] },
  { id: 'cloud-vs-spark', category: 'cloud-vs-local', name: 'Cloud vs DGX Spark', desc: 'Opus, Gemini Pro, GPT-5.4 vs 122B local models', columns: 3, models: ['cloud-opus46-1m','cloud-gemini31pro','cloud-gpt54','spark-qwen122b-ik','spark-qwen3-coder','spark-glm45'] },
  { id: 'cloud-vs-m4', category: 'cloud-vs-local', name: 'Cloud vs M4 Max', desc: 'Opus, Gemini Pro, GPT-5.4 vs portable local', columns: 3, models: ['cloud-opus46-1m','cloud-gemini31pro','cloud-gpt54','m4-gemma26b-q6','m4-gemma31b','m4-qwen9b'] },
  // Local platforms
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
  this.accepting = false;
  const deadline = Date.now() + timeoutMs;
  while (this.inUse.size > 0 && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 50));
  }
  if (this.inUse.size > 0) {
    console.warn(\`Forcing shutdown with \${this.inUse.size} active connections\`);
  }
  await Promise.allSettled(this.connections.map((c) => c.close()));
  this.connections = [];
  this.inUse.clear();
}
\`\`\`

And the test should verify concurrent access doesn't hand out the same connection:

\`\`\`typescript
test("concurrent acquire does not return same connection", async () => {
  const pool = new ConnectionPool({ maxSize: 2 });
  const [a, b] = await Promise.all([pool.acquire(), pool.acquire()]);
  expect(a.id).not.toBe(b.id);
  expect(pool.activeCount).toBe(2);
  pool.release(a);
  pool.release(b);
  expect(pool.activeCount).toBe(0);
});
\`\`\`

The changes are backward-compatible. The \`inUse\` tracking adds negligible overhead since it's a Set lookup (O(1)).

Now let me add the retry logic with exponential backoff for transient failures:

\`\`\`typescript
class RetryPolicy {
  constructor(
    private maxRetries = 3,
    private baseDelayMs = 100,
    private maxDelayMs = 5000
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error;
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (err) {
        lastError = err as Error;
        if (attempt === this.maxRetries) break;
        if (!this.isRetryable(err)) throw err;
        const delay = Math.min(
          this.baseDelayMs * Math.pow(2, attempt),
          this.maxDelayMs
        );
        await new Promise((r) => setTimeout(r, delay));
      }
    }
    throw lastError!;
  }

  private isRetryable(err: unknown): boolean {
    if (err instanceof ConnectionError) return true;
    if (err instanceof TimeoutError) return true;
    return false;
  }
}
\`\`\`

The migration script needs to handle the schema change gracefully:

\`\`\`diff
--- a/migrations/003_add_inuse_tracking.sql
+++ b/migrations/003_add_inuse_tracking.sql
@@ -1,4 +1,12 @@
 -- Migration: Add connection tracking metadata
+ALTER TABLE connections
+  ADD COLUMN in_use BOOLEAN DEFAULT FALSE,
+  ADD COLUMN acquired_at TIMESTAMPTZ,
+  ADD COLUMN acquired_by TEXT;
+
+CREATE INDEX idx_connections_in_use
+  ON connections (in_use)
+  WHERE in_use = TRUE;
+
+-- Backfill: mark all existing connections as available
+UPDATE connections SET in_use = FALSE WHERE in_use IS NULL;
\`\`\`

Finally, the monitoring dashboard query to track pool utilization:

\`\`\`typescript
async function getPoolMetrics(pool: ConnectionPool): Promise<PoolMetrics> {
  const total = pool.connections.length;
  const active = pool.connections.filter((c) => c.state === "active").length;
  const idle = total - active;
  const waiting = pool.waiting.length;

  return {
    total,
    active,
    idle,
    waiting,
    utilization: total > 0 ? active / total : 0,
    avgWaitMs: pool.getAverageWaitTime(),
    p99WaitMs: pool.getPercentileWaitTime(99),
  };
}

// Export as Prometheus metrics
app.get("/metrics", async (req, res) => {
  const metrics = await getPoolMetrics(pool);
  res.type("text/plain").send(\`
# HELP pool_connections_total Total connections in pool
pool_connections_total \${metrics.total}
# HELP pool_connections_active Currently active connections
pool_connections_active \${metrics.active}
# HELP pool_utilization Pool utilization ratio
pool_utilization \${metrics.utilization.toFixed(3)}
# HELP pool_wait_avg_ms Average wait time in ms
pool_wait_avg_ms \${metrics.avgWaitMs.toFixed(1)}
# HELP pool_wait_p99_ms 99th percentile wait time
pool_wait_p99_ms \${metrics.p99WaitMs.toFixed(1)}
  \`.trim());
});
\`\`\`

This gives us full observability into the connection pool behavior in production.
`.trim()

const tokenizeResponse = (text) => {
  const raw = text.split(/(?<=\s)|(?=\s)|(?<=[\`\{\}\(\)\[\];:,.<>+\-=!&|])|(?=[\`\{\}\(\)\[\];:,.<>+\-=!&|])/)
  return raw.filter(t => t.length > 0)
}
const RESPONSE_TOKENS = tokenizeResponse(CODE_RESPONSE)

const generateText = (tokenCount) => {
  const tokens = []
  for (let i = 0; i < tokenCount; i++) tokens.push(RESPONSE_TOKENS[i % RESPONSE_TOKENS.length])
  return tokens
}

// ── Markdown Renderer ──
const escapeHtml = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const renderMarkdown = (text) => {
  const lines = text.split('\n')
  let html = '', inCode = false, lang = ''
  for (const line of lines) {
    if (line.startsWith('```')) {
      if (inCode) { html += '</code></pre>'; inCode = false; lang = '' }
      else { lang = line.slice(3).trim(); html += '<pre class="md-pre"><code>'; inCode = true }
      continue
    }
    if (inCode) {
      const esc = escapeHtml(line)
      if (lang === 'diff') {
        if (line.startsWith('+')) html += `<span class="md-add">${esc}</span>\n`
        else if (line.startsWith('-')) html += `<span class="md-del">${esc}</span>\n`
        else if (line.startsWith('@@')) html += `<span class="md-hunk">${esc}</span>\n`
        else html += esc + '\n'
      } else html += esc + '\n'
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

// ── Constants ──
const SYSTEM_TOKENS = 12000

const PROMPT_PRESETS = [
  { label: 'Quick question', tokens: 500, desc: 'Short prompt, no context' },
  { label: 'Single file edit', tokens: 2000, desc: '~1 file + instructions' },
  { label: 'Multi-file task', tokens: 8000, desc: '3-5 files + conversation' },
  { label: 'Large refactor', tokens: 24000, desc: '10+ files + history' },
  { label: 'Deep exploration', tokens: 64000, desc: 'Many files + long conversation' },
  { label: 'Small C++ project', tokens: 120000, desc: '~50K LOC, CMake, few deps' },
  { label: 'Medium C++ project', tokens: 350000, desc: '~200K LOC, engine subsystem' },
  { label: 'Large C++ / UE5 project', tokens: 800000, desc: '~500K+ LOC, Unreal from source' },
]

const OUTPUT_PRESETS = [
  { label: 'Short answer', tokens: 200, desc: 'Quick explanation or fix' },
  { label: 'Single function', tokens: 500, desc: 'One function + explanation' },
  { label: 'File implementation', tokens: 1500, desc: 'Full file with tests' },
  { label: 'Multi-file change', tokens: 4000, desc: 'Several files + refactor' },
  { label: 'Large generation', tokens: 8000, desc: 'Major feature implementation' },
  { label: 'Max output', tokens: 16000, desc: 'Pushing output limits' },
]

// Tool steps: thinkTokens = short reasoning per call, parallel = grouped concurrent calls
// Thinking is distributed: short per tool call + remainder before final output
const TOOL_PRESETS = [
  { label: 'No tool calls', steps: [], desc: 'Single inference, no agent loop' },
  { label: 'Light agent (3)', steps: [
    { label: 'Read src/db/pool.ts', thinkTokens: 200, decodeTokens: 50, execMs: 200, resultTokens: 800 },
    { label: 'Grep "acquire" in src/', thinkTokens: 150, decodeTokens: 40, execMs: 300, resultTokens: 400 },
    { label: 'Edit src/db/pool.ts', thinkTokens: 400, decodeTokens: 80, execMs: 100, resultTokens: 200 },
  ], desc: 'Read, search, edit' },
  { label: 'Standard agent (6)', steps: [
    { label: 'Read src/db/pool.ts', thinkTokens: 200, decodeTokens: 50, execMs: 200, resultTokens: 800 },
    [  // parallel group
      { label: 'Read src/db/types.ts', thinkTokens: 100, decodeTokens: 40, execMs: 200, resultTokens: 600 },
      { label: 'Grep "Connection" in src/', thinkTokens: 100, decodeTokens: 45, execMs: 300, resultTokens: 500 },
    ],
    { label: 'Edit src/db/pool.ts', thinkTokens: 500, decodeTokens: 80, execMs: 100, resultTokens: 200 },
    { label: 'Run npm test', thinkTokens: 100, decodeTokens: 30, execMs: 3000, resultTokens: 1200 },
    { label: 'Edit src/db/pool.test.ts', thinkTokens: 400, decodeTokens: 90, execMs: 100, resultTokens: 300 },
  ], desc: 'Read, search, edit, test, fix' },
  { label: 'Deep exploration (10)', steps: [
    { label: 'Glob src/**/*.ts', thinkTokens: 100, decodeTokens: 30, execMs: 100, resultTokens: 300 },
    [  // parallel reads
      { label: 'Read src/db/pool.ts', thinkTokens: 80, decodeTokens: 50, execMs: 200, resultTokens: 800 },
      { label: 'Read src/db/types.ts', thinkTokens: 80, decodeTokens: 40, execMs: 200, resultTokens: 600 },
      { label: 'Read src/db/migrations.ts', thinkTokens: 80, decodeTokens: 45, execMs: 200, resultTokens: 900 },
    ],
    { label: 'Grep "acquire" in src/', thinkTokens: 150, decodeTokens: 40, execMs: 300, resultTokens: 500 },
    { label: 'Read src/server/handler.ts', thinkTokens: 150, decodeTokens: 50, execMs: 200, resultTokens: 700 },
    [  // parallel edits
      { label: 'Edit src/db/pool.ts', thinkTokens: 300, decodeTokens: 80, execMs: 100, resultTokens: 200 },
      { label: 'Edit src/db/pool.test.ts', thinkTokens: 300, decodeTokens: 90, execMs: 100, resultTokens: 300 },
    ],
    { label: 'Run npm test', thinkTokens: 100, decodeTokens: 30, execMs: 3000, resultTokens: 1200 },
    { label: 'Read test output', thinkTokens: 100, decodeTokens: 30, execMs: 100, resultTokens: 400 },
  ], desc: 'Full codebase exploration, multi-file edit, test' },
]

// Flatten tool steps: parallel groups become a single step with combined stats
const flattenSteps = (steps) => {
  const flat = []
  for (const s of steps) {
    if (Array.isArray(s)) {
      // Parallel group: think+decode is sum (emitted sequentially), exec is max (concurrent), results sum
      flat.push({
        label: s.map(t => t.label.split(' ')[0]).join(' + '),
        parallel: s.map(t => t.label),
        thinkTokens: s.reduce((sum, t) => sum + t.thinkTokens, 0),
        decodeTokens: s.reduce((sum, t) => sum + t.decodeTokens, 0),
        execMs: Math.max(...s.map(t => t.execMs)),
        resultTokens: s.reduce((sum, t) => sum + t.resultTokens, 0),
      })
    } else {
      flat.push(s)
    }
  }
  return flat
}

const TIER_COLORS = { S: '#fbbf24', A: '#34d399', B: '#60a5fa', C: '#a78bfa', D: '#f87171', F: '#6b7280' }
const formatTime = (s) => s < 1 ? `${Math.round(s * 1000)}ms` : s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`

// ── TokenStream Component ──
const TokenStream = ({ model, tokens, isRunning, isReset, tokenCount, promptTokens, toolSteps, timeScale, cumulative, onComplete, streamIndex }) => {
  const [displayedTokens, setDisplayedTokens] = useState([])
  const [phase, setPhase] = useState('idle')
  const [thinkingTokensGenerated, setThinkingTokensGenerated] = useState(0)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [prefillElapsed, setPrefillElapsed] = useState(0)
  const [compactElapsed, setCompactElapsed] = useState(0)
  const [prefillSize, setPrefillSize] = useState(0)
  const [toolResultTokens, setToolResultTokens] = useState(0)
  const [currentToolIdx, setCurrentToolIdx] = useState(-1)
  const [toolLog, setToolLog] = useState([])
  const intervalRef = useRef(null)
  const timerRef = useRef(null)
  const timeoutRef = useRef(null)
  const startTimeRef = useRef(null)
  const decodeStartRef = useRef(null)
  const totalIndexRef = useRef(0)
  const contentRef = useRef(null)
  const rafRef = useRef(null)
  const hasStartedRef = useRef(false)
  const toolResultsRef = useRef(0)

  const thinkingBudget = model.thinkingBudget
  const effectiveOutput = Math.round(tokenCount * (model.outputMul || 1))
  const totalTokens = thinkingBudget + effectiveOutput

  const scrollToBottom = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(() => {
      if (contentRef.current) contentRef.current.scrollTop = contentRef.current.scrollHeight
    })
  }, [])

  const clearTimers = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    if (timerRef.current) clearInterval(timerRef.current)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
  }, [])

  useEffect(() => () => clearTimers(), [clearTimers])

  useEffect(() => {
    if (isReset) {
      clearTimers()
      setDisplayedTokens([]); setPhase('idle'); setThinkingTokensGenerated(0)
      setElapsedTime(0); setPrefillElapsed(0); setCompactElapsed(0); setPrefillSize(0)
      setToolResultTokens(0); setCurrentToolIdx(-1); setToolLog([])
      totalIndexRef.current = 0; hasStartedRef.current = false
      startTimeRef.current = null; decodeStartRef.current = null; toolResultsRef.current = 0
      return
    }

    if (isRunning && !hasStartedRef.current) {
      hasStartedRef.current = true
      startTimeRef.current = Date.now()

      // Elapsed time shows simulated time (real time * timeScale)
      timerRef.current = setInterval(() => {
        if (startTimeRef.current) setElapsedTime(((Date.now() - startTimeRef.current) * ts / 1000).toFixed(1))
      }, 100)

      const maxCtx = parseInt(model.maxCtx) * 1000
      const totalToolResult = toolSteps.reduce((s, t) => s + t.resultTokens + t.decodeTokens, 0)
      const fullUsed = SYSTEM_TOKENS + promptTokens + totalToolResult + thinkingBudget + effectiveOutput
      const needsCompact = fullUsed / maxCtx > 0.8
      const compactTokens = needsCompact ? Math.max(0, fullUsed - maxCtx * 0.6) : 0
      const compactMs = needsCompact ? (compactTokens / (model.prefillRate * 0.5)) * 1000 : 0

      // Animate a progress bar over durationMs, then call next()
      const ts = timeScale || 1
      const animateBar = (setter, durationMs, next) => {
        const scaledMs = durationMs / ts
        const start = Date.now()
        const tick = () => {
          const p = Math.min((Date.now() - start) / scaledMs, 1)
          setter(p)
          if (p < 1) timeoutRef.current = setTimeout(tick, 16)
          else next()
        }
        tick()
      }

      // Thinking: distributed across tool steps + final output
      // Total thinking per tool step is step.thinkTokens (short reasoning per call)
      // Remaining thinking budget goes to the final output synthesis
      const totalStepThinking = toolSteps.reduce((s, t) => s + (t.thinkTokens || 0), 0)
      const finalThinking = thinkingBudget > 0
        ? Math.max(0, thinkingBudget - (toolSteps.length > 0 ? totalStepThinking : 0))
        : 0

      // Simulate thinking for N tokens, then call next()
      const doThinking = (thinkCount, next) => {
        if (thinkCount <= 0) { next(); return }
        setPhase('thinking')
        setThinkingTokensGenerated(0)
        const thinkMs = (thinkCount / model.tokPerSec) * 1000 / ts
        const thinkStart = Date.now()
        const tickThink = () => {
          const elapsed = Date.now() - thinkStart
          setThinkingTokensGenerated(Math.min(Math.floor((elapsed / thinkMs) * thinkCount), thinkCount))
          if (elapsed < thinkMs) timeoutRef.current = setTimeout(tickThink, 50)
          else { setThinkingTokensGenerated(thinkCount); next() }
        }
        tickThink()
      }

      // Stream N tokens of visible output, then call next()
      // Uses time-based batching: at high timeScales, emits multiple tokens per tick
      const streamTokens = (target, onDone) => {
        const tokPerMs = model.tokPerSec * ts / 1000
        const streamStart = Date.now()
        const baseIndex = totalIndexRef.current
        const tickInterval = Math.max(16, Math.floor(1000 / model.tokPerSec / ts))
        intervalRef.current = setInterval(() => {
          const elapsed = Date.now() - streamStart
          const shouldBe = Math.min(baseIndex + Math.floor(elapsed * tokPerMs), target, effectiveOutput)
          if (totalIndexRef.current < shouldBe) {
            const batch = []
            while (totalIndexRef.current < shouldBe) {
              batch.push(tokens[totalIndexRef.current])
              totalIndexRef.current++
            }
            setDisplayedTokens(prev => [...prev, ...batch])
          }
          if (totalIndexRef.current >= target || totalIndexRef.current >= effectiveOutput) {
            clearInterval(intervalRef.current)
            onDone()
          }
        }, tickInterval)
      }

      const streamChunk = (count, next) => {
        if (count <= 0) { next(); return }
        setPhase('streaming')
        streamTokens(totalIndexRef.current + count, next)
      }

      // Prefill then call next()
      const startPrefill = (contextSize, next) => {
        setPhase('prefill')
        setPrefillElapsed(0)
        setPrefillSize(contextSize)
        const prefillMs = (contextSize / model.prefillRate) * 1000
        animateBar(setPrefillElapsed, prefillMs, next)
      }

      // Distribute output across tool steps: ~15% per step, remainder at end
      const numSteps = toolSteps.length
      const perStepOutput = numSteps > 0 ? Math.floor(effectiveOutput * 0.15 / numSteps) : 0
      const finalOutput = effectiveOutput - (perStepOutput * numSteps)

      // Track tokens added in previous step for incremental prefill
      let lastStepTokens = 0

      // Run tool step i, then continue
      const runToolStep = (i) => {
        if (i >= numSteps) {
          // All tool calls done — final prefill + thinking + remaining output
          const doFinal = () => {
            if (!decodeStartRef.current) decodeStartRef.current = Date.now()
            const thinkAmount = numSteps === 0 ? thinkingBudget : finalThinking
            doThinking(thinkAmount, () => {
              streamChunk(finalOutput, () => {
                clearInterval(timerRef.current)
                if (startTimeRef.current) setElapsedTime(((Date.now() - startTimeRef.current) * ts / 1000).toFixed(1))
                setPhase('complete'); onComplete(streamIndex)
              })
            })
          }
          if (lastStepTokens > 0) startPrefill(lastStepTokens, doFinal)
          else doFinal()
          return
        }

        const step = toolSteps[i]
        setCurrentToolIdx(i)

        // First call: prefill user prompt (system tokens already cached in session).
        // Subsequent: only new tokens since last call (incremental KV).
        const prefillCtx = i === 0
          ? promptTokens
          : lastStepTokens

        // Prefill → think → tool-call decode → tool exec → stream output chunk → next
        startPrefill(prefillCtx, () => {
          const stepThink = thinkingBudget > 0 ? (step.thinkTokens || 0) : 0
          doThinking(stepThink, () => {
            // Stream a chunk of output (analysis/explanation before the tool call)
            if (!decodeStartRef.current) decodeStartRef.current = Date.now()
            streamChunk(perStepOutput, () => {
              setPhase('tool-decode')
              const decodeMs = (step.decodeTokens / model.tokPerSec) * 1000 / ts
              timeoutRef.current = setTimeout(() => {
                setPhase('tool-exec')
                const logEntries = step.parallel
                  ? step.parallel.map(l => ({ label: l, tokens: 0 }))
                  : [{ label: step.label, tokens: step.resultTokens }]
                if (step.parallel) {
                  logEntries[logEntries.length - 1].tokens = step.resultTokens
                }
                setToolLog(prev => [...prev, ...logEntries.map((e, idx) => ({
                  ...e,
                  tokens: step.parallel ? Math.round(step.resultTokens / step.parallel.length) : step.resultTokens,
                  parallel: step.parallel && idx > 0,
                }))])
                timeoutRef.current = setTimeout(() => {
                  const added = step.resultTokens + step.decodeTokens + stepThink + perStepOutput
                  toolResultsRef.current += added
                  lastStepTokens = added
                  setToolResultTokens(toolResultsRef.current)
                  runToolStep(i + 1)
                }, step.execMs / ts)
              }, decodeMs)
            })
          })
        })
      }

      const beginToolLoop = () => {
        if (toolSteps.length === 0) {
          // System tokens already cached, only prefill user prompt
          startPrefill(promptTokens, startFinalDecode)
        } else {
          runToolStep(0)
        }
      }

      if (needsCompact) {
        setPhase('compacting')
        animateBar(setCompactElapsed, compactMs, beginToolLoop)
      } else {
        beginToolLoop()
      }
    }

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isRunning, isReset, model, promptTokens, thinkingBudget, effectiveOutput, totalTokens, tokens, toolSteps, timeScale, onComplete, streamIndex, clearTimers])

  useEffect(() => {
    if (displayedTokens.length > 0 || toolLog.length > 0) scrollToBottom()
  }, [displayedTokens, toolLog, scrollToBottom])

  const totalProgress = totalTokens > 0 ? ((thinkingTokensGenerated + displayedTokens.length) / totalTokens) * 100 : 0
  // "Actual" tok/s should reflect simulated time (real wall time * timeScale)
  const ts = timeScale || 1
  const decodeElapsed = decodeStartRef.current ? (Date.now() - decodeStartRef.current) * ts / 1000 : 0
  const rate = displayedTokens.length > 0 && decodeElapsed > 0 ? (displayedTokens.length / decodeElapsed).toFixed(1) : null

  const phaseLabels = { idle: 'Ready', compacting: 'Compacting', prefill: 'Prefill', 'tool-decode': 'Tool Call', 'tool-exec': 'Executing', thinking: 'Thinking', streaming: 'Streaming', complete: 'Done' }
  const statusLabel = phaseLabels[phase]
  const isActive = phase !== 'idle' && phase !== 'complete'
  const cardClass = ['stream-card', isActive && 'is-running', phase === 'complete' && 'is-complete'].filter(Boolean).join(' ')
  const thinkingLabel = model.thinking ? `${model.thinkingBudget.toLocaleString()}` : 'Off'

  // Context bar — dynamic based on accumulated tool results
  const maxCtxTokens = parseInt(model.maxCtx) * 1000
  const usedTokens = SYSTEM_TOKENS + promptTokens + toolResultTokens + thinkingBudget + effectiveOutput
  const overflows = usedTokens > maxCtxTokens
  const pct = (n) => Math.min((n / maxCtxTokens) * 100, 100)
  const systemPct = pct(SYSTEM_TOKENS)
  const promptPct = Math.min(pct(promptTokens), 100 - systemPct)
  const toolPct = Math.min(pct(toolResultTokens), Math.max(0, 100 - systemPct - promptPct))
  const thinkPct = Math.min(pct(thinkingBudget), Math.max(0, 100 - systemPct - promptPct - toolPct))
  const outPct = Math.min(pct(effectiveOutput), Math.max(0, 100 - systemPct - promptPct - toolPct - thinkPct))

  // VRAM breakdown
  const hwTotal = HW_MEM[model.hardware] ?? 32
  // KV cache is allocated for the full context window, not just tokens in use
  const kvGB = (model.kvPerTokKB * maxCtxTokens) / 1024 / 1024
  const totalVram = model.weightGB + kvGB
  const weightDeg = (model.weightGB / hwTotal) * 360
  const kvDeg = (kvGB / hwTotal) * 360
  const vramOverflow = totalVram > hwTotal

  const markdownHtml = displayedTokens.length > 0 ? renderMarkdown(displayedTokens.join('')) : ''
  const currentStep = currentToolIdx >= 0 && currentToolIdx < toolSteps.length ? toolSteps[currentToolIdx] : null

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
        <span className="hw-spec">{model.maxCtx} ctx</span>
        <span className="hw-spec">{model.quality}</span>
      </div>

      <div className="ctx-bar-wrapper">
        <div className="ctx-bar-labels">
          <span>Context: {usedTokens.toLocaleString()} / {maxCtxTokens.toLocaleString()}{overflows ? ' — overflow!' : ''}</span>
        </div>
        <div className={`ctx-bar ${overflows ? 'ctx-overflow' : ''}`}>
          <span className="ctx-bar-tick" /><span className="ctx-bar-tick-label">compact</span>
          <div className="ctx-bar-fill-area">
            <div className="ctx-seg ctx-system" style={{ width: `${systemPct}%` }} />
            <div className="ctx-seg ctx-prompt" style={{ width: `${promptPct}%` }} />
            {toolResultTokens > 0 && <div className="ctx-seg ctx-tool" style={{ width: `${toolPct}%` }} />}
            {thinkingBudget > 0 && <div className="ctx-seg ctx-thinking" style={{ width: `${thinkPct}%` }} />}
            <div className="ctx-seg ctx-output" style={{ width: `${outPct}%` }} />
          </div>
        </div>
        <div className="ctx-bar-legend">
          <span className="ctx-legend-item"><span className="ctx-swatch ctx-system" />System</span>
          <span className="ctx-legend-item"><span className="ctx-swatch ctx-prompt" />Prompt</span>
          {(toolSteps.length > 0 || toolResultTokens > 0) && <span className="ctx-legend-item"><span className="ctx-swatch ctx-tool" />Tools</span>}
          {thinkingBudget > 0 && <span className="ctx-legend-item"><span className="ctx-swatch ctx-thinking" />Thinking</span>}
          <span className="ctx-legend-item"><span className="ctx-swatch ctx-output" />Output</span>
          <span className="ctx-legend-item ctx-free">{Math.max(0, maxCtxTokens - usedTokens).toLocaleString()} free</span>
        </div>
      </div>

      {hwTotal === 0 && model.costIn != null && (() => {
        const totalInput = SYSTEM_TOKENS + promptTokens + toolResultTokens + toolSteps.reduce((s, t) => s + (t.resultTokens || 0), 0)
        const outputTokens = effectiveOutput + thinkingBudget
        const totalToolDecodeTokens = toolSteps.reduce((s, t) => s + (t.decodeTokens || 0) + (t.thinkTokens || 0), 0)
        // Tiered pricing based on context window size, not input tokens
        // (e.g. GPT-5.4: $2.50/$15 at 272K window, $5/$22.50 at 1M window)
        const threshold = model.costInThreshold || Infinity
        const maxCtxTokens = parseInt(model.maxCtx) * 1000
        const useHighTier = maxCtxTokens > threshold
        const inRate = useHighTier && model.costInHigh ? model.costInHigh : model.costIn
        const outRate = useHighTier && model.costOutHigh ? model.costOutHigh : model.costOut
        const costInput = (totalInput / 1e6) * inRate
        const costOutput = ((outputTokens + totalToolDecodeTokens) / 1e6) * outRate
        const costTotal = costInput + costOutput
        const rateLabel = `$${inRate}/$${outRate} per 1M`
        return (
          <div className="cost-row">
            <div className="cost-total">${costTotal < 0.01 ? costTotal.toFixed(4) : costTotal.toFixed(2)}</div>
            <div className="cost-detail">
              <span className="cost-item">{rateLabel}</span>
              <span className="cost-item cost-breakdown">in ${costInput.toFixed(3)} + out ${costOutput.toFixed(3)}</span>
            </div>
          </div>
        )
      })()}

      {hwTotal > 0 && (
        <div className="vram-row">
          <div
            className={`vram-donut ${vramOverflow ? 'vram-overflow' : ''}`}
            style={{ background: `conic-gradient(#60a5fa 0deg ${weightDeg}deg, #f59e0b ${weightDeg}deg ${weightDeg + kvDeg}deg, var(--border) ${weightDeg + kvDeg}deg 360deg)` }}
          ><div className="vram-donut-hole" /></div>
          <div className="vram-text">
            <span className="vram-total">{totalVram.toFixed(1)} / {hwTotal} GB{vramOverflow ? ' !' : ''}</span>
            <span className="vram-detail">
              <span className="vram-item"><span className="vram-dot" style={{ background: '#60a5fa' }} />Weights {model.weightGB.toFixed(1)}</span>
              <span className="vram-item"><span className="vram-dot" style={{ background: '#f59e0b' }} />KV {kvGB.toFixed(1)}</span>
              <span className="vram-item vram-free">{Math.max(0, hwTotal - totalVram).toFixed(1)} free</span>
            </span>
          </div>
        </div>
      )}

      <div className="stats-row">
        <div className="stat"><span className="stat-label">Output{model.outputMul > 1 ? ` (${model.outputMul}x)` : ''}</span><span className="stat-value">{displayedTokens.length.toLocaleString()} / {effectiveOutput.toLocaleString()}</span></div>
        {toolSteps.length > 0 && <div className="stat"><span className="stat-label">Tool calls</span><span className="stat-value">{Math.min(toolLog.length, toolSteps.length)} / {toolSteps.length}</span></div>}
        <div className="stat"><span className="stat-label">Thinking</span><span className={`stat-value ${!model.thinking ? 'stat-dim' : ''}`}>{thinkingLabel}</span></div>
        <div className="stat"><span className="stat-label">Time</span><span className="stat-value">{elapsedTime}s</span></div>
        {rate && <div className="stat"><span className="stat-label">Actual</span><span className="stat-value">{rate} tok/s</span></div>}
        {cumulative && cumulative.loops > 0 && <div className="stat"><span className="stat-label">Total ({cumulative.loops} runs)</span><span className="stat-value">{(cumulative.input / 1000).toFixed(0)}K in / {(cumulative.output / 1000).toFixed(0)}K out</span></div>}
      </div>

      <div className="progress-track"><div className="progress-fill" style={{ width: `${totalProgress}%`, background: model.color }} /></div>

      {phase === 'compacting' && (
        <div className="compact-banner"><span className="compact-spinner" /><div className="compact-detail"><span>Compacting context</span><div className="compact-bar"><div className="compact-bar-fill" style={{ width: `${compactElapsed * 100}%` }} /></div></div></div>
      )}

      {phase === 'prefill' && (
        <div className="prefill-banner"><div className="prefill-bar"><div className="prefill-bar-fill" style={{ width: `${prefillElapsed * 100}%`, background: model.color }} /></div><span className="prefill-label">{toolResultTokens > 0 ? 'Incremental prefill' : 'Prefilling'} {prefillSize.toLocaleString()} tokens @ {model.prefillRate} tok/s — {formatTime(prefillSize / model.prefillRate)}</span></div>
      )}

      {(phase === 'tool-decode' || phase === 'tool-exec') && currentStep && (
        <div className="tool-banner">
          <span className={`tool-icon ${phase === 'tool-exec' ? 'tool-running' : ''}`}>{phase === 'tool-exec' ? '⚙' : '→'}</span>
          <span className="tool-label">{currentStep.parallel ? currentStep.parallel.join(', ') : currentStep.label}</span>
          {phase === 'tool-exec' && <span className="tool-exec-badge">{currentStep.parallel ? 'parallel' : 'executing'}</span>}
        </div>
      )}

      {phase === 'thinking' && (
        <div className="thinking-banner"><span className="thinking-spinner" /><div className="thinking-detail"><span>Thinking</span><span className="thinking-count">{thinkingTokensGenerated.toLocaleString()} tokens</span></div></div>
      )}

      <div ref={contentRef} className="stream-content">
        {displayedTokens.length === 0 && toolLog.length === 0 && phase === 'idle' && <div className="stream-empty">Waiting to start</div>}
        {displayedTokens.length === 0 && toolLog.length === 0 && (phase === 'prefill' || phase === 'compacting') && <div className="stream-empty">Processing...</div>}
        {displayedTokens.length === 0 && toolLog.length === 0 && phase === 'thinking' && <div className="stream-empty">Reasoning...</div>}
        {toolLog.length > 0 && (
          <div className="tool-log">
            {toolLog.map((entry, i) => (
              <div key={i} className={`tool-log-entry ${entry.parallel ? 'tool-log-parallel' : ''}`}>
                <span className="tool-log-icon">{entry.parallel ? '├' : '→'}</span>
                <span className="tool-log-label">{entry.label}</span>
                <span className="tool-log-tokens">+{entry.tokens.toLocaleString()} tok</span>
              </div>
            ))}
            {phase !== 'complete' && displayedTokens.length === 0 && currentToolIdx >= toolLog.length && (
              <div className="tool-log-entry tool-log-final"><span className="tool-log-icon">←</span><span className="tool-log-label">Generating response...</span></div>
            )}
          </div>
        )}
        {markdownHtml && <div className="md-content" dangerouslySetInnerHTML={{ __html: markdownHtml }} />}
        {phase === 'streaming' && <span className="cursor" />}
      </div>
    </div>
  )
}

// ── About Page ──
function AboutPage() {
  return (
    <div className="about-page">
      <h2>About this simulator</h2>
      <p>This tool visualizes what actually happens when you use an AI coding agent — the full pipeline from prompt processing to tool execution to token generation. It uses real benchmark data from <a href="https://github.com/gisenberg/local-model-eval" target="_blank" rel="noopener noreferrer">local-model-eval</a>, a systematic evaluation of local LLM inference across consumer and workstation hardware.</p>

      <h3>What it simulates</h3>
      <p>Each card runs through the real lifecycle of an inference request:</p>
      <ul>
        <li><strong>Context compaction</strong> — When conversation history exceeds 80% of the model's context window, older turns are summarized to make room. This adds latency before anything else can happen.</li>
        <li><strong>Prefill</strong> — The model processes all input tokens (system prompt, conversation history, file contents). This is bandwidth-bound and scales linearly with prompt size. Subsequent tool calls use incremental prefill — only new tokens need processing because the KV cache persists.</li>
        <li><strong>Thinking</strong> — Models with extended reasoning generate hidden chain-of-thought tokens before producing visible output. These are generated at the decode rate and consume context window space.</li>
        <li><strong>Tool calls</strong> — Coding agents don't generate one response. They read files, grep code, edit, run tests — each requiring a new prefill + decode cycle. The simulator models parallel tool grouping, incremental KV cache reuse, and context growth as tool results accumulate.</li>
        <li><strong>Decode</strong> — Visible output token generation, streamed at the model's measured throughput.</li>
      </ul>

      <h3>Hardware platforms</h3>
      <ul>
        <li><strong>RTX 5090</strong> (32 GB GDDR7, 1,792 GB/s) — Consumer GPU with the highest bandwidth. Best for MoE models where only a fraction of parameters are active per token.</li>
        <li><strong>MacBook Pro M4 Max</strong> (36 GB unified, 410 GB/s, ~30 GB Metal ceiling) — Portable inference. Lower bandwidth but unified memory avoids CPU-GPU transfers. TurboQuant KV is slower on Metal due to dequant compute overhead.</li>
        <li><strong>DGX Spark GB10</strong> (128 GB unified, 273 GB/s) — Massive memory lets you run 100B+ parameter models. Bandwidth-limited, not capacity-limited.</li>
        <li><strong>Cloud APIs</strong> (Anthropic, Google, OpenAI) — No local memory constraints. Performance data from <a href="https://artificialanalysis.ai" target="_blank" rel="noopener noreferrer">Artificial Analysis</a>.</li>
      </ul>

      <h3>Where the data comes from</h3>
      <p>Local model benchmarks use a standardized 3-task coding evaluation suite run at temperature 0 with zero code fixes — the model's output is extracted and tested as-is:</p>
      <ul>
        <li><strong>Expression Evaluator</strong> — Recursive descent parser with operator precedence (5 tests)</li>
        <li><strong>A* Pathfinding</strong> — Graph algorithms with heap usage and edge cases (6 tests)</li>
        <li><strong>LRU Cache with TTL</strong> — Doubly-linked list + hash map with time mocking (6 tests)</li>
      </ul>
      <p>VRAM measurements are captured from actual inference sessions. Throughput (tok/s) and TTFT are measured via streaming API calls. Prefill rates are derived from TTFT at known prompt sizes or from context-size deltas in long-context experiments.</p>
      <p>KV cache costs per token are computed from model architecture (attention layer count, KV head count, head dimension) and verified against measured VRAM at multiple context sizes. Models using TurboQuant KV quantization show ~3.6-5.1x compression vs f16.</p>

      <h3>Key findings</h3>
      <ul>
        <li><strong>Prefill dominates real-world latency</strong> — At 32K context on the Spark, prefill is 67% of total time. At 128K, it's 89%.</li>
        <li><strong>Tool call loops compound the cost</strong> — A 6-step agent workflow means 6 prefill+decode cycles. Even with KV cache reuse, context grows with each step.</li>
        <li><strong>MoE architectures win on throughput</strong> — Gemma 26B-A4B (4B active) runs at 139 tok/s vs Gemma 31B dense at 46 tok/s. Less bandwidth per token = faster decode.</li>
        <li><strong>Architecture determines KV cost</strong> — Mamba hybrids (16 KB/tok) vs dense transformers (256 KB/tok) is a 16x difference in context memory cost.</li>
        <li><strong>System prompts eat context</strong> — Coding agents consume ~12K tokens on system prompt + tool schemas before you type anything.</li>
      </ul>

      <p className="about-footer">Built with data from <a href="https://github.com/gisenberg/local-model-eval" target="_blank" rel="noopener noreferrer">gisenberg/local-model-eval</a>. Cloud model speeds from <a href="https://artificialanalysis.ai" target="_blank" rel="noopener noreferrer">Artificial Analysis</a>.</p>
    </div>
  )
}

// ── Hash Router ──
const getHashRoute = () => {
  const hash = window.location.hash.slice(1) // remove #
  if (hash === 'about') return { page: 'about', experiment: null }
  const exp = EXPERIMENTS.find(e => e.id === hash)
  return { page: 'sim', experiment: exp ? hash : 'cloud-all' }
}

// ── App ──
function App() {
  const [route, setRoute] = useState(getHashRoute)
  const [isRunning, setIsRunning] = useState(false)
  const [isReset, setIsReset] = useState(false)
  const [tokenCount, setTokenCount] = useState(4000)
  const [promptTokens, setPromptTokens] = useState(24000)
  const [toolPresetIdx, setToolPresetIdx] = useState(2)
  const [timeScale, setTimeScale] = useState(1)
  const [loopEnabled, setLoopEnabled] = useState(false)
  const [completedStreams, setCompletedStreams] = useState(new Set())
  const [cumulatives, setCumulatives] = useState({})
  const [accumulatedContext, setAccumulatedContext] = useState(0) // carries forward across loops

  useEffect(() => {
    const onHash = () => setRoute(getHashRoute())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const navigate = (hash) => { window.location.hash = hash }

  const activeExperiment = route.experiment || 'cloud-all'
  const experiment = EXPERIMENTS.find(e => e.id === activeExperiment)
  const selectedModels = experiment.models.map(id => MODELS.find(m => m.id === id))
  const maxThinkingBudget = Math.max(...selectedModels.map(m => m.thinkingBudget))
  const maxOutputMul = Math.max(...selectedModels.map(m => m.outputMul || 1))
  const maxTotalTokens = Math.round(tokenCount * maxOutputMul) + maxThinkingBudget
  const toolSteps = useMemo(() => flattenSteps(TOOL_PRESETS[toolPresetIdx].steps), [toolPresetIdx])

  const handleExperimentChange = (id) => {
    setIsRunning(false); setIsReset(true); setCompletedStreams(new Set()); setCumulatives({}); setAccumulatedContext(0)
    navigate(id)
    setTimeout(() => setIsReset(false), 100)
  }
  const handleComplete = useCallback((index) => {
    setCompletedStreams(prev => { const next = new Set(prev); next.add(index); return next })
  }, [])
  const handleStart = () => { setIsReset(false); setCompletedStreams(new Set()); setAccumulatedContext(0); setIsRunning(true) }
  const handleReset = () => { setIsRunning(false); setIsReset(true); setCompletedStreams(new Set()); setCumulatives({}); setAccumulatedContext(0); setTimeout(() => setIsReset(false), 100) }

  const tokens = useMemo(() => generateText(maxTotalTokens), [maxTotalTokens])
  const allComplete = completedStreams.size >= experiment.models.length
  const controlsDisabled = isRunning && !allComplete

  // Loop: when all complete, accumulate context from this turn, apply compaction, restart
  useEffect(() => {
    if (allComplete && loopEnabled && isRunning) {
      // Tokens generated this turn that become part of next turn's context
      const toolResultTotal = toolSteps.reduce((s, t) => s + (t.resultTokens || 0) + (t.decodeTokens || 0) + (t.thinkTokens || 0), 0)
      // Use the max output across models for context growth (conservative)
      const maxOutput = Math.max(...selectedModels.map(m => Math.round(tokenCount * (m.outputMul || 1)) + (m.thinkingBudget || 0)))
      const turnTokens = maxOutput + toolResultTotal

      setCumulatives(prev => {
        const next = { ...prev }
        selectedModels.forEach((m, i) => {
          const inputTok = SYSTEM_TOKENS + promptTokens + accumulatedContext + toolResultTotal
          const outputTok = Math.round(tokenCount * (m.outputMul || 1)) + (m.thinkingBudget || 0)
          const key = m.id + '-' + i
          if (!next[key]) next[key] = { loops: 0, input: 0, output: 0 }
          next[key].loops++
          next[key].input += inputTok
          next[key].output += outputTok
        })
        return next
      })

      // Accumulate context: previous turns + this turn's output + tool results
      // Check if compaction would trigger on next turn
      const maxCtxAll = Math.max(...selectedModels.map(m => parseInt(m.maxCtx) * 1000))
      const nextContext = SYSTEM_TOKENS + promptTokens + accumulatedContext + turnTokens
      const wouldCompact = nextContext / maxCtxAll > 0.8

      setAccumulatedContext(prev => {
        const newCtx = prev + turnTokens
        // Compaction reduces accumulated history by 70-90% (keep 10-30%)
        if (wouldCompact) return Math.round(newCtx * 0.2)
        return newCtx
      })

      const timer = setTimeout(() => {
        setIsReset(true)
        setTimeout(() => { setIsReset(false); setCompletedStreams(new Set()); setIsRunning(true) }, 150)
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [allComplete, loopEnabled, isRunning, selectedModels, promptTokens, tokenCount, toolSteps, accumulatedContext])

  const cols = experiment.columns
  const rows = []
  for (let i = 0; i < selectedModels.length; i += cols) rows.push({ start: i, models: selectedModels.slice(i, i + cols) })

  return (
    <div className="app-layout">
      <nav className="experiment-nav">
        <div className="nav-title">Experiments</div>
        {EXPERIMENT_CATEGORIES.map(cat => (
          <div key={cat.id} className="nav-category">
            <div className="nav-cat-label">{cat.label}</div>
            {EXPERIMENTS.filter(e => e.category === cat.id).map(exp => (
              <button key={exp.id} className={`nav-item ${activeExperiment === exp.id && route.page === 'sim' ? 'active' : ''}`} onClick={() => handleExperimentChange(exp.id)}>
                <span className="nav-item-name">{exp.name}</span>
                <span className="nav-item-count">{exp.models.length}</span>
              </button>
            ))}
          </div>
        ))}
        <div className="nav-footer">
          <button className={`nav-about ${route.page === 'about' ? 'active' : ''}`} onClick={() => navigate('about')}>About this simulator</button>
        </div>
      </nav>

      <main className="app-main">
        {route.page === 'about' ? <AboutPage /> : (
          <>
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
              <div className="control-group">
                <label>Agent tool calls<span>{toolSteps.length}</span></label>
                <select value={toolPresetIdx} onChange={(e) => setToolPresetIdx(parseInt(e.target.value))} disabled={controlsDisabled} className="prompt-select">
                  {TOOL_PRESETS.map((p, i) => <option key={i} value={i}>{p.label}</option>)}
                </select>
                <div className="prompt-desc">{TOOL_PRESETS[toolPresetIdx].desc}</div>
              </div>
            </div>

            <div className="action-bar">
              <button onClick={handleStart} disabled={controlsDisabled} className="btn-start">{controlsDisabled ? 'Running...' : 'Start'}</button>
              <button onClick={handleReset} className="btn-reset">Stop</button>
              <label className="loop-toggle">
                <input type="checkbox" checked={loopEnabled} onChange={(e) => setLoopEnabled(e.target.checked)} />
                Loop
              </label>
              <div className="time-scale">
                <span className="time-scale-label">{timeScale}x</span>
                <input type="range" min="0" max="4" step="1" value={[1,2,5,10,20].indexOf(timeScale)} onChange={(e) => setTimeScale([1,2,5,10,20][e.target.value])} className="time-scale-slider" />
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
                    promptTokens={promptTokens + accumulatedContext}
                    toolSteps={toolSteps}
                    timeScale={timeScale}
                    cumulative={cumulatives[model.id + '-' + (start + i)]}
                    onComplete={handleComplete}
                    streamIndex={start + i}
                  />
                ))}
              </div>
            ))}
          </>
        )}
      </main>
    </div>
  )
}

export default App

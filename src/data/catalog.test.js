import { describe, expect, it } from 'vitest'
import { EXPERIMENTS, HARDWARE, MODELS, MODEL_BY_ID } from './catalog.js'

describe('model catalog', () => {
  it('uses unique model and experiment identifiers', () => {
    expect(new Set(MODELS.map((model) => model.id)).size).toBe(MODELS.length)
    expect(new Set(EXPERIMENTS.map((experiment) => experiment.id)).size).toBe(EXPERIMENTS.length)
  })

  it('keeps every experiment model reference valid', () => {
    for (const experiment of EXPERIMENTS) {
      for (const modelId of experiment.models) expect(MODEL_BY_ID[modelId]).toBeDefined()
    }
  })

  it('provides balanced cloud and local comparison presets', () => {
    const comparisons = EXPERIMENTS.filter((experiment) => experiment.category === 'cloud-vs-local')
    expect(comparisons).toHaveLength(4)
    for (const comparison of comparisons) {
      const kinds = comparison.models.map((modelId) => HARDWARE[MODEL_BY_ID[modelId].hardware].kind)
      expect(kinds.filter((kind) => kind === 'cloud')).toHaveLength(3)
      expect(kinds.filter((kind) => kind === 'local')).toHaveLength(3)
    }
  })

  it('preserves measured RTX Pro 6000 speculative decode rates', () => {
    const model = MODEL_BY_ID['pro6000-qwen36-27b-fp8']
    const rates = Object.fromEntries(model.profiles.map((profile) => [profile.id, profile.decodeRate]))

    expect(rates).toMatchObject({
      baseline: 48.3,
      'mtp-1': 67.5,
      'mtp-2': 93.3,
      'dflash-k7': 170.8,
      'dflash-k11': 185.8,
      'dflash-k15': 197.5,
    })
  })

  it('keeps Claude Fable 5 in cloud and cloud-local views', () => {
    expect(MODEL_BY_ID['anthropic-fable-5']?.name).toBe('Claude Fable 5')
    const comparisons = EXPERIMENTS.filter((experiment) => experiment.category === 'cloud-vs-local')
    expect(comparisons.every((experiment) => experiment.models.includes('anthropic-fable-5'))).toBe(true)
  })

  it('represents every local platform in the best-local preset', () => {
    const bestLocal = EXPERIMENTS.find((experiment) => experiment.id === 'local-best')
    const represented = new Set(bestLocal.models.map((modelId) => MODEL_BY_ID[modelId].hardware))
    const localHardware = Object.entries(HARDWARE)
      .filter(([, hardware]) => hardware.kind === 'local')
      .map(([name]) => name)

    expect(represented).toEqual(new Set(localHardware))
  })
})

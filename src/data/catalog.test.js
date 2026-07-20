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
    expect(comparisons).toHaveLength(3)
    for (const comparison of comparisons) {
      const kinds = comparison.models.map((modelId) => HARDWARE[MODEL_BY_ID[modelId].hardware].kind)
      expect(kinds.filter((kind) => kind === 'cloud')).toHaveLength(3)
      expect(kinds.filter((kind) => kind === 'local')).toHaveLength(3)
    }
  })
})

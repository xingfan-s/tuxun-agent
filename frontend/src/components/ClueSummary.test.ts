import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { describe, expect, it } from 'vitest'
import { useTaskStore } from '@/stores/task'
import ClueSummary from './ClueSummary.vue'

describe('ClueSummary', () => {
  it('removes the full star-rating suffix without leaving empty parentheses', () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useTaskStore()
    store.steps = [{
      step: 4,
      type: 'clue_extraction',
      label: '线索提取',
      status: 'done',
      data: { top_clues: ['植被类型：椰子树（★★★）'] },
      elapsed_ms: 1,
    }]

    const wrapper = mount(ClueSummary, {
      global: {
        plugins: [pinia],
        stubs: {
          'el-icon': { template: '<span><slot /></span>' },
          'el-tag': { template: '<span><slot /></span>' },
        },
      },
    })

    expect(wrapper.text()).toContain('植被类型：椰子树')
    expect(wrapper.text()).not.toContain('（ ）')
    expect(wrapper.text()).not.toContain('★')
  })
})

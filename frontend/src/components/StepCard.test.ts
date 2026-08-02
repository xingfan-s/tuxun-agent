import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import StepCard from './StepCard.vue'

const step = {
  step: 1,
  type: 'safety_check' as const,
  label: '安全预检',
  status: 'done' as const,
  data: { passed: true, face_count: 0, raw_output: 'private' },
  elapsed_ms: 120,
}

describe('StepCard', () => {
  it('uses an accessible button and keeps raw JSON out of the default UI', async () => {
    const wrapper = mount(StepCard, {
      props: { step },
      global: {
        stubs: {
          'el-icon': { template: '<span><slot /></span>' },
          'el-collapse-transition': { template: '<div><slot /></div>' },
        },
      },
    })
    const button = wrapper.get('button')
    expect(button.attributes('aria-label')).toContain('已完成')
    await button.trigger('click')
    expect(wrapper.find('pre').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('private')
    expect(wrapper.find('dl').exists()).toBe(true)
  })
})

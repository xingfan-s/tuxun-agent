import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ConfidenceBar from './ConfidenceBar.vue'

describe('ConfidenceBar', () => {
  it('does not format an uncalibrated ranking score as a percentage', () => {
    const wrapper = mount(ConfidenceBar, {
      props: { province: '浙江', score: 0.73, calibrated: false },
      global: { stubs: ['el-progress', 'el-badge', 'el-tag'] },
    })
    expect(wrapper.get('.bar-score').text()).toBe('0.73')
  })

  it('formats calibrated values as probabilities', () => {
    const wrapper = mount(ConfidenceBar, {
      props: { province: '浙江', score: 0.73, calibrated: true },
      global: { stubs: ['el-progress', 'el-badge', 'el-tag'] },
    })
    expect(wrapper.get('.bar-score').text()).toBe('73%')
  })

  it('labels the candidate selected as the current conclusion', () => {
    const wrapper = mount(ConfidenceBar, {
      props: { province: '海南省', score: 0.55, selected: true },
      global: {
        stubs: {
          'el-progress': true,
          'el-badge': true,
          'el-tag': { template: '<span><slot /></span>' },
        },
      },
    })
    expect(wrapper.text()).toContain('当前结论')
  })
})

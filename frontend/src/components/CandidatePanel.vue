<template>
  <div class="candidate-panel" v-if="hypotheses.length > 0">
    <h4>{{ isCalibrated ? '候选置信度' : '候选对照（未校准得分）' }}</h4>
    <ConfidenceBar
      v-for="(h, idx) in hypotheses"
      :key="idx"
      :province="h.province"
      :score="h.score"
      :evidence-count="h.evidence_count"
      :calibrated="isCalibrated"
      :selected="h.selected"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useTaskStore } from '@/stores/task'
import type { Hypothesis } from '@/types'
import ConfidenceBar from './ConfidenceBar.vue'

const store = useTaskStore()

const hypotheses = computed<Hypothesis[]>(() => {
  return [...(store.result?.top_hypotheses || [])].sort(
    (a, b) => Number(Boolean(b.selected)) - Number(Boolean(a.selected))
  )
})
const isCalibrated = computed(() => store.result?.confidence_kind === 'calibrated')
</script>

<style scoped>
.candidate-panel {
  background: #fafbfc;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 14px 16px;
}
.candidate-panel h4 { margin: 0 0 8px 0; font-size: 15px; }
</style>

<template>
  <div class="confidence-bar">
    <span class="bar-label">{{ province }}</span>
    <el-progress
      :percentage="Math.round(score * 100)"
      :stroke-width="10"
      :color="barColor"
      :show-text="false"
    />
    <span class="bar-score">{{ calibrated ? `${Math.round(score * 100)}%` : score.toFixed(2) }}</span>
    <el-tag v-if="selected" type="primary" size="small" effect="plain">当前结论</el-tag>
    <el-badge
      v-if="evidenceCount > 0"
      :value="evidenceCount"
      type="info"
      class="evidence-badge"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  province: string
  score: number
  evidenceCount?: number
  calibrated?: boolean
  selected?: boolean
}>()

const evidenceCount = computed(() => props.evidenceCount ?? 0)

const barColor = computed(() => {
  if (props.score >= 0.75) return '#67c23a'
  if (props.score >= 0.45) return '#e6a23c'
  return '#909399'
})
</script>

<style scoped>
.confidence-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
}
.bar-label {
  min-width: 72px;
  font-weight: 500;
  font-size: 14px;
}
.bar-score {
  min-width: 36px;
  text-align: right;
  font-size: 13px;
  color: #666;
}
.confidence-bar :deep(.el-progress) {
  flex: 1;
  min-width: 48px;
}
.evidence-badge {
  margin-left: 2px;
}
</style>

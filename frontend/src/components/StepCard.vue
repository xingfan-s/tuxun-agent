<template>
  <div class="step-card" :class="`step-${step.status}`">
    <div class="step-header">
      <span class="step-icon">
        {{ step.status === 'done' ? '✅' : step.status === 'running' ? '⏳' : '❌' }}
      </span>
      <span class="step-label">{{ step.label }}</span>
      <span class="step-time">{{ step.elapsed_ms }}ms</span>
    </div>
    <div v-if="expanded && step.data" class="step-body">
      <pre>{{ JSON.stringify(step.data, null, 2) }}</pre>
    </div>
    <el-button text size="small" @click="expanded = !expanded">
      {{ expanded ? '收起' : '展开详情' }}
    </el-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { StepResult } from '@/types'

const props = defineProps<{ step: StepResult }>()
const expanded = ref(false)
</script>

<style scoped>
.step-card {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 8px;
}
.step-done { border-left: 3px solid #67c23a; }
.step-running { border-left: 3px solid #409eff; animation: pulse 1.5s infinite; }
.step-error { border-left: 3px solid #f56c6c; }
.step-header { display: flex; align-items: center; gap: 8px; }
.step-icon { font-size: 16px; }
.step-label { flex: 1; font-weight: 500; }
.step-time { color: #999; font-size: 12px; }
.step-body { margin-top: 8px; background: #f5f7fa; padding: 8px; border-radius: 4px; overflow-x: auto; }
.step-body pre { margin: 0; font-size: 12px; white-space: pre-wrap; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>

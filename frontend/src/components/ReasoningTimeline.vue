<template>
  <div class="reasoning-timeline" v-if="store.steps.length > 0 || isActive">
    <h3>推理过程</h3>
    <el-progress :percentage="store.progress" :stroke-width="6" />
    <div class="steps-list" ref="listRef">
      <StepCard v-for="step in store.steps" :key="`${step.step}-${step.type}`" :step="step" />
    </div>
    <div v-if="store.steps.length === 0 && isActive" class="waiting">
      <el-icon class="is-loading"><Loading /></el-icon>
      等待分析开始...
    </div>
    <div v-if="store.streamingText && isActive" class="streaming-text">
      <div class="streaming-label">分析状态</div>
      <div class="streaming-content">{{ store.streamingText }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch, nextTick } from 'vue'
import { Loading } from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'
import StepCard from './StepCard.vue'

const store = useTaskStore()
const listRef = ref<HTMLElement | null>(null)
const isActive = computed(() => store.status === 'queued' || store.status === 'analyzing')

watch(() => store.steps.length, () => {
  nextTick(() => {
    if (listRef.value) {
      listRef.value.scrollTop = listRef.value.scrollHeight
    }
  })
})
</script>

<style scoped>
.reasoning-timeline { margin-top: 16px; }
.steps-list {
  margin-top: 16px;
  max-height: 600px;
  overflow-y: auto;
  scroll-behavior: smooth;
}
.waiting { text-align: center; padding: 40px; color: #999; }
.streaming-text {
  margin-top: 12px;
  padding: 12px;
  background: #f0f7ff;
  border-radius: 8px;
  border-left: 3px solid #409eff;
}
.streaming-label {
  font-size: 12px;
  color: #409eff;
  margin-bottom: 6px;
  font-weight: 500;
}
.streaming-content {
  font-size: 13px;
  color: #555;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 200px;
  overflow-y: auto;
}
</style>

<template>
  <div class="reasoning-timeline">
    <h3>推理过程</h3>
    <el-progress :percentage="store.progress" :stroke-width="6" />
    <div class="steps-list">
      <StepCard v-for="step in store.steps" :key="`${step.step}-${step.type}`" :step="step" />
    </div>
    <div v-if="store.steps.length === 0 && store.status === 'analyzing'" class="waiting">
      <el-icon class="is-loading"><Loading /></el-icon>
      等待分析开始...
    </div>
  </div>
</template>

<script setup lang="ts">
import { Loading } from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'
import StepCard from './StepCard.vue'

const store = useTaskStore()
</script>

<style scoped>
.reasoning-timeline { margin-top: 16px; }
.steps-list { margin-top: 16px; }
.waiting { text-align: center; padding: 40px; color: #999; }
</style>

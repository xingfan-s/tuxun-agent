<template>
  <div class="clue-summary" v-if="topClues.length > 0">
    <h4><el-icon><Connection /></el-icon><span>关键线索</span></h4>
    <div class="clue-tags">
      <el-tag
        v-for="(clue, idx) in topClues"
        :key="idx"
        :type="reliabilityType(clue)"
        effect="plain"
        size="large"
        class="clue-tag"
      >
        {{ cleanClueText(clue) }}
      </el-tag>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Connection } from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'

const store = useTaskStore()

const topClues = computed(() => {
  const step = store.steps.find(s => s.type === 'clue_extraction' && s.status === 'done')
  if (!step || !step.data) return []
  const clues = step.data.top_clues || step.data.clues?.top_3_clues || []
  return Array.isArray(clues) ? clues : []
})

function reliabilityType(clue: string): 'danger' | 'warning' | 'info' {
  if (clue.includes('★★★★★') || clue.includes('决定性') || clue.includes('0.85') || clue.includes('0.9')) return 'danger'
  if (clue.includes('★★★★') || clue.includes('强线索') || clue.includes('0.55') || clue.includes('0.6') || clue.includes('0.7')) return 'warning'
  return 'info'
}

function cleanClueText(clue: string): string {
  return clue
    .replace(/[（(]\s*(?:★★★★★|★★★★|★★★|★★|★)\s*[）)]/g, '')
    .replace(/★★★★★|★★★★|★★★|★★|★|可信度[0-9.]+/g, '')
    .replace(/^\s*[-•]\s*/, '')
    .trim()
}
</script>

<style scoped>
.clue-summary {
  background: #fafbfc;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 14px 16px;
}
.clue-summary h4 {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 0 10px 0;
  font-size: 15px;
}
.clue-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.clue-tag { font-size: 13px; }
</style>

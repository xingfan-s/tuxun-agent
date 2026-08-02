<template>
  <!-- Success -->
  <div class="result-section" v-if="store.result">
    <el-alert type="warning" :closable="false" show-icon class="legal-warning">
      <template #title>
        <strong>重要提示：</strong>
        本工具仅用于地理学习与教学研究。推理结果仅供验证参考，禁止用于侵犯他人隐私、跟踪骚扰或其他非法用途。
      </template>
    </el-alert>

    <section class="result-summary" aria-label="定位结果摘要">
      <div class="summary-location">
        <span class="summary-label">定位结果</span>
        <strong>{{ store.result.address || '位置未确定' }}</strong>
      </div>
      <div class="summary-metric">
        <span>定位层级</span>
        <strong>{{ precisionLabel }}</strong>
      </div>
      <div class="summary-metric">
        <span>不确定半径</span>
        <strong>{{ formattedRadius }}</strong>
      </div>
      <div class="summary-metric">
        <span>{{ store.result.confidence_kind === 'calibrated' ? '置信度' : '候选得分' }}</span>
        <strong>{{ formattedConfidence }}</strong>
      </div>
    </section>

    <!-- Map -->
    <MapPanel />

    <!-- Clues + Candidates side by side -->
    <el-row :gutter="16" class="insight-row">
      <el-col :xs="24" :sm="12">
        <ClueSummary />
      </el-col>
      <el-col :xs="24" :sm="12">
        <CandidatePanel />
      </el-col>
    </el-row>

    <section v-if="supportEvidence.length || conflictEvidence.length" class="evidence-section" aria-labelledby="evidence-title">
      <h3 id="evidence-title">关键证据</h3>
      <div class="evidence-columns">
        <div v-if="supportEvidence.length" class="evidence-group">
          <h4>支持证据</h4>
          <ul>
            <li v-for="(item, index) in supportEvidence" :key="`support-${index}`">
              <span class="evidence-source">{{ item.source }}</span>
              <span>{{ item.summary || '提供正向定位信号' }}</span>
            </li>
          </ul>
        </div>
        <div v-if="conflictEvidence.length" class="evidence-group evidence-conflict">
          <h4>冲突证据</h4>
          <ul>
            <li v-for="(item, index) in conflictEvidence" :key="`conflict-${index}`">
              <span class="evidence-source">{{ item.source }}</span>
              <span>{{ item.summary || '与当前候选不一致' }}</span>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <!-- Address & stats -->
    <div class="result-info">
      <el-descriptions :column="descriptionColumns" border>
        <el-descriptions-item label="地址" :span="2">
          <strong>{{ store.result.address }}</strong>
        </el-descriptions-item>
        <el-descriptions-item label="国家">{{ store.result.country }}</el-descriptions-item>
        <el-descriptions-item label="省/州">{{ store.result.province || '-' }}</el-descriptions-item>
        <el-descriptions-item label="城市">{{ store.result.city || '-' }}</el-descriptions-item>
        <el-descriptions-item label="区/县">{{ store.result.district || '-' }}</el-descriptions-item>
        <el-descriptions-item label="置信度">{{ store.result.confidence != null && store.result.confidence_kind === 'calibrated' ? (store.result.confidence * 100).toFixed(0) + '%' : (store.result.confidence != null ? store.result.confidence.toFixed(2) + '（候选得分）' : '未评估') }}</el-descriptions-item>
        <el-descriptions-item label="耗时" v-if="store.result.total_elapsed_ms">{{ (store.result.total_elapsed_ms / 1000).toFixed(1) }}s</el-descriptions-item>
        <el-descriptions-item label="定位层级">{{ store.result.precision_level }}</el-descriptions-item>
        <el-descriptions-item label="不确定半径">{{ store.result.uncertainty_radius_m != null ? `${Math.round(store.result.uncertainty_radius_m / 1000)} km` : '-' }}</el-descriptions-item>
        <el-descriptions-item label="Tokens" v-if="store.result.tokens_used">{{ store.result.tokens_used }}</el-descriptions-item>
      </el-descriptions>

      <div v-if="store.result.tool_stats" class="tool-stats">
        <span>工具调用：{{ store.result.tool_stats.total_calls }} 次</span>
        <span class="stat-success">成功 {{ store.result.tool_stats.success }}</span>
        <span class="stat-timeout" v-if="store.result.tool_stats.timeout">超时 {{ store.result.tool_stats.timeout }}</span>
        <span class="stat-failed" v-if="store.result.tool_stats.failed">失败 {{ store.result.tool_stats.failed }}</span>
        <span class="stat-timeout" v-if="store.result.tool_stats.unavailable">不可用 {{ store.result.tool_stats.unavailable }}</span>
        <span class="stat-timeout" v-if="store.result.tool_stats.budget_skipped">额度跳过 {{ store.result.tool_stats.budget_skipped }}</span>
        <span class="stat-timeout" v-if="store.result.tool_stats.empty_result">空结果 {{ store.result.tool_stats.empty_result }}</span>
      </div>
    </div>

    <el-collapse v-if="diagnosticsEnabled" class="reasoning-collapse">
      <el-collapse-item title="推理依据">
        <p>{{ store.result.reasoning }}</p>
      </el-collapse-item>
    </el-collapse>
  </div>

  <!-- Rejected -->
  <div v-else-if="store.status === 'rejected'" class="result-card-placeholder">
    <el-result icon="error" title="安全预检未通过" :sub-title="store.safetyReason || '无法分析该图片'">
      <template #extra>
        <el-button type="primary" @click="store.reset()">重新上传</el-button>
      </template>
    </el-result>
  </div>

  <!-- Failed -->
  <div v-else-if="store.status === 'failed'" class="result-card-placeholder">
    <el-result icon="error" title="分析失败" :sub-title="store.error?.message || '未知错误'">
      <template #extra>
        <el-button type="primary" @click="store.reset()">重新上传</el-button>
      </template>
    </el-result>
  </div>

  <!-- Cancelled -->
  <div v-else-if="store.status === 'cancelled'" class="result-card-placeholder">
    <el-result icon="warning" title="任务已取消" :sub-title="store.error?.message || '分析已停止'">
      <template #extra>
        <el-button type="primary" @click="store.reset()">重新上传</el-button>
      </template>
    </el-result>
  </div>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent, onBeforeUnmount, onMounted, ref } from 'vue'
import { useTaskStore } from '@/stores/task'
import type { Evidence } from '@/types'
import ClueSummary from './ClueSummary.vue'
import CandidatePanel from './CandidatePanel.vue'

const MapPanel = defineAsyncComponent(() => import('./MapPanel.vue'))

const store = useTaskStore()
const diagnosticsEnabled = import.meta.env.DEV && import.meta.env.VITE_DIAGNOSTICS === 'true'
const descriptionColumns = ref(2)

const updateColumns = () => {
  descriptionColumns.value = window.innerWidth < 768 ? 1 : 2
}
onMounted(() => {
  updateColumns()
  window.addEventListener('resize', updateColumns)
})
onBeforeUnmount(() => window.removeEventListener('resize', updateColumns))

const precisionLabels: Record<string, string> = {
  country: '国家级', province: '省级', city: '城市级', district: '区县级',
  road: '道路级', poi: '地点级', unknown: '未确定',
}
const precisionLabel = computed(() => precisionLabels[store.result?.precision_level || 'unknown'])
const formattedRadius = computed(() => {
  const radius = store.result?.uncertainty_radius_m
  if (radius == null) return '未评估'
  return radius >= 1000 ? `${Math.round(radius / 1000)} km` : `${Math.round(radius)} m`
})
const formattedConfidence = computed(() => {
  const result = store.result
  if (!result || result.confidence == null) return '未评估'
  return result.confidence_kind === 'calibrated'
    ? `${Math.round(result.confidence * 100)}%`
    : result.confidence.toFixed(2)
})
const supportEvidence = computed<Evidence[]>(() =>
  (store.result?.evidence || []).filter(item => item.direction === 'support').slice(0, 5),
)
const conflictEvidence = computed<Evidence[]>(() =>
  (store.result?.evidence || []).filter(item => item.direction === 'contradict').slice(0, 5),
)
</script>

<style scoped>
.result-section { margin-top: 16px; }
.legal-warning { margin-bottom: 16px; }
.result-summary {
  display: grid;
  grid-template-columns: minmax(220px, 2fr) repeat(3, minmax(120px, 1fr));
  gap: 1px;
  margin-bottom: 16px;
  border-top: 1px solid #dcdfe6;
  border-bottom: 1px solid #dcdfe6;
  background: #dcdfe6;
}
.summary-location,
.summary-metric {
  min-width: 0;
  padding: 14px 16px;
  background: #fff;
}
.summary-location { display: flex; flex-direction: column; gap: 4px; }
.summary-location strong { overflow-wrap: anywhere; }
.summary-label,
.summary-metric span { color: #73767a; font-size: 12px; }
.summary-metric { display: flex; flex-direction: column; gap: 4px; }
.summary-metric strong { font-size: 16px; }
.insight-row { margin-top: 16px; }
.evidence-section { margin-top: 20px; }
.evidence-section h3 { margin: 0 0 10px; font-size: 16px; }
.evidence-columns { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.evidence-group h4 { margin: 0 0 8px; color: #2f6846; font-size: 14px; }
.evidence-conflict h4 { color: #a0443f; }
.evidence-group ul { margin: 0; padding: 0; list-style: none; }
.evidence-group li {
  display: grid;
  grid-template-columns: minmax(72px, auto) 1fr;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #ebeef5;
  color: #4b4f55;
  font-size: 13px;
  line-height: 1.5;
}
.evidence-source { color: #73767a; font-weight: 600; overflow-wrap: anywhere; }
.result-info { margin-top: 16px; }
.tool-stats {
  font-size: 13px; color: #666; margin-top: 10px;
  display: flex; gap: 16px; align-items: center;
}
.stat-success { color: #67c23a; }
.stat-timeout { color: #e6a23c; }
.stat-failed { color: #f56c6c; }
.reasoning-collapse { margin-top: 12px; }
.result-card-placeholder { margin-top: 24px; }

@media (max-width: 767px) {
  .result-summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .summary-location { grid-column: 1 / -1; }
  .evidence-columns { grid-template-columns: 1fr; gap: 14px; }
  .tool-stats { flex-wrap: wrap; gap: 8px 14px; }
}

@media (max-width: 420px) {
  .result-summary { grid-template-columns: 1fr; }
  .summary-location { grid-column: auto; }
  .evidence-group li { grid-template-columns: 1fr; gap: 2px; }
}
</style>

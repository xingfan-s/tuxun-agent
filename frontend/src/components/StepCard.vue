<template>
  <div class="step-card" :class="`step-${step.status}`">
    <button
      class="step-header"
      type="button"
      :aria-expanded="expanded"
      :aria-controls="detailId"
      :aria-label="`${step.label}，${statusText}`"
      @click="expanded = !expanded"
    >
      <el-icon class="step-icon" :class="{ 'is-loading': step.status === 'running' }">
        <component :is="statusIcon" />
      </el-icon>
      <span class="step-label">{{ step.label }}</span>
      <span class="step-status">{{ statusText }}</span>
      <span class="step-time">{{ (step.elapsed_ms / 1000).toFixed(1) }}s</span>
      <el-icon class="expand-icon"><ArrowDown v-if="!expanded" /><ArrowUp v-else /></el-icon>
    </button>

    <div class="step-summary">{{ summaryText }}</div>

    <el-collapse-transition>
      <div v-if="expanded && step.data" :id="detailId" class="step-detail">
        <dl v-if="detailRows.length" class="detail-list">
          <template v-for="row in detailRows" :key="row.label">
            <dt>{{ row.label }}</dt>
            <dd>{{ row.value }}</dd>
          </template>
        </dl>
        <pre v-if="diagnosticsEnabled">{{ JSON.stringify(step.data, null, 2) }}</pre>
      </div>
    </el-collapse-transition>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { ArrowDown, ArrowUp, CircleCheckFilled, Loading, WarningFilled } from '@element-plus/icons-vue'
import type { StepResult } from '@/types'

const props = defineProps<{ step: StepResult }>()
const detailId = `step-detail-${props.step.step}-${props.step.type}`
const diagnosticsEnabled = import.meta.env.DEV && import.meta.env.VITE_DIAGNOSTICS === 'true'
const expanded = ref(props.step.status === 'running' || props.step.status === 'error')
watch(() => props.step.status, (status) => {
  if (status === 'running' || status === 'error') expanded.value = true
})

const statusIcon = computed(() => {
  if (props.step.status === 'done') return CircleCheckFilled
  if (props.step.status === 'running') return Loading
  return WarningFilled
})
const statusText = computed(() => ({ done: '已完成', running: '进行中', error: '出错' })[props.step.status])

const detailRows = computed(() => Object.entries(props.step.data || {})
  .filter(([key, value]) => !['raw_output', 'thought', 'image_base64', 'base64'].includes(key)
    && (['string', 'number', 'boolean'].includes(typeof value)
      || (Array.isArray(value) && value.every(item => ['string', 'number', 'boolean'].includes(typeof item)))))
  .slice(0, 8)
  .map(([label, value]) => ({
    label: label.replace(/_/g, ' '),
    value: Array.isArray(value) ? value.slice(0, 5).join('、') : String(value),
  })))

const summaryText = computed(() => {
  const d = props.step.data || {}
  const type = props.step.type

  switch (type) {
    case 'safety_check': {
      if (d.face_count !== undefined) {
        const warning = Array.isArray(d.warnings) && d.warnings.length
          ? `，安全检查降级: ${d.warnings.join(', ')}`
          : ''
        return d.passed ? `通过（人脸: ${d.face_count}${warning}）` : `未通过: ${d.reason || '未知原因'}`
      }
      return d.status || ''
    }

    case 'exif': {
      if (d.has_gps && d.gps) {
        const g = d.gps
        return `有GPS (${g.lat?.toFixed(2)}, ${g.lng?.toFixed(2)})${d.device ? ' · ' + d.device : ''}${d.datetime ? ' · ' + d.datetime : ''}`
      }
      return '无GPS信息'
    }

    case 'vision_macro': {
      return `宏观分类: ${d.region || '未知'}`
    }

    case 'geoclip': {
      const preds = d.top_predictions
      if (Array.isArray(preds) && preds.length > 0) {
        return preds.slice(0, 3).join(' · ')
      }
      return d.status || '无预测结果'
    }

    case 'geoclip_anchor': {
      const lines = d.summary
      if (Array.isArray(lines) && lines.length > 0) {
        return lines.slice(0, 2).join(' | ')
      }
      return `锚点预搜: ${d.count ?? 0} 个有效锚点`
    }

    case 'clip_search': {
      if (d.matches_found !== undefined) {
        const top = d.top_match
        const loc = top ? ` (${top.lat?.toFixed(2)}, ${top.lon?.toFixed(2)})` : ''
        return `找到 ${d.matches_found} 个相似图片 · 图库 ${d.db_size} 张${loc}`
      }
      return d.status || ''
    }

    case 'vision_detail': {
      const desc = d.description || d.raw_output || ''
      const plain = typeof desc === 'string' ? desc.replace(/[\n\r]/g, ' ').trim() : ''
      return plain.length > 80 ? plain.slice(0, 80) + '...' : plain
    }

    case 'clue_extraction': {
      const tops = d.top_clues
      if (Array.isArray(tops) && tops.length > 0) {
        return tops.slice(0, 3).join(' · ')
      }
      return '线索提取完成'
    }

    case 'ocr': {
      const parts: string[] = []
      if (d.text_count) parts.push(`识别 ${d.text_count} 段文字`)
      if (d.plates) parts.push(`车牌 ${d.plates}`)
      if (d.area_codes) parts.push(`区号 ${d.area_codes}`)
      if (d.highways) parts.push(`公路 ${d.highways}`)
      return parts.length > 0 ? parts.join(' · ') : (d.summary || 'OCR 完成')
    }

    case 'ocr_fusion': {
      const qs = d.fused_queries
      if (Array.isArray(qs) && qs.length > 0) {
        return `生成 ${qs.length} 个融合查询 → ${qs.slice(0, 2).join(', ')}`
      }
      return d.fusion_strategy || 'OCR 上下文融合完成'
    }

    case 'search_strategy': {
      const cands = d.candidates || d.ranked_candidates
      if (Array.isArray(cands) && cands.length > 0) {
        return cands.slice(0, 3).join(' · ')
      }
      return `主要区域: ${d.primary_region || '未知'}`
    }

    case 'tool_call': {
      if (props.step.status === 'running') {
        return `调用中: ${d.tool_name || ''}`
      }
      if (props.step.status === 'error') {
        return `${d.tool_name || ''} 失败`
      }
      const output = d.output
      if (typeof output === 'string') {
        return output.length > 100 ? output.slice(0, 100) + '...' : output
      }
      if (Array.isArray(output) && output.length > 0) {
        const first = output[0]
        if (typeof first === 'object' && first !== null) {
          const parts: string[] = []
          if (first.title) parts.push(first.title)
          if (first.address) parts.push(first.address)
          if (first.display_name) parts.push(first.display_name)
          if (first.snippet) parts.push(first.snippet)
          const text = parts.join(' | ') || JSON.stringify(output[0])
          return text.length > 100 ? text.slice(0, 100) + '...' : text
        }
        return `返回 ${output.length} 条结果`
      }
      if (typeof output === 'object' && output !== null) {
        const flat = JSON.stringify(output)
        return flat.length > 100 ? flat.slice(0, 100) + '...' : flat
      }
      return `${d.tool_name || ''}: ${d.status || '完成'}`
    }

    case 'reasoning': {
      const action = d.action || ''
      if (action === 'redirect') return `${d.summary || '正在修正候选方向'} → ${d.new_target || ''}`
      return d.summary || (action === 'final_answer' ? '候选验证完成' : '正在评估候选证据')
    }

    case 'verification': {
      if (d.valid) return '验证通过'
      const contras = d.contradictions
      if (Array.isArray(contras) && contras.length > 0) {
        return `发现 ${contras.length} 个矛盾: ${contras[0]}`
      }
      return '验证完成'
    }

    case 'final': {
      return `结果: ${d.address || ''} · 置信度 ${d.confidence ?? '?'}`
    }

    case 'fine_localize':
    case 'result_enrichment': {
      const location = [d.district, ...(Array.isArray(d.landmarks) ? d.landmarks.slice(0, 2) : [])]
        .filter(Boolean)
        .join(' · ')
      return location || '结果丰富化完成'
    }

    default:
      return ''
  }
})
</script>

<style scoped>
.step-card {
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  padding: 10px 14px;
  margin-bottom: 8px;
  transition: border-color 0.2s;
}
.step-done { border-left: 3px solid #67c23a; }
.step-running { border-left: 3px solid #409eff; animation: pulse 1.5s infinite; }
.step-error { border-left: 3px solid #f56c6c; }
.step-header {
  display: flex;
  width: 100%;
  border: 0;
  background: transparent;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
  text-align: left;
  color: inherit;
  padding: 0;
}
.step-header:focus-visible { outline: 2px solid #409eff; outline-offset: 3px; }
.step-icon { font-size: 14px; }
.step-label { flex: 1; font-weight: 500; font-size: 14px; }
.step-status { color: #606266; font-size: 12px; }
.step-time { color: #999; font-size: 12px; }
.expand-icon { font-size: 12px; color: #999; }
.step-summary {
  margin-top: 4px;
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}
.step-detail {
  margin-top: 8px;
  background: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  overflow-x: auto;
}
.step-detail pre {
  margin: 0;
  font-size: 12px;
  white-space: pre-wrap;
  word-break: break-all;
}
.detail-list {
  display: grid;
  grid-template-columns: minmax(88px, auto) 1fr;
  gap: 6px 12px;
  margin: 0;
  font-size: 12px;
}
.detail-list dt { color: #73767a; text-transform: capitalize; }
.detail-list dd { margin: 0; overflow-wrap: anywhere; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
</style>

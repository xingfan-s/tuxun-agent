<template>
  <div class="result-card" v-if="store.result">
    <el-alert type="warning" :closable="false" show-icon class="legal-warning">
      <template #title>
        <strong>重要提示：</strong>
        本工具仅用于地理学习与教学研究。推理结果仅供验证参考，禁止用于侵犯他人隐私、跟踪骚扰或其他非法用途。任何人使用本工具均应遵守法律法规，违规后果自负。
      </template>
    </el-alert>

    <el-descriptions :column="2" border class="result-info">
      <el-descriptions-item label="地址" :span="2">
        <strong>{{ store.result.address }}</strong>
      </el-descriptions-item>
      <el-descriptions-item label="国家">{{ store.result.country }}</el-descriptions-item>
      <el-descriptions-item label="置信度">{{ (store.result.confidence * 100).toFixed(0) }}%</el-descriptions-item>
      <el-descriptions-item label="省/州">{{ store.result.province || '-' }}</el-descriptions-item>
      <el-descriptions-item label="耗时">{{ store.result.total_elapsed_ms }}ms</el-descriptions-item>
      <el-descriptions-item label="城市">{{ store.result.city || '-' }}</el-descriptions-item>
      <el-descriptions-item label="Token">{{ store.result.tokens_used }}</el-descriptions-item>
    </el-descriptions>

    <div v-if="store.result.tool_stats" class="tool-stats">
      <span>工具调用：{{ store.result.tool_stats.total_calls }} 次</span>
      <span class="stat-success">成功 {{ store.result.tool_stats.success }}</span>
      <span class="stat-timeout">超时 {{ store.result.tool_stats.timeout }}</span>
      <span class="stat-failed">失败 {{ store.result.tool_stats.failed }}</span>
    </div>

    <el-collapse>
      <el-collapse-item title="推理依据">
        <p>{{ store.result.reasoning }}</p>
      </el-collapse-item>
    </el-collapse>

    <div class="no-share">
      💡 出于隐私保护，本结果不支持分享
    </div>
  </div>

  <div v-else-if="store.status === 'rejected'" class="rejected-card">
    <el-result icon="error" title="安全预检未通过" :sub-title="store.safetyReason || '无法分析该图片'">
      <template #extra>
        <el-button type="primary" @click="store.reset()">重新上传</el-button>
      </template>
    </el-result>
  </div>

  <div v-else-if="store.status === 'failed'" class="error-card">
    <el-result icon="error" title="分析失败" :sub-title="store.error?.message || '未知错误'">
      <template #extra>
        <el-button type="primary" @click="store.reset()">重新上传</el-button>
      </template>
    </el-result>
  </div>
</template>

<script setup lang="ts">
import { useTaskStore } from '@/stores/task'
const store = useTaskStore()
</script>

<style scoped>
.result-card { margin-top: 16px; }
.legal-warning { margin-bottom: 16px; }
.result-info { margin-bottom: 16px; }
.tool-stats { font-size: 13px; color: #666; margin-bottom: 12px; display: flex; gap: 16px; }
.stat-success { color: #67c23a; }
.stat-timeout { color: #e6a23c; }
.stat-failed { color: #f56c6c; }
.no-share { text-align: center; color: #999; margin-top: 12px; font-size: 13px; }
.rejected-card, .error-card { margin-top: 24px; }
</style>

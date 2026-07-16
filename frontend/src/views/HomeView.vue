<template>
  <div class="home-view">
    <header class="app-header">
      <h1>图寻 Agent</h1>
      <p class="subtitle">上传图片，AI 自动推理拍摄地点</p>
    </header>

    <el-row :gutter="24">
      <el-col :span="8">
        <ImageUploader @analyze-start="onAnalyzeStart" />
      </el-col>
      <el-col :span="16">
        <div class="map-placeholder" v-if="!store.result">
          <el-empty description="分析完成后，结果将显示在地图上" />
        </div>
        <div v-else class="map-result-info">
          <p>📍 {{ store.result.address }}</p>
          <p>🌍 {{ store.result.lat.toFixed(4) }}, {{ store.result.lng.toFixed(4) }}</p>
        </div>
      </el-col>
    </el-row>

    <GlobalNoticeBar />

    <ReasoningTimeline />

    <ResultCard />

    <DisclaimerFooter />
  </div>
</template>

<script setup lang="ts">
import { useTaskStore } from '@/stores/task'
import ImageUploader from '@/components/ImageUploader.vue'
import GlobalNoticeBar from '@/components/GlobalNoticeBar.vue'
import ReasoningTimeline from '@/components/ReasoningTimeline.vue'
import ResultCard from '@/components/ResultCard.vue'
import DisclaimerFooter from '@/components/DisclaimerFooter.vue'

const store = useTaskStore()

function onAnalyzeStart(taskId: string) {
  // SSE connection handled inside ImageUploader
}
</script>

<style scoped>
.home-view { max-width: 1200px; margin: 0 auto; padding: 24px; }
.app-header { text-align: center; margin-bottom: 32px; }
.app-header h1 { font-size: 28px; margin-bottom: 4px; }
.subtitle { color: #999; }
.map-placeholder { height: 300px; display: flex; align-items: center; justify-content: center; background: #f5f7fa; border-radius: 8px; }
.map-result-info { padding: 16px; background: #f0f9eb; border-radius: 8px; }
</style>

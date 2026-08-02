<template>
  <div class="home-view">
    <header class="app-header">
      <h1>图寻 Agent</h1>
      <p class="subtitle">上传图片，AI 自动推理拍摄地点</p>
    </header>

    <GlobalNoticeBar />

    <!-- Phase 1: Upload -->
    <div v-if="isUploadPhase" class="upload-phase">
      <ImageUploader @analyze-start="onAnalyzeStart" />
    </div>

    <!-- Phase 2: Analyzing -->
    <div v-else-if="isAnalyzingPhase" class="analyze-phase">
      <el-row :gutter="24">
        <el-col :xs="24" :sm="9" :md="7">
          <div class="preview-panel">
            <img
              v-if="uploadedImageUrl"
              :src="uploadedImageUrl"
              class="preview-image"
              alt="预览图片"
            />
            <el-progress
              :percentage="store.progress"
              :stroke-width="6"
              class="preview-progress"
            />
          </div>
        </el-col>
        <el-col :xs="24" :sm="15" :md="17">
          <ReasoningTimeline />
        </el-col>
      </el-row>
    </div>

    <!-- Phase 3: Result -->
    <div v-else-if="isResultPhase" class="result-phase">
      <ResultCard />
      <ReasoningTimeline />
    </div>

    <!-- Phase 4: Idle / Error states -->
    <div v-else>
      <ResultCard />
    </div>

    <DisclaimerFooter />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useTaskStore } from '@/stores/task'
import { useSSE } from '@/composables/useSSE'
import { startTask, getTask } from '@/api'
import ImageUploader from '@/components/ImageUploader.vue'
import GlobalNoticeBar from '@/components/GlobalNoticeBar.vue'
import ReasoningTimeline from '@/components/ReasoningTimeline.vue'
import ResultCard from '@/components/ResultCard.vue'
import DisclaimerFooter from '@/components/DisclaimerFooter.vue'

const store = useTaskStore()
const { connect } = useSSE()

const uploadedImageUrl = computed(() => store.uploadedImageUrl)

const isUploadPhase = computed(() =>
  ['idle', 'uploading', 'uploaded'].includes(store.status)
)

const isAnalyzingPhase = computed(() =>
  ['queued', 'analyzing'].includes(store.status)
)

const isResultPhase = computed(() =>
  store.status === 'done'
)

function onAnalyzeStart(taskId: string) {
  void startTask(taskId).then((task) => {
    store.hydrate(task)
    void connect(taskId)
  }).catch((error: any) => {
    store.setError(error?.response?.data?.detail || '任务启动失败', false)
  })
}

onMounted(() => {
  if (!store.taskId) {
    store.taskId = sessionStorage.getItem('tuxun.taskId')
  }
  if (!store.taskId) return
  void getTask(store.taskId).then((task) => {
    store.hydrate(task)
    if (!['done', 'failed', 'rejected', 'cancelled', 'expired'].includes(task.status)) {
      void connect(task.task_id)
    }
  }).catch(() => undefined)
})
</script>

<style scoped>
.home-view {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  box-sizing: border-box;
}
.analyze-phase,
.result-phase { min-width: 0; }
.app-header { text-align: center; margin-bottom: 24px; }
.app-header h1 { font-size: 28px; margin-bottom: 4px; }
.subtitle { color: #999; }

.upload-phase {
  max-width: 480px;
  margin: 0 auto;
}

.preview-panel { margin-top: 8px; }
.preview-image {
  width: 100%;
  max-height: 280px;
  object-fit: contain;
  border-radius: 8px;
  border: 1px solid #e4e7ed;
}
.preview-progress { margin-top: 12px; }

@media (max-width: 767px) {
  .home-view { padding: 16px; }
  .app-header { margin-bottom: 18px; }
  .app-header h1 { font-size: 24px; }
  .preview-panel { margin-bottom: 18px; }
  .preview-image { max-height: 240px; }
}

@media (max-width: 420px) {
  .home-view { padding: 12px; }
  .subtitle { font-size: 13px; }
}
</style>

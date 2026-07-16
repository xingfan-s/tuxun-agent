<template>
  <div class="image-uploader">
    <el-upload
      class="upload-area"
      drag
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFileChange"
      accept="image/jpeg,image/png,image/webp"
      :disabled="store.status === 'analyzing'"
    >
      <div v-if="!previewUrl" class="upload-placeholder">
        <el-icon :size="48"><UploadFilled /></el-icon>
        <p>拖拽图片到此处，或点击上传</p>
        <p class="upload-hint">支持 JPG / PNG / WebP，最大 20MB</p>
      </div>
      <img v-else :src="previewUrl" class="preview-image" />
    </el-upload>
    <el-button
      v-if="previewUrl && store.status !== 'analyzing'"
      type="primary"
      size="large"
      class="analyze-btn"
      @click="startAnalyze"
      :loading="store.status === 'uploading'"
    >
      开始分析
    </el-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'
import { useSSE } from '@/composables/useSSE'
import { ElMessage } from 'element-plus'

const store = useTaskStore()
const { connect } = useSSE()
const previewUrl = ref<string | null>(null)
const selectedFile = ref<File | null>(null)

const emit = defineEmits<{
  analyzeStart: [taskId: string]
}>()

function handleFileChange(file: any) {
  const raw = file.raw as File
  if (!raw) return

  const validTypes = ['image/jpeg', 'image/png', 'image/webp']
  if (!validTypes.includes(raw.type)) {
    ElMessage.error('不支持的文件格式，仅支持 JPG/PNG/WebP')
    return
  }
  if (raw.size > 20 * 1024 * 1024) {
    ElMessage.error('文件大小超过 20MB 限制')
    return
  }

  selectedFile.value = raw
  previewUrl.value = URL.createObjectURL(raw)
}

async function startAnalyze() {
  if (!selectedFile.value) return
  store.reset()
  try {
    const taskId = await store.upload(selectedFile.value)
    emit('analyzeStart', taskId)
    connect(taskId)
  } catch {
    // Error handled in store
  }
}
</script>

<style scoped>
.image-uploader { width: 100%; }
.upload-placeholder { padding: 40px 0; color: #999; }
.upload-hint { font-size: 12px; color: #bbb; margin-top: 8px; }
.preview-image { max-width: 100%; max-height: 300px; object-fit: contain; }
.analyze-btn { width: 100%; margin-top: 12px; }
</style>

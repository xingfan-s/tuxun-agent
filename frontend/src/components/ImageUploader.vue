<template>
  <div class="image-uploader">
    <el-upload
      ref="uploadRef"
      class="upload-area"
      drag
      :auto-upload="false"
      :show-file-list="false"
      :on-change="handleFileChange"
      accept="image/jpeg,image/png,image/webp"
      :disabled="store.status === 'queued' || store.status === 'analyzing'"
    >
      <div v-if="!previewUrl" class="upload-placeholder">
        <el-icon :size="48"><UploadFilled /></el-icon>
        <p>拖拽图片到此处，或点击上传</p>
        <p class="upload-hint">支持 JPG / PNG / WebP，最大 20MB</p>
      </div>
      <img v-else :src="previewUrl" class="preview-image" alt="待分析的上传图片" />
    </el-upload>
    <el-button
      v-if="previewUrl && store.status !== 'queued' && store.status !== 'analyzing'"
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
import { onBeforeUnmount, ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'
import { ElMessage, type UploadFile, type UploadInstance } from 'element-plus'
import { resetNativeFileInput, validateImageFile } from '@/imageValidation'

const store = useTaskStore()
const uploadRef = ref<UploadInstance>()
const previewUrl = ref<string | null>(null)
const selectedFile = ref<File | null>(null)

const emit = defineEmits<{
  analyzeStart: [taskId: string]
}>()

function clearSelection() {
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = null
  selectedFile.value = null
  uploadRef.value?.clearFiles()
  resetNativeFileInput(uploadRef.value?.$el)
}

async function handleFileChange(file: UploadFile) {
  const raw = file.raw
  if (!raw) return

  clearSelection()
  const validationError = await validateImageFile(raw)
  if (validationError) {
    ElMessage.error(validationError)
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
  } catch {
    clearSelection()
  }
}

onBeforeUnmount(clearSelection)
</script>

<style scoped>
.image-uploader { width: 100%; }
.upload-placeholder { padding: 40px 0; color: #999; }
.upload-hint { font-size: 12px; color: #bbb; margin-top: 8px; }
.preview-image { max-width: 100%; max-height: 300px; object-fit: contain; }
.analyze-btn { width: 100%; margin-top: 12px; }
</style>

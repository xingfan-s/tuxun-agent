import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { StepResult, Result, ToolWarning, Notice } from '@/types'
import { uploadImage } from '@/api'

export const useTaskStore = defineStore('task', () => {
  const taskId = ref<string | null>(null)
  const status = ref<string>('idle')
  const progress = ref(0)
  const steps = ref<StepResult[]>([])
  const result = ref<Result | null>(null)
  const error = ref<{ message: string; recoverable: boolean } | null>(null)
  const safetyReason = ref<string | null>(null)
  const warnings = ref<ToolWarning[]>([])
  const notices = ref<Notice[]>([])
  const uploadedImageUrl = ref<string | null>(null)

  let noticeIdCounter = 0

  function addNotice(level: 'info' | 'warning' | 'error', message: string, closable = true, duration?: number) {
    const id = noticeIdCounter++
    notices.value.push({ id, level, message, closable, duration })
    if (notices.value.length > 3) {
      notices.value.shift()
    }
    return id
  }

  function removeNotice(id: number) {
    notices.value = notices.value.filter(n => n.id !== id)
  }

  async function upload(file: File) {
    try {
      status.value = 'uploading'
      const resp = await uploadImage(file)
      taskId.value = resp.task_id
      uploadedImageUrl.value = URL.createObjectURL(file)
      return resp.task_id
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || '上传失败'
      addNotice('error', msg)
      status.value = 'idle'
      throw e
    }
  }

  function addStep(step: StepResult) {
    const idx = steps.value.findIndex(s => s.step === step.step && s.type === step.type)
    if (idx >= 0) {
      steps.value[idx] = step
    } else {
      steps.value.push(step)
    }
  }

  function addWarning(warning: ToolWarning) {
    warnings.value.push(warning)
    addNotice('warning', warning.message)
  }

  function setResult(r: Result) {
    result.value = r
    status.value = 'done'
    progress.value = 100
  }

  function setError(msg: string, recoverable: boolean) {
    error.value = { message: msg, recoverable }
    if (!recoverable) {
      status.value = 'failed'
    }
  }

  function reset() {
    taskId.value = null
    status.value = 'idle'
    progress.value = 0
    steps.value = []
    result.value = null
    error.value = null
    safetyReason.value = null
    warnings.value = []
    notices.value = []
    if (uploadedImageUrl.value) {
      URL.revokeObjectURL(uploadedImageUrl.value)
      uploadedImageUrl.value = null
    }
  }

  return {
    taskId, status, progress, steps, result, error, safetyReason,
    warnings, notices, uploadedImageUrl,
    upload, addStep, addWarning, setResult, setError, reset,
    addNotice, removeNotice,
  }
})

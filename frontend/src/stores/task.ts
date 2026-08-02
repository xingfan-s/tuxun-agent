import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { StepResult, Result, ToolWarning, Notice, TaskStatus } from '@/types'
import { getImageUrl, uploadImage } from '@/api'

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
  const streamingText = ref<string>('')

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
      sessionStorage.setItem('tuxun.taskId', resp.task_id)
      status.value = 'uploaded'
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

  function hydrate(task: TaskStatus) {
    taskId.value = task.task_id
    status.value = task.status
    progress.value = task.progress ?? 0
    steps.value = [...(task.steps ?? [])]
    result.value = task.result
    safetyReason.value = task.safety_reason
    if (!uploadedImageUrl.value && ['uploaded', 'queued', 'analyzing'].includes(task.status)) {
      uploadedImageUrl.value = getImageUrl(task.task_id)
    }
    if (task.error) error.value = { message: task.error, recoverable: task.error_recoverable ?? false }
  }

  function addWarning(warning: ToolWarning) {
    warnings.value.push(warning)
    if (warning.message.includes('工具调用上限') &&
        notices.value.some(notice => notice.message.includes('工具调用上限'))) {
      return
    }
    addNotice('warning', warning.message)
  }

  function setResult(r: Result) {
    result.value = r
    status.value = 'done'
    progress.value = 100
  }

  function setStreamingText(text: string) {
    streamingText.value = text
  }

  function setError(msg: string, recoverable: boolean) {
    error.value = { message: msg, recoverable }
    if (!recoverable) {
      status.value = msg.includes('安全') || msg.includes('拒绝') ? 'rejected' : 'failed'
    }
  }

  function setRejected(reason: string) {
    safetyReason.value = reason
    error.value = { message: reason, recoverable: false }
    status.value = 'rejected'
  }

  function reset() {
    taskId.value = null
    sessionStorage.removeItem('tuxun.taskId')
    status.value = 'idle'
    progress.value = 0
    steps.value = []
    result.value = null
    error.value = null
    safetyReason.value = null
    warnings.value = []
    notices.value = []
    streamingText.value = ''
    if (uploadedImageUrl.value) {
      URL.revokeObjectURL(uploadedImageUrl.value)
      uploadedImageUrl.value = null
    }
  }

  return {
    taskId, status, progress, steps, result, error, safetyReason,
    warnings, notices, uploadedImageUrl, streamingText,
    upload, hydrate, addStep, addWarning, setResult, setError, setRejected, setStreamingText, reset,
    addNotice, removeNotice,
  }
})

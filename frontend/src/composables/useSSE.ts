import { ref, onUnmounted } from 'vue'
import { getStreamUrl } from '@/api'
import { useTaskStore } from '@/stores/task'
import type { StepResult, Result, ToolWarning } from '@/types'

export function useSSE() {
  const isConnected = ref(false)
  const reconnectAttempt = ref(0)
  let eventSource: EventSource | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  const store = useTaskStore()

  function connect(taskId: string) {
    if (eventSource) {
      eventSource.close()
    }

    const url = getStreamUrl(taskId)
    eventSource = new EventSource(url)
    isConnected.value = true
    store.status = 'analyzing'

    eventSource.addEventListener('step_update', (e: MessageEvent) => {
      const step: StepResult = JSON.parse(e.data)
      store.addStep(step)
    })

    eventSource.addEventListener('progress', (e: MessageEvent) => {
      const { progress } = JSON.parse(e.data)
      store.progress = progress
    })

    eventSource.addEventListener('tool_warning', (e: MessageEvent) => {
      const warning: ToolWarning = JSON.parse(e.data)
      store.addWarning(warning)
    })

    eventSource.addEventListener('result', (e: MessageEvent) => {
      const result: Result = JSON.parse(e.data)
      store.setResult(result)
      disconnect()
    })

    eventSource.addEventListener('error', (e: MessageEvent) => {
      if (e.data) {
        try {
          const { message, recoverable } = JSON.parse(e.data)
          store.setError(message, recoverable ?? false)
          if (!recoverable) {
            disconnect()
          }
        } catch {
          // SSE connection error (no data)
        }
      }
    })

    eventSource.onerror = () => {
      isConnected.value = false
      if (reconnectAttempt.value < 5) {
        reconnectAttempt.value++
        const retry = 3000 * reconnectAttempt.value
        store.addNotice('warning', `连接中断，${retry / 1000}s 后重连...`)
        reconnectTimer = setTimeout(() => connect(taskId), retry)
      } else {
        store.addNotice('error', '连接失败，请检查网络后刷新重试')
        store.status = 'failed'
      }
    }

    eventSource.onopen = () => {
      reconnectAttempt.value = 0
      isConnected.value = true
    }
  }

  function disconnect() {
    if (eventSource) {
      eventSource.close()
      eventSource = null
    }
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    isConnected.value = false
  }

  onUnmounted(() => {
    disconnect()
  })

  return { isConnected, connect, disconnect }
}

import { ref, onUnmounted } from 'vue'
import { getStreamUrl, getTask } from '@/api'
import { useTaskStore } from '@/stores/task'
import type { StepResult, Result, ToolWarning } from '@/types'

const DEBUG = false

/**
 * Parse a single SSE event block into { event, data } or null.
 *
 * Backend sends (one event = 5 lines, 2 trailing blanks):
 *   event: step_update\n
 *   data: {...}\n
 *   retry: 3000\n
 *   \n
 *   \n
 */
function parseSSEEvent(raw: string): { id?: number; event: string; data: Record<string, any> } | null {
  const lines = raw.split('\n')
  let eventType = ''
  let eventId: number | undefined
  let dataLines: string[] = []

  for (const line of lines) {
    if (line.startsWith('id: ')) eventId = Number(line.slice(4))
    if (line.startsWith('event: ')) {
      eventType = line.slice(7)
    } else if (line.startsWith('data: ')) {
      dataLines.push(line.slice(6))
    }
    // ignore retry:, comments, blank lines
  }

  if (!eventType || dataLines.length === 0) return null

  try {
    return { id: Number.isFinite(eventId) ? eventId : undefined, event: eventType, data: JSON.parse(dataLines.join('\n')) }
  } catch (e) {
    if (DEBUG) console.warn('[SSE] JSON parse failed', dataLines.join('\n').slice(0, 120))
    return null
  }
}

export function useSSE() {
  const isConnected = ref(false)
  let abortController: AbortController | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let reconnectCount = 0
  const store = useTaskStore()
  let lastEventId = 0
  const seenEventIds = new Set<number>()
  let connectedTaskId: string | null = null

  const MAX_RECONNECT = 5
  let connecting = false  // guard against double-connect

  async function connect(taskId: string) {
    if (connecting) {
      return
    }
    connecting = true
    if (connectedTaskId !== taskId) {
      connectedTaskId = taskId
      lastEventId = 0
      seenEventIds.clear()
      reconnectCount = 0
    }

    // Cancel any existing connection (but keep connecting=true)
    if (abortController) abortController.abort()
    abortController = null
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    isConnected.value = false
    // Don't reset reconnectCount here — keep it for reconnection limiting

    const url = getStreamUrl(taskId)
    lastEventId = Math.max(lastEventId, store.result ? lastEventId : 0)

    abortController = new AbortController()
    isConnected.value = true
    // The task snapshot determines status; opening a stream has no state side effect.

    try {
      // No custom headers — keep it a "simple" CORS request (no preflight)
      const task = await getTask(taskId)
      store.hydrate(task)
      lastEventId = Math.max(lastEventId, task.last_event_id || 0)
      const response = await fetch(url, {
        signal: abortController.signal,
        headers: lastEventId ? { 'Last-Event-ID': String(lastEventId) } : undefined,
      })

      if (DEBUG) console.log('[SSE] response', response.status, response.headers.get('content-type'))

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('Response body is not readable (CORS may be blocking)')
      }

      isConnected.value = true
      reconnectCount = 0  // connection healthy — reset reconnection counter

      const decoder = new TextDecoder()
      let buffer = ''

      let streamOpen = true
      while (streamOpen) {
        const { done, value } = await reader.read()
        if (done) {
          if (DEBUG) console.log('[SSE] stream ended')
          streamOpen = false
          break
        }

        buffer += decoder.decode(value, { stream: true })

        // SSE events are delimited by \n\n
        const parts = buffer.split('\n\n')
        buffer = parts.pop() || ''

        for (const part of parts) {
          const trimmed = part.trim()
          if (!trimmed) continue

          const parsed = parseSSEEvent(trimmed)
          if (!parsed) continue

          if (DEBUG && parsed.event === 'keepalive') {
            console.log('[SSE] keepalive')
          } else if (DEBUG) {
            console.log('[SSE] event:', parsed.event)
          }

          handleEvent(parsed.event, parsed.data, parsed.id)
        }
      }

      // Process remaining buffer (shouldn't normally happen)
      if (buffer.trim()) {
        const parsed = parseSSEEvent(buffer.trim())
        if (parsed) handleEvent(parsed.event, parsed.data, parsed.id)
      }
    } catch (err: any) {
      if (err.name === 'AbortError') {
        if (DEBUG) console.log('[SSE] aborted')
        connecting = false
        return
      }

      if (DEBUG) console.error('[SSE] fetch error:', err.message || err)

      isConnected.value = false

      // Check if task already finished while we were disconnected
      try {
        const taskStatus = await getTask(taskId)
        if (taskStatus.status === 'done' && taskStatus.result) {
          store.setResult(taskStatus.result)
          connecting = false
          return
        }
        if (['failed', 'rejected', 'cancelled', 'expired'].includes(taskStatus.status)) {
          store.hydrate(taskStatus)
          store.error = { message: taskStatus.error || '任务已结束', recoverable: false }
          connecting = false
          return
        }
      } catch {
        // Fall through to reconnect
      }

      if (reconnectCount < MAX_RECONNECT) {
        const delay = 3000 * (reconnectCount + 1)
        reconnectCount++
        store.addNotice('warning', `连接中断，${delay / 1000}s 后重连...`)
        connecting = false  // release guard before recursive call
        reconnectTimer = setTimeout(() => connect(taskId), delay)
      } else {
        store.addNotice('error', '连接失败，请检查网络后刷新重试')
        store.status = 'failed'
        connecting = false
      }
    }
  }

  function handleEvent(eventType: string, data: Record<string, any>, eventId?: number) {
    if (eventId !== undefined) {
      if (seenEventIds.has(eventId) || eventId <= lastEventId) return
      seenEventIds.add(eventId)
      lastEventId = eventId
    }
    reconnectCount = 0
    switch (eventType) {
      case 'step_update':
        store.addStep(data as StepResult)
        break
      case 'progress':
        store.progress = data.progress ?? store.progress
        if (data.status === 'queued' || data.status === 'analyzing') {
          store.status = data.status
        }
        break
      case 'tool_warning':
        store.addWarning(data as ToolWarning)
        break
      case 'result':
        store.setResult(data as Result)
        disconnect()
        break
      case 'error':
        if (data.cancelled) {
          store.error = { message: data.message || '任务已取消', recoverable: false }
          store.status = 'cancelled'
          disconnect()
          break
        }
        if (data.reason) {
          store.setRejected(data.reason)
          disconnect()
          break
        }
        store.setError(data.message || '未知错误', data.recoverable ?? false)
        if (!data.recoverable) disconnect()
        break
      case 'reasoning_summary':
        store.setStreamingText(data.text || '')
        break
      case 'keepalive':
        break
    }
  }

  function disconnect() {
    connecting = false
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    isConnected.value = false
    reconnectCount = 99
  }

  onUnmounted(() => {
    disconnect()
  })

  return { isConnected, connect, disconnect }
}

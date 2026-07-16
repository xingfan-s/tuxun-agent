export interface ToolStats {
  total_calls: number
  success: number
  timeout: number
  failed: number
}

export interface Result {
  address: string
  country: string
  province: string | null
  city: string | null
  district: string | null
  lat: number
  lng: number
  confidence: number
  reasoning: string
  tokens_used: number
  total_elapsed_ms: number
  tool_stats: ToolStats | null
}

export type StepType =
  | 'safety_check' | 'exif' | 'vision' | 'clue_extraction'
  | 'tool_call' | 'reasoning' | 'final'

export interface StepResult {
  step: number
  type: StepType
  label: string
  status: 'running' | 'done' | 'error'
  data: Record<string, any>
  elapsed_ms: number
}

export type TaskStatusType = 'uploaded' | 'analyzing' | 'done' | 'failed' | 'rejected'

export interface TaskStatus {
  task_id: string
  status: TaskStatusType
  progress: number
  steps: StepResult[]
  result: Result | null
  error: string | null
  error_recoverable: boolean | null
  safety_reason: string | null
  created_at: string
}

export interface UploadResponse {
  task_id: string
  status: string
  created_at: string
}

export interface SSEEvent {
  event: 'step_update' | 'progress' | 'tool_warning' | 'result' | 'error'
  data: Record<string, any>
}

export interface ToolWarning {
  tool: string
  reason: string
  message: string
}

export type NoticeLevel = 'info' | 'warning' | 'error'

export interface Notice {
  id: number
  level: NoticeLevel
  message: string
  closable: boolean
  duration?: number
}

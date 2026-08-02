export interface ToolStats {
  total_calls: number
  success: number
  timeout: number
  failed: number
  unavailable: number
  budget_skipped: number
  invalid_input: number
  upstream_error: number
  empty_result: number
}

export interface Evidence {
  source: string
  direction: 'support' | 'contradict' | 'context'
  locality: 'global' | 'country' | 'province' | 'city' | 'district' | 'road' | 'poi'
  reliability: number
  raw_score: number | null
  calibrated_contribution: number | null
  summary: string
  unique: boolean
  metadata: Record<string, unknown>
}

export interface Hypothesis {
  province: string
  score: number
  evidence_count: number
  selected?: boolean
}

export interface Result {
  address: string
  country: string
  province: string | null
  city: string | null
  district: string | null
  lat: number | null
  lng: number | null
  coord_system: 'WGS84' | 'GCJ-02' | 'BD-09' | 'unknown'
  precision_level: 'country' | 'province' | 'city' | 'district' | 'road' | 'poi' | 'unknown'
  uncertainty_radius_m: number | null
  confidence: number | null
  confidence_kind: 'calibrated' | 'ranking_score' | 'unknown'
  reasoning: string
  tokens_used: number
  model_calls: number
  model_usage: Record<string, { calls: number; completion_tokens: number }>
  estimated_cost: number
  total_elapsed_ms: number
  tool_stats: ToolStats | null
  evidence: Evidence[]
  top_hypotheses?: Hypothesis[]
}

export type StepType =
  | 'safety_check' | 'exif' | 'vision_macro' | 'geoclip'
  | 'clip_search' | 'anchor_search' | 'geoclip_anchor' | 'vision_detail'
  | 'clue_extraction' | 'ocr' | 'ocr_fusion'
  | 'search_strategy' | 'tool_call' | 'reasoning'
  | 'verification' | 'final' | 'fine_localize' | 'result_enrichment'

export interface StepResult {
  step: number
  type: StepType
  label: string
  status: 'running' | 'done' | 'error'
  data: Record<string, any>
  elapsed_ms: number
}

export type TaskStatusType = 'uploaded' | 'queued' | 'analyzing' | 'done' | 'failed' | 'rejected' | 'cancelled' | 'expired'

export interface TaskStatus {
  task_id: string
  status: TaskStatusType
  progress: number
  steps: StepResult[]
  result: Result | null
  error: string | null
  error_recoverable: boolean | null
  safety_reason: string | null
  cancel_requested: boolean
  last_event_id: number
  created_at: string
}

export interface UploadResponse {
  task_id: string
  status: string
  created_at: string
}

export interface SSEEvent {
  id?: number
  event: 'step_update' | 'progress' | 'tool_warning' | 'result' | 'error' | 'reasoning_summary'
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

import axios from 'axios'
import type { UploadResponse, TaskStatus } from '@/types'

// Use relative path → Vite dev proxy → localhost:8000 (same-origin, no CORS, no extension interference)
const apiBase = import.meta.env.VITE_API_BASE_URL || '/api'

const api = axios.create({
  baseURL: apiBase,
  timeout: 30000,
})

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<UploadResponse>('/upload', form, {
    timeout: 120000,  // 2min for large files through WSL proxy
  })
  return data
}

export async function getTask(taskId: string): Promise<TaskStatus> {
  const { data } = await api.get<TaskStatus>(`/task/${taskId}`)
  return data
}

export async function startTask(taskId: string): Promise<TaskStatus> {
  const { data } = await api.post<TaskStatus>(`/task/${taskId}/start`)
  return data
}

export async function deleteTask(taskId: string): Promise<void> {
  await api.delete(`/task/${taskId}`)
}

export function getStreamUrl(taskId: string): string {
  return `${apiBase}/task/${taskId}/stream`
}

export function getImageUrl(taskId: string): string {
  return `${apiBase}/task/${taskId}/image`
}

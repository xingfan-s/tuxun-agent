import axios from 'axios'
import type { UploadResponse, TaskStatus } from '@/types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  timeout: 30000,
})

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post<UploadResponse>('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function getTask(taskId: string): Promise<TaskStatus> {
  const { data } = await api.get<TaskStatus>(`/task/${taskId}`)
  return data
}

export async function deleteTask(taskId: string): Promise<void> {
  await api.delete(`/task/${taskId}`)
}

export function getStreamUrl(taskId: string): string {
  const base = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
  return `${base}/task/${taskId}/stream`
}

export function getImageUrl(taskId: string): string {
  const base = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api'
  return `${base}/task/${taskId}/image`
}

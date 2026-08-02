export const MAX_IMAGE_BYTES = 20 * 1024 * 1024

const SUPPORTED_IMAGE_TYPES = new Set([
  'image/jpeg',
  'image/png',
  'image/webp',
])

export async function validateImageFile(file: File): Promise<string | null> {
  if (!SUPPORTED_IMAGE_TYPES.has(file.type)) {
    return '不支持的文件格式，仅支持 JPG/PNG/WebP'
  }
  if (file.size > MAX_IMAGE_BYTES) {
    return '文件大小超过 20MB 限制'
  }

  const decode = globalThis.createImageBitmap
  if (typeof decode !== 'function') return null

  try {
    const bitmap = await decode(file)
    bitmap.close()
    return null
  } catch {
    return '文件内容不是有效的图片格式（JPG/PNG/WebP）'
  }
}

export function resetNativeFileInput(root: Element | null | undefined): void {
  const input = root?.querySelector<HTMLInputElement>('input[type="file"]')
  if (input) input.value = ''
}

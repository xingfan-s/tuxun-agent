import { afterEach, describe, expect, it, vi } from 'vitest'
import { MAX_IMAGE_BYTES, resetNativeFileInput, validateImageFile } from './imageValidation'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('validateImageFile', () => {
  it('rejects unsupported MIME types', async () => {
    const file = new File(['text'], 'notes.txt', { type: 'text/plain' })
    await expect(validateImageFile(file)).resolves.toContain('不支持的文件格式')
  })

  it('rejects files larger than 20MB', async () => {
    const file = new File(['image'], 'large.jpg', { type: 'image/jpeg' })
    Object.defineProperty(file, 'size', { value: MAX_IMAGE_BYTES + 1 })
    await expect(validateImageFile(file)).resolves.toContain('20MB')
  })

  it('rejects image MIME spoofing when decoding fails', async () => {
    vi.stubGlobal('createImageBitmap', vi.fn().mockRejectedValue(new Error('decode failed')))
    const file = new File(['not an image'], 'fake.jpg', { type: 'image/jpeg' })
    await expect(validateImageFile(file)).resolves.toContain('文件内容不是有效的图片格式')
  })

  it('accepts a decodable supported image', async () => {
    const close = vi.fn()
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue({ close }))
    const file = new File(['image'], 'valid.webp', { type: 'image/webp' })
    await expect(validateImageFile(file)).resolves.toBeNull()
    expect(close).toHaveBeenCalledOnce()
  })
})

describe('resetNativeFileInput', () => {
  it('clears the underlying file input value', () => {
    const input = { value: 'C:\\fakepath\\bad.jpg' }
    const root = { querySelector: () => input } as unknown as Element

    resetNativeFileInput(root)

    expect(input.value).toBe('')
  })
})

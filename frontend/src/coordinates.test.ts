import { describe, expect, it } from 'vitest'
import { wgs84ToGcj02 } from './coordinates'

describe('coordinate adapter', () => {
  it('leaves coordinates outside China unchanged', () => {
    expect(wgs84ToGcj02(40.7, -74)).toEqual([40.7, -74])
  })

  it('converts a WGS84 point for AMap display', () => {
    const [lat, lng] = wgs84ToGcj02(39.9042, 116.4074)
    expect(lat).toBeGreaterThan(39.9)
    expect(lng).toBeGreaterThan(116.4)
  })
})

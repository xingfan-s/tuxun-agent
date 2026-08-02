<template>
  <div class="map-panel" v-if="hasCoords">
    <div v-if="mapUnavailable" class="map-fallback" role="status">
      <el-icon :size="28"><Location /></el-icon>
      <div>
        <strong>推测坐标</strong>
        <span>{{ resultLat.toFixed(5) }}, {{ resultLng.toFixed(5) }}</span>
        <small>地图服务未配置</small>
      </div>
    </div>
    <div v-show="!mapUnavailable" :id="mapId" class="map-container"></div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { useTaskStore } from '@/stores/task'
import { wgs84ToGcj02 } from '@/coordinates'
import { Location } from '@element-plus/icons-vue'

const store = useTaskStore()

const mapId = `tuxun-map-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
const mapInstance = ref<any>(null)
const mapUnavailable = ref(false)

const hasCoords = computed(() => {
  const r = store.result
  return !!r && r.lat !== null && r.lng !== null
})

const resultLat = computed(() => store.result?.lat ?? 0)
const resultLng = computed(() => store.result?.lng ?? 0)

const circleRadius = computed(() => {
  return store.result?.uncertainty_radius_m ?? 0
})

// Watch the entire result object, not just lat — in case result is replaced
watch(() => store.result, () => {
  nextTick(() => initMap())
}, { deep: true })

onMounted(() => {
  nextTick(() => initMap())
})

onBeforeUnmount(() => {
  destroyMap()
})

function destroyMap() {
  if (mapInstance.value) {
    try {
      mapInstance.value.destroy()
    } catch {
      // ignore
    }
    mapInstance.value = null
  }
}

async function initMap() {
  if (!hasCoords.value) return
  const el = document.getElementById(mapId)
  if (!el) return

  // Destroy previous map before creating a new one on the same element
  destroyMap()
  mapUnavailable.value = false
  // Clear any leftover DOM from previous map render
  el.innerHTML = ''

  try {
    const { lazyAMapApiLoaderInstance } = await import('@vuemap/vue-amap')
    if (!lazyAMapApiLoaderInstance) {
      mapUnavailable.value = true
      return
    }

    const AMap = await lazyAMapApiLoaderInstance
    if (!AMap) {
      mapUnavailable.value = true
      return
    }

    const [lat, lng] = wgs84ToGcj02(resultLat.value, resultLng.value)

    const map = new AMap.Map(el, {
      center: [lng, lat],
      zoom: 10,
      resizeEnable: true,
    })

    const marker = new AMap.Marker({
      position: [lng, lat],
      title: store.result?.address || '推测位置',
      icon: new AMap.Icon({
        size: new AMap.Size(32, 42),
        image: 'https://webapi.amap.com/theme/v1.3/markers/n/mark_r.png',
        imageSize: new AMap.Size(32, 42),
      }),
    })
    map.add(marker)

    const circle = new AMap.Circle({
      center: [lng, lat],
      radius: circleRadius.value,
      strokeColor: '#409eff',
      strokeWeight: 2,
      strokeOpacity: 0.5,
      fillColor: '#409eff',
      fillOpacity: 0.12,
    })
    if (circleRadius.value > 0) map.add(circle)

    mapInstance.value = map
  } catch {
    mapUnavailable.value = true
  }
}
</script>

<style scoped>
.map-panel {
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #e4e7ed;
}
.map-container {
  width: 100%;
  height: 320px;
}
.map-fallback {
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #606266;
  background: #fafbfc;
}
.map-fallback > div {
  display: grid;
  gap: 4px;
}
.map-fallback span { font-variant-numeric: tabular-nums; }
.map-fallback small { color: #909399; }
@media (max-width: 767px) {
  .map-container { height: 260px; }
}
</style>

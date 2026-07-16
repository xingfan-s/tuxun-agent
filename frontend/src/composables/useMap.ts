import { ref, shallowRef } from 'vue'

interface MapInstance {
  setCenter(center: [number, number]): void
  setZoom(zoom: number): void
  addMarker(marker: { position: [number, number]; title?: string; icon?: string }): void
  clearMarkers(): void
}

export function useMap() {
  const mapInstance = shallowRef<MapInstance | null>(null)
  const markers = ref<Array<{ position: [number, number]; title: string; type: string }>>([])

  function initMap(containerId: string) {
    console.log('Map init:', containerId)
  }

  function setResultMarker(lat: number, lng: number, address: string) {
    markers.value.push({ position: [lng, lat], title: address, type: 'result' })
    flyTo(lat, lng, 14)
  }

  function setConfidenceCircle(lat: number, lng: number, radius: number) {
    console.log('Confidence circle:', lat, lng, radius)
  }

  function addClueMarker(lat: number, lng: number, label: string) {
    markers.value.push({ position: [lng, lat], title: label, type: 'clue' })
  }

  function flyTo(lat: number, lng: number, zoom: number) {
    console.log('Fly to:', lat, lng, zoom)
  }

  function clear() {
    markers.value = []
  }

  return { mapInstance, markers, initMap, setResultMarker, setConfidenceCircle, addClueMarker, flyTo, clear }
}

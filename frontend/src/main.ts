import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'
import router from './router'
import axios from 'axios'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(ElementPlus)
app.mount('#app')

// Fetch AMap key from backend and init loader
const apiBase = import.meta.env.VITE_API_BASE_URL || '/api'
axios.get(`${apiBase}/config`).then(({ data }) => {
  const key = data.amap_api_key
  if (key) {
    import('@vuemap/vue-amap').then(({ initAMapApiLoader }) => {
      initAMapApiLoader({
        key,
        version: '2.0',
      })
    })
  }
}).catch(() => {
  // Config fetch failed — map will be unavailable, other features unaffected
})

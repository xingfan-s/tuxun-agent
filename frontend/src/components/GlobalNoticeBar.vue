<template>
  <div class="global-notice-bar" v-if="store.notices.length > 0">
    <el-alert
      v-for="notice in store.notices"
      :key="notice.id"
      :type="notice.level"
      :closable="notice.closable"
      @close="store.removeNotice(notice.id)"
      show-icon
      class="notice-item"
    >
      {{ notice.message }}
    </el-alert>
  </div>
</template>

<script setup lang="ts">
import { useTaskStore } from '@/stores/task'
const store = useTaskStore()
</script>

<style scoped>
.global-notice-bar { margin-bottom: 16px; }
.notice-item { margin-bottom: 8px; }
</style>

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5274,
    proxy: {
      '/analyze': {
        target: 'http://127.0.0.1:8200',
        changeOrigin: false,
        // LLM 기반 보고서 생성은 오래 걸릴 수 있어 10분까지 대기한다.
        timeout: 600000,
        proxyTimeout: 600000,
      },
      '/settings': {
        target: 'http://127.0.0.1:8200',
        changeOrigin: false,
        timeout: 30000,
        proxyTimeout: 30000,
      },
    },
  },
})

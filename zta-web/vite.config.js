import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Todas las llamadas a /api se reenvían a la API de Python: así el navegador no necesita CORS.
const target = process.env.VITE_PROXY_TARGET || 'http://127.0.0.1:8000'
const proxy = {
  '/api': { target, changeOrigin: true, rewrite: (path) => path.replace(/^\/api/, '') },
}

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
})

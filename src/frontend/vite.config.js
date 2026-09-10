import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Build output goes to dist/, which is committed so a judge needs no Node.
// The dev server proxies API calls to uvicorn on 8000 so `npm run dev` and
// `uvicorn src.api:app` can run side by side.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/api': 'http://127.0.0.1:8000',
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  root: path.resolve(__dirname, './src/ui'),
  build: {
    outDir: path.resolve(__dirname, './dist-react'),
    emptyOutDir: true,
    rollupOptions: {
      input: path.resolve(__dirname, './src/ui/index.html')
    }
  },
  base: './',
    plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
      },
    },
  },
})
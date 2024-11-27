import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'static/dist',
    rollupOptions: {
      input: 'static/js/main.jsx',
      output: {
        entryFileNames: 'bundle.js',
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './static/js'),
    },
  },
});
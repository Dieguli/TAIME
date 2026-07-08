import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    build: {
        // Output to FastAPI static directory
        outDir: '../api/static',
        emptyOutDir: true,
    },
    server: {
        port: 5173,
        // Proxy API requests to FastAPI backend during development
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
            '/health': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
});

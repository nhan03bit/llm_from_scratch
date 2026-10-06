import devServer from '@hono/vite-dev-server'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Client bundle, served by the Hono app from dist/static/
  if (mode === 'client') {
    return {
      build: {
        rolldownOptions: {
          input: 'src/client/index.tsx',
          output: {
            entryFileNames: 'static/client.js',
            assetFileNames: 'static/[name].[ext]',
          },
        },
      },
    }
  }

  return {
    plugins: [devServer({ entry: 'src/index.tsx' })],
    build: {
      ssr: 'src/index.tsx',
      emptyOutDir: false,
    },
  }
})

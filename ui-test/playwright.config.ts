import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:8080',
    trace: 'on-first-retry'
  },
  webServer: {
    command: 'python ../serve.py',
    url: 'http://127.0.0.1:8080/demo/',
    reuseExistingServer: true,
    timeout: 120000
  }
});

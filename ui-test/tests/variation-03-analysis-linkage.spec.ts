import { expect, test } from '@playwright/test';

test('Variation 3: Ops variant keeps sticky nav and CTA-driven alert toggle', async ({ page }) => {
  await page.goto('/ui-test/fixtures/variation-03-ops.html');

  const headerPosition = await page.evaluate(() => getComputedStyle(document.querySelector('.header') as Element).position);
  expect(headerPosition).toBe('sticky');

  await expect(page.locator('#alert-state')).toHaveText('Status: Passive Monitoring');
  await page.locator('#toggle-alert').click();
  await expect(page.locator('#alert-state')).toHaveText('Status: Active Alerting (DI > 70)');

  const arrow = page.locator('#toggle-alert .arrow');
  const before = await arrow.evaluate((el) => getComputedStyle(el).transform);
  await page.locator('#toggle-alert').hover();
  const after = await arrow.evaluate((el) => getComputedStyle(el).transform);

  expect(before).toBe('none');
  expect(after).not.toBe('none');

  const footerColor = await page.evaluate(() => getComputedStyle(document.querySelector('.footer') as Element).backgroundColor);
  expect(footerColor).toBe('rgb(38, 33, 30)');
});

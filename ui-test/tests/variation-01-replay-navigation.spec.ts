import { expect, test } from '@playwright/test';

test('Variation 1: EnneaFlow replay page keeps style constraints and round flow', async ({ page }) => {
  await page.goto('/ui-test/fixtures/variation-01-replay.html');

  const bg = await page.evaluate(() => getComputedStyle(document.body).backgroundColor);
  expect(bg).toBe('rgb(250, 247, 242)');

  const ctaRadius = await page.evaluate(() => getComputedStyle(document.querySelector('.cta') as Element).borderRadius);
  expect(ctaRadius).toBe('0px');

  const squareTransform = await page.evaluate(() => getComputedStyle(document.getElementById('hero-square') as Element).transform);
  expect(squareTransform).not.toBe('none');

  await expect(page.locator('#round-state')).toHaveText('Round 1 / 3');
  await page.locator('#next-round').click();
  await expect(page.locator('#round-state')).toHaveText('Round 2 / 3');
  await expect(page.locator('#choice-state')).toContainText('Deception Index: 52');

  await page.locator('#next-round').click();
  await expect(page.locator('#round-state')).toHaveText('Round 3 / 3');
  await expect(page.locator('#choice-state')).toContainText('A: STEAL | B: STEAL');
});

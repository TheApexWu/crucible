import { expect, test } from '@playwright/test';

test('Variation 2: Ennea-Line analysis interactions and terracotta hover behavior', async ({ page }) => {
  await page.goto('/ui-test/fixtures/variation-02-analysis.html');

  const items = page.locator('#events .ennea-item');
  await expect(items).toHaveCount(3);

  const spineWidth = await page.evaluate(() => {
    const node = document.querySelector('.ennea-line');
    return getComputedStyle(node as Element, '::before').width;
  });
  expect(spineWidth).toBe('2px');

  const icon = page.locator('#events .ennea-item').first().locator('.ennea-icon');
  const preHover = await icon.evaluate((el) => getComputedStyle(el).borderColor);
  await items.first().hover();
  const postHover = await icon.evaluate((el) => getComputedStyle(el).borderColor);

  expect(preHover).toBe('rgba(26, 26, 26, 0.05)');
  expect(postHover).toBe('rgb(224, 120, 86)');

  await items.nth(1).click();
  await expect(page.locator('#event-detail')).toHaveText('Round 12: Agent A starts grim trigger');
});

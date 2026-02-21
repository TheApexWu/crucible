# ui-test

EnneaFlow-based frontend UI test repository for CRUCIBLE.

## Context Adaptation
CRUCIBLE is a deception-lab app with replay, strategy analysis, and observability workflows. These samples are not generic landing-page tests: each variation maps directly to one of those product surfaces and validates both behavior and visual-system constraints.

## What is included
- Shared EnneaFlow design tokens and component rules: `fixtures/enneaflow.css`
- Three CRUCIBLE-adapted UI variations:
- `fixtures/variation-01-replay.html`
- `fixtures/variation-02-analysis.html`
- `fixtures/variation-03-ops.html`
- Three Playwright tests (one per variation):
- `tests/variation-01-replay-navigation.spec.ts`
- `tests/variation-02-metrics-integrity.spec.ts`
- `tests/variation-03-analysis-linkage.spec.ts`
- Plan-first documentation: `docs/variation-plans.md`

## Run
```bash
cd /Users/evancorrea/Hackathons/SIA_Datadog_02_21_26/crucible/ui-test
npm install
npx playwright install
npm test
```

## React decision
Recommendation: use vanilla HTML/CSS/JS for these current variations, and defer React for now.

Why:
- These samples are mostly document-driven experiences with light interaction (state toggles, event detail swap, round stepping).
- Fast iteration on visual direction is easier with static files during design exploration.
- A React migration is higher ROI only when you need shared complex state across replay/analysis/ops panels with reusable data-driven components.

Use React when:
- You introduce real-time streaming updates (websocket state fan-out).
- You need synchronized multi-panel state and URL routing across many views.
- You want a reusable component library beyond prototype/testing scope.

# sessions_v2

Harness v2 agent logs kept on GitHub: **resolved** or **hit the step cap**, with at least one real fail → edit arc.

Left out: pull/setup errors, stuck loops (no useful edits), and `django-16408` (cap with 0 edits).

## Resolved

| Task | Notes |
|---|---|
| django__django-10924 | 3 X* points |
| django__django-12497 | 1 X* |
| django__django-12747 | 2 X*, 4 stages |
| django__django-13230 | 2 X* |
| django__django-15789 | 1 X* |
| django__django-16041 | 1 X* |
| matplotlib__matplotlib-25498 | 4 X* |
| sympy__sympy-24213 | 1 X* |
| pylint-dev__pylint-5859 | 1 X*; session passed its stage, official final flag is `resolved=False` because of a test-file naming quirk |

## Hit step cap (still useful F→E data)

| Task | X* points |
|---|---|
| django__django-11283 | 11 |
| django__django-13265 | 6 |
| django__django-13315 | 2 |
| django__django-13447 | 1 |
| django__django-14608 | 4 |
| pytest-dev__pytest-8365 | 8 |
| scikit-learn__scikit-learn-14087 | 11 |
| sympy__sympy-12454 | 4 |
| sympy__sympy-14308 | 8 |
| sympy__sympy-14817 | 12 |
| sympy__sympy-15308 | 12 |
| sympy__sympy-19007 | 2 |
| sympy__sympy-20442 | 7 |

Labels: `data/labeled/labeled_sessions_v2.json`. X* points: `data/labeled/xstar_v2.json` (104 points from these 22 sessions).

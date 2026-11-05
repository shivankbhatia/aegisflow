# Retrieval failure review

## Primary 15 multi-hop questions

The quota-free dense/structure-aware k=10 baseline retrieved all designated
gold documents for **12/15** questions. The three document-level misses:

- `m16_006`: retrieved `fastapi/tutorial/extra-models` but not
  `fastapi/advanced/advanced-python-types`. The `extra-models` page contains
  the PlaneItem/CarItem `Union` example and explains the `|` runtime error, so
  this is partly a strict duplicate-source gold-label issue.
- `mhand_001`: retrieved Psycopg pool pages and `fastapi/async`, but not
  `fastapi/advanced/events`, the designated FastAPI lifespan source.
- `mhand_003`: retrieved `fastapi/advanced/advanced-dependencies` plus the
  Psycopg pool docs, but not `fastapi/tutorial/dependencies/dependencies-with-yield`.
  The advanced dependency page covers closely related `yield` cleanup behavior.

These failures indicate both retrieval misses and gold-document redundancy.
Keep the requested all-gold-document rate as the objective metric, while
reviewing source equivalence before treating the metric as answerability.

## Independently authored multi-hop questions

The new source-checked set retrieved all designated gold documents for
**13/15**. Misses:

- `mnew_001`: found `fastapi/tutorial/extra-models`, which directly discusses
  the union response-model example and why `|` fails, but missed the related
  `advanced-python-types` page.
- `mnew_005`: found `fastapi/advanced/websockets` and
  `fastapi/reference/testclient`, but missed the specific
  `fastapi/advanced/testing-websockets` page.

No synthesis, judge, or quota-consuming model call was used for either report.

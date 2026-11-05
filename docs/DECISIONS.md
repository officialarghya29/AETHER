# Decisions — AETHER-ML

Each entry records the decision, why it was made, and what was rejected. Decisions are dated by the
v1.0.0 build.

## D-001 — One orchestration pipeline

**Decision:** every caller uses `ReliabilityPipeline`.
**Why:** the most common failure in reliability tooling is a demo path that differs from the served
path. Benchmark numbers would then describe code that never runs in production.
**Rejected:** per-endpoint inference logic; a separate research script that reimplements routing.

## D-002 — Monorepo modular monolith

**Decision:** one repository with `aether/`, `backend/`, `training/`, `experiments/`, `frontend/`.
**Why:** the components share types and artifacts; splitting them into services would add deployment
and versioning overhead with no demonstrated requirement.
**Rejected:** microservices, separate model-serving runtime.

## D-003 — Fine-tune from ImageNet at 64 px on CPU

**Decision:** MobileNetV3-Small (10 epochs) and ResNet50 (6 epochs) fine-tuned from pretrained
weights at 64 px.
**Why:** measured CPU training cost (3.47 ms/img lightweight, 9.34 ms/img advanced) made full CIFAR-10
training feasible without a GPU, while pretrained initialisation provides a strong base.
**Rejected:** training from scratch (slow, weaker), 224 px (unnecessary cost for 32 px inputs),
larger architectures (inference budget).

## D-004 — Dataset via Hugging Face Parquet mirror

**Decision:** fetch the official release; when the Toronto host is unreachable, convert the
`uoft-cs/cifar10` Parquet mirror into identical arrays and fingerprint the result.
**Why:** the official host delivered ~16 KB/s in this environment. Provenance is preserved by
recording the source and SHA-256 of images and labels.
**Rejected:** training on a subset (weakens the study), silently substituting a different dataset.

## D-005 — Five-way split with disjoint purposes

**Decision:** train 32k / validation 4k / calibration 4k / routing_dev 5k / holdout 5k.
**Why:** temperature fitting, policy development and drift reference construction each need data that
is not used for anything else; otherwise reported calibration or routing gains are contaminated.
**Rejected:** a single validation split reused for calibration and thresholds.

## D-006 — PSI + KS + MMD, combined with Bonferroni

**Decision:** per-feature PSI and KS on input statistics, permutation MMD on PCA embeddings,
behaviour tests on predictions; composite = max adjusted `-log10 p`.
**Why:** each test has different sensitivity; taking the maximum adjusted evidence is conservative
and interpretable. Permutation p-values avoid distributional assumptions for MMD.
**Rejected:** a single test presented as a universal drift detector; unadjusted p-values across
many features (inflated false positives).

## D-007 — Drift never implies accuracy loss

**Decision:** drift reports describe distribution movement only; accuracy impact is reported only
where labels exist.
**Why:** unlabelled shift is not evidence of error. Conflating them is the most common scientific
error in drift dashboards.
**Rejected:** converting drift scores into "accuracy drop" estimates.

## D-008 — Logistic risk models, not an LLM

**Decision:** error risk is a fitted logistic regression over six evidence features.
**Why:** deterministic, inspectable, cheap, trainable on 5k samples, and it cannot invent evidence.
The prompt explicitly forbids an LLM as the routing mechanism.
**Rejected:** gradient-boosted trees (marginal gain, extra dependency), neural router (data-hungry,
opaque), LLM routing (non-deterministic, unjustifiable).

## D-009 — Separate estimated and evaluated advanced risk

**Decision:** `RoutingState.risk_advanced` may hold a prediction, but `advanced_evaluated` must be
true before the rule policy returns the advanced answer.
**Why:** a pre-call estimate is useful for pricing escalation; treating it as a completed evaluation
lets the router accept a prediction the advanced model never made. This was found by a failing test.
**Rejected:** inferring evaluation status from `escalation_count` alone (fragile and implicit).

## D-010 — Explicit feasibility before utility

**Decision:** unaffordable or unavailable actions are removed from the option set, not merely
penalised in the utility.
**Why:** a large utility could otherwise buy an action that violates the latency budget or the
escalation cap. An over-budget answer is a failure, not a slow success.
**Rejected:** soft latency penalties.

## D-011 — Ten corruption families, not fifteen

**Decision:** implement gaussian/shot/impulse noise, defocus/gaussian/motion/zoom blur, brightness,
contrast, pixelate and JPEG; omit fog, frost, snow, spatter and elastic.
**Why:** faithful implementations of the omitted families are non-trivial to reproduce correctly;
shipping approximate versions would misrepresent CIFAR-10-C coverage.
**Rejected:** name-only stubs, or claiming full coverage.

## D-012 — Review gate before data

**Decision:** only resolved cases with a reviewer identity and an in-range label enter a candidate
dataset; promotion additionally needs a passing gate and a named approver.
**Why:** unverified feedback is a poisoning vector and an audit hazard.
**Rejected:** auto-adding feedback to training data.

## D-013 — SQLite default, PostgreSQL for deployment

**Decision:** SQLite for tests and local development; PostgreSQL via `DATABASE_URL` in compose.
**Why:** fast local iteration with a real SQLAlchemy/Alembic schema, and a production-grade database
without code changes.
**Rejected:** requiring PostgreSQL for `make test`; using an in-memory fake that hides migration bugs.

## D-014 — Canonical 32-character prediction ids

**Decision:** generate ids as `uuid4().hex` at the pipeline boundary.
**Why:** the database column is 32 characters, and previously the API returned a dashed UUID that
could never be looked up — every uncertainty/diagnostic request 404'd. One canonical form removes the
class of bug.
**Rejected:** widening the column and keeping two formats.

## D-015 — Rank-preserving uncertainty helpers

**Decision:** entropy/MSP/margin accept 1-D or 2-D input and return a scalar or array accordingly.
**Why:** the pipeline passes single vectors and batches; a length-1 array leaking into `float()`
crashed every prediction. Rank preservation makes the helpers safe in both contexts.
**Rejected:** requiring every caller to wrap inputs in a batch dimension.

## D-016 — Idempotent runtime configuration

**Decision:** apply `torch.set_num_interop_threads` once per process, guarded and logged.
**Why:** PyTorch refuses a second call after parallel work begins, which broke every second pipeline
construction in tests and would break in-process rebuilds.
**Rejected:** catching the error silently; per-request thread configuration.

## D-017 — Configurable costs, not hidden thresholds

**Decision:** all routing economics live in `routing.costs` with explicit defaults.
**Why:** the optimal policy depends on the cost of an error relative to a review; hiding that in code
makes results unreproducible.
**Rejected:** hardcoded constants.

## D-018 — Dashboard reads only the API

**Decision:** no static demo numbers anywhere in the frontend.
**Why:** a dashboard that fabricates data undermines the product's entire premise. Empty and error
states are explicit.
**Rejected:** seeded mock data for screenshots.

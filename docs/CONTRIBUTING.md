# Contributing — AETHER-ML

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
make data          # only needed for training/evaluation work
```

## 2. Rules that are not negotiable

1. **No label leakage.** Inference-time code may not read labels, test-split paths or future data.
2. **Split discipline.** Temperatures on calibration, policies on routing_dev, report on test.
3. **No fabricated evidence.** If a number is not produced by a command in this repository, it does
   not go in a document.
4. **Honest degradation.** Missing artifacts produce explicit states, never plausible defaults.
5. **No silent exceptions.** Log and re-raise or convert to a domain error.
6. **No placeholders.** A function either works or is absent.

## 3. Style

- Python 3.11+, type hints on public functions, `from __future__ import annotations`.
- Ruff for lint and format (`make lint`, `make format`), line length 100.
- Mypy over `aether/` (`make typecheck`).
- Comments explain *why*; the code explains *what*.
- Keep modules focused: detection logic in `aether/`, HTTP concerns in `backend/`.
- Add a dependency only with a demonstrated need and a compatibility check.

## 4. Tests

- Every behavioural change gets a test at the lowest level that can catch the regression.
- Numerical routines: include a stability case (zeros, extreme values, single class).
- Routing: include a determinism assertion and a feasibility assertion.
- API changes: update `tests/test_pipeline_api.py` and `docs/API.md`.
- Tests must not require the downloaded dataset or the trained checkpoints; use the fixtures in
  `tests/conftest.py`. Never weaken an assertion to make a suite pass.

```bash
make test
python -m pytest tests/test_routing.py -q
```

## 5. Commits

Conventional prefixes: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `perf:`, `ci:`, `build:`.
One logical change per commit; include the failing behaviour in the body when fixing a bug.

## 6. Pull request checklist

- [ ] `make lint` and `make typecheck` pass.
- [ ] `make test` passes.
- [ ] New behaviour has evidence (test, report or both).
- [ ] Documentation updated when behaviour, configuration or the API changes.
- [ ] No dataset, checkpoint, `.env` or credential in the diff.
- [ ] `docs/IMPLEMENTATION_STATUS.md` updated when a phase changes state.

## 7. Adding a model

1. Implement the `ModelAdapter` contract in `aether/adapters/`.
2. Register the architecture and checkpoint metadata loader.
3. Add the role to `configs/aether.yaml` and a test that loads a small checkpoint.
4. Confirm the class ordering matches CIFAR-10 before comparing predictions.
5. Record measured latency; do not assume it.

## 8. Adding a corruption family

1. Add the family to `CORRUPTION_FAMILIES` with an explicit severity table.
2. Guarantee determinism for a fixed seed and monotone severity.
3. Add a test asserting shape, dtype, determinism and severity sensitivity.
4. Update `docs/ALGORITHMS.md` and the coverage statement in `README.md`.

## 9. Reporting a problem

Include the command, the configuration, the observed output and whether artifacts were present.
For numerical disagreements, include the split used and the seed.

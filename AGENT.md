# Agent Operating Manual

## Read Order

1. `docs/notes.md` when it exists.
2. `docs/specs.md` when it exists.
3. `AGENT.md`.
4. The current user spec.

If an ignored docs file exists locally, read it when relevant but do not assume it is tracked.

## Project Purpose

PyNNS is a Python port and eventual Python-native replacement candidate for R NNS. The goal is public input/output semantic compatibility with R NNS where that behavior is stable, documented, and useful. The goal is not to copy R's internal imperative architecture.

## Parity Philosophy

Parity means public behavior matches, not internal middleware.

- Public Python inputs and outputs should match R NNS public behavior where R behavior is stable, documented, and useful.
- Numerical results should match within sensible absolute and relative tolerances.
- Output keys, shapes, labels, signs, class predictions, selected variables, and index orientation matter.
- Internal processing does not need to match R.
- Do not copy R's mutation-heavy loops, data-frame quirks, name-repair quirks, foreach internals, or broken branches unless they are required to reproduce stable public outputs.
- If R is broken or too implicit, PyNNS may intentionally diverge. Document the divergence and cover the Python contract with Python-native tests.
- Small floating-point differences, tiny relative percentage differences, and harmless ordering differences are not important unless they change public semantics.
- Installed R NNS is the parity ground truth. If installed R and `reference/NNS/` source disagree, match installed R and document the divergence.

Use these classifications for known R bugs or intentional differences:

- `R_COMPATIBLE`: Python intentionally matches stable public R behavior.
- `PYTHON_DIVERGENCE`: Python intentionally differs from R and documents/tests its own contract.
- `GUARDED_UNSUPPORTED`: Python rejects the path explicitly.
- `XFAIL_KNOWN_GAP`: Python returns the public structure but parity is not yet matched.

## Where Things Live

- `reference/`: upstream NNS R reference; read-only.
- `src/pynns/`: Python implementation.
- `tests/parity/`: R-vs-Python public behavior parity.
- `tests/invariants/`: mathematical/API invariants, guards, and Python-native contracts.
- `tests/property/`: property-based tests.
- `tests/benchmarks/`: performance tests.
- `tests/_r.py`: R parity bridge and versioned cache reader/writer.
- `tests/_r_cache.json`: committed parity cache for offline/CI use.
- `tests/_tolerances.py`: shared numerical tolerances.
- `docs/`: project context, conventions, deferred paths, benchmark summaries, parity reports, and source material.

## Workflow

1. Read the current user spec.
2. Read relevant docs and current state reports.
3. Read the relevant R source and man page.
4. List the R function's dependencies before implementing anything, including NNS functions, R helpers, and C++ kernels.
5. Sketch expected public Python call behavior.
6. Implement in `src/pynns/`.
7. Add or update parity tests.
8. Add or update invariant tests.
9. Add or update property tests when behavior has useful general laws.
10. Add or update benchmark tests when performance-sensitive.
11. Run `uv run ruff format .`.
12. Run `uv run ruff check . --fix`.
13. Run targeted pytest during development.
14. Run `uv run pytest`.
15. Run `uv run ruff check .`.
16. Run `uv run mypy`.
17. Report paths changed, tests passed/failed, ruff/mypy status, and deviations from spec.

Reports and scripts are diagnostic snapshots only. Enforced behavior belongs in normal tests:

- `tests/parity/`
- `tests/invariants/`
- `tests/property/`
- `tests/benchmarks/`

## Hard Rules

- No `subprocess` in `src/pynns/`.
- No `rpy2` in `src/pynns/`.
- Keep R calls isolated to the test harness.
- Do not modify `reference/`.
- Do not push to a remote unless the user explicitly asks.
- Use `uv`, not `pip`, for project commands.
- Types are required; mypy runs in strict mode.
- Prefer NumPy arrays at numeric API boundaries.
- Keep implementation comments sparse and useful.
- PyNNS is pure Python/NumPy/SciPy for now. Do not add native extensions unless explicitly requested and justified by benchmarks.
- Preserve GPL-3.0-only licensing.
- Watch R C-API distribution functions (`Rf_dexp`, `Rf_dnorm`, `Rf_dbinom`, etc.); verify at least one numeric case against installed R before trusting source-level parameter meaning.
- If a function being ported depends on another NNS function not yet implemented in PyNNS, stop and report. Do not silently substitute another algorithm.
- Before implementing a port, list the R function's dependencies, including other NNS functions, internal C++ kernels, and R helpers. Report any unported NNS dependency before writing code.
- Any new `NotImplementedError` path must be added to `docs/api_status.md` in the same commit.
- Do not accept fake paths. Either faithfully port public semantics, reject explicitly, or stop and report.

## Testing Philosophy

- Parity tests assert public semantic equivalence with R, not internal implementation equivalence.
- Do not compare whole objects or dictionaries when only a public numeric field matters.
- Normalize R/Python structure differences in tests when the semantic output is the same.
- Use tolerances from `tests/_tolerances.py`; do not invent unexplained magic tolerances.
- For stochastic paths, compare shape, keys, distributions, correlations, drift, selected labels, and public statistical behavior. Do not demand exact random streams unless docs give exact deterministic outputs.
- For known R bugs, classify behavior as `R_COMPATIBLE`, `PYTHON_DIVERGENCE`, `GUARDED_UNSUPPORTED`, or `XFAIL_KNOWN_GAP`.

Numeric parity should report or reason with:

- `max_abs_diff`
- `max_rel_pct_masked`
- `p95_rel_pct_masked`
- `median_rel_pct_masked`
- near-zero reference count

## Tolerances

Import tolerances from `tests/_tolerances.py`.

- `EXACT`: direct deterministic parity.
- `COMPOUND`: multi-step deterministic numerical work.
- `STOCHASTIC`: stochastic or simulation-heavy comparisons.

Do not write unexplained hardcoded tolerance values in parity tests.

## Cache

`PYNNS_OFFLINE=1` disables Rscript calls and forces cache-only parity tests.

`CI=true` and `PYNNS_R_CACHE_ONLY=1` also force cache-only mode.

Pytest defaults to 4 xdist workers. Set `PYNNS_PYTEST_WORKERS=<n>` to override the worker count for a local machine or CI runner.

For quick local loops that do not need stochastic structural checks, use `uv run pytest -q -m "not stochastic"`. The full suite still includes stochastic checks.

The cache schema is versioned. Schema mismatches fail. NNS version mismatches warn and refresh entries when online.

## Benchmarks

Default pytest excludes benchmarks. Run benchmarks serially because `pytest-benchmark` is not reliable under xdist:

```bash
uv run pytest -n0 -m benchmark --benchmark-enable \
  --benchmark-json=reports/benchmark_latest.json tests/benchmarks/
uv run python scripts/update_benchmarks_doc.py reports/benchmark_latest.json
```

Benchmark docs report `Python speed vs R = R baseline / Python mean`. Values above `1.00x` mean Python is faster; values below `1.00x` mean Python is slower.

## Reporting Format

Always include:

- Paths changed.
- Tests passed or failed.
- Ruff and mypy status.
- Deviations from the requested spec.

Keep reports terse and concrete unless the user requests a full investigation report.

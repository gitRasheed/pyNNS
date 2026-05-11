# Agent Operating Manual

## Read Order

1. `docs/notes.md`
2. `docs/specs.md` when it exists
3. `AGENT.md`
4. The current user spec

If an ignored docs file exists locally, read it but do not assume it is tracked.

## Where Things Live

- `reference/`: upstream NNS R reference; read-only.
- `src/pynns/`: Python implementation.
- `tests/parity/`: checks against R NNS behavior.
- `tests/invariants/`: mathematical and API invariants.
- `tests/property/`: property-based tests.
- `tests/_r.py`: R parity bridge and versioned cache reader/writer.
- `tests/_r_cache.json`: committed parity cache for offline/CI use.
- `tests/_tolerances.py`: shared numerical tolerances.
- `docs/`: project context, notes, specs, and source material.

## Workflow

1. Read the current spec.
2. Read the relevant R reference code and man page.
3. Sketch behavior at the Python call site.
4. Implement in `src/pynns/`.
5. Add or update parity tests.
6. Add or update invariant tests.
7. Add or update property tests when behavior has useful general laws.
8. Run `uv run pytest`.
9. Run `uv run ruff check .`.
10. Run `uv run mypy`.
11. Report paths changed, checks passed/failed, and deviations from spec.

## Hard Rules

- No `subprocess` in `src/pynns/`.
- No `rpy2` in `src/pynns/`.
- Keep R calls isolated to the test harness.
- Do not modify `reference/`.
- Do not push to a remote.
- Use `uv`, not `pip`, for project commands.
- Types are required; mypy runs in strict mode.
- Keep implementation comments sparse and useful.
- Prefer NumPy arrays at numeric API boundaries.
- Preserve GPL-3.0-only licensing.
- Installed R is ground truth, not the source in `reference/NNS/`; when parity tests fail, match the installed binary and document the divergence.
- Watch R C-API distribution functions (`Rf_dexp`, `Rf_dnorm`, `Rf_dbinom`, etc.); verify at least one numeric case against the installed binary before trusting source-level parameter meaning.
- If a function being ported depends on another NNS function not yet implemented in PyNNS, stop and report; do not substitute a different algorithm or mirror the workflow with different internals.
- Before implementing, list the R function's dependencies, including other NNS functions, internal C++ kernels, and R helpers; report any unported NNS dependency before writing code.
- Any new `NotImplementedError` path must be added to `docs/deferred_paths.md` in the same commit.
- Do not accept fake paths. Either faithfully port, reject explicitly, or stop and report.

## Tolerances

Import tolerances from `tests/_tolerances.py`.

- `EXACT`: direct deterministic parity.
- `COMPOUND`: multi-step deterministic numerical work.
- `STOCHASTIC`: stochastic or simulation-heavy comparisons.

Do not write unexplained magic tolerance numbers in parity tests.

## Cache

`PYNNS_OFFLINE=1` disables Rscript calls and forces cache-only parity tests.

`CI=true` and `PYNNS_R_CACHE_ONLY=1` also force cache-only mode.

Pytest defaults to 4 xdist workers. Set `PYNNS_PYTEST_WORKERS=<n>` to override
the worker count for a local machine or CI runner.

The cache schema is versioned. Schema mismatches fail. NNS version mismatches warn
and refresh entries when online.

## Reporting Format

Always include:

- Paths changed.
- Tests passed or failed.
- Ruff and mypy status.
- Deviations from the requested spec.

Keep the report terse and concrete.

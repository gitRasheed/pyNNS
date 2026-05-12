from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from numpy.typing import NDArray

_BENCHMARK_BASELINE_PATH = Path(__file__).parent / "benchmarks" / "_r_baseline.json"
_BENCHMARK_SCHEMA_VERSION = 1
_NNS_VERSION = "12.0"

JsonValue: TypeAlias = float | int | str | list["JsonValue"] | dict[str, "JsonValue"]
BenchmarkBaseline: TypeAlias = dict[str, JsonValue]

settings.register_profile(
    "fast",
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "thorough",
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "fast"))


def pytest_configure(config: pytest.Config) -> None:
    workers = os.environ.get("PYNNS_PYTEST_WORKERS")
    if workers:
        config.option.numprocesses = workers
    if config.getoption("benchmark_only", default=False):
        config.option.markexpr = "benchmark"


@dataclass(frozen=True)
class EdgeCase:
    name: str
    values: NDArray[np.float64] | NDArray[np.int64]


@pytest.fixture(
    params=[
        EdgeCase("empty", np.array([], dtype=np.float64)),
        EdgeCase("single-element", np.array([1.0], dtype=np.float64)),
        EdgeCase("all-identical", np.array([2.0, 2.0, 2.0], dtype=np.float64)),
        EdgeCase("all-zeros", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        EdgeCase("all-positive", np.array([1.0, 2.0, 3.0], dtype=np.float64)),
        EdgeCase("all-negative", np.array([-1.0, -2.0, -3.0], dtype=np.float64)),
        EdgeCase("contains-nan", np.array([1.0, np.nan, 3.0], dtype=np.float64)),
        EdgeCase("contains-inf", np.array([1.0, np.inf, 3.0], dtype=np.float64)),
        EdgeCase("very-large", np.array([1e15, 2e15, 3e15], dtype=np.float64)),
        EdgeCase("very-small", np.array([1e-15, 2e-15, 3e-15], dtype=np.float64)),
        EdgeCase("integer-dtype", np.array([1, 2, 3], dtype=np.int64)),
    ],
    ids=lambda case: case.name,
)
def edge_case(request: pytest.FixtureRequest) -> EdgeCase:
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def r_baseline() -> BenchmarkBaseline:
    cache = _read_benchmark_baseline()
    if "lpm_small_seconds" not in cache:
        cache["lpm_small_seconds"] = _time_r_lpm()
        _write_benchmark_baseline(cache)
    for n_variables in (10, 50, 100):
        key = f"pm_matrix_{n_variables}x500_seconds"
        if key not in cache:
            cache[key] = _time_r_pm_matrix(n_variables)
            _write_benchmark_baseline(cache)
    if "sd_efficient_set_50x252_degree2_seconds" not in cache:
        cache["sd_efficient_set_50x252_degree2_seconds"] = _time_r_sd_efficient_set()
        _write_benchmark_baseline(cache)
    if "nns_sd_cluster_252x50_degree2_seconds" not in cache:
        cache["nns_sd_cluster_252x50_degree2_seconds"] = _time_r_nns_sd_cluster()
        _write_benchmark_baseline(cache)
    if "nns_dep_1000_seconds" not in cache:
        cache["nns_dep_1000_seconds"] = _time_r_nns_dep()
        _write_benchmark_baseline(cache)
    if "nns_dep_asym_1000_seconds" not in cache:
        cache["nns_dep_asym_1000_seconds"] = _time_r_nns_dep_asym()
        _write_benchmark_baseline(cache)
    if "nns_copula_1000_seconds" not in cache:
        cache["nns_copula_1000_seconds"] = _time_r_nns_copula()
        _write_benchmark_baseline(cache)
    if "nns_causation_1000_seconds" not in cache:
        cache["nns_causation_1000_seconds"] = _time_r_nns_causation()
        _write_benchmark_baseline(cache)
    if "nns_norm_1000x3_seconds" not in cache:
        cache["nns_norm_1000x3_seconds"] = _time_r_nns_norm()
        _write_benchmark_baseline(cache)
    if "nns_distance_1000x3_seconds" not in cache:
        cache["nns_distance_1000x3_seconds"] = _time_r_nns_distance()
        _write_benchmark_baseline(cache)
    if "nns_distance_bulk_1000x3_100_seconds" not in cache:
        cache["nns_distance_bulk_1000x3_100_seconds"] = _time_r_nns_distance_bulk()
        _write_benchmark_baseline(cache)
    if "nns_distance_class_500x3_seconds" not in cache:
        cache["nns_distance_class_500x3_seconds"] = _time_r_nns_distance_class()
        _write_benchmark_baseline(cache)
    if "nns_distance_bulk_class_500x3_50_seconds" not in cache:
        cache["nns_distance_bulk_class_500x3_50_seconds"] = _time_r_nns_distance_bulk_class()
        _write_benchmark_baseline(cache)
    if "nns_diff_sin_seconds" not in cache:
        cache["nns_diff_sin_seconds"] = _time_r_nns_diff()
        _write_benchmark_baseline(cache)
    if "nns_anova_100x2_seconds" not in cache:
        cache["nns_anova_100x2_seconds"] = _time_r_nns_anova()
        _write_benchmark_baseline(cache)
    if "nns_part_500_seconds" not in cache:
        cache["nns_part_500_seconds"] = _time_r_nns_part()
        _write_benchmark_baseline(cache)
    if "nns_reg_500_seconds" not in cache:
        cache["nns_reg_500_seconds"] = _time_r_nns_reg()
        _write_benchmark_baseline(cache)
    if "nns_reg_200_ci_seconds" not in cache:
        cache["nns_reg_200_ci_seconds"] = _time_r_nns_reg_ci()
        _write_benchmark_baseline(cache)
    if "nns_reg_class_200_seconds" not in cache:
        cache["nns_reg_class_200_seconds"] = _time_r_nns_reg_class()
        _write_benchmark_baseline(cache)
    if "nns_reg_dimred_200x3_seconds" not in cache:
        cache["nns_reg_dimred_200x3_seconds"] = _time_r_nns_reg_dimred()
        _write_benchmark_baseline(cache)
    if "nns_m_reg_200x3_seconds" not in cache:
        cache["nns_m_reg_200x3_seconds"] = _time_r_nns_m_reg()
        _write_benchmark_baseline(cache)
    if "nns_m_reg_200x3_ci_seconds" not in cache:
        cache["nns_m_reg_200x3_ci_seconds"] = _time_r_nns_m_reg_ci()
        _write_benchmark_baseline(cache)
    if "nns_m_reg_class_200x3_seconds" not in cache:
        cache["nns_m_reg_class_200x3_seconds"] = _time_r_nns_m_reg_class()
        _write_benchmark_baseline(cache)
    if "nns_stack_100x3_seconds" not in cache:
        cache["nns_stack_100x3_seconds"] = _time_r_nns_stack()
        _write_benchmark_baseline(cache)
    if "nns_stack_100x3_pred_int_seconds" not in cache:
        cache["nns_stack_100x3_pred_int_seconds"] = _time_r_nns_stack_pred_int()
        _write_benchmark_baseline(cache)
    if "nns_stack_100x3_ts_test_seconds" not in cache:
        cache["nns_stack_100x3_ts_test_seconds"] = _time_r_nns_stack_ts_test()
        _write_benchmark_baseline(cache)
    if "nns_stack_class_100x3_seconds" not in cache:
        cache["nns_stack_class_100x3_seconds"] = _time_r_nns_stack_class()
        _write_benchmark_baseline(cache)
    if "nns_stack_class_balance_150x3_seconds" not in cache:
        cache["nns_stack_class_balance_150x3_seconds"] = _time_r_nns_stack_class_balance()
        _write_benchmark_baseline(cache)
    if "nns_boost_50x3_seconds" not in cache:
        cache["nns_boost_50x3_seconds"] = _time_r_nns_boost()
        _write_benchmark_baseline(cache)
    if "nns_boost_class_50x3_seconds" not in cache:
        cache["nns_boost_class_50x3_seconds"] = _time_r_nns_boost_class()
        _write_benchmark_baseline(cache)
    if "nns_boost_class_balance_80x3_seconds" not in cache:
        cache["nns_boost_class_balance_80x3_seconds"] = _time_r_nns_boost_class_balance()
        _write_benchmark_baseline(cache)
    if "nns_mode_continuous_1000_seconds" not in cache:
        cache["nns_mode_continuous_1000_seconds"] = _time_r_nns_mode_continuous()
        _write_benchmark_baseline(cache)
    if "nns_seas_1000_seconds" not in cache:
        cache["nns_seas_1000_seconds"] = _time_r_nns_seas(1000)
        _write_benchmark_baseline(cache)
    if "nns_seas_5000_seconds" not in cache:
        cache["nns_seas_5000_seconds"] = _time_r_nns_seas(5000)
        _write_benchmark_baseline(cache)
    if "nns_arma_500_auto_nonlin_seconds" not in cache:
        cache["nns_arma_500_auto_nonlin_seconds"] = _time_r_nns_arma(auto=True)
        _write_benchmark_baseline(cache)
    if "nns_arma_500_explicit12_nonlin_seconds" not in cache:
        cache["nns_arma_500_explicit12_nonlin_seconds"] = _time_r_nns_arma(auto=False)
        _write_benchmark_baseline(cache)
    if "nns_arma_200_explicit4_lin_predint_seconds" not in cache:
        cache["nns_arma_200_explicit4_lin_predint_seconds"] = _time_r_nns_arma_pred_int(
            auto=False,
            method="lin",
        )
        _write_benchmark_baseline(cache)
    if "nns_arma_200_auto_nonlin_predint_seconds" not in cache:
        cache["nns_arma_200_auto_nonlin_predint_seconds"] = _time_r_nns_arma_pred_int(
            auto=True,
            method="nonlin",
        )
        _write_benchmark_baseline(cache)
    if "nns_ss_1000_seconds" not in cache:
        cache["nns_ss_1000_seconds"] = _time_r_nns_ss()
        _write_benchmark_baseline(cache)
    if "nns_ss_200_ci_reps100_seconds" not in cache:
        cache["nns_ss_200_ci_reps100_seconds"] = _time_r_nns_ss_ci()
        _write_benchmark_baseline(cache)
    return cache


def _read_benchmark_baseline() -> BenchmarkBaseline:
    if not _BENCHMARK_BASELINE_PATH.exists():
        return {}

    cache = json.loads(_BENCHMARK_BASELINE_PATH.read_text(encoding="utf-8"))
    if not isinstance(cache, dict) or cache.get("schema_version") != _BENCHMARK_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported R benchmark baseline schema in {_BENCHMARK_BASELINE_PATH}."
        )
    if cache.get("nns_version") != _NNS_VERSION:
        raise RuntimeError(
            f"Unsupported NNS benchmark baseline version in {_BENCHMARK_BASELINE_PATH}."
        )

    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"Invalid R benchmark baseline entries in {_BENCHMARK_BASELINE_PATH}.")
    return cast(BenchmarkBaseline, entries)


def _write_benchmark_baseline(entries: BenchmarkBaseline) -> None:
    payload = {
        "nns_version": _NNS_VERSION,
        "schema_version": _BENCHMARK_SCHEMA_VERSION,
        "entries": entries,
    }
    _BENCHMARK_BASELINE_PATH.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _time_r_lpm() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 1000)\n"
        "invisible(NNS::LPM(1, 0, x))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(200)) invisible(NNS::LPM(1, 0, x))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 200)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_pm_matrix(n_variables: int) -> float:
    script = (
        "library(NNS)\n"
        "row <- seq_len(500)\n"
        f"col <- seq_len({n_variables})\n"
        "x <- outer(row, col, function(i, j) sin(i * j / 11) + cos((i + 1) / (j + 2)))\n"
        "invisible(NNS::PM.matrix(1, 1, target = NULL, variable = x, pop_adj = TRUE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(5)) {\n"
        "  invisible(NNS::PM.matrix(1, 1, target = NULL, variable = x, pop_adj = TRUE))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 5)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_sd_efficient_set() -> float:
    script = (
        "library(NNS)\n"
        "row <- seq_len(252)\n"
        "col <- seq_len(50)\n"
        "x <- outer(row, col, function(i, j) sin(i * j / 17) + cos((i + 3) / (j + 5)))\n"
        "invisible(NNS::NNS.SD.efficient.set(x, degree = 2, type = 'discrete', status = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(5)) {\n"
        "  invisible(NNS::NNS.SD.efficient.set(x, degree = 2, type = 'discrete', status = FALSE))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 5)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_sd_cluster() -> float:
    script = (
        "library(NNS)\n"
        "row <- seq(0, 251)\n"
        "x <- sapply(seq_len(50), function(i) sin(row / (i + 1)) + 0.01 * (i - 1))\n"
        "invisible(NNS::NNS.SD.cluster(x, degree = 2, min_cluster = 1, dendrogram = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(5)) {\n"
        "  invisible(NNS::NNS.SD.cluster(x, degree = 2, min_cluster = 1, dendrogram = FALSE))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 5)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_dep() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 1000)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "invisible(NNS::NNS.dep(x, y, asym = FALSE, p.value = FALSE, print.map = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(10)) {\n"
        "  invisible(NNS::NNS.dep(x, y, asym = FALSE, p.value = FALSE, print.map = FALSE))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 10)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_dep_asym() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 1000)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "invisible(NNS::NNS.dep(x, y, asym = TRUE, p.value = FALSE, print.map = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(10)) {\n"
        "  invisible(NNS::NNS.dep(x, y, asym = TRUE, p.value = FALSE, print.map = FALSE))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 10)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_copula() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 1000)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "xy <- cbind(x, y)\n"
        "run <- function() NNS::NNS.copula("
        "xy, target = NULL, continuous = TRUE, plot = FALSE, independence.overlay = FALSE)\n"
        "invisible(run())\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(10)) {\n"
        "  invisible(run())\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 10)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_causation() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 1000)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "invisible(NNS::NNS.caus(x, y, tau = 0, plot = FALSE, p.value = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(5)) {\n"
        "  invisible(NNS::NNS.caus(x, y, tau = 0, plot = FALSE, p.value = FALSE))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 5)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_norm() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 1000)\n"
        "X <- cbind(x + 3, x^2 + 1, sin(x) + 2)\n"
        "invisible(NNS::NNS.norm(X, linear = FALSE, chart.type = NULL))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(50)) {\n"
        "  invisible(NNS::NNS.norm(X, linear = FALSE, chart.type = NULL))\n"
        "}\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 50)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_distance() -> float:
    script = (
        "library(NNS)\n"
        "row <- seq_len(1000)\n"
        "rpm <- data.frame(x1 = sin(row / 3) + 1.5, x2 = cos(row / 5) + 2, "
        "x3 = row / 1000, y.hat = sin(row / 7))\n"
        "dest <- c(x1 = 1.25, x2 = 2.75, x3 = 0.4)\n"
        "invisible(NNS::NNS.distance(rpm, dest, k = 20))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(50)) invisible(NNS::NNS.distance(rpm, dest, k = 20))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 50)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_distance_bulk() -> float:
    script = (
        "library(NNS)\n"
        "row <- seq_len(1000)\n"
        "rpm <- data.frame(x1 = sin(row / 3) + 1.5, x2 = cos(row / 5) + 2, "
        "x3 = row / 1000, y.hat = sin(row / 7))\n"
        "test_row <- seq_len(100)\n"
        "Xtest <- data.frame(x1 = sin(test_row / 4) + 1.5, "
        "x2 = cos(test_row / 6) + 2, x3 = test_row / 100)\n"
        "invisible(NNS:::NNS.distance.bulk(rpm, Xtest, k = 20))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(20)) invisible(NNS:::NNS.distance.bulk(rpm, Xtest, k = 20))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 20)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_distance_class() -> float:
    script = (
        "library(NNS)\n"
        "row <- seq_len(500)\n"
        "rpm <- data.frame(x1 = sin(row / 3) + 1.5, x2 = cos(row / 5) + 2, "
        "x3 = row / 500, y.hat = (row %% 3) + 1)\n"
        "dest <- c(x1 = 1.25, x2 = 2.75, x3 = 0.4)\n"
        "invisible(NNS::NNS.distance(rpm, dest, k = 5, class = 'class'))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(100)) invisible(NNS::NNS.distance(rpm, dest, k = 5, class = 'class'))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 100)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_distance_bulk_class() -> float:
    script = (
        "library(NNS)\n"
        "row <- seq_len(500)\n"
        "rpm <- data.frame(x1 = sin(row / 3) + 1.5, x2 = cos(row / 5) + 2, "
        "x3 = row / 500, y.hat = (row %% 3) + 1)\n"
        "test_row <- seq_len(50)\n"
        "Xtest <- data.frame(x1 = sin(test_row / 4) + 1.5, "
        "x2 = cos(test_row / 6) + 2, x3 = test_row / 50)\n"
        "invisible(NNS:::NNS.distance.bulk(rpm, Xtest, k = 5, class = 'class'))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(50)) invisible(NNS:::NNS.distance.bulk("
        "rpm, Xtest, k = 5, class = 'class'))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 50)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_diff() -> float:
    script = (
        "library(NNS)\n"
        "f <- function(x) sin(x)\n"
        "invisible(NNS::NNS.diff(f, 1.0, plot = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(20)) invisible(NNS::NNS.diff(f, 1.0, plot = FALSE))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 20)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_anova() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 100) + 0.1 * sin(seq_len(100) / 3)\n"
        "y <- x + 0.25 + 0.05 * cos(seq_len(100) / 5)\n"
        "run <- function() NNS::NNS.ANOVA(x, y, confidence.interval = NULL, plot = FALSE)\n"
        "invisible(run())\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(20)) invisible(run())\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 20)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_part() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 500)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "invisible(NNS::NNS.part(x, y, Voronoi = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(20)) invisible(NNS::NNS.part(x, y, Voronoi = FALSE))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 20)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_reg() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 500)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "invisible(NNS::NNS.reg(x, y, factor.2.dummy = FALSE, plot = FALSE))\n"
        "times <- replicate(5, system.time(invisible(NNS::NNS.reg(x, y, "
        "factor.2.dummy = FALSE, plot = FALSE)))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_reg_ci() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 200)\n"
        "y <- sin(x) + 0.05 * cos(7 * x)\n"
        "point <- seq(-3, 3, length.out = 20)\n"
        "run <- function() NNS::NNS.reg(x, y, point.est = point, "
        "factor.2.dummy = FALSE, plot = FALSE, confidence.interval = 0.95)\n"
        "invisible(run())\n"
        "times <- replicate(5, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_reg_class() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 200)\n"
        "y <- rep(1:3, length.out = 200)\n"
        "point <- seq(-3, 3, length.out = 20)\n"
        "run <- function() NNS::NNS.reg(x, y, point.est = point, "
        "factor.2.dummy = FALSE, type = 'class', plot = FALSE)\n"
        "invisible(run())\n"
        "times <- replicate(5, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_reg_dimred() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 200)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "invisible(NNS::NNS.reg(X, y, factor.2.dummy = FALSE, "
        "dim.red.method = 'cor', plot = FALSE, ncores = 1))\n"
        "times <- replicate(5, system.time(invisible(NNS::NNS.reg(X, y, "
        "factor.2.dummy = FALSE, dim.red.method = 'cor', plot = FALSE, "
        "ncores = 1)))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_m_reg() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 200)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "invisible(NNS:::NNS.M.reg(X, y, factor.2.dummy = FALSE, plot = FALSE, "
        "residual.plot = FALSE, ncores = 1, confidence.interval = NULL))\n"
        "times <- replicate(5, system.time(invisible(NNS:::NNS.M.reg(X, y, "
        "factor.2.dummy = FALSE, plot = FALSE, residual.plot = FALSE, "
        "ncores = 1, confidence.interval = NULL)))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_m_reg_ci() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 200)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "run <- function() NNS:::NNS.M.reg(X, y, point.est = X[1:20,], "
        "factor.2.dummy = FALSE, plot = FALSE, residual.plot = FALSE, "
        "ncores = 1, confidence.interval = 0.95)\n"
        "invisible(run())\n"
        "times <- replicate(5, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_m_reg_class() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-3, 3, length.out = 200)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- rep(1:3, length.out = 200)\n"
        "run <- function() NNS:::NNS.M.reg(X, y, point.est = X[1:20,], "
        "factor.2.dummy = FALSE, type = 'class', plot = FALSE, "
        "residual.plot = FALSE, ncores = 1)\n"
        "invisible(run())\n"
        "times <- replicate(5, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_stack() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 100)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "run <- function() NNS::NNS.stack(X, y, IVs.test = X[1:20,], "
        "CV.size = 0.25, folds = 2, method = c(1, 2), stack = TRUE, "
        "dim.red.method = 'cor', status = FALSE, ncores = 1)\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_stack_pred_int() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 100)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "run <- function() NNS::NNS.stack(X, y, IVs.test = X[1:20,], "
        "CV.size = 0.25, folds = 1, method = c(1, 2), stack = TRUE, "
        "dim.red.method = 'cor', pred.int = 0.95, status = FALSE, ncores = 1)\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_stack_ts_test() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 100)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "run <- function() NNS::NNS.stack(X, y, IVs.test = X[1:20,], "
        "CV.size = 0.25, folds = 1, method = c(1, 2), stack = TRUE, "
        "dim.red.method = 'cor', ts.test = 20, status = FALSE, ncores = 1)\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_stack_class() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 100)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- ifelse(x < -0.5, 1, ifelse(x > 0.75, 3, 2))\n"
        "run <- function() NNS::NNS.stack(X, y, IVs.test = X[1:20,], "
        "CV.size = 0.25, folds = 1, method = c(1, 2), stack = TRUE, "
        "dim.red.method = 'cor', type = 'class', status = FALSE, ncores = 1)\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_stack_class_balance() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 150)\n"
        "X <- cbind(x, sin(x), cos(x))\n"
        "y <- ifelse(x < -0.75, 1, ifelse(x > 1.0, 3, 2))\n"
        "run <- function() { set.seed(42); NNS::NNS.stack(X, y, IVs.test = X[1:20,], "
        "CV.size = 0.25, folds = 1, method = c(1, 2), stack = TRUE, "
        "dim.red.method = 'cor', type = 'class', balance = TRUE, "
        "status = FALSE, ncores = 1) }\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_boost() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 50)\n"
        "X <- cbind(X1 = x, X2 = sin(x), X3 = cos(x))\n"
        "y <- x + sin(x) + 0.25 * cos(x)\n"
        "run <- function() NNS::NNS.boost(X, y, IVs.test = X[1:10,], "
        "learner.trials = 10, CV.size = 0.25, feature.importance = FALSE, "
        "status = FALSE)\n"
        "invisible(run())\n"
        "times <- replicate(2, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_boost_class() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 50)\n"
        "X <- cbind(X1 = x, X2 = sin(x), X3 = cos(x))\n"
        "y <- ifelse(x < -0.5, 1, ifelse(x > 0.75, 3, 2))\n"
        "run <- function() NNS::NNS.boost(X, y, IVs.test = X[1:10,], "
        "learner.trials = 10, CV.size = 0.25, depth = 2, type = 'class', "
        "feature.importance = FALSE, status = FALSE)\n"
        "invisible(run())\n"
        "times <- replicate(2, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_boost_class_balance() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 2, length.out = 80)\n"
        "X <- cbind(X1 = x, X2 = sin(x), X3 = cos(x))\n"
        "y <- ifelse(x < -0.75, 1, ifelse(x > 1.0, 3, 2))\n"
        "run <- function() { set.seed(42); NNS::NNS.boost(X, y, IVs.test = X[1:10,], "
        "learner.trials = 10, CV.size = 0.25, depth = 2, type = 'class', "
        "balance = TRUE, feature.importance = FALSE, status = FALSE) }\n"
        "invisible(run())\n"
        "times <- replicate(2, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_mode_continuous() -> float:
    script = (
        "library(NNS)\n"
        "x <- c(seq(-3, 3, length.out = 500), seq(1, 2, length.out = 500))\n"
        "invisible(NNS::NNS.mode(x, discrete = FALSE, multi = FALSE))\n"
        "start <- proc.time()[['elapsed']]\n"
        "for (i in seq_len(200)) invisible(NNS::NNS.mode(x, discrete = FALSE, multi = FALSE))\n"
        "elapsed <- proc.time()[['elapsed']] - start\n"
        "cat(elapsed / 200)\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_seas(n: int) -> float:
    script = (
        "library(NNS)\n"
        f"t <- seq_len({n})\n"
        "variable <- sin(2 * pi * t / 12) + 0.05 * cos(t / 3)\n"
        "invisible(NNS::NNS.seas(variable, plot = FALSE))\n"
        "times <- replicate(20, system.time(invisible(NNS::NNS.seas(variable, "
        "plot = FALSE)))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_arma(*, auto: bool) -> float:
    seasonal_factor = "TRUE" if auto else "12"
    script = (
        "library(NNS)\n"
        "t <- seq_len(500)\n"
        "variable <- sin(2 * pi * t / 12) + 0.05 * cos(t / 3) + 2\n"
        f"run <- function() NNS::NNS.ARMA(variable, h = 12, seasonal.factor = {seasonal_factor}, "
        "method = 'nonlin', plot = FALSE, seasonal.plot = FALSE)\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_arma_pred_int(*, auto: bool, method: str) -> float:
    seasonal_factor = "TRUE" if auto else "c(3, 4)"
    script = (
        "library(NNS)\n"
        "t <- seq_len(200)\n"
        "variable <- sin(2 * pi * t / 12) + 0.05 * cos(t / 3) + 2\n"
        f"run <- function() NNS::NNS.ARMA(variable, h = 5, seasonal.factor = {seasonal_factor}, "
        f"method = '{method}', pred.int = 0.95, plot = FALSE, seasonal.plot = FALSE)\n"
        "set.seed(123); invisible(run())\n"
        "times <- replicate(5, { set.seed(123); system.time(invisible(run()))[['elapsed']] })\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_ss() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 3, length.out = 1000) + 0.2 * sin(seq_len(1000))\n"
        "y <- seq(-1.5, 2.5, length.out = 1000) + 0.3 * cos(seq_len(1000))\n"
        "invisible(NNS::NNS.SS(x, y))\n"
        "times <- replicate(50, system.time(invisible(NNS::NNS.SS(x, y)))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _time_r_nns_ss_ci() -> float:
    script = (
        "library(NNS)\n"
        "x <- seq(-2, 3, length.out = 200) + 0.2 * sin(seq_len(200))\n"
        "y <- seq(-1.5, 2.5, length.out = 200) + 0.3 * cos(seq_len(200))\n"
        "run <- function() { set.seed(123); NNS::NNS.SS(x, y, "
        "confidence.interval = TRUE, reps = 100, rho = 1) }\n"
        "invisible(run())\n"
        "times <- replicate(3, system.time(invisible(run()))[['elapsed']])\n"
        "cat(max(mean(times), .Machine$double.eps))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return float(completed.stdout)


def _r_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    return env

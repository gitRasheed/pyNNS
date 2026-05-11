from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias, cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray

_CACHE_PATH = Path(__file__).with_name("_r_cache.json")
_LOCK_PATH = _CACHE_PATH.with_suffix(".lock")
_SCHEMA_VERSION = 1
_NNS_VERSION = "12.0"

JsonValue: TypeAlias = None | str | float | list["JsonValue"] | dict[str, "JsonValue"]
RValue: TypeAlias = None | str | list[str] | NDArray[np.float64] | dict[str, "RValue"]
Cache: TypeAlias = dict[str, JsonValue]

_CACHE: Cache | None = None
_CACHE_REFRESH = False


def nns(function: str, *args: Any) -> RValue:
    key = _cache_key(function, args)
    cache, refresh = _cache_state()

    if key in cache:
        return _decode(cache[key])

    if _offline():
        raise RuntimeError(
            f"R cache miss for NNS::{function} with key {key}. "
            f"Run without CI/PYNNS_R_CACHE_ONLY/PYNNS_OFFLINE to populate {_CACHE_PATH}."
        )

    return _uncached_nns(function, args, key, refresh)


def nns_stack_numeric(
    x: list[list[float]],
    y: list[float],
    x_test: list[list[float]],
    *,
    cv_size: float,
    folds: int,
    method: list[int],
    order: int | str | None,
    stack: bool,
    dim_red_method: str | list[float],
    ts_test: int | None = None,
    pred_int: float | None = None,
) -> RValue:
    args = {
        "x": x,
        "y": y,
        "x_test": x_test,
        "cv_size": cv_size,
        "folds": folds,
        "method": method,
        "order": order,
        "stack": stack,
        "dim_red_method": dim_red_method,
        "ts_test": ts_test,
        "pred_int": pred_int,
    }
    key = _cache_key("NNS.stack.numeric", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.stack.numeric with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_stack_numeric(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_boost_numeric(
    x: list[list[float]],
    y: list[float],
    x_test: list[list[float]],
    *,
    learner_trials: int,
    cv_size: float,
    depth: int | str | None,
    features_only: bool,
) -> RValue:
    args = {
        "x": x,
        "y": y,
        "x_test": x_test,
        "learner_trials": learner_trials,
        "cv_size": cv_size,
        "depth": depth,
        "features_only": features_only,
    }
    key = _cache_key("NNS.boost.numeric", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.boost.numeric with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_boost_numeric(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_meboot_diagnostics(
    x: list[float],
    *,
    rho: float,
    reps: int = 2,
    drift: bool = True,
    trim: float = 0.1,
    xmin: float | None = None,
    xmax: float | None = None,
    sym: bool = False,
    scl_adjustment: bool = False,
    seed: int = 1,
) -> RValue:
    args = {
        "x": x,
        "rho": rho,
        "reps": reps,
        "drift": drift,
        "trim": trim,
        "xmin": xmin,
        "xmax": xmax,
        "sym": sym,
        "scl_adjustment": scl_adjustment,
        "seed": seed,
    }
    key = _cache_key("NNS.meboot.diagnostics", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.meboot.diagnostics with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_meboot_diagnostics(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_meboot_stat_summary(
    x: list[float],
    *,
    rho: float,
    reps: int = 100,
    seed: int = 1,
) -> RValue:
    args = {"x": x, "rho": rho, "reps": reps, "seed": seed}
    key = _cache_key("NNS.meboot.stat_summary", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.meboot.stat_summary with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_meboot_stat_summary(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_mc_grid(
    *,
    lower_rho: float,
    upper_rho: float,
    by: float,
    exp: float,
) -> RValue:
    args = {"lower_rho": lower_rho, "upper_rho": upper_rho, "by": by, "exp": exp}
    key = _cache_key("NNS.MC.grid", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.MC.grid with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_mc_grid(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_mc_stat_summary(
    x: list[float],
    *,
    reps: int,
    lower_rho: float,
    upper_rho: float,
    by: float,
    seed: int,
) -> RValue:
    args = {
        "x": x,
        "reps": reps,
        "lower_rho": lower_rho,
        "upper_rho": upper_rho,
        "by": by,
        "seed": seed,
    }
    key = _cache_key("NNS.MC.stat_summary", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.MC.stat_summary with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_mc_stat_summary(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_anova_custom(payload: dict[str, Any]) -> RValue:
    key = _cache_key("NNS.ANOVA.custom", (payload,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.ANOVA.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_anova_custom(payload)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_distance_bulk_custom(
    rpm: dict[str, list[float]],
    x_test: dict[str, list[float]],
    k: int | str,
) -> RValue:
    args = {"rpm": rpm, "x_test": x_test, "k": k}
    key = _cache_key("NNS.distance.bulk.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.distance.bulk.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_distance_bulk_custom(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_diff_custom(name: str, point: float) -> RValue:
    args = {"name": name, "point": point}
    key = _cache_key("NNS.diff.custom", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.diff.custom with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_diff_custom(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def nns_arma_pred_int(
    variable: list[float],
    *,
    h: int,
    seasonal_factor: int | list[int] | bool,
    method: str,
    pred_int: float,
    seed: int,
) -> RValue:
    args = {
        "variable": variable,
        "h": h,
        "seasonal_factor": seasonal_factor,
        "method": method,
        "pred_int": pred_int,
        "seed": seed,
    }
    key = _cache_key("NNS.ARMA.pred_int", (args,))
    cache, refresh = _cache_state()
    if key in cache:
        return _decode(cache[key])
    if _offline():
        raise RuntimeError(f"R cache miss for NNS.ARMA.pred_int with key {key}.")
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
        if key in disk_cache:
            return _decode(disk_cache[key])
        result = _call_r_arma_pred_int(args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        return result


def _uncached_nns(
    function: str,
    args: tuple[Any, ...],
    key: str,
    refresh: bool,
) -> RValue:
    global _CACHE, _CACHE_REFRESH
    with _cache_lock():
        disk_cache, disk_refresh = _read_cache_from_disk()
        if refresh or disk_refresh:
            disk_cache = {}
            _CACHE_REFRESH = False
        if key in disk_cache:
            _CACHE = disk_cache
            return _decode(disk_cache[key])

        result = _call_r(function, args)
        disk_cache[key] = _encode(result)
        _write_cache(disk_cache)
        _CACHE = disk_cache
        return result


def _cache_key(function: str, args: tuple[Any, ...]) -> str:
    payload = json.dumps(
        {"function": function, "args": args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_state() -> tuple[Cache, bool]:
    global _CACHE, _CACHE_REFRESH
    if _CACHE is None:
        _CACHE, _CACHE_REFRESH = _read_cache_from_disk()
    return _CACHE, _CACHE_REFRESH


def _read_cache_from_disk() -> tuple[Cache, bool]:
    if not _CACHE_PATH.exists():
        return {}, False

    cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    if not isinstance(cache, dict) or cache.get("schema_version") != _SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported R cache schema in {_CACHE_PATH}.")

    nns_version = cache.get("nns_version")
    if nns_version != _NNS_VERSION:
        warn(
            f"R cache was built for NNS {nns_version}; "
            f"expected {_NNS_VERSION}. Refreshing entries.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}, True

    entries = cache.get("entries")
    if not isinstance(entries, dict):
        raise RuntimeError(f"Invalid R cache entries in {_CACHE_PATH}.")
    return cast(Cache, entries), False


def _write_cache(entries: Cache) -> None:
    payload = {
        "nns_version": _NNS_VERSION,
        "schema_version": _SCHEMA_VERSION,
        "entries": entries,
    }
    tmp_path = _CACHE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(_CACHE_PATH)


@contextmanager
def _cache_lock() -> Iterator[None]:
    _LOCK_PATH.touch(exist_ok=True)
    with _LOCK_PATH.open("r+") as lock_file:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _offline() -> bool:
    return (
        os.environ.get("CI") == "true"
        or os.environ.get("PYNNS_R_CACHE_ONLY") == "1"
        or os.environ.get("PYNNS_OFFLINE") == "1"
    )


def _call_r(function: str, args: tuple[Any, ...]) -> RValue:
    if not function.replace(".", "").replace("_", "").isalnum():
        raise ValueError(f"Unsupported NNS function name: {function!r}")

    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        f"result <- do.call(getFromNamespace('{function}', 'NNS'), args)\n"
        "encode <- function(x) {\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_stack_numeric(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "mat <- function(z) do.call(rbind, lapply(z, as.numeric))\n"
        "order_arg <- args$order\n"
        "if (length(order_arg) == 0) order_arg <- NULL\n"
        "dim_arg <- args$dim_red_method\n"
        "if (is.list(dim_arg)) dim_arg <- as.numeric(unlist(dim_arg))\n"
        "ts_arg <- args$ts_test\n"
        "if (length(ts_arg) == 0) ts_arg <- NULL else ts_arg <- as.integer(ts_arg)\n"
        "pred_arg <- args$pred_int\n"
        "if (length(pred_arg) == 0) pred_arg <- NULL else pred_arg <- as.numeric(pred_arg)\n"
        "result <- NNS::NNS.stack("
        "mat(args$x), as.numeric(unlist(args$y)), IVs.test = mat(args$x_test), "
        "CV.size = as.numeric(args$cv_size), folds = as.integer(args$folds), "
        "method = as.numeric(unlist(args$method)), order = order_arg, "
        "stack = isTRUE(as.logical(unlist(args$stack))), "
        "dim.red.method = dim_arg, pred.int = pred_arg, ts.test = ts_arg, "
        "status = FALSE, ncores = 1)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_boost_numeric(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "mat <- function(z) {\n"
        "  out <- do.call(rbind, lapply(z, as.numeric))\n"
        "  colnames(out) <- paste0('X', seq_len(ncol(out)))\n"
        "  out\n"
        "}\n"
        "depth_arg <- args$depth\n"
        "if (length(depth_arg) == 0) depth_arg <- NULL\n"
        "result <- NNS::NNS.boost("
        "mat(args$x), as.numeric(unlist(args$y)), IVs.test = mat(args$x_test), "
        "learner.trials = as.integer(args$learner_trials), "
        "CV.size = as.numeric(args$cv_size), depth = depth_arg, "
        "features.only = isTRUE(as.logical(unlist(args$features_only))), "
        "feature.importance = FALSE, status = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_meboot_diagnostics(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "f <- get('FUN', envir = environment(NNS::NNS.meboot))\n"
        "set.seed(as.integer(args$seed))\n"
        "nullify <- function(v) if (length(v) == 0) NULL else as.numeric(v)\n"
        "result <- f("
        "x = as.numeric(unlist(args$x)), reps = as.integer(args$reps), "
        "rho = as.numeric(args$rho), drift = isTRUE(as.logical(args$drift)), "
        "trim = as.numeric(args$trim), xmin = nullify(args$xmin), xmax = nullify(args$xmax), "
        "expand.sd = FALSE, force.clt = FALSE, "
        "scl.adjustment = isTRUE(as.logical(args$scl_adjustment)), "
        "sym = isTRUE(as.logical(args$sym)))\n"
        "picked <- result[c('x','xx','z','dv','dvtrim','xmin','xmax','desintxb','ordxx','kappa')]\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(picked), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_meboot_stat_summary(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "f <- get('FUN', envir = environment(NNS::NNS.meboot))\n"
        "set.seed(as.integer(args$seed))\n"
        "result <- f(x = as.numeric(unlist(args$x)), reps = as.integer(args$reps), "
        "rho = as.numeric(args$rho))\n"
        "replicates <- result$replicates\n"
        "summary <- c(mean_ensemble = mean(result$ensemble), sd_ensemble = sd(result$ensemble), "
        "median_rep_means = median(colMeans(replicates)), "
        "median_rep_sds = median(apply(replicates, 2, sd)))\n"
        "cat(jsonlite::toJSON(as.numeric(summary), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_mc_grid(args: dict[str, Any]) -> RValue:
    script = (
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "rhos <- seq(as.numeric(args$lower_rho), as.numeric(args$upper_rho), "
        "as.numeric(args$by))\n"
        "neg_rhos <- abs(rhos[rhos <= 0])\n"
        "pos_rhos <- rhos[rhos > 0]\n"
        "exp_rhos <- rev(c((neg_rhos^as.numeric(args$exp)) * -1, "
        "pos_rhos^(1/as.numeric(args$exp))))\n"
        "result <- list(values = as.numeric(exp_rhos), names = paste0('rho = ', exp_rhos))\n"
        "cat(jsonlite::toJSON(result, auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_mc_stat_summary(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "set.seed(as.integer(args$seed))\n"
        "result <- NNS::NNS.MC("
        "x = as.numeric(unlist(args$x)), reps = as.integer(args$reps), "
        "lower_rho = as.numeric(args$lower_rho), upper_rho = as.numeric(args$upper_rho), "
        "by = as.numeric(args$by))\n"
        "replicates <- result$replicates\n"
        "block_sds <- vapply(replicates, function(m) median(apply(m, 2, sd)), numeric(1))\n"
        "summary <- c(mean_ensemble = mean(result$ensemble), sd_ensemble = sd(result$ensemble), "
        "median_block_sd = median(block_sds))\n"
        "cat(jsonlite::toJSON(as.numeric(summary), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_anova_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "if (args$mode == 'binary') {\n"
        "  ci <- if (isTRUE(args$robust)) 0.95 else NULL\n"
        "  result <- NNS::NNS.ANOVA(as.numeric(unlist(args$control)), "
        "as.numeric(unlist(args$treatment)), means.only = args$means_only, "
        "medians = args$medians, confidence.interval = ci, robust = args$robust, "
        "plot = FALSE)\n"
        "} else {\n"
        "  groups <- lapply(args$groups, function(x) as.numeric(unlist(x)))\n"
        "  result <- NNS::NNS.ANOVA(groups, means.only = args$means_only, "
        "medians = args$medians, confidence.interval = NULL, "
        "pairwise = args$pairwise, plot = FALSE)\n"
        "}\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_distance_bulk_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "rpm <- as.data.frame(args$rpm)\n"
        "x_test <- as.data.frame(args$x_test)\n"
        "result <- NNS:::NNS.distance.bulk(rpm, x_test, args$k, class = NULL)\n"
        "cat(jsonlite::toJSON(as.numeric(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_diff_custom(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "f <- switch(args$name,\n"
        "  square = function(x) x^2,\n"
        "  sin = function(x) sin(x),\n"
        "  exp = function(x) exp(x),\n"
        "  constant = function(x) 5,\n"
        "  identity = function(x) x)\n"
        "result <- NNS::NNS.diff(f, args$point, plot = FALSE)\n"
        "payload <- as.numeric(result[, 1])\n"
        "names(payload) <- rownames(result)\n"
        "cat(jsonlite::toJSON(as.list(payload), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _call_r_arma_pred_int(args: dict[str, Any]) -> RValue:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'), "
        "simplifyVector = FALSE)\n"
        "seasonal <- args$seasonal_factor\n"
        "if (is.list(seasonal)) seasonal <- as.numeric(unlist(seasonal))\n"
        "set.seed(as.integer(args$seed))\n"
        "result <- NNS::NNS.ARMA("
        "as.numeric(unlist(args$variable)), h = as.integer(args$h), "
        "seasonal.factor = seasonal, method = args$method, "
        "pred.int = as.numeric(args$pred_int), plot = FALSE, seasonal.plot = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(NULL)\n"
        "  if (is.matrix(x) || is.data.frame(x)) {\n"
        "    out <- lapply(seq_along(x), function(i) as.numeric(x[[i]]))\n"
        "    names(out) <- names(x)\n"
        "    return(out)\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  if (is.character(x)) return(as.character(x))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA, null = 'null'))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=_r_env(),
        input=json.dumps(args),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _r_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    return env


def _decode(value: JsonValue) -> RValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return cast(list[str], value)
    if isinstance(value, list):
        converted: list[JsonValue] = []
        has_numeric_special = False
        for item in value:
            if item == "NA":
                converted.append(float("nan"))
                has_numeric_special = True
            elif item == "Inf":
                converted.append(float("inf"))
                has_numeric_special = True
            elif item == "-Inf":
                converted.append(float("-inf"))
                has_numeric_special = True
            else:
                converted.append(item)
        if has_numeric_special:
            return np.asarray(converted, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _encode(value: RValue) -> JsonValue:
    if value is None:
        return None
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return cast(JsonValue, value)
    encoded = value.tolist()
    return cast(JsonValue, encoded)

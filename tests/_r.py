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
        "result <- NNS::NNS.stack("
        "mat(args$x), as.numeric(unlist(args$y)), IVs.test = mat(args$x_test), "
        "CV.size = as.numeric(args$cv_size), folds = as.integer(args$folds), "
        "method = as.numeric(unlist(args$method)), order = order_arg, "
        "stack = isTRUE(as.logical(unlist(args$stack))), "
        "dim.red.method = dim_arg, status = FALSE, ncores = 1)\n"
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

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, TypeAlias, cast
from warnings import warn

import numpy as np
from numpy.typing import NDArray

_CACHE_PATH = Path(__file__).with_name("_r_cache.json")
_SCHEMA_VERSION = 1
_NNS_VERSION = "12.0"

JsonValue: TypeAlias = float | list["JsonValue"] | dict[str, "JsonValue"]
RValue: TypeAlias = NDArray[np.float64] | dict[str, "RValue"]
Cache: TypeAlias = dict[str, JsonValue]


def nns(function: str, *args: Any) -> RValue:
    key = _cache_key(function, args)
    cache, refresh = _read_cache()

    if key in cache:
        return _decode(cache[key])

    if _offline():
        raise RuntimeError(
            f"R cache miss for NNS::{function} with key {key}. "
            f"Run without CI/PYNNS_R_CACHE_ONLY/PYNNS_OFFLINE to populate {_CACHE_PATH}."
        )

    result = _call_r(function, args)
    if refresh:
        cache = {}
    cache[key] = _encode(result)
    _write_cache(cache)
    return result


def _cache_key(function: str, args: tuple[Any, ...]) -> str:
    payload = json.dumps(
        {"function": function, "args": args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache() -> tuple[Cache, bool]:
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
    _CACHE_PATH.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


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
        "args <- jsonlite::fromJSON(commandArgs(trailingOnly = TRUE)[[1]])\n"
        f"result <- do.call(NNS::{function}, args)\n"
        "encode <- function(x) {\n"
        "  if (is.matrix(x)) {\n"
        "    return(unname(lapply(seq_len(nrow(x)), function(i) as.numeric(x[i, ]))))\n"
        "  }\n"
        "  if (is.list(x)) return(lapply(x, encode))\n"
        "  as.numeric(x)\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script, json.dumps(args)],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    return _decode(json.loads(completed.stdout))


def _r_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    return env


def _decode(value: JsonValue) -> RValue:
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    return np.asarray(value, dtype=np.float64)


def _encode(value: RValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    encoded = value.tolist()
    return cast(JsonValue, encoded)

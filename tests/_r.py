from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

_CACHE_PATH = Path(__file__).with_name("_r_cache.json")


def nns(function: str, *args: Any) -> NDArray[np.float64]:
    key = _cache_key(function, args)
    cache = _read_cache()

    if key in cache:
        return np.asarray(cache[key], dtype=np.float64)

    if _cache_only():
        raise RuntimeError(
            f"R cache miss for NNS::{function} with key {key}. "
            f"Run without CI/PYNNS_R_CACHE_ONLY to populate {_CACHE_PATH}."
        )

    result = _call_r(function, args)
    cache[key] = result.tolist()
    _CACHE_PATH.write_text(json.dumps(cache, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def _cache_key(function: str, args: tuple[Any, ...]) -> str:
    payload = json.dumps(
        {"function": function, "args": args},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache() -> dict[str, list[float]]:
    if not _CACHE_PATH.exists():
        return {}
    return cast(dict[str, list[float]], json.loads(_CACHE_PATH.read_text(encoding="utf-8")))


def _cache_only() -> bool:
    return os.environ.get("CI") == "true" or os.environ.get("PYNNS_R_CACHE_ONLY") == "1"


def _call_r(function: str, args: tuple[Any, ...]) -> NDArray[np.float64]:
    if not function.replace(".", "").replace("_", "").isalnum():
        raise ValueError(f"Unsupported NNS function name: {function!r}")

    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(commandArgs(trailingOnly = TRUE)[[1]])\n"
        f"result <- do.call(NNS::{function}, args)\n"
        "cat(jsonlite::toJSON(as.numeric(result), auto_unbox = TRUE, digits = NA))\n"
    )
    completed = subprocess.run(
        ["Rscript", "-e", script, json.dumps(args)],
        check=True,
        capture_output=True,
        env=_r_env(),
        text=True,
    )
    values = json.loads(completed.stdout)
    if isinstance(values, int | float):
        values = [values]
    return np.asarray(values, dtype=np.float64)


def _r_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    return env

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import nns_distance, nns_distance_bulk


@pytest.mark.parity
@pytest.mark.parametrize("k", [1, 2, 3, "all"])
def test_nns_distance_matches_r(k: int | Literal["all"]) -> None:
    rpm, dest = _rpm_and_target()

    expected = nns("NNS.distance", _rpm_dict(rpm), dest.tolist(), k, None)
    assert isinstance(expected, np.ndarray)
    actual = nns_distance(rpm, dest, k=k)

    np.testing.assert_allclose(actual, float(expected), atol=EXACT)


@pytest.mark.parity
@pytest.mark.parametrize("k", [1, 2, "all"])
def test_nns_distance_bulk_matches_r(k: int | Literal["all"]) -> None:
    rpm, _ = _rpm_and_target()
    x_test = rpm[:4, :-1] + np.array([0.05, -0.03, 0.02])

    expected = _r_distance_bulk(rpm, x_test, k)
    actual = nns_distance_bulk(rpm, x_test, k=k)

    np.testing.assert_allclose(actual, expected, atol=EXACT)


def _rpm_and_target() -> tuple[np.ndarray, np.ndarray]:
    row = np.arange(1, 13, dtype=np.float64)
    features = np.column_stack(
        (
            np.sin(row / 3.0) + 1.5,
            np.cos(row / 5.0) + 2.0,
            row / 10.0 + 0.5,
        )
    )
    y_hat = np.sin(row / 4.0) + row / 20.0
    return np.column_stack((features, y_hat)), np.array([1.25, 2.75, 1.4])


def _rpm_dict(rpm: np.ndarray) -> dict[str, list[float]]:
    return {
        "x1": rpm[:, 0].tolist(),
        "x2": rpm[:, 1].tolist(),
        "x3": rpm[:, 2].tolist(),
        "y.hat": rpm[:, 3].tolist(),
    }


def _r_distance_bulk(rpm: np.ndarray, x_test: np.ndarray, k: int | str) -> np.ndarray:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "rpm <- as.data.frame(args$rpm)\n"
        "x_test <- as.data.frame(args$x_test)\n"
        "result <- NNS:::NNS.distance.bulk(rpm, x_test, args$k, class = NULL)\n"
        "cat(jsonlite::toJSON(as.numeric(result), auto_unbox = TRUE, digits = NA))\n"
    )
    payload = {
        "rpm": _rpm_dict(rpm),
        "x_test": {
            "x1": x_test[:, 0].tolist(),
            "x2": x_test[:, 1].tolist(),
            "x3": x_test[:, 2].tolist(),
        },
        "k": k,
    }
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=env,
        input=json.dumps(payload),
        text=True,
    )
    return np.asarray(json.loads(completed.stdout), dtype=np.float64)

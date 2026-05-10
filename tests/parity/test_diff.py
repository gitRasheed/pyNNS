from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _tolerances import EXACT

from pynns import nns_diff

DIFF_PARITY = 1e-5


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "func", "point"),
    [
        ("square", lambda x: x * x, 2.0),
        ("sin", np.sin, 1.0),
        ("exp", np.exp, 0.5),
        ("constant", lambda x: 5.0, 2.0),
        ("identity", lambda x: x, 3.0),
    ],
)
def test_nns_diff_derivative_matches_r(
    name: str,
    func: Any,
    point: float,
) -> None:
    expected = _r_nns_diff(name, point)
    actual = nns_diff(func, point)

    np.testing.assert_allclose(actual["DERIVATIVE"], expected["DERIVATIVE"], atol=DIFF_PARITY)
    np.testing.assert_allclose(
        actual["Value of f(x) at point"],
        expected["Value of f(x) at point"],
        atol=EXACT,
    )


def _r_nns_diff(name: str, point: float) -> dict[str, float]:
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
    env = os.environ.copy()
    env.setdefault("R_LIBS_USER", str(Path.home() / "R" / "library"))
    completed = subprocess.run(
        ["Rscript", "-e", script],
        check=True,
        capture_output=True,
        env=env,
        input=json.dumps({"name": name, "point": point}),
        text=True,
    )
    return {
        key: float("nan") if value == "NA" else float(value)
        for key, value in json.loads(completed.stdout).items()
    }

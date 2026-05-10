from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from pynns import nns_anova

ANOVA_PARITY = 3e-5
SIZES = [30, 100, 500]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize(
    ("means_only", "medians"),
    [(False, False), (True, False), (False, True)],
)
def test_nns_anova_binary_matches_r(size: int, means_only: bool, medians: bool) -> None:
    control, treatment = _groups(size)

    expected = _r_anova_binary(control, treatment, means_only=means_only, medians=medians)
    actual = nns_anova(
        control,
        treatment,
        means_only=means_only,
        medians=medians,
        confidence_interval=None,
    )

    assert isinstance(actual, dict)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=ANOVA_PARITY)


@pytest.mark.parity
def test_nns_anova_binary_unequal_sizes_matches_r() -> None:
    control, treatment = _groups(100)
    treatment = treatment[:73]

    expected = _r_anova_binary(control, treatment)
    actual = nns_anova(control, treatment, confidence_interval=None)

    assert isinstance(actual, dict)
    for key, value in expected.items():
        np.testing.assert_allclose(actual[key], value, atol=ANOVA_PARITY)


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
def test_nns_anova_multi_group_certainty_matches_r(size: int) -> None:
    groups = _multi_groups(size)

    expected = _r_anova_groups(groups, pairwise=False)
    actual = nns_anova(groups, confidence_interval=None)

    assert isinstance(actual, dict)
    assert isinstance(expected, float)
    np.testing.assert_allclose(actual["Certainty"], expected, atol=ANOVA_PARITY)


@pytest.mark.parity
def test_nns_anova_pairwise_matches_r() -> None:
    groups = _multi_groups(100)

    expected = _r_anova_groups(groups, pairwise=True)
    actual = nns_anova(groups, confidence_interval=None, pairwise=True)

    assert isinstance(actual, np.ndarray)
    assert isinstance(expected, np.ndarray)
    np.testing.assert_allclose(actual, expected, atol=ANOVA_PARITY)


@pytest.mark.parity
def test_nns_anova_robust_structure_matches_r_shape() -> None:
    control, treatment = _groups(30)

    expected = _r_anova_binary(control, treatment, robust=True)
    actual = nns_anova(control, treatment, robust=True, random_seed=123)

    assert isinstance(actual, dict)
    assert set(expected).issuperset(
        {"Control", "Treatment", "Grand_Statistic", "Control_CDF", "Treatment_CDF", "Certainty"}
    )
    assert set(actual) == {
        "Control",
        "Treatment",
        "Grand_Statistic",
        "Control_CDF",
        "Treatment_CDF",
        "Certainty",
        "Effect_Size_LB",
        "Effect_Size_UB",
        "Confidence_Level",
        "Robust Certainty Estimate",
        "Lower Bound Robust Certainty",
        "Upper Bound Robust Certainty",
    }
    assert 0.0 <= actual["Robust Certainty Estimate"] <= 1.0
    assert 0.0 <= actual["Lower Bound Robust Certainty"] <= 1.0
    assert 0.0 <= actual["Upper Bound Robust Certainty"] <= 1.0


def _groups(size: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(size, dtype=np.float64)
    x = np.linspace(-2.0, 2.0, size) + 0.1 * np.sin(idx / 3.0)
    y = x + 0.25 + 0.05 * np.cos(idx / 5.0)
    return x, y


def _multi_groups(size: int) -> list[np.ndarray]:
    x, y = _groups(size)
    z = np.cos(np.linspace(0.0, 3.0, size)) + 0.1 * np.sin(np.arange(size) / 7.0)
    return [x, y, z]


def _r_anova_binary(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    means_only: bool = False,
    medians: bool = False,
    robust: bool = False,
) -> dict[str, float]:
    result = _r_anova(
        {
            "mode": "binary",
            "control": control.tolist(),
            "treatment": treatment.tolist(),
            "means_only": means_only,
            "medians": medians,
            "robust": robust,
        }
    )
    assert isinstance(result, dict)
    return result


def _r_anova_groups(groups: list[np.ndarray], *, pairwise: bool) -> float | np.ndarray:
    result = _r_anova(
        {
            "mode": "groups",
            "groups": [group.tolist() for group in groups],
            "means_only": False,
            "medians": False,
            "pairwise": pairwise,
        }
    )
    assert isinstance(result, float | np.ndarray)
    return result


def _r_anova(payload: dict[str, Any]) -> dict[str, float] | np.ndarray | float:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON("
        "paste(readLines('stdin'), collapse = '\\n'), simplifyVector = FALSE)\n"
        "if (args$mode == 'binary') {\n"
        "  ci <- if (isTRUE(args$robust)) 0.95 else NULL\n"
        "  result <- NNS::NNS.ANOVA(unlist(args$control), unlist(args$treatment), "
        "means.only = args$means_only, medians = args$medians, "
        "confidence.interval = ci, robust = args$robust, plot = FALSE)\n"
        "} else {\n"
        "  groups <- lapply(args$groups, unlist)\n"
        "  result <- NNS::NNS.ANOVA(groups, means.only = args$means_only, "
        "medians = args$medians, confidence.interval = NULL, "
        "pairwise = args$pairwise, plot = FALSE)\n"
        "}\n"
        "encode <- function(x) {\n"
        "  if (is.matrix(x)) return(list(type = 'matrix', value = unname(lapply("
        "seq_len(nrow(x)), function(i) as.numeric(x[i, ])))))\n"
        "  if (is.list(x)) return(list(type = 'list', value = lapply(x, encode)))\n"
        "  list(type = 'numeric', value = as.numeric(x))\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
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
    return cast(dict[str, float] | np.ndarray | float, _decode_r(json.loads(completed.stdout)))


def _decode_r(value: dict[str, Any]) -> Any:
    if value["type"] == "matrix":
        return np.asarray(value["value"], dtype=np.float64)
    if value["type"] == "numeric":
        numeric = np.asarray(value["value"], dtype=np.float64)
        return float(numeric.reshape(-1)[0]) if numeric.size == 1 else numeric
    decoded = {key: _decode_r(item) for key, item in value["value"].items()}
    return {key: float(item) for key, item in decoded.items()}

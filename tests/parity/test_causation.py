from __future__ import annotations

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT

from pynns import causal_matrix, nns_causation

SIZES = [50, 200, 1000]
RELATIONSHIPS = ["linear", "independent", "quadratic", "sin", "asymmetric"]


@pytest.mark.parity
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("relationship", RELATIONSHIPS)
def test_nns_causation_matches_r(
    rng: np.random.Generator,
    size: int,
    relationship: str,
) -> None:
    x, y = _relationship(relationship, size, rng)

    expected = nns("NNS.caus", x.tolist(), y.tolist(), False, 0, False, False)
    actual = np.fromiter(nns_causation(x, y).values(), dtype=np.float64)

    np.testing.assert_allclose(actual, _vector(expected), atol=EXACT)


@pytest.mark.parity
def test_causal_matrix_matches_r() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=100)
    variable = np.column_stack(
        (
            x,
            x**2 + 0.1 * rng.normal(size=100),
            np.sin(x) + 0.05 * rng.normal(size=100),
        )
    )

    expected = nns("NNS.caus", variable.tolist(), None, False, 0, False, False)
    actual = causal_matrix(variable)

    np.testing.assert_allclose(actual, _matrix(expected), atol=EXACT)


def _vector(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value.astype(np.float64)


def _matrix(value: object) -> np.ndarray:
    assert isinstance(value, np.ndarray)
    return value.astype(np.float64)


def _relationship(
    relationship: str,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=size)
    if relationship == "linear":
        noise = rng.normal(size=size)
        return x, 0.7 * x + np.sqrt(1.0 - 0.7**2) * noise
    if relationship == "independent":
        return x, rng.normal(size=size)
    if relationship == "quadratic":
        return x, x * x + 0.1 * rng.normal(size=size)
    if relationship == "sin":
        return x, np.sin(x) + 0.05 * rng.normal(size=size)
    return x, x * x + 0.1 * rng.normal(size=size)

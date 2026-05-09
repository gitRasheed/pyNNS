from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _r import nns
from _tolerances import EXACT
from numpy.typing import NDArray


@pytest.mark.parity
def test_co_lpm_smoke() -> None:
    result = nns("Co.LPM", 1, [-1, 1], [-1, 1], 0, 0)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array(0.5), atol=EXACT)


@pytest.mark.parity
def test_pm_matrix_smoke() -> None:
    result = nns("PM.matrix", 1, 1, [0, 0], [[-1, -1], [1, 1]], True)

    assert isinstance(result, dict)
    assert set(result) == {"cupm", "dupm", "dlpm", "clpm", "cov.matrix"}
    for value in result.values():
        assert isinstance(value, np.ndarray)
        assert value.shape == (2, 2)

    cupm = cast(NDArray[np.float64], result["cupm"])
    clpm = cast(NDArray[np.float64], result["clpm"])
    dupm = cast(NDArray[np.float64], result["dupm"])
    dlpm = cast(NDArray[np.float64], result["dlpm"])
    cov_matrix = cast(NDArray[np.float64], result["cov.matrix"])

    expected_half = np.full((2, 2), 0.5)
    expected_zero = np.zeros((2, 2))
    expected_cov = np.ones((2, 2))

    assert np.allclose(cupm, expected_half, atol=EXACT)
    assert np.allclose(clpm, expected_half, atol=EXACT)
    assert np.allclose(dupm, expected_zero, atol=EXACT)
    assert np.allclose(dlpm, expected_zero, atol=EXACT)
    assert np.allclose(cov_matrix, expected_cov, atol=EXACT)

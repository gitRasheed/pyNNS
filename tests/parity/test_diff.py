from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import dy_dx_overall, nns_diff_custom
from _tolerances import EXACT

from pynns import dy_dx, nns_diff

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


@pytest.mark.parity
def test_dy_dx_overall_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    y = x + np.sin(x)

    expected = float(np.asarray(dy_dx_overall(x.tolist(), y.tolist()), dtype=np.float64))
    actual = dy_dx(x, y, eval_point="overall")

    assert actual == pytest.approx(expected, abs=EXACT)


def _r_nns_diff(name: str, point: float) -> dict[str, float]:
    result = nns_diff_custom(name, point)
    assert isinstance(result, dict)
    return {
        key: float(np.asarray(value).reshape(-1)[0])
        for key, value in result.items()
        if isinstance(value, np.ndarray)
    }

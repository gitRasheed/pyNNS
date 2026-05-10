from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_diff


def test_nns_diff_constant_derivative_is_zero() -> None:
    result = nns_diff(lambda x: 12.0, 3.0)

    assert result["DERIVATIVE"] == pytest.approx(0.0)


def test_nns_diff_identity_derivative_is_one() -> None:
    result = nns_diff(lambda x: x, -2.0)

    assert result["DERIVATIVE"] == pytest.approx(1.0)


def test_nns_diff_smooth_function_derivative_has_bounded_error() -> None:
    point = 1.25
    result = nns_diff(np.sin, point)

    assert result["DERIVATIVE"] == pytest.approx(np.cos(point), abs=1e-6)

from __future__ import annotations

import numpy as np
from _tolerances import EXACT

from pynns import lpm, upm


def test_mean_decomposes_into_upm_minus_lpm() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])

    assert np.isclose(x.mean(), upm(1, 0, x) - lpm(1, 0, x), atol=EXACT)


def test_population_variance_decomposes_into_second_partial_moments() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    target = x.mean()

    assert np.isclose(np.var(x, ddof=0), upm(2, target, x) + lpm(2, target, x), atol=EXACT)


def test_lpm_zero_at_sorted_points_is_empirical_cdf() -> None:
    x = np.array([-3.0, -1.0, 0.5, 2.0, 4.0])
    sorted_x = np.sort(x)
    expected = np.arange(1, x.size + 1) / x.size

    np.testing.assert_allclose(lpm(0, sorted_x, x), expected, atol=EXACT)


def test_lpm_is_non_negative() -> None:
    x = np.array([-2.0, 0.0, 3.0])

    assert lpm(2, 1.0, x) >= 0


def test_upm_is_non_negative() -> None:
    x = np.array([-2.0, 0.0, 3.0])

    assert upm(2, 1.0, x) >= 0


def test_partial_moment_sum_positive_unless_constant_equal_to_target() -> None:
    x = np.array([-2.0, 0.0, 3.0])

    assert lpm(2, 1.0, x) + upm(2, 1.0, x) > 0
    assert lpm(2, 2.0, np.array([2.0, 2.0])) + upm(2, 2.0, np.array([2.0, 2.0])) == 0


def test_lpm_upm_symmetry() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0])
    target = 0.75

    assert np.isclose(lpm(2, target, x), upm(2, -target, -x), atol=EXACT)

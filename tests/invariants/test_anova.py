from __future__ import annotations

import numpy as np

from pynns import nns_anova


def test_nns_anova_identical_degenerate_groups_match_r_nan_convention() -> None:
    x = np.linspace(-2.0, 2.0, 100)
    constant = np.ones_like(x)

    result = nns_anova(constant, constant, confidence_interval=None)

    assert isinstance(result, dict)
    assert np.isnan(result["Certainty"])
    assert result["Control"] == result["Treatment"]


def test_nns_anova_binary_output_structure_without_ci_matches_r_shape() -> None:
    x = np.linspace(-1.0, 1.0, 50)
    y = x + 0.2

    result = nns_anova(x, y, confidence_interval=None)

    assert list(result) == [
        "Control",
        "Treatment",
        "Grand_Statistic",
        "Control_CDF",
        "Treatment_CDF",
        "Certainty",
    ]


def test_nns_anova_pairwise_matrix_is_symmetric_with_unit_diagonal() -> None:
    x = np.linspace(-2.0, 2.0, 80)
    groups = [x, x + 0.2, np.sin(x)]

    result = nns_anova(groups, confidence_interval=None, pairwise=True)

    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_allclose(np.diag(result), np.ones(3))
    assert np.all((0.0 <= result) & (result <= 1.0))

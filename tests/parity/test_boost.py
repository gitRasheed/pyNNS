from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _r import nns_boost_numeric
from _tolerances import COMPOUND

from pynns import nns_boost
from pynns.boost import _all_feature_sets, _learner_scores, _sse


@pytest.mark.parity
@pytest.mark.parametrize("depth", [None, 1, 2])
def test_nns_boost_numeric_matches_r(depth: int | None) -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)
    point = variable[:5]

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        point.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        point,
        learner_trials=10,
        cv_size=0.25,
        depth=depth,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_ivs_test_none_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 24)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable.tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        learner_trials=10,
        cv_size=0.25,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


@pytest.mark.parity
def test_nns_boost_deterministic_wider_feature_set_matches_r() -> None:
    x = np.linspace(-1.0, 1.0, 24)
    variable = np.column_stack((x, 2.0 * x, np.sin(x), np.cos(x)))
    y = x + 0.2 * np.sin(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:4].tolist(),
        learner_trials=100,
        cv_size=0.25,
        depth=None,
        features_only=False,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:4],
        learner_trials=100,
        cv_size=0.25,
        feature_importance=False,
        random_seed=4,
    )

    _assert_boost_matches(actual, expected)


def test_nns_boost_depth_1_learner_scores_match_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    scores = _learner_scores(
        variable,
        y,
        _all_feature_sets(3),
        depth=1,
        cv_size=0.25,
        objective_fn=_sse,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [
                0.40188115414190784,
                0.78285682511456822,
                35.935805501222866,
                1.0515124585617981,
                16.21790224565731,
                17.371713848236109,
                9.018035482048569,
            ]
        ),
        atol=COMPOUND,
    )


@pytest.mark.parity
def test_nns_boost_features_only_matches_r() -> None:
    x = np.linspace(-2.0, 2.0, 30)
    variable = np.column_stack((x, np.sin(x), np.cos(x)))
    y = x + np.sin(x) + 0.25 * np.cos(x)

    expected = nns_boost_numeric(
        variable.tolist(),
        y.tolist(),
        variable[:5].tolist(),
        learner_trials=10,
        cv_size=0.25,
        depth=None,
        features_only=True,
    )
    actual = nns_boost(
        variable,
        y,
        variable[:5],
        learner_trials=10,
        cv_size=0.25,
        features_only=True,
        feature_importance=False,
    )

    _assert_boost_matches(actual, expected)


def _assert_boost_matches(actual: dict[str, Any], expected: Any) -> None:
    assert isinstance(expected, dict)
    assert set(actual) == set(expected)
    for key in actual:
        if actual[key] is None:
            assert expected[key] is None
        else:
            np.testing.assert_allclose(
                np.asarray(actual[key], dtype=np.float64),
                np.asarray(expected[key], dtype=np.float64),
                atol=COMPOUND,
            )

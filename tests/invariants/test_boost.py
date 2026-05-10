from __future__ import annotations

import numpy as np

from pynns import nns_boost


def test_nns_boost_predictions_match_test_length() -> None:
    x_train, y_train, x_test = _data(100, 50, 3)

    result = nns_boost(
        x_train,
        y_train,
        x_test,
        epochs=10,
        learner_trials=10,
        cv_size=0.25,
        random_seed=123,
    )

    assert isinstance(result["results"], np.ndarray)
    assert result["results"].shape == (50,)
    assert isinstance(result["feature.weights"], np.ndarray)
    assert np.isclose(np.sum(result["feature.weights"]), 1.0)


def test_nns_boost_training_predictions_track_training_y() -> None:
    x_train = np.linspace(-2.0, 2.0, 100).reshape(-1, 1)
    y_train = 1.5 * x_train[:, 0] - 0.25

    result = nns_boost(
        x_train,
        y_train,
        x_train,
        epochs=10,
        learner_trials=10,
        cv_size=0.25,
        random_seed=123,
    )

    predictions = result["results"]
    assert isinstance(predictions, np.ndarray)
    assert np.mean(np.abs(predictions - y_train)) < 0.5


def test_nns_boost_features_only_structure() -> None:
    x_train, y_train, x_test = _data(100, 50, 5)

    result = nns_boost(
        x_train,
        y_train,
        x_test,
        epochs=10,
        learner_trials=10,
        cv_size=0.25,
        features_only=True,
        random_seed=123,
    )

    assert set(result) == {"feature.weights", "feature.frequency"}
    assert isinstance(result["feature.weights"], np.ndarray)
    assert isinstance(result["feature.frequency"], np.ndarray)


def _data(n_train: int, n_test: int, n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_row = np.linspace(-2.0, 2.0, n_train)
    test_row = np.linspace(-1.8, 1.8, n_test)
    x_train = np.column_stack([train_row ** (power + 1) for power in range(n_features)])
    x_test = np.column_stack([test_row ** (power + 1) for power in range(n_features)])
    y_train = train_row**2 + np.sin(train_row)
    return x_train, y_train, x_test

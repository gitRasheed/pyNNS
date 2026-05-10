from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from pynns import nns_boost


@pytest.mark.parity
@pytest.mark.parametrize("n_features", [3, 5])
@pytest.mark.parametrize("epochs", [10, 50])
def test_nns_boost_output_structure_matches_r(n_features: int, epochs: int) -> None:
    x_train, y_train, x_test = _linear_case(100, 50, n_features)

    expected = _r_boost(x_train, y_train, x_test, epochs=epochs, learner_trials=10)
    actual = nns_boost(
        x_train,
        y_train,
        x_test,
        epochs=epochs,
        learner_trials=10,
        cv_size=0.25,
        feature_importance=False,
        random_seed=123,
    )

    assert set(expected) == {
        "results",
        "pred.int",
        "feature.weights",
        "feature.frequency",
        "n.best",
    }
    assert set(actual) == set(expected)
    assert isinstance(actual["results"], np.ndarray)
    assert actual["results"].shape == (x_test.shape[0],)
    assert actual["pred.int"] is None
    assert isinstance(actual["feature.weights"], np.ndarray)
    assert isinstance(actual["feature.frequency"], np.ndarray)


@pytest.mark.parity
def test_nns_boost_single_feature_structure_without_r_baseline() -> None:
    x_train, y_train, x_test = _linear_case(100, 50, 1)

    actual = nns_boost(
        x_train,
        y_train,
        x_test,
        epochs=10,
        learner_trials=10,
        cv_size=0.25,
        feature_importance=False,
        random_seed=123,
    )

    assert set(actual) == {"results", "pred.int", "feature.weights", "feature.frequency", "n.best"}
    assert isinstance(actual["results"], np.ndarray)
    assert actual["results"].shape == (50,)


@pytest.mark.parity
def test_nns_boost_features_only_matches_r_shape() -> None:
    x_train, y_train, x_test = _nonlinear_case(100, 50, 3)

    expected = _r_boost(
        x_train,
        y_train,
        x_test,
        epochs=10,
        learner_trials=10,
        features_only=True,
    )
    actual = nns_boost(
        x_train,
        y_train,
        x_test,
        epochs=10,
        learner_trials=10,
        cv_size=0.25,
        features_only=True,
        random_seed=123,
    )

    assert set(expected) == {"feature.weights", "feature.frequency"}
    assert set(actual) == set(expected)
    assert isinstance(actual["feature.weights"], np.ndarray)
    assert isinstance(actual["feature.frequency"], np.ndarray)


def _linear_case(
    n_train: int,
    n_test: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_row = np.linspace(-2.0, 2.0, n_train)
    test_row = np.linspace(-1.8, 1.8, n_test)
    x_train = _features(train_row, n_features)
    x_test = _features(test_row, n_features)
    beta = np.arange(1, n_features + 1, dtype=np.float64) / n_features
    y_train = x_train @ beta + 0.05 * np.sin(np.arange(n_train))
    return x_train, y_train, x_test


def _nonlinear_case(
    n_train: int,
    n_test: int,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_row = np.linspace(-2.0, 2.0, n_train)
    test_row = np.linspace(-1.8, 1.8, n_test)
    x_train = _features(train_row, n_features)
    x_test = _features(test_row, n_features)
    y_train = train_row**2 + np.sin(train_row) + 0.03 * np.cos(np.arange(n_train))
    return x_train, y_train, x_test


def _features(row: np.ndarray, n_features: int) -> np.ndarray:
    columns = [row]
    if n_features >= 2:
        columns.append(row**2)
    if n_features >= 3:
        columns.append(np.sin(row))
    if n_features >= 4:
        columns.append(np.cos(row))
    if n_features >= 5:
        columns.append(row**3)
    return np.column_stack(columns[:n_features])


def _r_boost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    epochs: int,
    learner_trials: int,
    features_only: bool = False,
) -> dict[str, object]:
    script = (
        "library(NNS)\n"
        "args <- jsonlite::fromJSON(paste(readLines('stdin'), collapse = '\\n'))\n"
        "set.seed(123)\n"
        "result <- NNS::NNS.boost(args$x_train, unlist(args$y_train), args$x_test, "
        "epochs = args$epochs, learner.trials = args$learner_trials, CV.size = 0.25, "
        "features.only = args$features_only, feature.importance = FALSE, status = FALSE)\n"
        "encode <- function(x) {\n"
        "  if (is.null(x)) return(list(type = 'null', value = NULL))\n"
        "  if (is.list(x)) return(list(type = 'list', value = lapply(x, encode)))\n"
        "  list(type = 'numeric', value = as.numeric(x))\n"
        "}\n"
        "cat(jsonlite::toJSON(encode(result), auto_unbox = TRUE, digits = NA))\n"
    )
    payload = {
        "x_train": x_train.tolist(),
        "y_train": y_train.tolist(),
        "x_test": x_test.tolist(),
        "epochs": epochs,
        "learner_trials": learner_trials,
        "features_only": features_only,
    }
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
    return cast(dict[str, object], _decode_r(json.loads(completed.stdout)))


def _decode_r(value: dict[str, Any]) -> dict[str, object] | np.ndarray | None:
    if value["type"] == "null":
        return None
    if value["type"] == "numeric":
        return np.asarray(value["value"], dtype=np.float64)
    return {key: _decode_r(item) for key, item in value["value"].items()}

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pynns.dependence import _gravity

BoostResult = dict[str, NDArray[np.float64] | None | int]
Objective = Literal["min", "max"]


def nns_boost(
    ivs_train: NDArray[np.float64],
    dv_train: NDArray[np.float64],
    ivs_test: NDArray[np.float64] | None = None,
    *,
    type: str | None = None,
    depth: int | str | None = None,
    learner_trials: int = 100,
    epochs: int | None = None,
    cv_size: float | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    threshold: float | None = None,
    objective: Objective | str = "min",
    extreme: bool = False,
    features_only: bool = False,
    feature_importance: bool = True,
    pred_int: float | None = None,
    status: bool = True,
    random_seed: int | None = None,
) -> BoostResult:
    """NNS-style feature-subset boosting for continuous numeric data."""
    del depth, feature_importance, pred_int, status
    if type is not None:
        raise NotImplementedError("type='CLASS' is not ported; continuous numeric boosting only.")
    if balance:
        raise NotImplementedError("balance=True requires R's classification sampling path.")
    if ts_test is not None:
        raise NotImplementedError("ts_test uses R's DTW/time-series path and is not ported.")

    x = _as_2d(ivs_train, "ivs_train")
    y = _as_1d(dv_train, "dv_train")
    if x.shape[0] != y.size:
        raise ValueError("ivs_train and dv_train must have the same number of rows.")
    z = x if ivs_test is None else _as_2d(ivs_test, "ivs_test")
    if z.shape[1] != x.shape[1]:
        raise ValueError("ivs_test must have the same number of columns as ivs_train.")

    objective_value = objective.lower()
    if objective_value not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    if learner_trials < 1:
        raise ValueError("learner_trials must be >= 1.")

    rng = np.random.default_rng(random_seed)
    n_features = x.shape[1]
    if epochs is None:
        epochs = 2 * y.size
    cv_fraction = float(np.round(rng.uniform(0.2, 1.0 / 3.0), 3)) if cv_size is None else cv_size
    if not 0.0 < cv_fraction < 1.0:
        raise ValueError("cv_size must be in (0, 1).")

    feature_sets = _feature_sets(n_features)
    deterministic = (len(feature_sets) < y.size) or n_features <= 10
    if deterministic:
        trial_sets = feature_sets
    else:
        trial_sets = [
            tuple(
                sorted(
                    rng.choice(n_features, size=rng.integers(2, n_features + 1), replace=False)
                )
            )
            for _ in range(learner_trials)
        ]

    trial_scores = np.array(
        [
            _score_feature_set(x, y, features, cv_fraction, index, objective_value, rng)
            for index, features in enumerate(trial_sets, start=1)
        ],
        dtype=np.float64,
    )
    if threshold is None:
        if extreme:
            threshold_value = (
                float(np.nanmax(trial_scores))
                if objective_value == "max"
                else float(np.nanmin(trial_scores))
            )
        else:
            fivenum = np.percentile(trial_scores[np.isfinite(trial_scores)], [0, 25, 50, 75, 100])
            threshold_value = float(fivenum[3] if objective_value == "max" else fivenum[1])
    else:
        threshold_value = float(threshold)

    if extreme:
        best_index = _best_index(trial_scores, objective_value)
        keeper_features = [trial_sets[best_index]]
    else:
        if objective_value == "max":
            keeper_features = [
                features
                for features, score in zip(trial_sets, trial_scores, strict=True)
                if score >= threshold_value
            ]
        else:
            keeper_features = [
                features
                for features, score in zip(trial_sets, trial_scores, strict=True)
                if score <= threshold_value
            ]

    if not deterministic and epochs > 0:
        pool = _feature_pool(keeper_features, n_features)
        epoch_keepers: list[tuple[int, ...]] = []
        for index in range(1, epochs + 1):
            k = int(rng.integers(1, n_features + 1))
            features = tuple(sorted(set(rng.choice(pool, size=k, replace=True).tolist())))
            if not features:
                features = (int(rng.integers(0, n_features)),)
            score = _score_feature_set(x, y, features, cv_fraction, index, objective_value, rng)
            if objective_value == "max":
                if np.isnan(score):
                    score = 0.99 * threshold_value
                passes = score >= threshold_value
            else:
                if np.isnan(score):
                    score = 1.01 * threshold_value
                passes = score <= threshold_value
            if passes:
                epoch_keepers.append(features)
        if epoch_keepers:
            keeper_features = epoch_keepers

    if not keeper_features:
        best_index = _best_index(trial_scores, objective_value)
        keeper_features = [trial_sets[best_index]]

    frequency = _feature_frequency(keeper_features, n_features)
    weights = frequency / np.sum(frequency)
    if features_only:
        return {"feature.weights": weights, "feature.frequency": frequency}

    xstar_train, xstar_test = _project_weighted_features(x, z, weights)
    estimates = _interp_predict(xstar_train, y, xstar_test)
    if np.any(~np.isfinite(estimates)):
        fill = _gravity(estimates)
        estimates = np.where(np.isfinite(estimates), estimates, fill)

    return {
        "results": estimates,
        "pred.int": None,
        "feature.weights": weights,
        "feature.frequency": frequency,
        "n.best": 1,
    }


def _score_feature_set(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    features: tuple[int, ...],
    cv_fraction: float,
    iteration: int,
    objective: str,
    rng: np.random.Generator,
) -> float:
    cv_index = _cv_index(y.size, cv_fraction, iteration, rng)
    train_mask = np.ones(y.size, dtype=bool)
    train_mask[cv_index] = False
    train_x_raw = x[train_mask][:, features]
    train_x = np.vstack((_augment_with_fivenum(train_x_raw), train_x_raw))
    train_y = np.concatenate((_fivenum(y[train_mask]), y[train_mask]))
    rpm = np.column_stack((train_x, train_y))
    predicted = _knn_predict(rpm[:, :-1], rpm[:, -1], x[cv_index][:, features])
    if np.any(~np.isfinite(predicted)):
        fill = _gravity(predicted)
        predicted = np.where(np.isfinite(predicted), predicted, fill)
    actual = y[cv_index]
    if objective == "max":
        return float(np.mean(predicted == actual))
    return float(np.sum((predicted - actual) ** 2))


def _best_index(scores: NDArray[np.float64], objective: str) -> int:
    return int(np.nanargmax(scores) if objective == "max" else np.nanargmin(scores))


def _cv_index(
    n: int,
    cv_fraction: float,
    iteration: int,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    size = max(1, int(cv_fraction * n))
    if iteration <= n / 4:
        return np.unique(np.linspace(iteration - 1, n - 1, num=size).astype(np.intp))
    return np.asarray(rng.choice(n, size=size, replace=False), dtype=np.intp)


def _feature_sets(n_features: int) -> list[tuple[int, ...]]:
    return [
        tuple(combo)
        for size in range(1, n_features + 1)
        for combo in combinations(range(n_features), size)
    ]


def _feature_pool(keeper_features: list[tuple[int, ...]], n_features: int) -> NDArray[np.int64]:
    frequency = _feature_frequency(keeper_features, n_features)
    positive = frequency[frequency > 0]
    if positive.size == 0:
        return np.arange(n_features, dtype=np.int64)
    scale = frequency / np.min(positive)
    counts = np.where(scale % 1.0 < 0.5, np.floor(scale), np.ceil(scale)).astype(np.int64)
    counts = np.maximum(counts, 0)
    pool = np.repeat(np.arange(n_features, dtype=np.int64), counts)
    if pool.size == 0:
        return np.arange(n_features, dtype=np.int64)
    return pool


def _feature_frequency(
    keeper_features: list[tuple[int, ...]],
    n_features: int,
) -> NDArray[np.float64]:
    frequency = np.zeros(n_features, dtype=np.float64)
    for features in keeper_features:
        for feature in features:
            frequency[feature] += 1.0
    if not np.any(frequency):
        frequency[:] = 1.0
    return frequency


def _project_weighted_features(
    x: NDArray[np.float64],
    z: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    joint = np.vstack((z, x))
    mins = np.min(joint, axis=0)
    ranges = np.ptp(joint, axis=0)
    ranges = np.where(ranges == 0.0, 1.0, ranges)
    joint_norm = (joint - mins) / ranges
    active = max(1, int(np.sum(weights != 0.0)))
    projected = joint_norm @ weights / active
    return projected[z.shape[0] :], projected[: z.shape[0]]


def _interp_predict(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.float64],
    test_x: NDArray[np.float64],
) -> NDArray[np.float64]:
    order = np.argsort(train_x)
    x_sorted = train_x[order]
    y_sorted = train_y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    unique_y = np.zeros_like(unique_x)
    counts = np.bincount(inverse)
    np.add.at(unique_y, inverse, y_sorted)
    unique_y = unique_y / counts
    if unique_x.size == 1:
        return np.full(test_x.shape, unique_y[0], dtype=np.float64)
    return np.interp(test_x, unique_x, unique_y)


def _knn_predict(
    train_x: NDArray[np.float64],
    train_y: NDArray[np.float64],
    test_x: NDArray[np.float64],
) -> NDArray[np.float64]:
    distances = np.linalg.norm(test_x[:, np.newaxis, :] - train_x[np.newaxis, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return np.asarray(train_y[nearest], dtype=np.float64)


def _augment_with_fivenum(x: NDArray[np.float64]) -> NDArray[np.float64]:
    columns = [_fivenum(x[:, col]) for col in range(x.shape[1])]
    return np.column_stack(columns)


def _fivenum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.percentile(values, [0, 25, 50, 75, 100]).astype(np.float64)


def _as_2d(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D.")
    if values.size == 0 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_1d(value: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pynns.categorical import _balance_class_training, _dense_factor_codes
from pynns.dependence import _gravity
from pynns.regression import (
    Order,
    _normalize_type,
    _prepare_y_values,
    _r_minmax_columns,
    _round_clamp_classes,
    nns_reg,
)
from pynns.stack import nns_stack

Objective = Literal["min", "max"]
BoostResult = dict[str, Any]


def nns_boost(
    ivs_train: NDArray[np.float64],
    dv_train: NDArray[np.float64],
    ivs_test: NDArray[np.float64] | None = None,
    *,
    type: str | None = None,
    depth: Order = None,
    learner_trials: int = 100,
    epochs: int | None = None,
    cv_size: float | None = None,
    balance: bool = False,
    ts_test: int | None = None,
    threshold: float | None = None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None = None,
    objective: Objective = "min",
    extreme: bool = False,
    features_only: bool = False,
    feature_importance: bool = True,
    pred_int: float | None = None,
    status: bool = False,
    random_seed: int | None = None,
    class_levels: list[object] | None = None,
) -> BoostResult:
    """Deterministic NNS.boost port using real NNS.reg and NNS.stack internals."""
    del status
    type_value = _normalize_type(type)
    if balance:
        type_value = "class"
    if ts_test is not None:
        raise NotImplementedError("ts_test requires the time-series boost path, deferred.")
    if pred_int is not None and type_value == "class":
        raise NotImplementedError("nns_boost pred_int for classification is not yet ported.")

    x_train = _as_matrix(ivs_train, "ivs_train")
    x_test = x_train.copy() if ivs_test is None else _as_point_matrix(ivs_test, x_train.shape[1])
    if balance:
        y_train, class_codes = _dense_factor_codes(dv_train, levels=class_levels)
    elif type_value == "class":
        y_train, _ = _prepare_y_values(
            dv_train,
            type_value=type_value,
            class_levels=class_levels,
        )
        class_codes = np.unique(y_train[np.isfinite(y_train)])
    else:
        y_train = _as_vector(dv_train, "dv_train")
        class_codes = np.empty(0, dtype=np.float64)
    if x_train.shape[0] != y_train.size:
        raise ValueError("ivs_train and dv_train must have the same row count.")
    if x_train.shape[1] > 10:
        raise NotImplementedError(
            "nns_boost for n_features > 10 requires R's stochastic epoch keeper loop, "
            "which is not yet ported. Use <=10 features or port the epoch loop first."
        )
    rng = np.random.default_rng(random_seed)
    if balance:
        x_train, y_train = _balance_class_training(
            x_train,
            y_train,
            classes=class_codes,
            rng=rng,
        )

    try:
        return _nns_boost_core(
            x_train,
            y_train,
            x_test,
            type_value=type_value,
            depth=depth,
            learner_trials=learner_trials,
            cv_size=cv_size,
            threshold=threshold,
            obj_fn=obj_fn,
            objective=objective,
            extreme=extreme,
            features_only=features_only,
            feature_importance=feature_importance,
            pred_int=pred_int,
            rng=rng,
        )
    except NotImplementedError:
        raise
    except Exception:
        if not balance:
            raise
        warnings.warn(
            "[retry] First attempt failed; retrying with balance = False",
            RuntimeWarning,
            stacklevel=2,
        )
        return nns_boost(
            ivs_train,
            dv_train,
            ivs_test,
            type=type,
            depth=depth,
            learner_trials=learner_trials,
            epochs=epochs,
            cv_size=cv_size,
            balance=False,
            ts_test=ts_test,
            threshold=threshold,
            obj_fn=obj_fn,
            objective=objective,
            extreme=extreme,
            features_only=features_only,
            feature_importance=feature_importance,
            pred_int=pred_int,
            random_seed=random_seed,
            class_levels=class_levels,
        )


def _nns_boost_core(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    *,
    type_value: str | None,
    depth: Order,
    learner_trials: int,
    cv_size: float | None,
    threshold: float | None,
    obj_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] | None,
    objective: Objective,
    extreme: bool,
    features_only: bool,
    feature_importance: bool,
    pred_int: float | None,
    rng: np.random.Generator,
) -> BoostResult:
    objective_l = objective.lower()
    if objective_l not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'.")
    objective_value: Objective = "min" if objective_l == "min" else "max"
    if type_value == "class" and obj_fn is None:
        objective_value = "max"
        objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float] = _accuracy
    else:
        objective_fn = _sse if obj_fn is None else obj_fn

    n_rows, n_cols = x_train.shape
    feature_sets = _all_feature_sets(n_cols)
    deterministic = (len(feature_sets) < n_rows) or n_cols <= 10
    if deterministic:
        trial_sets = feature_sets
        learner_trials = len(trial_sets)
    else:
        learner_trials = min(learner_trials, len(feature_sets))
        trial_sets = [_random_feature_set(n_cols, rng, min_size=2) for _ in range(learner_trials)]

    if threshold is None:
        cv_fraction = 0.25 if cv_size is None else float(cv_size)
        scores = _learner_scores(
            x_train,
            y_train,
            trial_sets,
            depth=depth,
            cv_size=cv_fraction,
            objective_fn=objective_fn,
            rng=rng,
            type_value=type_value,
        )
    else:
        scores = np.asarray([threshold], dtype=np.float64)

    threshold_value = _threshold(scores, objective_value, extreme)
    if threshold is None:
        keepers = _keeper_sets(trial_sets, scores, threshold_value, objective_value, extreme)
    else:
        keepers = trial_sets
    if not keepers:
        if threshold is not None:
            if objective_value == "min":
                raise ValueError("Please increase threshold.")
            raise ValueError("Please reduce threshold.")
        best_index = int(np.nanargmin(scores) if objective_value == "min" else np.nanargmax(scores))
        keepers = [trial_sets[best_index]]

    counts = _feature_counts(keepers, n_cols)
    if np.sum(counts) == 0.0:
        counts[:] = 1.0
    weights = counts / float(np.sum(counts))
    order_idx = np.flatnonzero(counts > 0.0)
    if features_only or feature_importance:
        order_idx = order_idx[np.argsort(-counts[order_idx], kind="mergesort")]

    if features_only:
        return {
            "feature.weights": weights[order_idx],
            "feature.frequency": counts[order_idx],
        }

    coef = weights.copy()
    xstar_fit = nns_reg(
        x_train,
        y_train,
        dim_red_method=coef,
        order=depth,
        point_only=False,
    )
    xstar_train = np.asarray(xstar_fit["x.star"]["x"], dtype=np.float64)
    xstar_train = _fill_nan_with_gravity(xstar_train)
    xstar_test = _project_xstar(x_train, x_test, coef)
    xstar_test = _fill_nan_with_gravity(xstar_test)

    final_fit = nns_stack(
        np.column_stack((xstar_train, xstar_train)),
        y_train,
        np.column_stack((xstar_test, xstar_test)),
        method=1,
        objective=objective_value,
        cv_size=0.25 if cv_size is None else cv_size,
        type=type_value,
        pred_int=pred_int,
    )
    estimates = np.asarray(final_fit["stack"], dtype=np.float64)
    if estimates.size == 0 or np.any(np.isnan(estimates)):
        estimates = np.asarray(final_fit["reg"], dtype=np.float64)
    estimates = _fill_nan_with_gravity(estimates)
    if type_value == "class":
        estimates = _round_clamp_classes(estimates, y_train)

    return {
        "results": estimates,
        "pred.int": final_fit["pred.int"],
        "feature.weights": weights[order_idx],
        "feature.frequency": counts[order_idx],
        "n.best": final_fit["NNS.reg.n.best"],
    }


def _all_feature_sets(n_cols: int) -> list[tuple[int, ...]]:
    return [
        combo
        for size in range(1, n_cols + 1)
        for combo in itertools.combinations(range(n_cols), size)
    ]


def _random_feature_set(
    n_cols: int,
    rng: np.random.Generator,
    *,
    min_size: int,
) -> tuple[int, ...]:
    low = min(min_size, n_cols)
    size = int(rng.integers(low, n_cols + 1))
    return tuple(sorted(rng.choice(n_cols, size=size, replace=False).astype(int).tolist()))


def _boost_cv_split(
    n_rows: int,
    iteration: int,
    cv_size: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    test_count = max(1, int(cv_size * n_rows))
    if iteration <= n_rows / 4.0:
        one_based = np.linspace(iteration, n_rows, test_count).astype(np.int64)
        test_idx = np.clip(one_based - 1, 0, n_rows - 1)
    else:
        test_idx = rng.choice(n_rows, size=test_count, replace=False).astype(np.int64)
    mask = np.ones(n_rows, dtype=bool)
    mask[np.unique(test_idx)] = False
    return np.flatnonzero(mask).astype(np.int64), test_idx


def _learner_scores(
    x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    feature_sets: list[tuple[int, ...]],
    *,
    depth: Order,
    cv_size: float,
    objective_fn: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    rng: np.random.Generator,
    type_value: str | None = None,
) -> NDArray[np.float64]:
    scores = np.empty(len(feature_sets), dtype=np.float64)
    for idx, features in enumerate(feature_sets, start=1):
        train_idx, test_idx = _boost_cv_split(y_train.size, idx, cv_size, rng)
        aug_x, aug_y = _augmented_training(x_train[train_idx], y_train[train_idx])
        train_subset = aug_x[:, features]
        point_subset = x_train[test_idx][:, features]
        if len(features) == 1:
            predicted = nns_reg(
                train_subset.reshape(-1),
                aug_y,
                point_est=point_subset.reshape(-1),
                order=depth,
                point_only=False,
                type=type_value,
            )["Point.est"]
        else:
            predicted = nns_reg(
                train_subset,
                aug_y,
                point_est=point_subset,
                dim_red_method="equal",
                order=depth,
                point_only=False,
                type=type_value,
            )["Point.est"]
        pred = _fill_nan_with_gravity(np.asarray(predicted, dtype=np.float64))
        if type_value == "class":
            pred = _round_clamp_classes(pred, y_train)
        scores[idx - 1] = objective_fn(pred, y_train[test_idx])
    return scores


def _augmented_training(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    joined = np.column_stack((y, x))
    five = np.column_stack([_fivenum(joined[:, col]) for col in range(joined.shape[1])])
    aug = np.vstack((five[:, 1:], x))
    aug_y = np.concatenate((five[:, 0], y))
    return aug, aug_y


def _fivenum(values: NDArray[np.float64]) -> NDArray[np.float64]:
    sorted_values = np.sort(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    n = sorted_values.size
    if n == 0:
        return np.full(5, np.nan, dtype=np.float64)
    n4 = math.floor((n + 3) / 2.0) / 2.0
    positions = np.array([1.0, n4, (n + 1) / 2.0, n + 1.0 - n4, float(n)])
    lower = np.floor(positions).astype(np.int64) - 1
    upper = np.ceil(positions).astype(np.int64) - 1
    return np.asarray(0.5 * (sorted_values[lower] + sorted_values[upper]), dtype=np.float64)


def _threshold(scores: NDArray[np.float64], objective: Objective, extreme: bool) -> float:
    clean = scores[np.isfinite(scores)]
    if clean.size == 0:
        return math.nan
    if extreme:
        return float(np.max(clean) if objective == "max" else np.min(clean))
    five = _fivenum(clean)
    return float(five[3] if objective == "max" else five[1])


def _keeper_sets(
    feature_sets: list[tuple[int, ...]],
    scores: NDArray[np.float64],
    threshold: float,
    objective: Objective,
    extreme: bool,
) -> list[tuple[int, ...]]:
    if extreme:
        target = float(np.nanmax(scores) if objective == "max" else np.nanmin(scores))
        return [feature_sets[int(np.flatnonzero(scores == target)[0])]]
    keepers: list[tuple[int, ...]] = []
    for features, score in zip(feature_sets, scores, strict=True):
        if objective == "max" and score >= threshold:
            keepers.append(features)
        if objective == "min" and score <= threshold:
            keepers.append(features)
    return keepers


def _feature_counts(feature_sets: list[tuple[int, ...]], n_cols: int) -> NDArray[np.float64]:
    counts = np.zeros(n_cols, dtype=np.float64)
    for features in feature_sets:
        for feature in features:
            counts[feature] += 1.0
    return counts


def _project_xstar(
    x_train: NDArray[np.float64],
    x_test: NDArray[np.float64],
    coef: NDArray[np.float64],
) -> NDArray[np.float64]:
    active = int(np.sum(np.abs(coef) > 0.0))
    if active == 0:
        active = 1
    joint = np.vstack((x_test, x_train))
    norm = _r_minmax_columns(joint, zero_guard=True)
    return np.asarray(norm[: x_test.shape[0]] @ coef / active, dtype=np.float64)


def _fill_nan_with_gravity(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    if np.any(np.isnan(out)):
        finite = out[np.isfinite(out)]
        fill = _gravity(finite) if finite.size else 0.0
        out[np.isnan(out)] = fill
    return out


def _sse(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.sum((predicted - actual) ** 2))


def _accuracy(predicted: NDArray[np.float64], actual: NDArray[np.float64]) -> float:
    return float(np.mean(predicted == actual))


def _as_matrix(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty numeric vector or matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values


def _as_point_matrix(x: NDArray[np.float64], n_cols: int) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        if n_cols == 1:
            values = values.reshape(-1, 1)
        else:
            values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != n_cols:
        raise ValueError("ivs_test must have the same column count as ivs_train.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ivs_test must contain only finite values.")
    return values


def _as_vector(x: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    return values

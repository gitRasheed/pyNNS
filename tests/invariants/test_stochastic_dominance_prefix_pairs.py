from __future__ import annotations

import numpy as np
import pytest

from pynns import nns_sd_cluster, sd_efficient_set
from pynns import stochastic_dominance as sd


@pytest.mark.parametrize(
    ("degree", "discrete"),
    [
        (1, True),
        (1, False),
        (2, True),
        (3, True),
    ],
)
@pytest.mark.parametrize(
    "returns",
    [
        np.asarray(
            [
                [0.0, 0.0, -1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, -1.0, 2.0],
                [2.0, 2.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 2.0, 2.0, 2.0],
                [4.0, 4.0, 3.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [
                [0.0, 0.2, -1.0, 0.7],
                [0.0, 0.3, 2.0, -0.2],
                [1.0, 0.5, -0.5, 1.4],
                [1.0, 0.8, 2.5, -0.1],
                [2.0, 1.1, 0.0, 0.2],
                [2.0, 1.3, 1.2, 1.9],
            ],
            dtype=np.float64,
        ),
    ],
)
def test_prefix_pair_dominance_matrix_matches_global_grid(
    degree: int,
    discrete: bool,
    returns: np.ndarray,
) -> None:
    global_precomputed = sd._precompute_sd_table(returns, degree, discrete=discrete)
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=discrete)

    expected = sd._dominance_matrix_from_precomputed(global_precomputed, degree)
    actual = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=discrete,
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_prefix_pair_dominance_matrix_matches_random_global_grid(degree: int) -> None:
    rng = np.random.default_rng(20260517 + degree)
    returns = rng.normal(size=(11, 9))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.25
    returns[:, 3] = np.round(returns[:, 3], 1)

    global_precomputed = sd._precompute_sd_table(returns, degree, discrete=True)
    prefix_precomputed = sd._prefix_sd_precompute(returns, degree, discrete=True)

    expected = sd._dominance_matrix_from_precomputed(global_precomputed, degree)
    actual = sd._dominance_matrix_from_prefix_pairs(
        prefix_precomputed,
        degree,
        discrete=True,
    )

    np.testing.assert_array_equal(actual, expected)


def test_sd_efficient_set_prefix_path_matches_lazy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = sd_efficient_set(returns, 2)

    monkeypatch.setattr(sd, "_SD_PREFIX_PAIR_MATRIX_MIN_COLUMNS", 1)
    prefix = sd_efficient_set(returns, 2)

    assert prefix == lazy


@pytest.mark.parametrize("dendrogram", [False, True])
def test_nns_sd_cluster_prefix_path_matches_lazy_path(
    monkeypatch: pytest.MonkeyPatch,
    dendrogram: bool,
) -> None:
    returns = _large_fixture()

    monkeypatch.setattr(sd, "_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS", returns.shape[1] + 1)
    lazy = nns_sd_cluster(returns, degree=2, min_cluster=1, dendrogram=dendrogram)

    monkeypatch.setattr(sd, "_SD_CLUSTER_DOMINANCE_MATRIX_MIN_COLUMNS", 1)
    prefix = nns_sd_cluster(returns, degree=2, min_cluster=1, dendrogram=dendrogram)

    if not dendrogram:
        assert prefix == lazy
        return

    assert prefix["Clusters"] == lazy["Clusters"]
    prefix_dendrogram = prefix["Dendrogram"]
    lazy_dendrogram = lazy["Dendrogram"]
    assert isinstance(prefix_dendrogram, dict)
    assert isinstance(lazy_dendrogram, dict)
    for key in ("merge", "height", "order", "labels"):
        np.testing.assert_array_equal(prefix_dendrogram[key], lazy_dendrogram[key])
    assert prefix_dendrogram["method"] == lazy_dendrogram["method"]
    assert prefix_dendrogram["dist.method"] == lazy_dendrogram["dist.method"]


def _large_fixture() -> np.ndarray:
    rng = np.random.default_rng(17)
    returns = rng.normal(size=(16, 80))
    returns[:, 1] = returns[:, 0]
    returns[:, 2] = returns[:, 0] + 0.2
    returns[:, 3] = np.linspace(-1.0, 1.0, returns.shape[0])
    returns[:, 4] = returns[:, 3][::-1]
    returns[:, 5:10] = np.round(returns[:, 5:10], 1)
    return returns

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray


@dataclass(frozen=True)
class EdgeCase:
    name: str
    values: NDArray[np.float64] | NDArray[np.int64]


@pytest.fixture(
    params=[
        EdgeCase("empty", np.array([], dtype=np.float64)),
        EdgeCase("single-element", np.array([1.0], dtype=np.float64)),
        EdgeCase("all-identical", np.array([2.0, 2.0, 2.0], dtype=np.float64)),
        EdgeCase("all-zeros", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        EdgeCase("all-positive", np.array([1.0, 2.0, 3.0], dtype=np.float64)),
        EdgeCase("all-negative", np.array([-1.0, -2.0, -3.0], dtype=np.float64)),
        EdgeCase("contains-nan", np.array([1.0, np.nan, 3.0], dtype=np.float64)),
        EdgeCase("contains-inf", np.array([1.0, np.inf, 3.0], dtype=np.float64)),
        EdgeCase("very-large", np.array([1e15, 2e15, 3e15], dtype=np.float64)),
        EdgeCase("very-small", np.array([1e-15, 2e-15, 3e-15], dtype=np.float64)),
        EdgeCase("integer-dtype", np.array([1, 2, 3], dtype=np.int64)),
    ],
    ids=lambda case: case.name,
)
def edge_case(request: pytest.FixtureRequest) -> EdgeCase:
    return request.param  # type: ignore[no-any-return]

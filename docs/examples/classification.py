from __future__ import annotations

import numpy as np

from pynns import nns_reg


def main() -> None:
    x = np.linspace(-2.0, 2.0, 60, dtype=np.float64)
    y = np.where(x < -0.5, 1.0, np.where(x > 0.75, 3.0, 2.0))
    points = np.array([-1.0, 0.0, 1.25], dtype=np.float64)

    fit = nns_reg(x, y, type="class", point_est=points, confidence_interval=None)
    predictions = np.asarray(fit["Point.est"], dtype=np.float64)

    assert predictions.shape == points.shape
    assert set(predictions).issubset(set(np.unique(y)))

    print("points:", points)
    print("predicted class codes:", predictions)


if __name__ == "__main__":
    main()

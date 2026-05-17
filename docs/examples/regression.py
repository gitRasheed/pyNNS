from __future__ import annotations

import numpy as np

from pynns import nns_reg


def main() -> None:
    x = np.linspace(-3.0, 3.0, 80, dtype=np.float64)
    y = np.sin(x) + 0.2 * x
    points = np.array([-1.5, 0.0, 1.5], dtype=np.float64)

    fit = nns_reg(x, y, point_est=points, confidence_interval=None)

    fitted = fit["Fitted.xy"]
    assert fitted["x"].shape == x.shape
    assert fitted["y.hat"].shape == y.shape
    assert fit["Point.est"].shape == points.shape
    assert 0.0 <= fit["R2"] <= 1.0

    print("R2:", fit["R2"])
    print("point estimates:")
    print(np.column_stack((points, fit["Point.est"])))


if __name__ == "__main__":
    main()

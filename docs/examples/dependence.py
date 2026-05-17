from __future__ import annotations

import numpy as np

from pynns import nns_cor, nns_dep


def main() -> None:
    x = np.linspace(-2.0, 2.0, 101, dtype=np.float64)
    linear_y = 2.0 * x
    nonlinear_y = x**2

    linear = nns_dep(x, linear_y)
    nonlinear = nns_dep(x, nonlinear_y)

    np.testing.assert_allclose(nns_cor(x, linear_y), linear["Correlation"])
    assert linear["Dependence"] > 0.95
    assert nonlinear["Dependence"] > abs(nonlinear["Correlation"])

    print("linear relationship:", linear)
    print("nonlinear relationship:", nonlinear)


if __name__ == "__main__":
    main()

from __future__ import annotations

import numpy as np

from pynns import lpm, nns_moments, upm


def main() -> None:
    x = np.array([-2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    target = float(np.mean(x))

    lower_degree_zero = lpm(0, target, x)
    upper_degree_zero = upm(0, target, x)
    variance_from_partials = lpm(2, target, x) + upm(2, target, x)

    np.testing.assert_allclose(lower_degree_zero + upper_degree_zero, 1.0)
    np.testing.assert_allclose(variance_from_partials, np.var(x, ddof=0))

    print("target:", target)
    print("P(x <= target):", lower_degree_zero)
    print("P(x > target):", upper_degree_zero)
    print("population variance from partial moments:", variance_from_partials)
    print("NNS moments:", nns_moments(x))


if __name__ == "__main__":
    main()

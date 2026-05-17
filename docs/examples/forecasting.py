from __future__ import annotations

import numpy as np

from pynns import nns_arma, nns_arma_optim, nns_var


def main() -> None:
    t = np.arange(1, 48, dtype=np.float64)
    series = 10.0 + np.sin(t / 3.0) + 0.05 * t

    arma = nns_arma(series, h=3, seasonal_factor=4, method="lin")
    optim = nns_arma_optim(
        series,
        h=3,
        seasonal_factor=[3, 4, 5],
        lin_only=True,
        print_trace=False,
    )

    panel = np.column_stack((series, series * 0.8 + 1.0))
    var = nns_var(panel, h=2, tau=2)

    assert arma.shape == (3,)
    assert optim["results"].shape == (3,)
    assert var["ensemble"].shape == (2, 2)

    print("ARMA forecast:", arma)
    print("optimized ARMA forecast:", optim["results"])
    print("VAR ensemble forecast:")
    print(var["ensemble"])


if __name__ == "__main__":
    main()

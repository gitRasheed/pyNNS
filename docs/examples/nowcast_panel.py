from __future__ import annotations

from collections import OrderedDict

import numpy as np

from pynns import nns_nowcast_panel


def main() -> None:
    t = np.arange(1, 25, dtype=np.float64)
    panel = OrderedDict(
        (
            ("employment", 100.0 + 0.3 * t + np.sin(t / 4.0)),
            ("inflation", 3.0 + 0.05 * np.cos(t / 3.0)),
            ("production", 80.0 + 0.5 * t + np.cos(t / 5.0)),
        )
    )
    dates = [f"2024-{month:02d}" for month in range(1, 13)] + [
        f"2025-{month:02d}" for month in range(1, 13)
    ]

    result = nns_nowcast_panel(panel, h=2, tau=2, dates=dates)

    assert result["names"] == list(panel)
    assert result["ensemble"].shape == (2, 3)
    assert result["dates"]["forecast"] == ["2026-01", "2026-02"]

    print("series:", result["names"])
    print("forecast dates:", result["dates"]["forecast"])
    print("ensemble forecast:")
    print(result["ensemble"])


if __name__ == "__main__":
    main()

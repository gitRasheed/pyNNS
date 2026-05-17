from __future__ import annotations

from collections import OrderedDict
from tempfile import NamedTemporaryFile

import numpy as np

from pynns import nns_nowcast, nns_nowcast_panel
from pynns.providers import CsvNowcastProvider


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
    matrix_result = nns_nowcast_panel(
        np.column_stack(tuple(panel.values())),
        h=1,
        tau=[1, 2, 2],
        names=list(panel),
        naive_weights=True,
    )

    # Default live fetching stays guarded; explicit providers make the boundary visible.
    try:
        nns_nowcast(h=1)
    except NotImplementedError as exc:
        guarded_message = str(exc)
    else:  # pragma: no cover - this would change the documented provider contract.
        raise AssertionError("default nns_nowcast unexpectedly fetched live data")

    with NamedTemporaryFile("w", suffix=".csv", delete=True) as handle:
        handle.write("date,employment,inflation,production\n")
        for row, month in enumerate(dates):
            handle.write(
                f"{month},{panel['employment'][row]},"
                f"{panel['inflation'][row]},{panel['production'][row]}\n"
            )
        handle.flush()
        provider_result = nns_nowcast(
            h=1,
            fetch=True,
            provider_backend=CsvNowcastProvider(handle.name),
            start_date="2024-01",
            keep_data=True,
            naive_weights=True,
        )

    assert result["names"] == list(panel)
    assert result["ensemble"].shape == (2, 3)
    assert result["dates"]["forecast"] == ["2026-01", "2026-02"]
    assert matrix_result["names"] == list(panel)
    assert matrix_result["dates"]["forecast"] == ["t+1"]
    assert "not implemented" in guarded_message
    assert provider_result["metadata"]["source"] == "provider"
    assert provider_result["ensemble"].shape == (1, 3)
    assert "raw_panel" in provider_result

    print("series:", result["names"])
    print("forecast dates:", result["dates"]["forecast"])
    print("ensemble forecast:")
    print(result["ensemble"])
    print("matrix-input next-step forecast:")
    print(matrix_result["ensemble"])
    print("guarded default message:", guarded_message)
    print("provider-backed forecast:")
    print(provider_result["ensemble"])


if __name__ == "__main__":
    main()

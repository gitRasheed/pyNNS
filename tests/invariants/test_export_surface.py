from __future__ import annotations

from pathlib import Path

import pytest

import pynns

ROOT = Path(__file__).resolve().parents[2]
API_STATUS = ROOT / "docs" / "api_status.md"

GUARDED_EXPORTS = {
    "nns_nowcast": "NNS.nowcast",
}


def test_guarded_exports_are_public_and_documented() -> None:
    docs = API_STATUS.read_text(encoding="utf-8")

    for python_name, r_name in GUARDED_EXPORTS.items():
        assert python_name in pynns.__all__
        assert hasattr(pynns, python_name)
        assert python_name in docs
        assert r_name in docs


@pytest.mark.parametrize("python_name", sorted(GUARDED_EXPORTS))
def test_guarded_exports_raise_notimplemented(python_name: str) -> None:
    function = getattr(pynns, python_name)

    with pytest.raises(NotImplementedError):
        if python_name == "nns_var":
            function([[1.0, 2.0], [2.0, 3.0]], h=1)
        elif python_name == "nns_nowcast":
            function(h=1)
        else:  # pragma: no cover - keeps the parametrized dispatch exhaustive.
            raise AssertionError(f"Unhandled guarded export: {python_name}")

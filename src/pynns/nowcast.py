from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def nns_nowcast(
    h: int = 1,
    additional_regressors: Sequence[str] | None = None,
    additional_sources: Sequence[str] | None = None,
    naive_weights: bool = False,
    specific_regressors: Sequence[int] | None = None,
    start_date: str = "2000-01-03",
    keep_data: bool = False,
    status: bool = True,
    ncores: int | None = None,
) -> dict[str, Any]:
    """Guarded placeholder for R's NNS.nowcast wrapper."""
    del (
        h,
        additional_regressors,
        additional_sources,
        naive_weights,
        specific_regressors,
        start_date,
        keep_data,
        status,
        ncores,
    )
    raise NotImplementedError(
        "nns_nowcast depends on nns_var, which depends on nns_arma_optim and "
        "nns_reg smooth=True; it is not yet ported."
    )

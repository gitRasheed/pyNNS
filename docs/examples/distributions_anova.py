from __future__ import annotations

import numpy as np

from pynns import nns_anova, nns_cdf


def main() -> None:
    control = np.linspace(-1.0, 1.0, 25, dtype=np.float64)
    treatment = control + 0.35

    cdf = nns_cdf(control, degree=0)
    comparison = nns_anova(control, treatment, confidence_interval=None)

    function = cdf["Function"]
    assert isinstance(function, dict)
    assert set(function) == {"x", "CDF"}
    assert 0.0 <= comparison["Certainty"] <= 1.0

    print("first CDF rows:")
    print(np.column_stack((function["x"][:5], function["CDF"][:5])))
    print("ANOVA certainty:", comparison["Certainty"])


if __name__ == "__main__":
    main()

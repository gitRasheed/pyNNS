from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TESTS = ROOT / "tests" / "benchmarks" / "test_lpm.py"
R_BASELINE = ROOT / "tests" / "benchmarks" / "_r_baseline.json"
BENCHMARK_DOC = ROOT / "docs" / "benchmarks.md"
REALISTIC_SD_R_PLACEHOLDERS = {
    ("sd_efficient_set", 252, 50, 1): 0.0023,
    ("sd_efficient_set", 252, 50, 2): 0.0022,
    ("nns_sd_cluster", 252, 50, 1): 0.0026,
    ("nns_sd_cluster", 252, 50, 2): 0.0073,
    ("sd_efficient_set", 252, 100, 1): 0.0052,
    ("sd_efficient_set", 252, 100, 2): 0.0046,
    ("nns_sd_cluster", 252, 100, 1): 0.0059,
    ("nns_sd_cluster", 252, 100, 2): 0.0155,
    ("sd_efficient_set", 252, 250, 2): 0.0146,
    ("nns_sd_cluster", 252, 250, 2): 0.0579,
    ("sd_efficient_set", 1257, 100, 2): 0.0199,
    ("sd_efficient_set", 252, 479, 2): 0.037,
    ("nns_sd_cluster", 252, 479, 2): 0.182,
    ("sd_efficient_set", 1257, 250, 2): 0.068,
    ("nns_sd_cluster", 1257, 250, 2): 0.186,
    ("sd_efficient_set", 1257, 479, 2): 0.167,
    ("nns_sd_cluster", 1257, 479, 2): 0.545,
}
LABEL_OVERRIDES = {
    **{
        f"test_dy_d_scalar_wrt1_100x2[{eval_points}]": (
            f"`dy_d`, scalar wrt=1, eval_points={eval_points}, N=2, T_obs=100"
        )
        for eval_points in ("mean", "median", "last", "obs", "apd")
    },
    **{
        f"test_nns_var_80x3_h3_tau2[{method}]": (
            f"`nns_var`, dim_red_method={method}, N=3, T_obs=80, h=3, tau=2"
        )
        for method in ("cor", "NNS.dep", "NNS.caus", "all")
    },
}


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    label: str
    python_seconds: float
    r_seconds: float


@dataclass(frozen=True)
class RealisticSDRow:
    function_name: str
    rows: int
    columns: int
    degree: int
    python_seconds: float
    r_seconds: float | None
    r_source: str


@dataclass(frozen=True)
class PythonOnlyRow:
    label: str
    python_seconds: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update docs/benchmarks.md from pytest-benchmark JSON and R baselines."
    )
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DOC)
    parser.add_argument(
        "--realistic-sd-r-csv",
        type=Path,
        default=None,
        help="CSV emitted by scripts/benchmark_realistic_sd_r.R.",
    )
    args = parser.parse_args()

    benchmark_payload = _read_json(args.benchmark_json)
    r_baseline = _read_json(R_BASELINE)["entries"]
    realistic_r = _read_realistic_sd_r_csv(args.realistic_sd_r_csv)
    key_by_test = _r_baseline_keys_by_test()

    rows: list[BenchmarkRow] = []
    realistic_rows: list[RealisticSDRow] = []
    python_only_rows: list[PythonOnlyRow] = []
    for benchmark in benchmark_payload["benchmarks"]:
        name = str(benchmark["name"])
        python_seconds = float(benchmark["stats"]["mean"])
        realistic_case = _realistic_sd_case_from_benchmark_name(name)
        if realistic_case is not None:
            r_seconds = realistic_r.get(realistic_case)
            r_source = "measured"
            if r_seconds is None:
                r_seconds = REALISTIC_SD_R_PLACEHOLDERS.get(realistic_case)
                r_source = "placeholder"
            realistic_rows.append(
                RealisticSDRow(
                    function_name=realistic_case[0],
                    rows=realistic_case[1],
                    columns=realistic_case[2],
                    degree=realistic_case[3],
                    python_seconds=python_seconds,
                    r_seconds=r_seconds,
                    r_source=r_source,
                )
            )
            continue
        python_only_label = _realistic_python_only_label(name)
        if python_only_label is not None:
            python_only_rows.append(
                PythonOnlyRow(label=python_only_label, python_seconds=python_seconds)
            )
            continue
        r_key = _r_baseline_key(name, key_by_test)
        r_seconds = float(r_baseline[r_key])
        label = LABEL_OVERRIDES.get(name, _fallback_label(name))
        rows.append(
            BenchmarkRow(
                name=name,
                label=label,
                python_seconds=python_seconds,
                r_seconds=r_seconds,
            )
        )

    args.output.write_text(_render(rows, realistic_rows, python_only_rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _read_realistic_sd_r_csv(
    path: Path | None,
) -> dict[tuple[str, int, int, int], float]:
    if path is None or not path.exists():
        return {}
    rows: dict[tuple[str, int, int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            function_name = row["function_name"]
            rows_count = int(row["rows"])
            columns = int(row["columns"])
            degree = int(row["degree"])
            rows[(function_name, rows_count, columns, degree)] = float(row["mean_seconds"])
    return rows


def _r_baseline_keys_by_test() -> dict[str, str]:
    tree = ast.parse(BENCHMARK_TESTS.read_text(encoding="utf-8"))
    keys: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Subscript)
                and isinstance(child.value, ast.Name)
                and child.value.id == "r_baseline"
            ):
                key = _literal_subscript(child.slice)
                if key is not None:
                    keys[node.name] = key
                    break
    return keys


def _literal_subscript(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _r_baseline_key(name: str, key_by_test: dict[str, str]) -> str:
    base_name, param = _split_benchmark_name(name)
    if base_name == "test_pm_matrix_scale":
        if param is None:
            raise KeyError(f"Missing parameter for {name}.")
        return f"pm_matrix_{param}x500_seconds"
    if base_name == "test_dy_d_scalar_wrt1_100x2":
        if param is None:
            raise KeyError(f"Missing parameter for {name}.")
        return f"dy_d_scalar_{param}_100x2_seconds"
    if base_name == "test_nns_var_80x3_h3_tau2":
        if param is None:
            raise KeyError(f"Missing parameter for {name}.")
        return f"nns_var_80x3_h3_tau2_{param.lower().replace('.', '_')}_seconds"
    if base_name in key_by_test:
        return key_by_test[base_name]
    raise KeyError(f"No R baseline key mapping found for {name}.")


def _split_benchmark_name(name: str) -> tuple[str, str | None]:
    match = re.fullmatch(r"(?P<base>.+)\[(?P<param>.+)]", name)
    if match:
        return match.group("base"), match.group("param")
    return name, None


def _benchmark_names_from_tests() -> list[str]:
    tree = ast.parse(BENCHMARK_TESTS.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if node.name == "test_pm_matrix_scale":
            names.extend([f"{node.name}[{value}]" for value in (10, 50, 100)])
        elif node.name == "test_dy_d_scalar_wrt1_100x2":
            names.extend(
                [f"{node.name}[{value}]" for value in ("mean", "median", "last", "obs", "apd")]
            )
        elif node.name == "test_nns_var_80x3_h3_tau2":
            names.extend(
                [f"{node.name}[{value}]" for value in ("cor", "NNS.dep", "NNS.caus", "all")]
            )
        else:
            names.append(node.name)
    return names


def _fallback_label(name: str) -> str:
    base_name, param = _split_benchmark_name(name)
    label = base_name.removeprefix("test_").replace("_", " ")
    if param is not None:
        label = f"{label}, {param}"
    return f"`{label}`"


def _render(
    rows: list[BenchmarkRow],
    realistic_rows: list[RealisticSDRow],
    python_only_rows: list[PythonOnlyRow],
) -> str:
    lines = [
        "# Benchmarks",
        "",
        "Run with:",
        "",
        "```bash",
        "mkdir -p docs/benchmark_reports",
        "uv run pytest -n0 -m benchmark --benchmark-enable \\",
        "  --benchmark-json=docs/benchmark_reports/benchmark_latest.json tests/benchmarks/",
        "Rscript scripts/benchmark_realistic_sd_r.R \\",
        "  --repeats=3 --max-repeats=1 \\",
        "  --output=docs/benchmark_reports/realistic_sd_r_latest.csv",
        "uv run python scripts/update_benchmarks_doc.py "
        "docs/benchmark_reports/benchmark_latest.json \\",
        "  --realistic-sd-r-csv=docs/benchmark_reports/realistic_sd_r_latest.csv",
        "```",
        "",
        "## Results",
        "",
        "`Python speed vs R` is computed as `R baseline / Python mean`. Values above `1.00x` "
        "mean Python is faster; values below `1.00x` mean Python is slower.",
        "",
        "| Benchmark | Python mean | R baseline | Python speed vs R |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row.label,
                    _format_ms(row.python_seconds),
                    _format_ms(row.r_seconds),
                    _format_speed_ratio(row.python_seconds, row.r_seconds),
                ]
            )
            + " |"
        )
    if realistic_rows:
        lines.extend(_render_realistic_sd(realistic_rows, python_only_rows))
    return "\n".join(lines) + "\n"


def _render_realistic_sd(
    realistic_rows: list[RealisticSDRow],
    python_only_rows: list[PythonOnlyRow],
) -> list[str]:
    sorted_rows = sorted(
        realistic_rows,
        key=lambda row: (row.rows, row.columns, row.degree, row.function_name),
    )
    lines = [
        "",
        "## Realistic Finance SD North Stars",
        "",
        "These benchmarks use the static daily-return fixture at",
        "`tests/fixtures/finance/sp500_daily_returns_2019_2023.csv`. The fixture contains",
        "1257 daily return rows and 479 clean return columns after dropping tickers with",
        "missing or non-finite returns.",
        "",
        "Python timings come from `pytest-benchmark`. R timings come from",
        "`scripts/benchmark_realistic_sd_r.R` when `--realistic-sd-r-csv` is supplied to",
        "the updater. Rows marked `manual placeholder` use the last manually recorded R",
        "baseline so Python/R comparisons remain visible when R has not been rerun.",
        "",
        "Run only the realistic Python benchmarks with:",
        "",
        "```bash",
        "PYNNS_OFFLINE=1 uv run pytest -q -n0 -m benchmark --benchmark-enable \\",
        "  --benchmark-json=docs/benchmark_reports/realistic_sd_python_latest.json \\",
        "  tests/benchmarks/test_stochastic_dominance_realistic.py",
        "```",
        "",
        "Run matching R baselines with:",
        "",
        "```bash",
        "Rscript scripts/benchmark_realistic_sd_r.R \\",
        "  --repeats=3 --max-repeats=1 \\",
        "  --output=docs/benchmark_reports/realistic_sd_r_latest.csv",
        "```",
        "",
        "`Python/R slowdown` is computed as `Python mean / R mean`. Values above `1.00x`",
        "mean Python is slower than R.",
        "",
        "| Realistic benchmark | Python mean | R mean | R source | Python/R slowdown |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for row in sorted_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _realistic_label(row),
                    _format_seconds(row.python_seconds),
                    _format_seconds(row.r_seconds) if row.r_seconds is not None else "n/a",
                    "measured" if row.r_source == "measured" else "manual placeholder",
                    _format_slowdown(row.python_seconds, row.r_seconds),
                ]
            )
            + " |"
        )
    if python_only_rows:
        lines.extend(
            [
                "",
                "Additional Python-only realistic building-block benchmarks from the same file:",
                "",
                "| Benchmark | Python mean |",
                "| --- | ---: |",
            ]
        )
        for row in sorted(python_only_rows, key=lambda item: item.label):
            lines.append(f"| {row.label} | {_format_seconds(row.python_seconds)} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Guarded prefix-pair evaluation skips curve work for min/mean/identical",
            "  impossible pairs, and the standalone efficient-set path only checks",
            "  already-kept candidates.",
            "- The implementation deliberately follows R's C++ SD algorithmic structure:",
            "  sorted columns, prefix sums, pair-threshold dominance checks, exact guards, and",
            "  no tolerance-based shortcuts.",
            "- Full-fixture PyNNS runs are feasible for research iteration, but R's C++ SD",
            "  core remains materially faster on the largest cluster cases.",
        ]
    )
    return lines


def _realistic_label(row: RealisticSDRow) -> str:
    return (
        f"`{row.function_name}`, degree={row.degree}, "
        f"N={row.columns}, T_obs={row.rows}"
    )


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1.0:
        return f"{seconds * 1000.0:.3f} ms"
    return f"{seconds:.3f} s"


def _format_slowdown(python_seconds: float, r_seconds: float | None) -> str:
    if r_seconds is None:
        return "n/a"
    return f"{python_seconds / r_seconds:.2f}x"


def _realistic_sd_case_from_benchmark_name(
    name: str,
) -> tuple[str, int, int, int] | None:
    base_name, param = _split_benchmark_name(name)
    if base_name == "test_sd_efficient_set_sp500_daily_returns":
        if param is None:
            return None
        degree, column_count = _parse_degree_column_param(param)
        return ("sd_efficient_set", 252, column_count, degree)
    if base_name == "test_nns_sd_cluster_sp500_daily_returns":
        if param is None:
            return None
        degree, column_count = _parse_degree_column_param(param)
        return ("nns_sd_cluster", 252, column_count, degree)
    if base_name == "test_sd_efficient_set_sp500_daily_returns_252x250_degree2":
        return ("sd_efficient_set", 252, 250, 2)
    if base_name == "test_nns_sd_cluster_sp500_daily_returns_252x250_degree2":
        return ("nns_sd_cluster", 252, 250, 2)
    if base_name == "test_sd_efficient_set_sp500_daily_returns_1257x100_degree2":
        return ("sd_efficient_set", 1257, 100, 2)
    if base_name == "test_sd_efficient_set_sp500_daily_returns_full_fixture_degree2":
        if param is None:
            return None
        rows, columns = _parse_rows_columns_param(param)
        return ("sd_efficient_set", rows, columns, 2)
    if base_name == "test_nns_sd_cluster_sp500_daily_returns_full_fixture_degree2":
        if param is None:
            return None
        rows, columns = _parse_rows_columns_param(param)
        return ("nns_sd_cluster", rows, columns, 2)
    return None


def _parse_degree_column_param(param: str) -> tuple[int, int]:
    degree_text, column_text = param.split("-", maxsplit=1)
    return int(degree_text.removeprefix("degree")), int(column_text.removeprefix("n"))


def _parse_rows_columns_param(param: str) -> tuple[int, int]:
    rows_text, columns_text = param.split("x", maxsplit=1)
    columns = 479 if columns_text == "max" else int(columns_text)
    return int(rows_text), columns


def _realistic_python_only_label(name: str) -> str | None:
    labels = {
        "test_magnificent_seven_downside_stress_components": (
            "Magnificent Seven downside stress components with SPY"
        ),
        "test_lower_upper_constituent_dispersion_ratio": (
            "Lower/upper constituent dispersion ratio, N=100, T_obs=252"
        ),
    }
    return labels.get(name)


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def _format_speed_ratio(python_seconds: float, r_seconds: float) -> str:
    return f"{r_seconds / python_seconds:.2f}x"


if __name__ == "__main__":
    main()

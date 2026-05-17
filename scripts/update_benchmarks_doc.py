from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_TESTS = ROOT / "tests" / "benchmarks" / "test_lpm.py"
R_BASELINE = ROOT / "tests" / "benchmarks" / "_r_baseline.json"
BENCHMARK_DOC = ROOT / "docs" / "benchmarks.md"
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update docs/benchmarks.md from pytest-benchmark JSON and R baselines."
    )
    parser.add_argument("benchmark_json", type=Path)
    parser.add_argument("--output", type=Path, default=BENCHMARK_DOC)
    args = parser.parse_args()

    benchmark_payload = _read_json(args.benchmark_json)
    r_baseline = _read_json(R_BASELINE)["entries"]
    key_by_test = _r_baseline_keys_by_test()
    existing_labels = _existing_doc_labels_by_name(args.output)

    rows: list[BenchmarkRow] = []
    for benchmark in benchmark_payload["benchmarks"]:
        name = str(benchmark["name"])
        python_seconds = float(benchmark["stats"]["mean"])
        r_key = _r_baseline_key(name, key_by_test)
        r_seconds = float(r_baseline[r_key])
        label = LABEL_OVERRIDES.get(name, existing_labels.get(name, _fallback_label(name)))
        rows.append(
            BenchmarkRow(
                name=name,
                label=label,
                python_seconds=python_seconds,
                r_seconds=r_seconds,
            )
        )

    args.output.write_text(_render(rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


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


def _existing_doc_labels_by_name(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts:
            labels.append(parts[0])
    return dict(zip(_benchmark_names_from_tests(), labels, strict=False))


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


def _render(rows: list[BenchmarkRow]) -> str:
    lines = [
        "# Benchmarks",
        "",
        "Run with:",
        "",
        "```bash",
        "uv run pytest -n0 -m benchmark --benchmark-enable \\",
        "  --benchmark-json=reports/benchmark_latest.json tests/benchmarks/",
        "uv run python scripts/update_benchmarks_doc.py reports/benchmark_latest.json",
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
    return "\n".join(lines) + "\n"


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def _format_speed_ratio(python_seconds: float, r_seconds: float) -> str:
    return f"{r_seconds / python_seconds:.2f}x"


if __name__ == "__main__":
    main()

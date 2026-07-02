"""Summarize per-fold no-hint and Reqo candidate runtimes.

Inputs:
  - fold_*/reqo_fold_*_split.csv
  - fold_*/reqo_fold_*_query_selection.csv
  - original_plain_runtime.csv
  - robdp_plain_runtime.csv

Outputs are intentionally fold-oriented and do not require template metadata.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Reqo fold runtime CSVs with no-hint baselines."
    )
    parser.add_argument("--fold-results-dir", type=Path, required=True)
    parser.add_argument("--original-runtime-csv", type=Path, required=True)
    parser.add_argument("--robdp-runtime-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary CSVs. Default: fold results directory.",
    )
    parser.add_argument(
        "--fold-query-summary-csv",
        type=Path,
        default=None,
        help="Override path for per-query summary CSV.",
    )
    parser.add_argument(
        "--fold-summary-csv",
        type=Path,
        default=None,
        help="Override path for per-fold summary CSV.",
    )
    parser.add_argument(
        "--overall-summary-csv",
        type=Path,
        default=None,
        help="Override path for overall summary CSV.",
    )
    return parser.parse_args()


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir or args.fold_results_dir
    return (
        args.fold_query_summary_csv or output_dir / "fold_query_summary.csv",
        args.fold_summary_csv or output_dir / "fold_summary.csv",
        args.overall_summary_csv or output_dir / "overall_summary.csv",
    )


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required file does not exist: {path}")


def as_int(value: Any) -> int:
    return int(float(str(value)))


def as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def query_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        as_int(row["fold_id"]),
        as_int(row["fold_query_idx"]),
        str(row["query_id"]),
    )


def find_csvs(fold_results_dir: Path, suffix: str) -> list[Path]:
    csvs = sorted(fold_results_dir.glob(f"fold_*/reqo_fold_*_{suffix}.csv"))
    if not csvs:
        raise ValueError(f"No {suffix} CSVs found under: {fold_results_dir}")
    return csvs


def load_split_rows(fold_results_dir: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows = {}
    for split_csv in find_csvs(fold_results_dir, "split"):
        with split_csv.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("split") != "test":
                    continue
                key = query_key(row)
                rows[key] = {
                    "fold_id": as_int(row["fold_id"]),
                    "fold_query_idx": as_int(row["fold_query_idx"]),
                    "global_query_idx": as_int(row["global_query_idx"]),
                    "query_id": str(row["query_id"]),
                    "candidate_count": as_int(row["candidate_count"]),
                }
    return rows


def load_selection_rows(fold_results_dir: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    rows = {}
    for selection_csv in find_csvs(fold_results_dir, "query_selection"):
        with selection_csv.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                key = query_key(row)
                rows[key] = {
                    "optimal_runtime_ms": as_float_or_none(row["optimal_runtime_ms"]),
                    "min_cost_pg_runtime_ms": as_float_or_none(row["postgres_runtime_ms"]),
                    "reqo_runtime_ms": as_float_or_none(row["model_runtime_ms"]),
                    "postgres_candidate_idx": row["postgres_candidate_idx"],
                    "model_candidate_idx": row["model_candidate_idx"],
                    "optimal_candidate_idx": row["optimal_candidate_idx"],
                }
    return rows


def load_plain_runtime_rows(runtime_csv: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    require_file(runtime_csv)
    runtimes_by_key = defaultdict(list)
    status_by_key = defaultdict(list)

    with runtime_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = query_key(row)
            status = row.get("status", "")
            status_by_key[key].append(status)
            if status == "ok":
                runtime = as_float_or_none(row.get("runtime_ms"))
                if runtime is not None:
                    runtimes_by_key[key].append(runtime)

    result = {}
    for key in set(status_by_key) | set(runtimes_by_key):
        runtimes = runtimes_by_key.get(key, [])
        statuses = status_by_key.get(key, [])
        result[key] = {
            "runtime_ms": mean(runtimes) if runtimes else None,
            "rounds_ok": len(runtimes),
            "rounds_total": len(statuses),
            "status": "ok" if runtimes and len(runtimes) == len(statuses) else "partial_or_error",
        }
    return result


def mean_or_blank(values: list[float | None]) -> float | str:
    present = [value for value in values if value is not None]
    return mean(present) if present else ""


def ratio_or_blank(numerator: float | None, denominator: float | None) -> float | str:
    if numerator is None or denominator in (None, 0):
        return ""
    return numerator / denominator


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_fold_query_summary(
        split_rows: dict[tuple[int, int, str], dict[str, Any]],
        selection_rows: dict[tuple[int, int, str], dict[str, Any]],
        original_plain: dict[tuple[int, int, str], dict[str, Any]],
        robdp_plain: dict[tuple[int, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(split_rows):
        split_row = split_rows[key]
        selection_row = selection_rows.get(key, {})
        original_row = original_plain.get(key, {})
        robdp_row = robdp_plain.get(key, {})

        original_runtime = original_row.get("runtime_ms")
        robdp_runtime = robdp_row.get("runtime_ms")
        optimal_runtime = selection_row.get("optimal_runtime_ms")
        min_cost_runtime = selection_row.get("min_cost_pg_runtime_ms")
        reqo_runtime = selection_row.get("reqo_runtime_ms")

        rows.append({
            **split_row,
            "original_runtime_ms": original_runtime if original_runtime is not None else "",
            "original_rounds_ok": original_row.get("rounds_ok", 0),
            "original_rounds_total": original_row.get("rounds_total", 0),
            "robdp_runtime_ms": robdp_runtime if robdp_runtime is not None else "",
            "robdp_rounds_ok": robdp_row.get("rounds_ok", 0),
            "robdp_rounds_total": robdp_row.get("rounds_total", 0),
            "optimal_runtime_ms": optimal_runtime if optimal_runtime is not None else "",
            "min_cost_pg_runtime_ms": min_cost_runtime if min_cost_runtime is not None else "",
            "reqo_runtime_ms": reqo_runtime if reqo_runtime is not None else "",
            "robdp_vs_original_ratio": ratio_or_blank(robdp_runtime, original_runtime),
            "reqo_vs_min_cost_pg_ratio": ratio_or_blank(reqo_runtime, min_cost_runtime),
            "reqo_vs_optimal_ratio": ratio_or_blank(reqo_runtime, optimal_runtime),
            "postgres_candidate_idx": selection_row.get("postgres_candidate_idx", ""),
            "model_candidate_idx": selection_row.get("model_candidate_idx", ""),
            "optimal_candidate_idx": selection_row.get("optimal_candidate_idx", ""),
        })
    return rows


RUNTIME_COLUMNS = [
    "original_runtime_ms",
    "robdp_runtime_ms",
    "optimal_runtime_ms",
    "min_cost_pg_runtime_ms",
    "reqo_runtime_ms",
]


def build_group_summary(rows: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_name]].append(row)

    summary_rows = []
    for group_value in sorted(grouped, key=lambda value: int(value)):
        group_rows = grouped[group_value]
        summary = {
            group_name: group_value,
            "query_count": len(group_rows),
        }
        for column in RUNTIME_COLUMNS:
            summary[column.replace("_runtime_ms", "_avg_ms")] = mean_or_blank([
                as_float_or_none(row[column])
                for row in group_rows
            ])
        summary["robdp_vs_original_avg_ratio"] = ratio_or_blank(
            as_float_or_none(summary["robdp_avg_ms"]),
            as_float_or_none(summary["original_avg_ms"]),
        )
        summary["reqo_vs_min_cost_pg_avg_ratio"] = ratio_or_blank(
            as_float_or_none(summary["reqo_avg_ms"]),
            as_float_or_none(summary["min_cost_pg_avg_ms"]),
        )
        summary["reqo_vs_optimal_avg_ratio"] = ratio_or_blank(
            as_float_or_none(summary["reqo_avg_ms"]),
            as_float_or_none(summary["optimal_avg_ms"]),
        )
        summary_rows.append(summary)
    return summary_rows


def build_overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = {
        "scope": "overall",
        "query_count": len(rows),
    }
    for column in RUNTIME_COLUMNS:
        summary[column.replace("_runtime_ms", "_avg_ms")] = mean_or_blank([
            as_float_or_none(row[column])
            for row in rows
        ])
    summary["robdp_vs_original_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["robdp_avg_ms"]),
        as_float_or_none(summary["original_avg_ms"]),
    )
    summary["reqo_vs_min_cost_pg_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["reqo_avg_ms"]),
        as_float_or_none(summary["min_cost_pg_avg_ms"]),
    )
    summary["reqo_vs_optimal_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["reqo_avg_ms"]),
        as_float_or_none(summary["optimal_avg_ms"]),
    )
    return [summary]


def main() -> None:
    args = parse_args()
    fold_query_summary_csv, fold_summary_csv, overall_summary_csv = output_paths(args)

    split_rows = load_split_rows(args.fold_results_dir)
    selection_rows = load_selection_rows(args.fold_results_dir)
    original_plain = load_plain_runtime_rows(args.original_runtime_csv)
    robdp_plain = load_plain_runtime_rows(args.robdp_runtime_csv)

    missing_selection = sorted(set(split_rows) - set(selection_rows))
    if missing_selection:
        print(f"Warning: {len(missing_selection)} test queries lack query_selection rows")

    fold_query_rows = build_fold_query_summary(
        split_rows=split_rows,
        selection_rows=selection_rows,
        original_plain=original_plain,
        robdp_plain=robdp_plain,
    )
    fold_rows = build_group_summary(fold_query_rows, "fold_id")
    overall_rows = build_overall_summary(fold_query_rows)

    fold_query_fields = [
        "fold_id",
        "fold_query_idx",
        "global_query_idx",
        "query_id",
        "candidate_count",
        "original_runtime_ms",
        "original_rounds_ok",
        "original_rounds_total",
        "robdp_runtime_ms",
        "robdp_rounds_ok",
        "robdp_rounds_total",
        "optimal_runtime_ms",
        "min_cost_pg_runtime_ms",
        "reqo_runtime_ms",
        "robdp_vs_original_ratio",
        "reqo_vs_min_cost_pg_ratio",
        "reqo_vs_optimal_ratio",
        "postgres_candidate_idx",
        "model_candidate_idx",
        "optimal_candidate_idx",
    ]
    fold_summary_fields = [
        "fold_id",
        "query_count",
        "original_avg_ms",
        "robdp_avg_ms",
        "optimal_avg_ms",
        "min_cost_pg_avg_ms",
        "reqo_avg_ms",
        "robdp_vs_original_avg_ratio",
        "reqo_vs_min_cost_pg_avg_ratio",
        "reqo_vs_optimal_avg_ratio",
    ]
    overall_summary_fields = ["scope"] + fold_summary_fields[1:]

    write_csv(fold_query_summary_csv, fold_query_fields, fold_query_rows)
    write_csv(fold_summary_csv, fold_summary_fields, fold_rows)
    write_csv(overall_summary_csv, overall_summary_fields, overall_rows)

    print(f"Wrote per-query summary: {fold_query_summary_csv}")
    print(f"Wrote per-fold summary: {fold_summary_csv}")
    print(f"Wrote overall summary: {overall_summary_csv}")


if __name__ == "__main__":
    main()

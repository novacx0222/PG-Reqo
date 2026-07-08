"""Summarize pre-split runner runtimes and trained Reqo fold outputs.

Expected new-flow inputs:

  - folds/original_fold_<k>.csv
  - results/original/<template_id>/<query_id>/results_*.txt
  - RobDP runtime results from run_imdb_with_robdp.py:
      <robdp_runtime>/<template_id>/<query_id>/results_*.txt
  - optional results/reqo_guc/<template_id>/<query_id>/results_*.txt
  - RobDP last-level trained fold outputs:
      <dir>/fold_<k>/reqo_fold_<k>_query_selection.csv
  - Reqo-GUC trained fold outputs:
      <dir>/fold_<k>/reqo_fold_<k>_query_selection.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fold runtimes for the pre-split Reqo pipeline."
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        required=True,
        help="Directory containing <source>_fold_<k>.csv membership files.",
    )
    parser.add_argument(
        "--fold-source",
        default="original",
        help="Fold membership source prefix. Default: original.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Root written by run_imdb_with_pg/robdp/reqo_guc.py.",
    )
    parser.add_argument("--original-results-dir", type=Path, default=None)
    parser.add_argument(
        "--robdp-runtime-results-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing RobDP no-hint runtime results for one "
            "parameter group, usually written by run_imdb_with_robdp.py."
        ),
    )
    parser.add_argument("--reqo-guc-results-dir", type=Path, default=None)
    parser.add_argument(
        "--robdp-trained-results-dir",
        type=Path,
        required=True,
        help="Directory containing RobDP last-level trained fold_<k>/ outputs.",
    )
    parser.add_argument(
        "--reqo-guc-trained-results-dir",
        type=Path,
        required=True,
        help="Directory containing Reqo-GUC trained fold_<k>/ outputs.",
    )
    parser.add_argument(
        "--runtime-agg",
        choices=("mean", "min", "median"),
        default="mean",
        help="How to combine original/RobDP repeated rounds. Default: mean.",
    )
    parser.add_argument(
        "--reqo-guc-runner-agg",
        choices=("min", "mean", "median"),
        default="min",
        help="How to combine raw Reqo-GUC runner candidates/rounds. Default: min.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for summary CSVs. Default: folds-dir.",
    )
    parser.add_argument("--fold-query-summary-csv", type=Path, default=None)
    parser.add_argument("--fold-summary-csv", type=Path, default=None)
    parser.add_argument("--overall-summary-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_input_dirs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    results_path = args.results_path.expanduser().resolve()
    original_dir = args.original_results_dir or results_path / "original"
    robdp_dir = args.robdp_runtime_results_dir
    reqo_guc_dir = args.reqo_guc_results_dir or results_path / "reqo_guc"
    return original_dir, robdp_dir, reqo_guc_dir


def output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = args.output_dir or args.folds_dir
    return (
        args.fold_query_summary_csv or output_dir / "fold_query_summary.csv",
        args.fold_summary_csv or output_dir / "fold_summary.csv",
        args.overall_summary_csv or output_dir / "overall_summary.csv",
    )


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise ValueError(f"Required {label} directory does not exist: {path}")


def as_int(value: Any) -> int:
    return int(float(str(value)))


def as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def aggregate(values: Iterable[float], method: str) -> float | None:
    present = sorted(values)
    if not present:
        return None
    if method == "mean":
        return mean(present)
    if method == "min":
        return present[0]
    if method == "median":
        mid = len(present) // 2
        if len(present) % 2:
            return present[mid]
        return (present[mid - 1] + present[mid]) / 2
    raise ValueError(f"Unknown aggregate method: {method}")


def ratio_or_blank(numerator: float | None, denominator: float | None) -> float | str:
    if numerator is None or denominator in (None, 0):
        return ""
    return numerator / denominator


def mean_or_blank(values: list[float | None]) -> float | str:
    present = [value for value in values if value is not None]
    return mean(present) if present else ""


def fold_csvs(folds_dir: Path, fold_source: str) -> list[Path]:
    csvs = sorted(folds_dir.glob(f"{fold_source}_fold_*.csv"))
    if not csvs:
        raise ValueError(f"No fold CSVs found: {folds_dir}/{fold_source}_fold_*.csv")
    return csvs


def load_fold_rows(folds_dir: Path, fold_source: str) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = {}
    for fold_csv in fold_csvs(folds_dir, fold_source):
        with fold_csv.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row.get("split") != "test":
                    continue
                fold_id = as_int(row["fold_id"])
                template_id = as_int(row["template_id"])
                original_query_id = as_int(row["original_query_id"])
                rows[(fold_id, template_id, original_query_id)] = {
                    "fold_id": fold_id,
                    "fold_query_idx": as_int(row["fold_query_idx"]),
                    "global_query_idx": as_int(row["global_query_idx"]),
                    "query_group_id": row["query_group_id"],
                    "template_id": template_id,
                    "original_query_id": original_query_id,
                    "candidate_count": as_int(row["candidate_count"]),
                }
    return rows


def query_result_files(result_root: Path, template_id: int, query_id: int) -> list[Path]:
    query_dir = result_root / str(template_id) / str(query_id)
    return sorted(query_dir.glob("results_*.txt"))


def parse_result_blocks(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current_headers: dict[str, str] = {}
    current_payload: list[str] = []

    def flush() -> None:
        nonlocal current_headers, current_payload
        if current_headers or current_payload:
            blocks.append({
                "headers": current_headers,
                "payload": "\n".join(current_payload).strip(),
            })
        current_headers = {}
        current_payload = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        header_match = re.match(r"^([^:]+):\s*(.*)$", stripped)
        if header_match and not current_payload:
            current_headers[header_match.group(1)] = header_match.group(2)
        else:
            current_payload.append(line)
    flush()
    return blocks


def parse_explain_payload(payload: str) -> dict[str, Any] | None:
    if not payload:
        return None
    candidate = payload.strip()
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start >= 0 and end >= start:
        candidate = candidate[start:end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return parsed[0]
    if isinstance(parsed, dict):
        return parsed
    return None


def extract_execution_time_ms(payload: str) -> float | None:
    explain_doc = parse_explain_payload(payload)
    if explain_doc is None:
        return None
    return as_float_or_none(explain_doc.get("Execution Time"))


def load_runner_runtime(
        result_root: Path,
        template_id: int,
        query_id: int,
        agg_method: str,
        reqo_guc: bool = False,
) -> dict[str, Any]:
    files = query_result_files(result_root, template_id, query_id)
    runtimes: list[float] = []
    failed_blocks = 0
    total_blocks = 0
    missing_execution_time = 0

    for result_file in files:
        with result_file.open("r", encoding="utf-8") as file:
            for block in parse_result_blocks(file.read()):
                headers = block["headers"]
                payload = block["payload"]
                if reqo_guc and headers.get("Status") == "failed":
                    failed_blocks += 1
                    total_blocks += 1
                    continue
                runtime = extract_execution_time_ms(payload)
                total_blocks += 1
                if runtime is None:
                    missing_execution_time += 1
                else:
                    runtimes.append(runtime)

    return {
        "runtime_ms": aggregate(runtimes, agg_method),
        "rounds_ok": len(runtimes),
        "rounds_total": total_blocks,
        "failed_blocks": failed_blocks,
        "missing_execution_time_blocks": missing_execution_time,
        "status": "ok" if runtimes and len(runtimes) == total_blocks else (
            "missing" if not files else "partial_or_error"
        ),
    }


def load_selection_rows(
        trained_results_dir: Path,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    rows = {}
    for selection_csv in sorted(
            trained_results_dir.glob("fold_*/reqo_fold_*_query_selection.csv")
    ):
        with selection_csv.open("r", encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                fold_id = as_int(row["fold_id"])
                template_id = as_int(row["template_id"])
                original_query_id = as_int(row["original_query_id"])
                rows[(fold_id, template_id, original_query_id)] = {
                    "oracle_runtime_ms": as_float_or_none(row["optimal_runtime_ms"]),
                    "min_cost_runtime_ms": as_float_or_none(row["postgres_runtime_ms"]),
                    "reqo_runtime_ms": as_float_or_none(row["model_runtime_ms"]),
                    "postgres_candidate_idx": row["postgres_candidate_idx"],
                    "model_candidate_idx": row["model_candidate_idx"],
                    "optimal_candidate_idx": row["optimal_candidate_idx"],
                }
    return rows


def prefixed_selection_fields(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    oracle = row.get("oracle_runtime_ms")
    min_cost = row.get("min_cost_runtime_ms")
    reqo = row.get("reqo_runtime_ms")
    return {
        f"{prefix}_oracle_runtime_ms": oracle if oracle is not None else "",
        f"{prefix}_min_cost_runtime_ms": min_cost if min_cost is not None else "",
        f"{prefix}_reqo_runtime_ms": reqo if reqo is not None else "",
        f"{prefix}_reqo_vs_min_cost_ratio": ratio_or_blank(reqo, min_cost),
        f"{prefix}_reqo_vs_oracle_ratio": ratio_or_blank(reqo, oracle),
        f"{prefix}_postgres_candidate_idx": row.get("postgres_candidate_idx", ""),
        f"{prefix}_model_candidate_idx": row.get("model_candidate_idx", ""),
        f"{prefix}_optimal_candidate_idx": row.get("optimal_candidate_idx", ""),
    }


def build_fold_query_summary(
        fold_rows: dict[tuple[int, int, int], dict[str, Any]],
        original_dir: Path,
        robdp_dir: Path,
        reqo_guc_dir: Path,
        robdp_selection: dict[tuple[int, int, int], dict[str, Any]],
        reqo_guc_selection: dict[tuple[int, int, int], dict[str, Any]],
        runtime_agg: str,
        reqo_guc_runner_agg: str,
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(fold_rows):
        fold_row = fold_rows[key]
        template_id = fold_row["template_id"]
        query_id = fold_row["original_query_id"]

        original_row = load_runner_runtime(
            original_dir, template_id, query_id, runtime_agg
        )
        robdp_row = load_runner_runtime(
            robdp_dir, template_id, query_id, runtime_agg
        )
        reqo_guc_runner_row = load_runner_runtime(
            reqo_guc_dir,
            template_id,
            query_id,
            reqo_guc_runner_agg,
            reqo_guc=True,
        )

        original_runtime = original_row.get("runtime_ms")
        robdp_runtime = robdp_row.get("runtime_ms")
        reqo_guc_runner_runtime = reqo_guc_runner_row.get("runtime_ms")

        rows.append({
            **fold_row,
            "original_runtime_ms": original_runtime if original_runtime is not None else "",
            "original_rounds_ok": original_row.get("rounds_ok", 0),
            "original_rounds_total": original_row.get("rounds_total", 0),
            "original_status": original_row.get("status", ""),
            "robdp_runtime_ms": robdp_runtime if robdp_runtime is not None else "",
            "robdp_rounds_ok": robdp_row.get("rounds_ok", 0),
            "robdp_rounds_total": robdp_row.get("rounds_total", 0),
            "robdp_status": robdp_row.get("status", ""),
            "reqo_guc_runner_runtime_ms": (
                reqo_guc_runner_runtime
                if reqo_guc_runner_runtime is not None
                else ""
            ),
            "reqo_guc_runner_rounds_ok": reqo_guc_runner_row.get("rounds_ok", 0),
            "reqo_guc_runner_rounds_total": reqo_guc_runner_row.get("rounds_total", 0),
            "reqo_guc_runner_failed_blocks": reqo_guc_runner_row.get("failed_blocks", 0),
            "reqo_guc_runner_status": reqo_guc_runner_row.get("status", ""),
            "robdp_vs_original_ratio": ratio_or_blank(robdp_runtime, original_runtime),
            "reqo_guc_runner_vs_original_ratio": ratio_or_blank(
                reqo_guc_runner_runtime,
                original_runtime,
            ),
            **prefixed_selection_fields(
                "robdp_last_level",
                robdp_selection.get(key, {}),
            ),
            **prefixed_selection_fields(
                "reqo_guc",
                reqo_guc_selection.get(key, {}),
            ),
        })
    return rows


RUNTIME_COLUMNS = [
    "original_runtime_ms",
    "robdp_runtime_ms",
    "reqo_guc_runner_runtime_ms",
    "robdp_last_level_oracle_runtime_ms",
    "robdp_last_level_min_cost_runtime_ms",
    "robdp_last_level_reqo_runtime_ms",
    "reqo_guc_oracle_runtime_ms",
    "reqo_guc_min_cost_runtime_ms",
    "reqo_guc_reqo_runtime_ms",
]


def avg_column_name(runtime_column: str) -> str:
    return runtime_column[:-len("_runtime_ms")] + "_avg_ms"


def build_group_summary(rows: list[dict[str, Any]], group_name: str) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_name]].append(row)

    summary_rows = []
    for group_value in sorted(grouped, key=lambda value: int(value)):
        group_rows = grouped[group_value]
        summary = {group_name: group_value, "query_count": len(group_rows)}
        for column in RUNTIME_COLUMNS:
            summary[avg_column_name(column)] = mean_or_blank([
                as_float_or_none(row[column])
                for row in group_rows
            ])
        add_summary_ratios(summary)
        summary_rows.append(summary)
    return summary_rows


def add_summary_ratios(summary: dict[str, Any]) -> None:
    summary["robdp_vs_original_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["robdp_avg_ms"]),
        as_float_or_none(summary["original_avg_ms"]),
    )
    summary["reqo_guc_runner_vs_original_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["reqo_guc_runner_avg_ms"]),
        as_float_or_none(summary["original_avg_ms"]),
    )
    summary["robdp_last_level_reqo_vs_min_cost_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["robdp_last_level_reqo_avg_ms"]),
        as_float_or_none(summary["robdp_last_level_min_cost_avg_ms"]),
    )
    summary["robdp_last_level_reqo_vs_oracle_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["robdp_last_level_reqo_avg_ms"]),
        as_float_or_none(summary["robdp_last_level_oracle_avg_ms"]),
    )
    summary["reqo_guc_reqo_vs_min_cost_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["reqo_guc_reqo_avg_ms"]),
        as_float_or_none(summary["reqo_guc_min_cost_avg_ms"]),
    )
    summary["reqo_guc_reqo_vs_oracle_avg_ratio"] = ratio_or_blank(
        as_float_or_none(summary["reqo_guc_reqo_avg_ms"]),
        as_float_or_none(summary["reqo_guc_oracle_avg_ms"]),
    )


def build_overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = {"scope": "overall", "query_count": len(rows)}
    for column in RUNTIME_COLUMNS:
        summary[avg_column_name(column)] = mean_or_blank([
            as_float_or_none(row[column])
            for row in rows
        ])
    add_summary_ratios(summary)
    return [summary]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


FOLD_QUERY_FIELDS = [
    "fold_id",
    "fold_query_idx",
    "global_query_idx",
    "query_group_id",
    "template_id",
    "original_query_id",
    "candidate_count",
    "original_runtime_ms",
    "original_rounds_ok",
    "original_rounds_total",
    "original_status",
    "robdp_runtime_ms",
    "robdp_rounds_ok",
    "robdp_rounds_total",
    "robdp_status",
    "reqo_guc_runner_runtime_ms",
    "reqo_guc_runner_rounds_ok",
    "reqo_guc_runner_rounds_total",
    "reqo_guc_runner_failed_blocks",
    "reqo_guc_runner_status",
    "robdp_vs_original_ratio",
    "reqo_guc_runner_vs_original_ratio",
    "robdp_last_level_oracle_runtime_ms",
    "robdp_last_level_min_cost_runtime_ms",
    "robdp_last_level_reqo_runtime_ms",
    "robdp_last_level_reqo_vs_min_cost_ratio",
    "robdp_last_level_reqo_vs_oracle_ratio",
    "robdp_last_level_postgres_candidate_idx",
    "robdp_last_level_model_candidate_idx",
    "robdp_last_level_optimal_candidate_idx",
    "reqo_guc_oracle_runtime_ms",
    "reqo_guc_min_cost_runtime_ms",
    "reqo_guc_reqo_runtime_ms",
    "reqo_guc_reqo_vs_min_cost_ratio",
    "reqo_guc_reqo_vs_oracle_ratio",
    "reqo_guc_postgres_candidate_idx",
    "reqo_guc_model_candidate_idx",
    "reqo_guc_optimal_candidate_idx",
]

SUMMARY_FIELDS = [
    "query_count",
    "original_avg_ms",
    "robdp_avg_ms",
    "reqo_guc_runner_avg_ms",
    "robdp_last_level_oracle_avg_ms",
    "robdp_last_level_min_cost_avg_ms",
    "robdp_last_level_reqo_avg_ms",
    "reqo_guc_oracle_avg_ms",
    "reqo_guc_min_cost_avg_ms",
    "reqo_guc_reqo_avg_ms",
    "robdp_vs_original_avg_ratio",
    "reqo_guc_runner_vs_original_avg_ratio",
    "robdp_last_level_reqo_vs_min_cost_avg_ratio",
    "robdp_last_level_reqo_vs_oracle_avg_ratio",
    "reqo_guc_reqo_vs_min_cost_avg_ratio",
    "reqo_guc_reqo_vs_oracle_avg_ratio",
]


def main() -> None:
    args = parse_args()
    original_dir, robdp_dir, reqo_guc_dir = resolve_input_dirs(args)
    folds_dir = args.folds_dir.expanduser().resolve()
    robdp_trained_dir = args.robdp_trained_results_dir.expanduser().resolve()
    reqo_guc_trained_dir = args.reqo_guc_trained_results_dir.expanduser().resolve()

    require_dir(folds_dir, "folds")
    require_dir(original_dir, "original results")
    require_dir(robdp_dir, "RobDP runtime results")
    require_dir(reqo_guc_dir, "Reqo-GUC runner results")
    require_dir(robdp_trained_dir, "RobDP trained results")
    require_dir(reqo_guc_trained_dir, "Reqo-GUC trained results")

    fold_query_summary_csv, fold_summary_csv, overall_summary_csv = output_paths(args)

    fold_rows = load_fold_rows(folds_dir, args.fold_source)
    robdp_selection = load_selection_rows(robdp_trained_dir)
    reqo_guc_selection = load_selection_rows(reqo_guc_trained_dir)

    missing_robdp = sorted(set(fold_rows) - set(robdp_selection))
    missing_reqo_guc = sorted(set(fold_rows) - set(reqo_guc_selection))
    if missing_robdp:
        print(f"Warning: {len(missing_robdp)} test queries lack RobDP selection rows")
    if missing_reqo_guc:
        print(f"Warning: {len(missing_reqo_guc)} test queries lack Reqo-GUC selection rows")

    fold_query_rows = build_fold_query_summary(
        fold_rows=fold_rows,
        original_dir=original_dir,
        robdp_dir=robdp_dir,
        reqo_guc_dir=reqo_guc_dir,
        robdp_selection=robdp_selection,
        reqo_guc_selection=reqo_guc_selection,
        runtime_agg=args.runtime_agg,
        reqo_guc_runner_agg=args.reqo_guc_runner_agg,
    )
    fold_rows_summary = build_group_summary(fold_query_rows, "fold_id")
    overall_rows = build_overall_summary(fold_query_rows)

    write_csv(fold_query_summary_csv, FOLD_QUERY_FIELDS, fold_query_rows)
    write_csv(fold_summary_csv, ["fold_id"] + SUMMARY_FIELDS, fold_rows_summary)
    write_csv(overall_summary_csv, ["scope"] + SUMMARY_FIELDS, overall_rows)

    print(f"Fold source: {args.fold_source}")
    print(f"Original results: {original_dir}")
    print(f"RobDP runtime results: {robdp_dir}")
    print(f"Reqo-GUC runner results: {reqo_guc_dir}")
    print(f"RobDP trained results: {robdp_trained_dir}")
    print(f"Reqo-GUC trained results: {reqo_guc_trained_dir}")
    print(f"Wrote per-query summary: {fold_query_summary_csv}")
    print(f"Wrote per-fold summary: {fold_summary_csv}")
    print(f"Wrote overall summary: {overall_summary_csv}")


if __name__ == "__main__":
    main()

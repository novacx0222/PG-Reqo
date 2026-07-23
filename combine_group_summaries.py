"""Combine per-parameter-group summary CSVs into one wide report.

Each group summary is produced by summarize_fold_runtimes.py. This script keeps
shared baselines once and expands RobDP-parameter-specific metrics with a group
prefix.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


SHARED_COLUMNS = [
    "query_count",
    "original_avg_ms",
    "reqo_guc_runner_avg_ms",
    "reqo_guc_oracle_avg_ms",
    "reqo_guc_min_cost_avg_ms",
    "reqo_guc_reqo_avg_ms",
    "original_planning_avg_ms",
    "reqo_guc_runner_planning_avg_ms",
    "reqo_guc_runner_vs_original_avg_ratio",
    "reqo_guc_reqo_vs_min_cost_avg_ratio",
    "reqo_guc_reqo_vs_oracle_avg_ratio",
]

ROBDP_GROUP_COLUMNS = [
    "robdp_avg_ms",
    "robdp_planning_avg_ms",
    "robdp_last_level_oracle_avg_ms",
    "robdp_last_level_min_cost_avg_ms",
    "robdp_last_level_reqo_avg_ms",
    "robdp_vs_original_avg_ratio",
    "robdp_last_level_reqo_vs_min_cost_avg_ratio",
    "robdp_last_level_reqo_vs_oracle_avg_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine per-group fold/overall summary CSVs into wide CSVs."
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        required=True,
        help="Root containing <group>/fold_summary.csv and overall_summary.csv.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Group names such as 1x1__0x0 8x1__0x0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for combined summary CSVs.",
    )
    parser.add_argument(
        "--fold-summary-filename",
        default="fold_summary.csv",
    )
    parser.add_argument(
        "--overall-summary-filename",
        default="overall_summary.csv",
    )
    parser.add_argument(
        "--combined-fold-summary-csv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--combined-overall-summary-csv",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def group_prefix(group: str) -> str:
    prefix = group.replace("/", "__")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError(f"Invalid group name for CSV prefix: {group}")
    return prefix


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required summary CSV does not exist: {path}")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def row_key(row: dict[str, str], key_column: str) -> str:
    value = row.get(key_column)
    if value is None or value == "":
        raise ValueError(f"Missing {key_column} in row: {row}")
    return value


def add_shared_columns(
        output_row: dict[str, Any],
        source_row: dict[str, str],
) -> None:
    for column in SHARED_COLUMNS:
        output_row[column] = source_row.get(column, "")


def add_group_columns(
        output_row: dict[str, Any],
        group: str,
        source_row: dict[str, str],
) -> None:
    prefix = group_prefix(group)
    for column in ROBDP_GROUP_COLUMNS:
        output_row[f"{prefix}_{column}"] = source_row.get(column, "")


def combined_fieldnames(key_column: str, groups: list[str]) -> list[str]:
    fields = [key_column] + SHARED_COLUMNS
    for group in groups:
        prefix = group_prefix(group)
        fields.extend([
            f"{prefix}_{column}"
            for column in ROBDP_GROUP_COLUMNS
        ])
    return fields


def combine_summary_kind(
        summary_root: Path,
        groups: list[str],
        filename: str,
        key_column: str,
) -> list[dict[str, Any]]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    seen_keys_by_group: dict[str, set[str]] = {}

    for group_idx, group in enumerate(groups):
        csv_path = summary_root / group / filename
        rows = read_csv_rows(csv_path)
        seen_keys: set[str] = set()
        for row in rows:
            key = row_key(row, key_column)
            seen_keys.add(key)
            output_row = rows_by_key.setdefault(key, {key_column: key})
            if group_idx == 0:
                add_shared_columns(output_row, row)
            add_group_columns(output_row, group, row)
        seen_keys_by_group[group] = seen_keys

    expected_keys = set().union(*seen_keys_by_group.values())
    for group, seen_keys in seen_keys_by_group.items():
        missing = sorted(expected_keys - seen_keys)
        if missing:
            raise ValueError(
                f"Group {group} is missing {key_column} values: {missing}"
            )

    return [
        rows_by_key[key]
        for key in sorted(
            rows_by_key,
            key=lambda value: int(value) if value.isdigit() else value,
        )
    ]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary_root = args.summary_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    fold_output = (
        args.combined_fold_summary_csv
        or output_dir / "combined_fold_summary.csv"
    )
    overall_output = (
        args.combined_overall_summary_csv
        or output_dir / "combined_overall_summary.csv"
    )

    fold_rows = combine_summary_kind(
        summary_root=summary_root,
        groups=args.groups,
        filename=args.fold_summary_filename,
        key_column="fold_id",
    )
    overall_rows = combine_summary_kind(
        summary_root=summary_root,
        groups=args.groups,
        filename=args.overall_summary_filename,
        key_column="scope",
    )

    write_csv(
        fold_output,
        combined_fieldnames("fold_id", args.groups),
        fold_rows,
    )
    write_csv(
        overall_output,
        combined_fieldnames("scope", args.groups),
        overall_rows,
    )

    print(f"Wrote combined fold summary: {fold_output}")
    print(f"Wrote combined overall summary: {overall_output}")


if __name__ == "__main__":
    main()

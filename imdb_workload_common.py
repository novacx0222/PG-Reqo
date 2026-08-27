"""Shared helpers for running the IMDb workload against PostgreSQL.

The runner scripts use this module to load SQL queries, configure common
command-line options, execute EXPLAIN modes, and persist returned rows. Backend-
specific behavior, such as RobDP score files and error profiles, stays in the
individual runner.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

# SQLs are grouped as: template_id -> query_id -> SQL string.
SQLGroups = dict[int, dict[int, str]]
GUCDict = dict[str, str | int | float]
QueryKey = tuple[int, int]
LEGACY_FOLD_COLUMNS = {"eval_fold_id", "train_fold_id", "split", "query_fold_id"}

RUN_MODE_PREFIXES = {
    "none": "",
    "explain-json": "EXPLAIN (FORMAT JSON)",
    "explain-text": "EXPLAIN (FORMAT TEXT)",
    "explain-analyze-json": "EXPLAIN (ANALYZE, FORMAT JSON)",
    "explain-analyze-text": "EXPLAIN (ANALYZE, FORMAT TEXT)",
}


def create_argument_parser(description: str) -> argparse.ArgumentParser:
    """Create a parser with options shared by both IMDb runners."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dbname", required=True, help="PostgreSQL database name.")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host.")
    parser.add_argument("--port", default="5432", help="PostgreSQL port.")
    parser.add_argument("--user", required=True, help="PostgreSQL username.")
    parser.add_argument(
        "--password",
        default=None,
        help="PostgreSQL password. Optional.",
    )
    parser.add_argument(
        "--statement-timeout",
        default="60s",
        help="PostgreSQL statement timeout. Default: 60s.",
    )
    parser.add_argument(
        "--sqls-dir",
        type=Path,
        required=True,
        help="Directory containing {template_id}-0_{workload_name} subdirectories.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Base directory for query results.",
    )
    parser.add_argument(
        "--workload-name",
        required=True,
        help="Workload name to load, such as kepler, csv, or cardinality.",
    )
    parser.add_argument(
        "--skip-template-id-vals",
        type=int,
        nargs="+",
        default=[],
        help="Template IDs to skip. Default: none.",
    )
    parser.add_argument(
        "--query-id-limit",
        type=int,
        default=None,
        help="Keep query IDs in [0, limit). Default: keep all queries.",
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=None,
        help=(
            "Optional folds directory containing <fold-source>_fold_<fold-id>.csv. "
            "When set with --fold-id, only the selected fold split is loaded."
        ),
    )
    parser.add_argument(
        "--fold-source",
        default="original",
        help="Fold membership source prefix. Default: original.",
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        default=None,
        help="Optional fold id to load from --folds-dir.",
    )
    parser.add_argument(
        "--fold-split",
        choices=("train", "test", "all"),
        default="test",
        help=(
            "Fold ownership slice to load when --fold-id is set. With 2 folds, "
            "test reads fold_id, train reads the other fold, and all reads both. "
            "Default: test."
        ),
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of times to execute each SQL query. Default: 3.",
    )
    parser.add_argument(
        "--run-mode",
        choices=RUN_MODE_PREFIXES,
        default="explain-analyze-json",
        help=(
            "SQL execution mode. Default: explain-analyze-json. "
            "Choices: none, explain-json, explain-text, "
            "explain-analyze-json, explain-analyze-text."
        ),
    )
    return parser


def validate_common_args(args: argparse.Namespace) -> None:
    """Validate options shared by both runners."""
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.query_id_limit is not None and args.query_id_limit < 0:
        raise ValueError("--query-id-limit must be non-negative.")
    if (args.folds_dir is None) != (args.fold_id is None):
        raise ValueError("--folds-dir and --fold-id must be used together.")
    if args.fold_id is not None and args.fold_id <= 0:
        raise ValueError("--fold-id must be positive.")
    if args.fold_id is not None and args.fold_id not in {1, 2}:
        raise ValueError(
            "This separate-error-profile workflow expects --fold-id to be 1 or 2; "
            f"got {args.fold_id}."
        )


def csv_int(value: Any, column_name: str, csv_path: Path) -> int:
    """Parse integer-ish CSV values while keeping error messages useful."""
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer value for {column_name!r} in {csv_path}: {value!r}"
        ) from exc


def load_fold_query_keys(
        folds_dir: Path,
        fold_source: str,
        fold_id: int,
        fold_split: str,
) -> set[QueryKey]:
    """Load selected (template_id, original_query_id) keys from ownership CSVs."""
    folds_dir = folds_dir.expanduser().resolve()
    fold_csvs = sorted(folds_dir.glob(f"{fold_source}_fold_*.csv"))
    if len(fold_csvs) != 2:
        raise ValueError(
            "This separate-error-profile branch expects exactly 2 fold files "
            f"for source={fold_source!r}; found {len(fold_csvs)} in {folds_dir}."
        )

    if fold_split == "test":
        owner_fold_ids = [fold_id]
    elif fold_split == "train":
        owner_fold_ids = [3 - fold_id]
    else:
        owner_fold_ids = [1, 2]

    selected_keys: set[QueryKey] = set()
    required_columns = {
        "fold_id",
        "template_id",
        "original_query_id",
    }
    for owner_fold_id in owner_fold_ids:
        fold_csv = folds_dir / f"{fold_source}_fold_{owner_fold_id}.csv"
        if not fold_csv.is_file():
            raise ValueError(f"Fold membership CSV does not exist: {fold_csv}")

        with fold_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            missing_columns = sorted(required_columns - set(reader.fieldnames or []))
            if missing_columns:
                raise ValueError(
                    f"{fold_csv} is missing required columns: {missing_columns}"
                )
            legacy_columns = sorted(LEGACY_FOLD_COLUMNS & set(reader.fieldnames or []))
            if legacy_columns:
                raise ValueError(
                    f"{fold_csv} uses the old train/test fold schema with columns "
                    f"{legacy_columns}. Regenerate folds with the ownership schema."
                )

            for row in reader:
                row_fold_id = csv_int(row["fold_id"], "fold_id", fold_csv)
                if row_fold_id != owner_fold_id:
                    raise ValueError(
                        f"{fold_csv} contains row for fold_id={row_fold_id}; "
                        f"expected {owner_fold_id}."
                    )
                selected_keys.add((
                    csv_int(row["template_id"], "template_id", fold_csv),
                    csv_int(row["original_query_id"], "original_query_id", fold_csv),
                ))

    if not selected_keys:
        raise ValueError(
            f"No query keys selected from {folds_dir} for split={fold_split!r}."
        )
    return selected_keys


def load_sql_groups(
        sqls_dir: Path,
        workload_name: str,
        skip_template_id_vals: list[int],
        query_id_limit: int | None,
        query_filter_keys: set[QueryKey] | None = None,
) -> SQLGroups:
    """Load SQLs as template_id -> query_id -> SQL string."""
    if not sqls_dir.is_dir():
        raise ValueError(f"SQL directory does not exist: {sqls_dir}")
    if query_id_limit is not None and query_id_limit < 0:
        raise ValueError("--query-id-limit must be non-negative.")

    sql_groups: SQLGroups = {}
    skipped_template_ids = set(skip_template_id_vals)
    # The "-0" segment is a fixed placeholder and is not a query ID.
    directory_pattern = re.compile(
        r"^(?P<template_id>\d+)-0_(?P<workload_name>.+)$"
    )

    parsed_directories = []
    for sql_group_dir in sqls_dir.iterdir():
        if not sql_group_dir.is_dir():
            continue

        match = directory_pattern.fullmatch(sql_group_dir.name)
        if match is None or match.group("workload_name") != workload_name:
            continue

        parsed_directories.append(
            (int(match.group("template_id")), sql_group_dir)
        )

    for template_id, sql_group_dir in sorted(parsed_directories):
        if template_id in skipped_template_ids:
            continue

        testing_json_path = (
                sql_group_dir
                / "raw_data"
                / f"{template_id}-0_testing.json"
        )
        if not testing_json_path.is_file():
            raise ValueError(
                f"Testing SQL file does not exist: {testing_json_path}"
            )

        with testing_json_path.open("r", encoding="utf-8") as testing_json_file:
            sql_by_key = json.load(testing_json_file)
        if not isinstance(sql_by_key, dict):
            raise ValueError(f"Expected a JSON object in: {testing_json_path}")

        # The numeric suffix is the actual query ID.
        key_pattern = re.compile(
            rf"^{template_id}-0_testing_(?P<query_id>\d+)$"
        )
        sql_by_query_id: dict[int, str] = {}
        for query_key, sql_string in sql_by_key.items():
            match = key_pattern.fullmatch(query_key)
            if match is None:
                raise ValueError(
                    f"Unexpected query key {query_key!r} in: {testing_json_path}"
                )
            if not isinstance(sql_string, str):
                raise ValueError(
                    f"Expected SQL string for key {query_key!r} in: "
                    f"{testing_json_path}"
                )

            query_id = int(match.group("query_id"))
            if query_id_limit is not None and query_id >= query_id_limit:
                continue
            if (
                    query_filter_keys is not None
                    and (template_id, query_id) not in query_filter_keys
            ):
                continue

            # Handle SQL strings that were escaped more than once before loading.
            sql_by_query_id[query_id] = (
                sql_string
                .replace("\\r\\n", "\n")
                .replace("\\n", "\n")
            )

        if sql_by_query_id or query_filter_keys is None:
            sql_groups[template_id] = dict(sorted(sql_by_query_id.items()))

    return sql_groups


def load_sql_groups_from_args(args: argparse.Namespace) -> SQLGroups:
    """Load SQL groups, optionally restricted by the selected fold split."""
    query_filter_keys: set[QueryKey] | None = None
    if args.fold_id is not None:
        query_filter_keys = load_fold_query_keys(
            folds_dir=args.folds_dir,
            fold_source=args.fold_source,
            fold_id=args.fold_id,
            fold_split=args.fold_split,
        )
        print(
            "Fold filter: "
            f"source={args.fold_source}, "
            f"fold={args.fold_id}, "
            f"split={args.fold_split}, "
            f"query_keys={len(query_filter_keys)}"
        )

    return load_sql_groups(
        sqls_dir=args.sqls_dir,
        workload_name=args.workload_name,
        skip_template_id_vals=args.skip_template_id_vals,
        query_id_limit=args.query_id_limit,
        query_filter_keys=query_filter_keys,
    )


def print_sql_group_statistics(
        sql_groups: SQLGroups,
        workload_name: str,
) -> None:
    """Print a concise summary of the loaded SQL workload."""
    template_count = len(sql_groups)
    sql_counts = [
        len(sql_by_query_id)
        for sql_by_query_id in sql_groups.values()
    ]
    total_sql_count = sum(sql_counts)

    print("SQL workload summary:")
    print(f"  Workload name: {workload_name}")
    print(f"  Templates loaded: {template_count}")
    print(f"  Total SQL queries: {total_sql_count}")

    if not sql_counts:
        print("  No SQL queries were loaded.")
        return

    template_ids = sorted(sql_groups)
    average_sql_count = total_sql_count / template_count
    print(f"  Template ID range: {template_ids[0]} to {template_ids[-1]}")
    print(
        "  SQL queries per template: "
        f"min={min(sql_counts)}, "
        f"max={max(sql_counts)}, "
        f"average={average_sql_count:.2f}"
    )


def build_executable_sql(sql_string: str, run_mode: str) -> str:
    """Add the EXPLAIN prefix selected by the execution mode."""
    sql_prefix = RUN_MODE_PREFIXES[run_mode]
    if not sql_prefix:
        return sql_string
    return f"{sql_prefix} {sql_string}"


def format_query_results(rows: list[tuple]) -> str:
    """Convert cursor result rows into a readable text representation."""
    if not rows:
        return ""

    if len(rows) == 1 and len(rows[0]) == 1:
        result_value = rows[0][0]
        if isinstance(result_value, (dict, list)):
            return json.dumps(result_value, indent=2, default=str)

    formatted_rows = []
    for row in rows:
        formatted_values = []
        for value in row:
            if isinstance(value, (dict, list)):
                formatted_values.append(json.dumps(value, default=str))
            else:
                formatted_values.append(str(value))
        formatted_rows.append("\t".join(formatted_values))
    return "\n".join(formatted_rows)


def execute_query(
        cursor: Any,
        sql_string: str,
        run_mode: str,
) -> str:
    """Execute one query and return its formatted result rows."""
    cursor.execute(build_executable_sql(sql_string, run_mode))
    result_rows = cursor.fetchall() if cursor.description is not None else []
    return format_query_results(result_rows)


def save_query_results(
        results_filename: Path,
        header_lines: list[str],
        result_text: str,
) -> None:
    """Append one execution result and its metadata to a text file."""
    results_filename.parent.mkdir(parents=True, exist_ok=True)
    with results_filename.open("a", encoding="utf-8") as results_file:
        for header_line in header_lines:
            results_file.write(f"{header_line}\n")
        results_file.write(result_text)
        results_file.write("\n\n")


def set_guc_dict(
        cursor: Any,
        guc_dict: GUCDict,
) -> None:
    """Set all PostgreSQL GUC values in insertion order."""
    for guc_name, guc_val in guc_dict.items():
        if isinstance(guc_val, str):
            escaped_guc_val = guc_val.replace("'", "''")
            guc_stmt = f"SET {guc_name} = '{escaped_guc_val}';"
        else:
            guc_stmt = f"SET {guc_name} = {guc_val};"
        cursor.execute(guc_stmt)


def open_connection(
        dbname: str,
        host: str,
        port: str,
        user: str,
        password: str | None,
):
    """Open a PostgreSQL connection."""
    import psycopg2

    conn_kwargs = {
        "dbname": dbname,
        "host": host,
        "port": port,
        "user": user,
    }
    if password is not None:
        conn_kwargs["password"] = password
    return psycopg2.connect(**conn_kwargs)

"""Shared helpers for running the IMDb workload against PostgreSQL.

The runner scripts use this module to load SQL queries, configure common
command-line options, execute EXPLAIN modes, and persist returned rows. Backend-
specific behavior, such as RobDP score files and error profiles, stays in the
individual runner.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

# SQLs are grouped as: template_id -> query_id -> SQL string.
SQLGroups = dict[int, dict[int, str]]
GUCDict = dict[str, str | int | float]

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


def load_sql_groups(
        sqls_dir: Path,
        workload_name: str,
        skip_template_id_vals: list[int],
        query_id_limit: int | None,
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

            # Handle SQL strings that were escaped more than once before loading.
            sql_by_query_id[query_id] = (
                sql_string
                .replace("\\r\\n", "\n")
                .replace("\\n", "\n")
            )

        sql_groups[template_id] = dict(sorted(sql_by_query_id.items()))

    return sql_groups


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

"""Run no-hint SQLs for the test queries recorded in Reqo fold splits.

The fold split CSVs produced by train.py contain query_group_id values that
match the sequential query group id used by the hint SQL CSV builder. This
script reconstructs that group id -> original SQL mapping, executes only
split == test queries, and writes one runtime row per fold/query/round.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from imdb_workload_common import (
    SQLGroups,
    load_sql_groups,
    open_connection,
    set_guc_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run plain no-hint SQLs for test queries from Reqo folds."
    )
    parser.add_argument("--fold-results-dir", type=Path, required=True)
    parser.add_argument("--sqls-dir", type=Path, required=True)
    parser.add_argument("--workload-name", required=True)
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="5432")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--statement-timeout", default="60s")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--query-id-limit",
        type=int,
        default=None,
        help="Keep original workload query IDs in [0, limit).",
    )
    parser.add_argument(
        "--skip-template-id-vals",
        type=int,
        nargs="+",
        default=[],
        help="Template IDs skipped when the dataset was built.",
    )
    parser.add_argument(
        "--parameter-group-dir",
        type=Path,
        default=None,
        help=(
            "Optional hint result parameter group directory used to build the "
            "training CSV, such as results/1x13/1x13. If omitted, "
            "query_group_id is reconstructed from all loaded SQLs sorted by "
            "template/original query id."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive.")
    if args.query_id_limit is not None and args.query_id_limit < 0:
        raise ValueError("--query-id-limit must be non-negative.")
    if not args.fold_results_dir.is_dir():
        raise ValueError(f"Fold results directory does not exist: {args.fold_results_dir}")


def discover_query_dirs(parameter_group_dir: Path) -> list[tuple[int, int]]:
    """Return sorted (template_id, original_query_id) pairs from hint dirs."""
    query_ids = []
    if not parameter_group_dir.is_dir():
        raise ValueError(f"Parameter group directory does not exist: {parameter_group_dir}")

    for template_dir in parameter_group_dir.iterdir():
        if not template_dir.is_dir() or not template_dir.name.isdigit():
            continue
        template_id = int(template_dir.name)
        for query_dir in template_dir.iterdir():
            if query_dir.is_dir() and query_dir.name.isdigit():
                query_ids.append((template_id, int(query_dir.name)))
    return sorted(query_ids, key=lambda item: (item[0], item[1]))


def build_sql_by_group_id(
        sql_groups: SQLGroups,
        parameter_group_dir: Path | None,
) -> dict[int, dict[str, Any]]:
    """Build train.py query_group_id -> original no-hint SQL and metadata."""
    if parameter_group_dir is None:
        ordered_query_ids = [
            (template_id, original_query_id)
            for template_id in sorted(sql_groups)
            for original_query_id in sorted(sql_groups[template_id])
        ]
    else:
        ordered_query_ids = discover_query_dirs(parameter_group_dir)

    sql_by_group_id = {}
    for query_group_id, (template_id, original_query_id) in enumerate(ordered_query_ids):
        try:
            sql_by_group_id[query_group_id] = {
                "sql": sql_groups[template_id][original_query_id],
                "template_id": template_id,
                "original_query_id": original_query_id,
            }
        except KeyError as exc:
            raise KeyError(
                "Cannot find original SQL for reconstructed mapping: "
                f"query_group_id={query_group_id}, template_id={template_id}, "
                f"original_query_id={original_query_id}"
            ) from exc
    return sql_by_group_id


def parse_fold_id_from_path(path: Path) -> int:
    match = re.search(r"reqo_fold_(\d+)_split\.csv$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse fold id from split CSV name: {path}")
    return int(match.group(1))


def find_split_csvs(fold_results_dir: Path) -> list[Path]:
    split_csvs = sorted(fold_results_dir.glob("fold_*/reqo_fold_*_split.csv"))
    if not split_csvs:
        raise ValueError(f"No fold split CSVs found under: {fold_results_dir}")
    return split_csvs


def load_test_queries(fold_results_dir: Path) -> list[dict[str, Any]]:
    """Load split == test rows from every fold split CSV."""
    rows = []
    for split_csv in find_split_csvs(fold_results_dir):
        fallback_fold_id = parse_fold_id_from_path(split_csv)
        with split_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("split") != "test":
                    continue
                fold_id = int(row.get("fold_id") or fallback_fold_id)
                rows.append({
                    "fold_id": fold_id,
                    "fold_query_idx": int(row["fold_query_idx"]),
                    "global_query_idx": int(row["global_query_idx"]),
                    "query_group_id": int(row["query_group_id"]),
                    "template_id": row["template_id"],
                    "original_query_id": row["original_query_id"],
                })
    return sorted(rows, key=lambda item: (item["fold_id"], item["fold_query_idx"]))


def parse_explain_analyze_json(value: Any) -> dict[str, Any]:
    """Normalize psycopg2's EXPLAIN JSON cell into the top-level document."""
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError(f"Unexpected EXPLAIN JSON shape: {type(value)!r}")
        return value[0]
    if isinstance(value, dict):
        return value
    raise ValueError(f"Unexpected EXPLAIN JSON value type: {type(value)!r}")


def run_explain_analyze_json(cursor: Any, sql: str) -> float:
    cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}")
    rows = cursor.fetchall()
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError(f"Unexpected EXPLAIN row shape: {len(rows)} rows")
    explain_doc = parse_explain_analyze_json(rows[0][0])
    execution_time = explain_doc.get("Execution Time")
    if execution_time is None:
        raise ValueError("EXPLAIN JSON did not contain Execution Time")
    return float(execution_time)


def write_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()


def append_row(output_csv: Path, row: dict[str, Any]) -> None:
    with output_csv.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerow(row)


FIELDNAMES = [
    "system_name",
    "fold_id",
    "fold_query_idx",
    "global_query_idx",
    "query_group_id",
    "template_id",
    "original_query_id",
    "round",
    "runtime_ms",
    "status",
    "error",
]


def main() -> None:
    args = parse_args()
    validate_args(args)

    sql_groups = load_sql_groups(
        sqls_dir=args.sqls_dir,
        workload_name=args.workload_name,
        skip_template_id_vals=args.skip_template_id_vals,
        query_id_limit=args.query_id_limit,
    )
    sql_by_group_id = build_sql_by_group_id(sql_groups, args.parameter_group_dir)
    test_queries = load_test_queries(args.fold_results_dir)

    print(f"Loaded {len(test_queries)} test query rows from {args.fold_results_dir}")
    print(f"Reconstructed {len(sql_by_group_id)} query_group_id -> SQL mappings")
    write_header(args.output_csv)

    with open_connection(
            dbname=args.dbname,
            host=args.host,
            port=str(args.port),
            user=args.user,
            password=args.password,
    ) as conn:
        with conn.cursor() as cursor:
            for query_row in test_queries:
                query_group_id = query_row["query_group_id"]
                sql_entry = sql_by_group_id.get(query_group_id)
                if sql_entry is None:
                    base_row = {
                        "system_name": args.system_name,
                        **query_row,
                        "runtime_ms": "",
                        "status": "missing_sql",
                        "error": f"No SQL mapping for query_group_id={query_group_id}",
                    }
                    for round_number in range(1, args.rounds + 1):
                        append_row(args.output_csv, {**base_row, "round": round_number})
                    continue
                sql = sql_entry["sql"]
                if query_row["template_id"] == "":
                    query_row["template_id"] = sql_entry["template_id"]
                if query_row["original_query_id"] == "":
                    query_row["original_query_id"] = sql_entry["original_query_id"]

                for round_number in range(1, args.rounds + 1):
                    print(
                        "Running plain SQL: "
                        f"system={args.system_name}, "
                        f"fold={query_row['fold_id']}, "
                        f"fold_query_idx={query_row['fold_query_idx']}, "
                        f"query_group_id={query_group_id}, "
                        f"round={round_number}/{args.rounds}"
                    )
                    try:
                        set_guc_dict(cursor, {"statement_timeout": args.statement_timeout})
                        runtime_ms = run_explain_analyze_json(cursor, sql)
                        status = "ok"
                        error = ""
                    except Exception as exc:
                        conn.rollback()
                        runtime_ms = ""
                        status = "error"
                        error = repr(exc)

                    append_row(args.output_csv, {
                        "system_name": args.system_name,
                        **query_row,
                        "round": round_number,
                        "runtime_ms": runtime_ms,
                        "status": status,
                        "error": error,
                    })

    print(f"Wrote plain runtime CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

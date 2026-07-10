#!/usr/bin/env python3
"""
Encode many SQL queries into Reqo raw plan-graph encodings.

Input:
  A text file where each non-empty line is one complete SQL query.

Output:
  A dataset containing one graph per SQL:
    - x:          [num_plan_nodes, raw_node_feature_dim]
    - edge_index: [2, num_edges], child -> parent edges, postorder node ids
    - metadata
    - optional runtime label if --analyze is used

Important:
  By default this script uses EXPLAIN (FORMAT JSON), which plans each SQL but does
  not execute the query. If you pass --analyze, it uses EXPLAIN (ANALYZE, FORMAT JSON),
  which really executes every SQL line separately.

Example:
  python3 reqo_encode_sql_lines.py \
    --analyze
    --sql-file ../Test/queries/group.sql \
    --stats-dir ../Data/postgres/database_statistics \
    --dbname postgres \
    --host localhost \
    --port 5432 \
    --user novacx0222 \
    --output-dir ../Test \
    --norm-stats-output ../Test/norm_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import torch
from pandas import DataFrame
from torch_geometric.data import Data
from tqdm import tqdm


def normalize_id_value(value: Any) -> Any:
    """Normalize CSV id values while preserving non-numeric ids."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return int(value)
    value_str = str(value)
    if value_str.isdigit():
        return int(value_str)
    return value


def normalize_sql_rows(sql_rows_df: DataFrame) -> List[Dict[str, Any]]:
    """Normalize metadata-rich SQL CSV rows."""
    required_columns = {
        "query_group_id",
        "template_id",
        "original_query_id",
        "candidate_id",
        "sql_text",
    }
    missing_columns = sorted(required_columns - set(sql_rows_df.columns))
    if missing_columns:
        raise ValueError(f"SQL CSV is missing required columns: {missing_columns}")

    rows = []
    for row_number, row in enumerate(sql_rows_df.to_dict("records"), start=1):
        query_group_id = normalize_id_value(row["query_group_id"])
        candidate_id = normalize_id_value(row["candidate_id"])
        template_id = normalize_id_value(row.get("template_id"))
        original_query_id = normalize_id_value(row.get("original_query_id"))

        rows.append({
            "row_number": row_number,
            "query_group_id": query_group_id,
            "candidate_id": candidate_id,
            "template_id": template_id,
            "original_query_id": original_query_id,
            "sql_text": row["sql_text"],
        })
    return rows


def metadata_value(metadata: Dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    return "" if value is None else value


def query_metadata_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "query_group_id": rec["query_group_id"],
        "template_id": rec.get("template_id"),
        "original_query_id": rec.get("original_query_id"),
    }


def write_query_metadata_csv(path: Path, query_metadata: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["query_group_id", "template_id", "original_query_id"],
        )
        writer.writeheader()
        for metadata in query_metadata:
            writer.writerow({
                "query_group_id": metadata_value(metadata, "query_group_id"),
                "template_id": metadata_value(metadata, "template_id"),
                "original_query_id": metadata_value(metadata, "original_query_id"),
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode one SQL-per-line file into Reqo raw graph encodings."
    )
    parser.add_argument("--sql-file", default=None, help="Path to a metadata-rich SQL CSV.")
    parser.add_argument("--dbname", required=True, help="PostgreSQL database name.")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host.")
    parser.add_argument("--port", default="5432", help="PostgreSQL port.")
    parser.add_argument("--user", required=True, help="PostgreSQL username.")
    parser.add_argument("--password", default=None, help="PostgreSQL password. Optional.")

    parser.add_argument(
        "--stats-dir",
        required=True,
        help="Directory containing Reqo database_statistics/*.npy files. ",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output path. Format: *.pt and *.npy",
    )
    parser.add_argument(
        "--reqo-dataset-dir",
        default=None,
        help=(
            "Directory for original Reqo .npy dataset files. Default: "
            "Data/<dbname>/datasets. Use fold-specific directories to avoid "
            "overwriting other experiments."
        ),
    )
    parser.add_argument(
        "--no-save-original-reqo-dataset",
        action="store_true",
        help="Only save encode.pt and skip original Reqo .npy dataset files.",
    )
    parser.add_argument(
        "--min-candidates-per-query",
        type=int,
        default=2,
        help=(
            "Minimum successful candidates required to keep a query group in "
            "the original Reqo .npy dataset. Default: 2."
        ),
    )
    parser.add_argument(
        "--norm-stats-output",
        default=None,
        help="Optional path to save global normalization stats computed from all SQL plans.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help=(
            "Optional JSON file with existing training normalization stats. "
            "If provided, this script will use it instead of auto-computing stats."
        ),
    )
    parser.add_argument(
        "--plans-cache-input",
        default=None,
        help=(
            "Optional raw EXPLAIN JSON cache produced by --plans-cache-output. "
            "When set, this script skips PostgreSQL and encodes plans from the cache."
        ),
    )
    parser.add_argument(
        "--plans-cache-output",
        default=None,
        help="Optional JSON path where raw EXPLAIN plans will be saved before encoding.",
    )
    parser.add_argument(
        "--plans-cache-only",
        action="store_true",
        help=(
            "Collect and save --plans-cache-output, then exit without encoding. "
            "Useful for running EXPLAIN ANALYZE exactly once per source."
        ),
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help=(
            "Use EXPLAIN ANALYZE. This executes every SQL line and records root actual runtime. "
            "Use this for training labels, not normal inference."
        ),
    )
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=0,
        help="Optional PostgreSQL statement_timeout in milliseconds. 0 means no change.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip SQL lines that fail EXPLAIN/encoding instead of stopping the whole script.",
    )
    parser.add_argument(
        "--query-id-prefix",
        default="q",
        help="Prefix for generated query ids. Default: q, producing q000001, q000002, ...",
    )
    return parser.parse_args()


def plan_cache_key(item: Dict[str, Any]) -> tuple[Any, Any, Any]:
    """Use stable IMDb/candidate identity instead of fold-local query_group_id."""
    return (
        normalize_id_value(item.get("template_id")),
        normalize_id_value(item.get("original_query_id")),
        normalize_id_value(item.get("candidate_id")),
    )


def save_plans_cache(path: Path, plans_by_row: List[Dict[str, Any]]) -> None:
    """Save raw plans before fold-specific normalization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plans_by_row),
        encoding="utf-8",
    )
    print(f"Saved raw plans cache to: {path}")


def load_plans_from_cache(
        path: Path,
        sql_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Load raw plans and select/relabel rows for one fold split."""
    cache_rows = json.loads(path.read_text(encoding="utf-8"))
    cache_by_key: Dict[tuple[Any, Any, Any], Dict[str, Any]] = {}
    for cached in cache_rows:
        key = plan_cache_key(cached)
        if key in cache_by_key:
            raise ValueError(f"Duplicate plan cache key in {path}: {key}")
        cache_by_key[key] = cached

    selected = []
    missing_keys = []
    for row in sql_rows:
        key = plan_cache_key(row)
        cached = cache_by_key.get(key)
        if cached is None:
            missing_keys.append(key)
            continue
        selected.append({
            **cached,
            **row,
            "sql": row["sql_text"],
            "plan": cached["plan"],
        })

    if missing_keys:
        preview = ", ".join(str(key) for key in missing_keys[:5])
        raise KeyError(
            f"{len(missing_keys)} SQL rows were not found in plan cache {path}. "
            f"First missing keys: {preview}"
        )
    return selected


def load_database_info_from_dir(stats_dir: Path):
    """Load the same metadata arrays that Reqo's load_database_info(dbname) loads."""
    required = [
        "tables_index.npy",
        "tables_index_all.npy",
        "columns_index.npy",
        "columns_list.npy",
        "attribute_range.npy",
        "postgresql_nodestypes_all.npy",
    ]
    missing = [name for name in required if not (stats_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Reqo database statistics in {stats_dir}: {missing}. "
            "Generate them first with postgresql_database_statistic_generator.py "
            "or pass the correct --stats-dir to an existing database_statistics directory."
        )

    tables_index = np.load(stats_dir / "tables_index.npy", allow_pickle=True).item()
    tables_index_all = np.load(stats_dir / "tables_index_all.npy", allow_pickle=True).item()
    columns_index = np.load(stats_dir / "columns_index.npy", allow_pickle=True).item()
    columns_list = np.load(stats_dir / "columns_list.npy", allow_pickle=True)
    attribute_range = np.load(stats_dir / "attribute_range.npy", allow_pickle=True).item()
    nodes = np.load(stats_dir / "postgresql_nodestypes_all.npy", allow_pickle=True).item()
    return tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes


def open_connection(
        dbname: str,
        host: str,
        port: str,
        user: str,
        password: Optional[str],
):
    """Open a PostgreSQL connection."""
    conn_kwargs = {
        "dbname": dbname,
        "host": host,
        "port": port,
        "user": user,
    }
    if password is not None:
        conn_kwargs["password"] = password
    return psycopg2.connect(**conn_kwargs)


def run_explain_json_with_cursor(
        cur,
        sql: str,
        analyze: bool,
) -> Dict[str, Any]:
    """Return PostgreSQL's top-level EXPLAIN JSON document for one SQL."""
    options = "ANALYZE, FORMAT JSON" if analyze else "FORMAT JSON"
    explain_sql = f"EXPLAIN ({options}) {sql}"
    cur.execute(explain_sql)
    raw = cur.fetchone()[0]

    # psycopg2 normally returns a Python list for JSON, but keep string fallback.
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Unexpected EXPLAIN JSON result: {type(raw)}")
    return raw[0]


def is_statement_timeout(exc: Exception) -> bool:
    """Return true for PostgreSQL statement_timeout cancellations."""
    return (
        isinstance(exc, psycopg2.errors.QueryCanceled)
        or getattr(exc, "pgcode", None) == "57014"
        or "statement timeout" in str(exc).lower()
    )


def collect_timeout_fallback_plan(
        cur,
        sql: str,
        timeout_ms: int,
) -> Dict[str, Any]:
    """Collect a plan shape after EXPLAIN ANALYZE timed out.

    The timed-out execution is assigned the configured timeout as its runtime
    label. We then run plain EXPLAIN to keep the plan graph available for Reqo
    features without executing the query a second time.
    """
    cur.execute("SET statement_timeout = 0")
    explain_doc = run_explain_json_with_cursor(
        cur=cur,
        sql=sql,
        analyze=False,
    )
    if timeout_ms > 0:
        cur.execute("SET statement_timeout = %s", (timeout_ms,))
    return explain_doc


def visit_plan_nodes(plan: Dict[str, Any], out: List[Dict[str, Any]]) -> None:
    """Collect all plan nodes recursively."""
    out.append(plan)
    for child in plan.get("Plans", []):
        visit_plan_nodes(child, out)


def collect_global_plan_stats(plans: List[Dict[str, Any]]) -> List[Any]:
    """
    Compute Reqo-style log min/max normalization stats across all plans.

    This is the 'auto norm' mode requested by the user.
    """
    costs: List[float] = []
    rows: List[float] = []

    for plan in plans:
        nodes: List[Dict[str, Any]] = []
        visit_plan_nodes(plan, nodes)
        for node in nodes:
            costs.append(float(node.get("Total Cost", 0.0)))
            rows.append(float(node.get("Plan Rows", 0.0)))

    if not costs or not rows:
        raise ValueError("Cannot compute normalization stats: no plan nodes found.")

    log_costs = np.log(np.asarray(costs, dtype=float) + 1.0)
    log_rows = np.log(np.asarray(rows, dtype=float) + 1.0)

    cmin, cmax = float(log_costs.min()), float(log_costs.max())
    rmin, rmax = float(log_rows.min()), float(log_rows.max())

    eps = 1e-9
    if abs(cmax - cmin) < eps:
        cmax = cmin + eps
    if abs(rmax - rmin) < eps:
        rmax = rmin + eps

    return [["Total Cost", "Plan Rows"], [cmin, rmin], [cmax, rmax]]


def norm_stats_to_json(norm_stats: List[Any]) -> Dict[str, float]:
    """Convert Reqo norm stats list into a clearer JSON object."""
    return {
        "log_total_cost_min": float(norm_stats[1][0]),
        "log_total_cost_max": float(norm_stats[2][0]),
        "log_plan_rows_min": float(norm_stats[1][1]),
        "log_plan_rows_max": float(norm_stats[2][1]),
    }


def load_norm_stats_json(path: Path) -> List[Any]:
    """Load normalization stats from JSON."""
    obj = json.loads(path.read_text(encoding="utf-8"))
    cmin = float(obj["log_total_cost_min"])
    cmax = float(obj["log_total_cost_max"])
    rmin = float(obj["log_plan_rows_min"])
    rmax = float(obj["log_plan_rows_max"])

    eps = 1e-9
    if abs(cmax - cmin) < eps:
        cmax = cmin + eps
    if abs(rmax - rmin) < eps:
        rmax = rmin + eps

    return [["Total Cost", "Plan Rows"], [cmin, rmin], [cmax, rmax]]


def encode_plan(plan: Dict[str, Any], stats_dir: Path, norm_stats: List[Any]):
    """Use Reqo's own parser/encoder to produce node features and edges."""
    from query_plan_feature_extraction import encoding_generate, replace_aliases_and_columns  # type: ignore

    (
        tables_index,
        tables_index_all,
        columns_index,
        columns_list,
        attribute_range,
        nodes,
    ) = load_database_info_from_dir(stats_dir)

    normalized_plan = replace_aliases_and_columns(plan, columns_list)

    _, node_features, edges, _, post_t_index_dic = encoding_generate(
        normalized_plan,
        0,
        0,
        {},
        [],
        [],
        tables_index,
        tables_index_all,
        columns_index,
        attribute_range,
        nodes,
        norm_stats,
    )

    remapped_edges = [[post_t_index_dic[src], post_t_index_dic[dst]] for src, dst in edges]

    x = np.asarray(node_features, dtype=np.float32)
    if remapped_edges:
        edge_index = np.asarray(remapped_edges, dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    metadata = {
        "num_nodes": int(x.shape[0]),
        "raw_node_feature_dim": int(x.shape[1]) if x.ndim == 2 and x.shape[0] else 0,
        "num_edges": int(edge_index.shape[1]),
        "table_num": len(tables_index),
        "column_num": len(columns_index),
        "node_type_num": len(nodes),
        "expected_raw_node_feature_dim": int(len(nodes) + 2 + len(tables_index) + 8 * len(columns_index)),
        "postgres_total_cost": float(normalized_plan.get("Total Cost", 0.0)),
        "postgres_plan_rows": float(normalized_plan.get("Plan Rows", 0.0)),
        "root_node_type": normalized_plan.get("Node Type"),
    }

    if "Actual Total Time" in normalized_plan:
        metadata["root_actual_total_time_ms"] = float(normalized_plan["Actual Total Time"])

    return x, edge_index, normalized_plan, metadata


def save_dataset(
        path: Path,
        records: List[Dict[str, Any]],
        norm_stats: List[Any],
        analyze: bool,
        dbname: str,
        save_original_reqo_dataset: bool = True,
        reqo_dataset_dir: Path | None = None,
        min_candidates_per_query: int = 3,
) -> None:
    """
    Save encoded records.

    This function saves two formats:

    1. .pt format:
       Used by custom inference/debug scripts.
       Contains PyG Data objects.

    2. Original Reqo .npy format, optional:
       Used directly by the author's original train.py.
       Required files:
         - postgresql_<dbname>_executed_query_plans_dataset.npy
         - postgresql_<dbname>_executed_query_index.npy
         - postgresql_<dbname>_executed_query_plans_index.npy
         - postgresql_<dbname>_executed_query_plans_index_num.npy
         - postgresql_<dbname>_executed_query_plans_postgres_cost.npy
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "num_records": len(records),
        "norm_stats": norm_stats_to_json(norm_stats),
        "analyze": analyze,
    }

    # -------------------------
    # 1. Save .pt format
    # -------------------------
    data_list = []
    for rec in records:
        data = Data(
            x=torch.tensor(rec["x"], dtype=torch.float32),
            edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
        )

        # Store useful metadata directly on the PyG Data object.
        data.query_group_id = rec["query_group_id"]
        data.template_id = rec.get("template_id")
        data.original_query_id = rec.get("original_query_id")
        data.candidate_id = rec["candidate_id"]
        data.sql = rec["sql"]
        data.metadata = rec["metadata"]
        data.plan = rec["plan"]

        # If EXPLAIN ANALYZE was used, root actual runtime becomes a training label.
        if "root_actual_total_time_ms" in rec["metadata"]:
            data.y = torch.tensor(
                [rec["metadata"]["root_actual_total_time_ms"]],
                dtype=torch.float32,
            )

        data_list.append(data)

    payload = {
        "data_list": data_list,
        "metadata": dataset_metadata,
    }
    torch.save(payload, path)
    print(f"Saved .pt dataset to: {path}")

    # -------------------------
    # 2. Optionally save original Reqo .npy format
    # -------------------------
    if not save_original_reqo_dataset:
        return

    if not analyze:
        raise ValueError(
            "Original Reqo training dataset requires runtime labels. "
            "Please run encoding with --analyze."
        )

    if reqo_dataset_dir is None:
        reqo_dataset_dir = (Path("Data") / dbname / "datasets").resolve()
    else:
        reqo_dataset_dir = reqo_dataset_dir.expanduser().resolve()
    reqo_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Group records by query_group_id.
    # Original Reqo needs query_plans_index_num to know how many candidate
    # plans belong to each query.
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["query_group_id"], []).append(rec)

    dataset_rows = []
    query_index = []
    query_metadata = []
    query_plans_index = []
    query_plans_index_num = []
    query_plans_postgres_cost = []

    dropped_groups = 0
    dropped_plans = 0

    for query_group_id in sorted(grouped.keys()):
        group = grouped[query_group_id]

        # For real plan selector training, this should usually be >= 2.
        if len(group) < min_candidates_per_query:
            dropped_groups += 1
            dropped_plans += len(group)
            continue

        start_len = len(dataset_rows)
        plan_ids = []
        postgres_costs = []

        for rec in group:
            runtime = rec["metadata"].get("root_actual_total_time_ms")
            if runtime is None:
                dropped_plans += 1
                continue

            x = rec["x"].astype(np.float32).tolist()

            # PyG edge_index is [2, E].
            # Original Reqo train.py expects row[1] as [E, 2],
            # because it later calls torch.LongTensor(row[1]).t().
            edge_index = rec["edge_index"]
            if edge_index.shape[0] == 2:
                edge_list_e_by_2 = edge_index.T.astype(np.int64).tolist()
            elif edge_index.shape[1] == 2:
                edge_list_e_by_2 = edge_index.astype(np.int64).tolist()
            else:
                raise ValueError(
                    f"Unexpected edge_index shape for query_group_id={query_group_id}: "
                    f"{edge_index.shape}"
                )

            dataset_rows.append([
                x,
                edge_list_e_by_2,
                float(runtime),
            ])

            plan_ids.append(rec["candidate_id"])
            postgres_costs.append(
                float(rec["metadata"].get("postgres_total_cost", np.nan))
            )

        kept = len(dataset_rows) - start_len

        if kept >= min_candidates_per_query:
            query_index.append(query_group_id)
            query_metadata.append(query_metadata_from_record(group[0]))
            query_plans_index.append(plan_ids)
            query_plans_index_num.append(kept)
            query_plans_postgres_cost.append(postgres_costs)
        else:
            # Remove partially added rows for this query group.
            dataset_rows = dataset_rows[:start_len]
            dropped_groups += 1
            dropped_plans += len(group)

    if not dataset_rows:
        raise RuntimeError(
            "No rows written to original Reqo dataset. "
            "Use --analyze and make sure each query group has enough candidate plans."
        )

    prefix = reqo_dataset_dir / f"postgresql_{dbname}_executed_query"

    np.save(
        f"{prefix}_plans_dataset.npy",
        np.array(dataset_rows, dtype=object),
    )
    np.save(
        f"{prefix}_index.npy",
        np.array(query_index, dtype=object),
    )
    np.save(
        f"{prefix}_metadata.npy",
        np.array(query_metadata, dtype=object),
    )
    np.save(
        f"{prefix}_plans_index.npy",
        np.array(query_plans_index, dtype=object),
    )
    np.save(
        f"{prefix}_plans_index_num.npy",
        np.array(query_plans_index_num, dtype=object),
    )
    np.save(
        f"{prefix}_plans_postgres_cost.npy",
        np.array(query_plans_postgres_cost, dtype=object),
    )
    metadata_csv_path = reqo_dataset_dir / f"postgresql_{dbname}_executed_query_metadata.csv"
    write_query_metadata_csv(metadata_csv_path, query_metadata)

    summary = {
        "reqo_dataset_dir": str(reqo_dataset_dir.resolve()),
        "written_plans": len(dataset_rows),
        "written_query_groups": len(query_index),
        "dropped_groups": dropped_groups,
        "dropped_plans": dropped_plans,
        "min_candidates_per_query": min_candidates_per_query,
        "files": {
            "plans_dataset": f"{prefix}_plans_dataset.npy",
            "index": f"{prefix}_index.npy",
            "metadata": f"{prefix}_metadata.npy",
            "metadata_csv": str(metadata_csv_path),
            "plans_index": f"{prefix}_plans_index.npy",
            "plans_index_num": f"{prefix}_plans_index_num.npy",
            "plans_postgres_cost": f"{prefix}_plans_postgres_cost.npy",
        },
    }

    summary_path = reqo_dataset_dir / f"postgresql_{dbname}_executed_query_conversion_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4, default=str),
        encoding="utf-8",
    )

    print("Saved original Reqo .npy dataset files:")
    print(json.dumps(summary, indent=4, default=str))


def main() -> None:
    args = parse_args()
    if args.min_candidates_per_query <= 0:
        raise ValueError("--min-candidates-per-query must be positive.")
    if args.sql_file is None:
        raise ValueError("--sql-file is required.")
    if args.plans_cache_only and args.plans_cache_output is None:
        raise ValueError("--plans-cache-only requires --plans-cache-output.")
    if args.plans_cache_input is not None and args.plans_cache_output is not None:
        raise ValueError("Use either --plans-cache-input or --plans-cache-output, not both.")

    stats_dir = Path(args.stats_dir).resolve()
    sql_file_path = Path(args.sql_file).resolve()
    sql_rows_df: DataFrame = pd.read_csv(sql_file_path)
    sql_rows = normalize_sql_rows(sql_rows_df)

    print(f"Loaded {len(sql_rows)} SQL queries from {args.sql_file}")
    print(f"Database statistics dir: {stats_dir}")
    if args.plans_cache_input is not None:
        print("Mode: raw plan cache input, PostgreSQL will not be contacted")
    else:
        print(
            "Mode:",
            "EXPLAIN ANALYZE, every SQL will be executed"
            if args.analyze
            else "EXPLAIN only, queries are planned but not executed",
        )

    plans_by_row: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if args.plans_cache_input is not None:
        cache_path = Path(args.plans_cache_input).resolve()
        plans_by_row = load_plans_from_cache(cache_path, sql_rows)
        print(f"Loaded {len(plans_by_row)} plans from cache: {cache_path}")
    else:
        with open_connection(
                dbname=args.dbname,
                host=args.host,
                port=str(args.port),
                user=args.user,
                password=args.password,
        ) as conn:
            with conn.cursor() as cur:
                if args.statement_timeout_ms and args.statement_timeout_ms > 0:
                    cur.execute("SET statement_timeout = %s", (args.statement_timeout_ms,))

                for item in tqdm(sql_rows, desc="Explaining SQLs", total=len(sql_rows)):
                    try:
                        explain_doc = run_explain_json_with_cursor(
                            cur=cur,
                            sql=item["sql_text"],
                            analyze=args.analyze,
                        )
                        plans_by_row.append({
                            **item,
                            "sql": item["sql_text"],
                            "plan": explain_doc["Plan"],
                        })
                    except Exception as exc:
                        conn.rollback()
                        if (
                                args.analyze
                                and args.statement_timeout_ms
                                and args.statement_timeout_ms > 0
                                and is_statement_timeout(exc)
                        ):
                            try:
                                explain_doc = collect_timeout_fallback_plan(
                                    cur=cur,
                                    sql=item["sql_text"],
                                    timeout_ms=args.statement_timeout_ms,
                                )
                                plans_by_row.append({
                                    **item,
                                    "sql": item["sql_text"],
                                    "plan": explain_doc["Plan"],
                                    "timeout_runtime_ms": args.statement_timeout_ms,
                                    "timed_out": True,
                                    "timeout_error": repr(exc),
                                })
                                print(
                                    "\nStatement timed out; using timeout as "
                                    f"runtime for line {item['row_number']}."
                                )
                                continue
                            except Exception as fallback_exc:
                                conn.rollback()
                                exc = fallback_exc
                        error_obj = {
                            "query_group_id": item["query_group_id"],
                            "template_id": item["template_id"],
                            "original_query_id": item["original_query_id"],
                            "candidate_id": item["candidate_id"],
                            "sql": item["sql_text"],
                            "error": repr(exc),
                        }
                        if args.skip_errors:
                            errors.append(error_obj)
                            print(f"\nSkipped line {item['row_number']}: {exc}")
                            continue
                        raise RuntimeError(
                            f"Failed on row {item['row_number']}: {item['sql_text']}\n{exc}"
                        ) from exc

    if not plans_by_row:
        raise RuntimeError("No SQL plans were successfully collected.")
    if args.plans_cache_output is not None:
        save_plans_cache(Path(args.plans_cache_output).resolve(), plans_by_row)
    if args.plans_cache_only:
        if errors:
            error_path = Path(args.plans_cache_output).resolve().with_suffix(".errors.json")
            error_path.write_text(json.dumps(errors, indent=4), encoding="utf-8")
            print(f"Saved skipped SQL errors to: {error_path}")
        return

    if args.norm_stats:
        norm_stats = load_norm_stats_json(Path(args.norm_stats))
        print(f"Loaded normalization stats from: {args.norm_stats}")
    else:
        norm_stats = collect_global_plan_stats([item["plan"] for item in plans_by_row])
        print("Auto-computed normalization stats from all collected plans:")
        print(json.dumps(norm_stats_to_json(norm_stats), indent=4))

    if args.norm_stats_output:
        norm_stats_path = Path(args.norm_stats_output)
        norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
        norm_stats_path.write_text(json.dumps(norm_stats_to_json(norm_stats), indent=4), encoding="utf-8")
        print(f"Saved normalization stats to: {norm_stats_path}")

    records: List[Dict[str, Any]] = []
    for item in tqdm(plans_by_row, desc="Encoding plans", total=len(plans_by_row)):
        try:
            x, edge_index, normalized_plan, metadata = encode_plan(
                plan=item["plan"],
                stats_dir=stats_dir,
                norm_stats=norm_stats,
            )
            metadata["query_group_id"] = item["query_group_id"]
            metadata["template_id"] = item["template_id"]
            metadata["original_query_id"] = item["original_query_id"]
            metadata["candidate_id"] = item["candidate_id"]
            if "timeout_runtime_ms" in item:
                # Timeout rows keep their plan shape but use the configured
                # statement_timeout as the supervised runtime label.
                metadata["root_actual_total_time_ms"] = float(item["timeout_runtime_ms"])
                metadata["timed_out"] = bool(item.get("timed_out", True))
                metadata["timeout_error"] = item.get("timeout_error")

            records.append({
                "query_group_id": item["query_group_id"],
                "template_id": item["template_id"],
                "original_query_id": item["original_query_id"],
                "candidate_id": item["candidate_id"],
                "sql": item["sql"],
                "x": x,
                "edge_index": edge_index,
                "metadata": metadata,
                "plan": normalized_plan,
            })
        except Exception as exc:
            error_obj = {
                "query_group_id": item["query_group_id"],
                "template_id": item["template_id"],
                "original_query_id": item["original_query_id"],
                "candidate_id": item["candidate_id"],
                "sql": item["sql"],
                "error": repr(exc),
            }
            if args.skip_errors:
                errors.append(error_obj)
                print(f"\nSkipped encoding line {item['row_number']}: {exc}")
                continue
            raise RuntimeError(
                f"Failed encoding line {item['row_number']}: {item['sql']}\n{exc}"
            ) from exc

    if not records:
        raise RuntimeError("No SQL plans were successfully encoded.")

    out_path = Path(args.output_dir) / "encode.pt"
    save_dataset(
        out_path,
        records,
        norm_stats,
        analyze=args.analyze or args.plans_cache_input is not None,
        dbname=args.dbname,
        save_original_reqo_dataset=not args.no_save_original_reqo_dataset,
        reqo_dataset_dir=(
            Path(args.reqo_dataset_dir)
            if args.reqo_dataset_dir is not None
            else None
        ),
        min_candidates_per_query=args.min_candidates_per_query,
    )

    if errors:
        error_path = out_path.with_suffix(out_path.suffix + ".errors.json")
        error_path.write_text(json.dumps(errors, indent=4), encoding="utf-8")
        print(f"Saved {len(errors)} skipped errors to: {error_path}")

    print(f"Saved encoded dataset to: {out_path}")
    print(f"Encoded records: {len(records)}")

    if not args.analyze:
        print(
            "Note: this output has no runtime label. Use --analyze for training labels, "
            "or merge labels from your own execution logs."
        )


if __name__ == "__main__":
    main()

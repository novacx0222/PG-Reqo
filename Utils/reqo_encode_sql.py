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
  python reqo_encode_sql_lines.py \
    --repo-root /path/to/Reqo-on-PostgreSQL \
    --sql-file ./queries.sql \
    --dbname imdb \
    --host localhost \
    --port 5432 \
    --user postgres \
    --password 123456 \
    --output ./encoded_queries.pt \
    --norm-stats-output ./norm_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import psycopg2

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable: Iterable, **kwargs):  # type: ignore
        """Small fallback when tqdm is not installed."""
        total = kwargs.get("total", None)
        desc = kwargs.get("desc", "")
        for index, item in enumerate(iterable, start=1):
            if total:
                print(f"{desc}: {index}/{total}")
            else:
                print(f"{desc}: {index}")
            yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode one SQL-per-line file into Reqo raw graph encodings."
    )
    parser.add_argument("--sql-file", required=True, help="Path to a file. Each non-empty line is one SQL.")
    parser.add_argument("--dbname", required=True, help="PostgreSQL database name.")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host.")
    parser.add_argument("--port", default="5432", help="PostgreSQL port.")
    parser.add_argument("--user", required=True, help="PostgreSQL username.")
    parser.add_argument("--password", default=None, help="PostgreSQL password. Optional.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help=(
            "Path to Reqo-on-PostgreSQL repo root, or directly to its Utils directory. "
            "Default: current directory."
        ),
    )
    parser.add_argument(
        "--stats-dir",
        default=None,
        help=(
            "Directory containing Reqo database_statistics/*.npy files. "
            "Default: <repo-root>/../Data/<dbname>/database_statistics"
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path. Supported extensions: .pt, .npz, .json. Recommended: .pt.",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "pt", "npz", "json"],
        default="auto",
        help="Output format. Default infers from --output extension.",
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


def add_reqo_utils_to_path(repo_root: Path) -> Path:
    """
    Add Reqo's Utils directory to sys.path.

    The uploaded one-query version expected query_plan_feature_extraction.py directly
    under --repo-root. This version supports both:
      - --repo-root /path/to/Reqo-on-PostgreSQL
      - --repo-root /path/to/Reqo-on-PostgreSQL/Utils
    """
    repo_root = repo_root.resolve()

    candidates = [
        repo_root / "Utils",
        repo_root,
    ]

    for candidate in candidates:
        if (candidate / "query_plan_feature_extraction.py").exists():
            sys.path.insert(0, str(candidate))
            return candidate

    raise FileNotFoundError(
        "Cannot find query_plan_feature_extraction.py. "
        "Pass --repo-root /path/to/Reqo-on-PostgreSQL or /path/to/Reqo-on-PostgreSQL/Utils."
    )


def resolve_stats_dir(repo_root: Path, utils_dir: Path, dbname: str, stats_dir_arg: Optional[str]) -> Path:
    """Resolve the database_statistics directory."""
    if stats_dir_arg:
        return Path(stats_dir_arg).resolve()

    # If repo_root is Utils, then its parent is the repo root.
    possible_repo_root = utils_dir.parent if utils_dir.name == "Utils" else repo_root.resolve()

    candidates = [
        possible_repo_root / ".." / "Data" / dbname / "database_statistics",
        possible_repo_root / "Data" / dbname / "database_statistics",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Return Reqo's original default even if it does not exist, so the error message is useful.
    return candidates[0].resolve()


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
            "or pass --stats-dir to an existing database_statistics directory."
        )

    tables_index = np.load(stats_dir / "tables_index.npy", allow_pickle=True).item()
    tables_index_all = np.load(stats_dir / "tables_index_all.npy", allow_pickle=True).item()
    columns_index = np.load(stats_dir / "columns_index.npy", allow_pickle=True).item()
    columns_list = np.load(stats_dir / "columns_list.npy", allow_pickle=True)
    attribute_range = np.load(stats_dir / "attribute_range.npy", allow_pickle=True).item()
    nodes = np.load(stats_dir / "postgresql_nodestypes_all.npy", allow_pickle=True).item()
    return tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes


def read_sql_lines(sql_file: Path) -> List[Tuple[int, str]]:
    """
    Read one SQL query per non-empty line.

    Lines starting with -- are ignored. Inline comments are not stripped because they
    may appear inside strings; keep your input as one complete SQL per line.
    """
    sqls: List[Tuple[int, str]] = []
    for line_number, raw_line in enumerate(sql_file.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("--"):
            continue
        sql = line.rstrip().rstrip(";")
        if sql:
            sqls.append((line_number, sql))

    if not sqls:
        raise ValueError(f"No SQL queries found in {sql_file}. Expected one SQL per non-empty line.")
    return sqls


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


def infer_output_format(path: Path, fmt: str) -> str:
    """Infer output format from extension."""
    if fmt != "auto":
        return fmt
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return "pt"
    if suffix == ".json":
        return "json"
    return "npz"


def save_dataset(
        path: Path,
        fmt: str,
        records: List[Dict[str, Any]],
        norm_stats: List[Any],
        analyze: bool,
) -> None:
    """Save all encoded records in the requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)

    dataset_metadata = {
        "num_records": len(records),
        "norm_stats": norm_stats_to_json(norm_stats),
        "analyze": analyze,
    }

    if fmt == "pt":
        try:
            import torch
            from torch_geometric.data import Data
        except Exception as exc:
            raise RuntimeError(
                "Saving .pt requires torch and torch_geometric. "
                "Use --output-format npz if you only need arrays."
            ) from exc

        data_list = []
        for rec in records:
            data = Data(
                x=torch.tensor(rec["x"], dtype=torch.float32),
                edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
            )

            # Store useful metadata directly on the PyG Data object.
            data.query_id = rec["query_id"]
            data.sql_id = rec["query_id"]
            data.line_number = rec["line_number"]
            data.sql = rec["sql"]
            data.metadata = rec["metadata"]
            data.plan = rec["plan"]

            # If EXPLAIN ANALYZE was used, root actual runtime becomes a training label.
            if "root_actual_total_time_ms" in rec["metadata"]:
                data.y = torch.tensor([rec["metadata"]["root_actual_total_time_ms"]], dtype=torch.float32)

            data_list.append(data)

        payload = {
            "data_list": data_list,
            "metadata": dataset_metadata,
        }
        torch.save(payload, path)

    elif fmt == "npz":
        np.savez_compressed(
            path,
            x_list=np.asarray([rec["x"] for rec in records], dtype=object),
            edge_index_list=np.asarray([rec["edge_index"] for rec in records], dtype=object),
            query_ids=np.asarray([rec["query_id"] for rec in records], dtype=object),
            line_numbers=np.asarray([rec["line_number"] for rec in records], dtype=np.int64),
            sqls=np.asarray([rec["sql"] for rec in records], dtype=object),
            metadata_json=json.dumps(dataset_metadata, default=str),
            record_metadata_json=json.dumps([rec["metadata"] for rec in records], default=str),
            plans_json=json.dumps([rec["plan"] for rec in records], default=str),
        )

    elif fmt == "json":
        payload = {
            "metadata": dataset_metadata,
            "records": [
                {
                    "query_id": rec["query_id"],
                    "line_number": rec["line_number"],
                    "sql": rec["sql"],
                    "x": rec["x"].tolist(),
                    "edge_index": rec["edge_index"].tolist(),
                    "metadata": rec["metadata"],
                    "plan": rec["plan"],
                }
                for rec in records
            ],
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    else:
        raise ValueError(f"Unknown output format: {fmt}")


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root)
    utils_dir = add_reqo_utils_to_path(repo_root)
    stats_dir = resolve_stats_dir(repo_root, utils_dir, args.dbname, args.stats_dir)

    sql_rows = read_sql_lines(Path(args.sql_file))

    print(f"Loaded {len(sql_rows)} SQL queries from {args.sql_file}")
    print(f"Reqo Utils dir: {utils_dir}")
    print(f"Database statistics dir: {stats_dir}")
    print("Mode:",
          "EXPLAIN ANALYZE, every SQL will be executed" if args.analyze else "EXPLAIN only, queries are planned but not executed")

    plans_by_row: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

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

            for idx, (line_number, sql_text) in enumerate(
                    tqdm(sql_rows, desc="Explaining SQLs", total=len(sql_rows)),
                    start=1,
            ):
                query_id = f"{args.query_id_prefix}{idx:06d}"
                try:
                    explain_doc = run_explain_json_with_cursor(
                        cur=cur,
                        sql=sql_text,
                        analyze=args.analyze,
                    )
                    plans_by_row.append(
                        {
                            "query_id": query_id,
                            "line_number": line_number,
                            "sql": sql_text,
                            "plan": explain_doc["Plan"],
                        }
                    )
                except Exception as exc:
                    conn.rollback()
                    error_obj = {
                        "query_id": query_id,
                        "line_number": line_number,
                        "sql": sql_text,
                        "error": repr(exc),
                    }
                    if args.skip_errors:
                        errors.append(error_obj)
                        print(f"\nSkipped line {line_number}: {exc}")
                        continue
                    raise RuntimeError(f"Failed on line {line_number}: {sql_text}\n{exc}") from exc

    if not plans_by_row:
        raise RuntimeError("No SQL plans were successfully collected.")

    if args.norm_stats:
        norm_stats = load_norm_stats_json(Path(args.norm_stats))
        print(f"Loaded normalization stats from: {args.norm_stats}")
    else:
        norm_stats = collect_global_plan_stats([item["plan"] for item in plans_by_row])
        print("Auto-computed normalization stats from all collected plans:")
        print(json.dumps(norm_stats_to_json(norm_stats), indent=2))

    if args.norm_stats_output:
        norm_stats_path = Path(args.norm_stats_output)
        norm_stats_path.parent.mkdir(parents=True, exist_ok=True)
        norm_stats_path.write_text(json.dumps(norm_stats_to_json(norm_stats), indent=2), encoding="utf-8")
        print(f"Saved normalization stats to: {norm_stats_path}")

    records: List[Dict[str, Any]] = []
    for item in tqdm(plans_by_row, desc="Encoding plans", total=len(plans_by_row)):
        try:
            x, edge_index, normalized_plan, metadata = encode_plan(
                plan=item["plan"],
                stats_dir=stats_dir,
                norm_stats=norm_stats,
            )
            metadata["line_number"] = item["line_number"]
            metadata["query_id"] = item["query_id"]

            records.append(
                {
                    "query_id": item["query_id"],
                    "line_number": item["line_number"],
                    "sql": item["sql"],
                    "x": x,
                    "edge_index": edge_index,
                    "metadata": metadata,
                    "plan": normalized_plan,
                }
            )
        except Exception as exc:
            error_obj = {
                "query_id": item["query_id"],
                "line_number": item["line_number"],
                "sql": item["sql"],
                "error": repr(exc),
            }
            if args.skip_errors:
                errors.append(error_obj)
                print(f"\nSkipped encoding line {item['line_number']}: {exc}")
                continue
            raise RuntimeError(f"Failed encoding line {item['line_number']}: {item['sql']}\n{exc}") from exc

    if not records:
        raise RuntimeError("No SQL plans were successfully encoded.")

    out_path = Path(args.output)
    fmt = infer_output_format(out_path, args.output_format)
    save_dataset(out_path, fmt, records, norm_stats, analyze=args.analyze)

    if errors:
        error_path = out_path.with_suffix(out_path.suffix + ".errors.json")
        error_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"Saved {len(errors)} skipped errors to: {error_path}")

    print(f"Saved encoded dataset to: {out_path}")
    print(f"Encoded records: {len(records)}")
    print(f"Output format: {fmt}")

    if not args.analyze:
        print(
            "Note: this output has no runtime label. Use --analyze for training labels, "
            "or merge labels from your own execution logs."
        )


if __name__ == "__main__":
    main()

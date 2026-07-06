#!/usr/bin/env python3
"""Rank candidate SQL plans from a CSV with a trained Reqo model.

This script is for inference only. It does not use EXPLAIN ANALYZE and does not
need actual runtimes. It gets PostgreSQL plans with EXPLAIN (FORMAT JSON),
encodes them into Reqo graph features, runs the trained model, and ranks
candidate SQLs by the model's integrated score (pred_iv, lower is better).

Input CSV format:
  query_group_id,template_id,original_query_id,candidate_id,sql_text

The sql_text column should already contain any pg_hint_plan hint comment plus
the original SQL.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank candidate SQL plans with a trained Reqo model."
    )
    parser.add_argument(
        "--candidate-csv",
        required=True,
        help=(
            "CSV with columns query_group_id,template_id,"
            "original_query_id,candidate_id,sql_text."
        ),
    )

    parser.add_argument("--dbname", required=True, help="PostgreSQL database name.")
    parser.add_argument("--host", default="localhost", help="PostgreSQL host.")
    parser.add_argument("--port", default="5432", help="PostgreSQL port.")
    parser.add_argument("--user", required=True, help="PostgreSQL username.")
    parser.add_argument("--password", default=None, help="PostgreSQL password. Optional.")
    parser.add_argument(
        "--statement-timeout-ms",
        type=int,
        default=0,
        help="Optional PostgreSQL statement_timeout in milliseconds. 0 means no change.",
    )

    parser.add_argument(
        "--stats-dir",
        required=True,
        help="Directory containing database_statistics/*.npy files.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to a saved reqo_fold_*_model.pth state dict.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help=(
            "Optional normalization JSON from training/encoding. Strongly "
            "recommended. If omitted, stats are computed from candidate plans."
        ),
    )
    parser.add_argument(
        "--norm-stats-output",
        default=None,
        help="Optional path to save auto-computed normalization stats.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Where to save ranked inference results.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip candidates that fail EXPLAIN/encoding instead of stopping.",
    )

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoder-attention-heads", type=int, default=8)
    parser.add_argument("--encoder-conv-layers", type=int, default=4)
    parser.add_argument("--encoder-gnn-embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-gnn-dropout-rate", type=float, default=0.1)
    parser.add_argument("--encoder-dirgnn-alpha", type=float, default=0.3)
    parser.add_argument("--encoder-node-type-embedding-dim", type=int, default=16)
    parser.add_argument("--encoder-column-embedding-dim", type=int, default=8)
    parser.add_argument("--estimator-fcn-layers", type=int, default=4)
    parser.add_argument("--estimator-estimation-embedding-dim", type=int, default=512)
    parser.add_argument("--estimator-fcn-dropout-rate", type=float, default=0.1)

    return parser.parse_args()


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Load candidate SQLs from a CSV."""
    with Path(args.candidate_csv).open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {
            "query_group_id",
            "template_id",
            "original_query_id",
            "candidate_id",
            "sql_text",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Candidate CSV must contain columns: {sorted(required)}"
            )
        return [
            {
                "query_group_id": row["query_group_id"],
                "template_id": row["template_id"],
                "original_query_id": row["original_query_id"],
                "candidate_id": row["candidate_id"],
                "sql_text": row["sql_text"],
            }
            for row in reader
        ]


def build_reqo_model(
        args: argparse.Namespace,
        table_columns_number: Any,
        device: Any,
) -> Any:
    """Construct the Reqo model using the same hyperparameters as training."""
    import torch

    from Models.reqo_model import Reqo

    encoder_params = {
        "encoder_attention_heads": args.encoder_attention_heads,
        "encoder_conv_layers": args.encoder_conv_layers,
        "encoder_gnn_embedding_dim": args.encoder_gnn_embedding_dim,
        "encoder_gnn_dropout_rate": args.encoder_gnn_dropout_rate,
        "encoder_dirgnn_alpha": args.encoder_dirgnn_alpha,
        "encoder_node_type_embedding_dim": args.encoder_node_type_embedding_dim,
        "encoder_column_embedding_dim": args.encoder_column_embedding_dim,
        "encoder_table_num": len(table_columns_number),
        "encoder_column_num": int(sum(table_columns_number)),
    }
    estimator_params = {
        "estimator_fcn_layers": args.estimator_fcn_layers,
        "estimator_estimation_embedding_dim": args.estimator_estimation_embedding_dim,
        "estimator_fcn_dropout_rate": args.estimator_fcn_dropout_rate,
    }

    model = Reqo(encoder_params=encoder_params, estimator_params=estimator_params)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def collect_candidate_plans(
        args: argparse.Namespace,
        candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run EXPLAIN JSON for candidates.

    This intentionally does not ANALYZE, so inference does not execute the
    candidate SQLs or require actual runtimes.
    """
    from reqo_encode_sql import open_connection, run_explain_json_with_cursor  # type: ignore

    plans = []
    errors = []
    with open_connection(
            dbname=args.dbname,
            host=args.host,
            port=str(args.port),
            user=args.user,
            password=args.password,
    ) as conn:
        with conn.cursor() as cur:
            if args.statement_timeout_ms and args.statement_timeout_ms > 0:
                cur.execute("SET statement_timeout = %s", args.statement_timeout_ms)

            for candidate in candidates:
                try:
                    explain_doc = run_explain_json_with_cursor(
                        cur=cur,
                        sql=candidate["sql_text"],
                        analyze=False,
                    )
                    plans.append({
                        **candidate,
                        "plan": explain_doc["Plan"],
                    })
                except Exception as exc:
                    conn.rollback()
                    error_obj = {
                        **candidate,
                        "error": repr(exc),
                    }
                    if args.skip_errors:
                        errors.append(error_obj)
                        continue
                    raise RuntimeError(
                        "Failed to EXPLAIN candidate "
                        f"query_group_id={candidate['query_group_id']} "
                        f"candidate_id={candidate['candidate_id']}\n{exc}"
                    ) from exc
    return plans, errors


def encode_candidates(
        plans: list[dict[str, Any]],
        stats_dir: Path,
        norm_stats: list[Any],
) -> list[Any]:
    """Encode EXPLAIN plans into PyG Data objects for model inference."""
    import torch
    from torch_geometric.data import Data

    from reqo_encode_sql import encode_plan  # type: ignore

    data_list = []
    for item in plans:
        x, edge_index, normalized_plan, metadata = encode_plan(
            plan=item["plan"],
            stats_dir=stats_dir,
            norm_stats=norm_stats,
        )
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
        )
        data.query_group_id = item["query_group_id"]
        data.template_id = item["template_id"]
        data.original_query_id = item["original_query_id"]
        data.candidate_id = item["candidate_id"]
        data.sql = item["sql_text"]
        data.metadata = metadata
        data.plan = normalized_plan
        data_list.append(data)
    return data_list


def run_model_inference(
        model: Any,
        data_list: list[Any],
        table_columns_number: Any,
        batch_size: int,
        device: Any,
) -> list[dict[str, Any]]:
    """Score candidates with Reqo. Lower pred_iv is better."""
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    table_columns_number = np.asarray(table_columns_number)
    scored_rows = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_ev, pred_va, pred_iv = model(batch, table_columns_number)
            pred_ev = pred_ev.view(-1).detach().cpu().numpy()
            pred_va = pred_va.view(-1).detach().cpu().numpy()
            pred_iv = pred_iv.view(-1).detach().cpu().numpy()

            for i in range(batch.num_graphs):
                scored_rows.append({
                    "query_group_id": str(batch.query_group_id[i]),
                    "template_id": str(batch.template_id[i]),
                    "original_query_id": str(batch.original_query_id[i]),
                    "candidate_id": str(batch.candidate_id[i]),
                    "sql_text": str(batch.sql[i]),
                    "postgres_total_cost": float(batch.metadata[i].get("postgres_total_cost", np.nan)),
                    "pred_ev": float(pred_ev[i]),
                    "pred_va": float(pred_va[i]),
                    "pred_iv": float(pred_iv[i]),
                })
    return scored_rows


def rank_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates within each query group by pred_iv ascending."""
    grouped = defaultdict(list)
    for row in scored_rows:
        grouped[row["query_group_id"]].append(row)

    ranked_rows = []
    for query_group_id in sorted(grouped.keys(), key=str):
        rows = sorted(grouped[query_group_id], key=lambda row: row["pred_iv"])
        for rank, row in enumerate(rows, start=1):
            ranked_rows.append({
                "query_group_id": query_group_id,
                "rank": rank,
                **row,
            })
    return ranked_rows


def write_ranked_csv(output_csv: Path, ranked_rows: list[dict[str, Any]]) -> None:
    """Write ranked candidates to a CSV file."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_group_id",
        "template_id",
        "original_query_id",
        "rank",
        "candidate_id",
        "pred_iv",
        "pred_ev",
        "pred_va",
        "postgres_total_cost",
        "sql_text",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked_rows:
            writer.writerow({name: row[name] for name in fieldnames})


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    from Utils.reqo_encode_sql import (  # type: ignore
        collect_global_plan_stats,
        load_norm_stats_json,
        norm_stats_to_json,
    )

    stats_dir = Path(args.stats_dir).resolve()
    table_columns_number = np.load(stats_dir / "table_columns_number.npy")

    candidates = load_candidates(args)
    print(f"Loaded candidates: {len(candidates)}")

    plans, errors = collect_candidate_plans(args, candidates)
    if not plans:
        raise RuntimeError("No candidate plans were successfully collected.")
    print(f"Collected EXPLAIN plans: {len(plans)}")

    if args.norm_stats:
        norm_stats = load_norm_stats_json(Path(args.norm_stats))
        print(f"Loaded normalization stats: {args.norm_stats}")
    else:
        norm_stats = collect_global_plan_stats([item["plan"] for item in plans])
        print("Auto-computed normalization stats from candidate plans.")
        if args.norm_stats_output:
            norm_stats_output = Path(args.norm_stats_output)
            norm_stats_output.parent.mkdir(parents=True, exist_ok=True)
            norm_stats_output.write_text(
                json.dumps(norm_stats_to_json(norm_stats), indent=4),
                encoding="utf-8",
            )
            print(f"Saved normalization stats: {norm_stats_output}")

    data_list = encode_candidates(
        plans=plans,
        stats_dir=stats_dir,
        norm_stats=norm_stats,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_reqo_model(
        args=args,
        table_columns_number=table_columns_number,
        device=device,
    )
    scored_rows = run_model_inference(
        model=model,
        data_list=data_list,
        table_columns_number=table_columns_number,
        batch_size=args.batch_size,
        device=device,
    )
    ranked_rows = rank_rows(scored_rows)
    write_ranked_csv(Path(args.output_csv), ranked_rows)

    best_by_query_group_id = {}
    for row in ranked_rows:
        if row["rank"] == 1:
            best_by_query_group_id[row["query_group_id"]] = row

    print(f"Saved ranked results: {args.output_csv}")
    for query_group_id, row in best_by_query_group_id.items():
        print(
            "Best candidate: "
            f"query_group_id={query_group_id}, "
            f"candidate_id={row['candidate_id']}, "
            f"pred_iv={row['pred_iv']:.6f}"
        )

    if errors:
        error_path = Path(args.output_csv).with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=4), encoding="utf-8")
        print(f"Saved skipped errors: {error_path}")


if __name__ == "__main__":
    main()

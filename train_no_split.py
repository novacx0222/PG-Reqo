"""Train Reqo on an already split train/test fold.

Use this when folds were created before encoding. The script reads two Reqo
.npy dataset directories and writes the same per-fold result files as train.py:

  - reqo_fold_<id>_split.csv
  - reqo_fold_<id>_results.txt
  - reqo_fold_<id>_query_selection.csv
  - reqo_fold_<id>_candidate_scores.csv
  - reqo_fold_<id>_model.pth, when --save-model is set
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def build_reqo_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the config expected by train.train."""
    return {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "pairrankingloss_margin": args.pairrankingloss_margin,
        "encoder_attention_heads": args.encoder_attention_heads,
        "encoder_conv_layers": args.encoder_conv_layers,
        "encoder_gnn_embedding_dim": args.encoder_gnn_embedding_dim,
        "encoder_gnn_dropout_rate": args.encoder_gnn_dropout_rate,
        "encoder_dirgnn_alpha": args.encoder_dirgnn_alpha,
        "encoder_node_type_embedding_dim": args.encoder_node_type_embedding_dim,
        "encoder_column_embedding_dim": args.encoder_column_embedding_dim,
        "estimator_fcn_layers": args.estimator_fcn_layers,
        "estimator_estimation_embedding_dim": args.estimator_estimation_embedding_dim,
        "estimator_fcn_dropout_rate": args.estimator_fcn_dropout_rate,
        "explainer_fcn_layers": args.explainer_fcn_layers,
        "explainer_explanation_embedding_dim": args.explainer_explanation_embedding_dim,
        "explainer_fcn_dropout_rate": args.explainer_fcn_dropout_rate,
    }


def blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def normalize_query_metadata(metadata: Any, fallback_query_group_id: Any) -> dict[str, Any]:
    if metadata is None:
        return {
            "query_group_id": fallback_query_group_id,
            "template_id": None,
            "original_query_id": None,
        }
    if hasattr(metadata, "item") and not isinstance(metadata, dict):
        try:
            metadata = metadata.item()
        except ValueError:
            pass
    if not isinstance(metadata, dict):
        return {
            "query_group_id": fallback_query_group_id,
            "template_id": None,
            "original_query_id": None,
        }
    return {
        "query_group_id": metadata.get("query_group_id", fallback_query_group_id),
        "template_id": metadata.get("template_id"),
        "original_query_id": metadata.get("original_query_id"),
    }


def metadata_for_index(query_metadata: Any, query_index: Any, idx: int) -> dict[str, Any]:
    query_group_id = query_index[idx] if idx < len(query_index) else idx
    if query_metadata is None or idx >= len(query_metadata):
        return normalize_query_metadata(None, query_group_id)
    return normalize_query_metadata(query_metadata[idx], query_group_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Reqo training on one pre-split fold."
    )
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--train-dataset-dir", type=Path, required=True)
    parser.add_argument("--test-dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for reqo_fold_<id>_* result files.",
    )
    parser.add_argument(
        "--database-statistics-dir",
        type=Path,
        default=None,
        help="Default: Data/<dbname>/database_statistics.",
    )
    parser.add_argument("--save-model", action="store_true")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--pairrankingloss-margin", type=float, default=0.0)

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

    parser.add_argument("--explainer-fcn-layers", type=int, default=4)
    parser.add_argument("--explainer-explanation-embedding-dim", type=int, default=512)
    parser.add_argument("--explainer-fcn-dropout-rate", type=float, default=0.1)
    return parser.parse_args()


def dataset_path(dataset_dir: Path, dbname: str, suffix: str) -> Path:
    return dataset_dir / f"postgresql_{dbname}_executed_query_{suffix}.npy"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required dataset file does not exist: {path}")


def load_dataset_bundle(dataset_dir: Path, dbname: str) -> dict[str, Any]:
    import numpy as np

    dataset_dir = dataset_dir.expanduser().resolve()
    paths = {
        "dataset": dataset_path(dataset_dir, dbname, "plans_dataset"),
        "query_index": dataset_path(dataset_dir, dbname, "index"),
        "query_metadata": dataset_path(dataset_dir, dbname, "metadata"),
        "query_plans_index": dataset_path(dataset_dir, dbname, "plans_index"),
        "query_plans_index_num": dataset_path(dataset_dir, dbname, "plans_index_num"),
        "query_postgres_cost": dataset_path(dataset_dir, dbname, "plans_postgres_cost"),
    }
    for path in paths.values():
        require_file(path)
    return {
        name: np.load(path, allow_pickle=True)
        for name, path in paths.items()
    }


def write_pre_split_details(
        filename: Path,
        fold_id: int,
        train_bundle: dict[str, Any],
        test_bundle: dict[str, Any],
) -> None:
    fieldnames = [
        "fold_id",
        "split",
        "global_query_idx",
        "fold_query_idx",
        "query_group_id",
        "template_id",
        "original_query_id",
        "candidate_count",
    ]
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        global_query_idx = 0
        for split_name, bundle in (
                ("train", train_bundle),
                ("test", test_bundle),
        ):
            fold_query_idx = 0
            query_index = bundle["query_index"]
            query_metadata = bundle["query_metadata"]
            query_plans_index_num = bundle["query_plans_index_num"]
            for local_idx, _query_group_id in enumerate(query_index):
                metadata = metadata_for_index(
                    query_metadata,
                    query_index,
                    local_idx,
                )
                writer.writerow({
                    "fold_id": fold_id,
                    "split": split_name,
                    "global_query_idx": global_query_idx,
                    "fold_query_idx": fold_query_idx if split_name == "test" else "",
                    "query_group_id": metadata["query_group_id"],
                    "template_id": blank_if_none(metadata["template_id"]),
                    "original_query_id": blank_if_none(metadata["original_query_id"]),
                    "candidate_count": int(query_plans_index_num[local_idx]),
                })
                global_query_idx += 1
                if split_name == "test":
                    fold_query_idx += 1


def main() -> None:
    args = parse_args()
    if args.fold_id <= 0:
        raise ValueError("--fold-id must be positive.")

    reqo_config = build_reqo_config(args)
    train_bundle = load_dataset_bundle(args.train_dataset_dir, args.dbname)
    test_bundle = load_dataset_bundle(args.test_dataset_dir, args.dbname)
    from train import train

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not str(output_dir).endswith("/"):
        save_path = str(output_dir) + "/"
    else:
        save_path = str(output_dir)

    write_pre_split_details(
        filename=output_dir / f"reqo_fold_{args.fold_id}_split.csv",
        fold_id=args.fold_id,
        train_bundle=train_bundle,
        test_bundle=test_bundle,
    )

    results, _runtime_per_query = train(
        args.dbname,
        reqo_config,
        args.fold_id,
        train_bundle["dataset"],
        test_bundle["dataset"],
        save_path,
        test_bundle["query_plans_index_num"],
        test_bundle["query_postgres_cost"],
        args.save_model,
        test_bundle["query_index"],
        test_bundle["query_metadata"],
        test_bundle["query_plans_index"],
        database_statistics_dir=args.database_statistics_dir,
    )
    print(f"Fold {args.fold_id} results: {results}")
    print(f"Wrote fold results to: {output_dir}")


if __name__ == "__main__":
    main()

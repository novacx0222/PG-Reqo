#!/usr/bin/env python3
"""Convert a saved Reqo encode.pt file into original Reqo .npy datasets.

This is useful when ``reqo_encode_sql.py`` successfully saved ``encode.pt`` but
ran out of memory while writing the original Reqo training files. The converter
does not connect to PostgreSQL and does not re-run EXPLAIN ANALYZE.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Reqo encode.pt into original Reqo .npy files."
    )
    parser.add_argument(
        "--pt-file",
        type=Path,
        required=True,
        help="Path to encode.pt produced by Utils/reqo_encode_sql.py.",
    )
    parser.add_argument(
        "--dbname",
        required=True,
        help="Database name used in the original Reqo dataset filenames.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for .npy files. Default: Data/<dbname>/datasets, "
            "relative to the current working directory."
        ),
    )
    parser.add_argument(
        "--min-candidates-per-query",
        type=int,
        default=3,
        help="Drop query groups with fewer candidates. Default: 3.",
    )
    return parser.parse_args()


def load_pt_payload(pt_file: Path) -> dict[str, Any]:
    """Load torch payload, handling both old and new torch.load signatures."""
    try:
        payload = torch.load(pt_file, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(pt_file, map_location="cpu")

    if not isinstance(payload, dict) or "data_list" not in payload:
        raise ValueError(f"Expected a dict with data_list in: {pt_file}")
    return payload


def get_data_attr(data: Any, name: str, default: Any = None) -> Any:
    """Read a PyG Data attribute without assuming normal object attributes."""
    return getattr(data, name, default)


def query_sort_key(query_group_id: Any) -> tuple[int, Any]:
    """Sort numeric-looking query IDs numerically and others lexicographically."""
    query_group_id_str = str(query_group_id)
    if query_group_id_str.isdigit():
        return 0, int(query_group_id_str)
    return 1, query_group_id_str


def build_edge_list_e_by_2(edge_index: Any, query_group_id: Any) -> np.ndarray:
    """Convert PyG edge_index into the [E, 2] shape expected by train.py."""
    edge_array = edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    if edge_array.ndim != 2:
        raise ValueError(
            f"Unexpected edge_index rank for query_group_id={query_group_id}: "
            f"{edge_array.shape}"
        )
    if edge_array.shape[0] == 2:
        return edge_array.T
    if edge_array.shape[1] == 2:
        return edge_array
    raise ValueError(
        f"Unexpected edge_index shape for query_group_id={query_group_id}: "
        f"{edge_array.shape}"
    )


def get_runtime_ms(data: Any, metadata: dict[str, Any]) -> float | None:
    """Get the actual runtime label from data.y or metadata."""
    y = get_data_attr(data, "y")
    if y is not None:
        return float(y.detach().cpu().view(-1)[0].item())

    runtime = metadata.get("root_actual_total_time_ms")
    if runtime is None:
        return None
    return float(runtime)


def save_object_rows(path: Path, rows: list[list[Any]]) -> None:
    """Save ragged rows as an object ndarray without list-to-array expansion."""
    array = np.empty((len(rows), 3), dtype=object)
    for row_idx, row in enumerate(rows):
        array[row_idx, 0] = row[0]
        array[row_idx, 1] = row[1]
        array[row_idx, 2] = row[2]
    np.save(path, array)


def save_object_vector(path: Path, values: list[Any]) -> None:
    """Save a list as a one-dimensional object ndarray."""
    array = np.empty(len(values), dtype=object)
    for idx, value in enumerate(values):
        array[idx] = value
    np.save(path, array)


def metadata_value(metadata: dict[str, Any], key: str) -> Any:
    value = metadata.get(key)
    return "" if value is None else value


def write_query_metadata_csv(path: Path, query_metadata: list[dict[str, Any]]) -> None:
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


def get_query_group_id(data: Any) -> Any:
    """Read the Reqo query group id from encode.pt."""
    query_group_id = get_data_attr(data, "query_group_id")
    if query_group_id is not None:
        return query_group_id

    metadata = get_data_attr(data, "metadata", {}) or {}
    return metadata.get("query_group_id")


def get_candidate_id(data: Any, metadata: dict[str, Any]) -> Any:
    """Read the candidate/hint id."""
    candidate_id = get_data_attr(data, "candidate_id")
    if candidate_id is not None:
        return candidate_id
    if "candidate_id" in metadata:
        return metadata["candidate_id"]
    raise ValueError("Found a data item without candidate_id.")


def query_metadata_from_group(query_group_id: Any, group: list[Any]) -> dict[str, Any]:
    metadata = get_data_attr(group[0], "metadata", {}) or {}
    return {
        "query_group_id": query_group_id,
        "template_id": metadata.get("template_id", get_data_attr(group[0], "template_id")),
        "original_query_id": metadata.get(
            "original_query_id",
            get_data_attr(group[0], "original_query_id"),
        ),
    }


def convert_pt_to_reqo_npy(
        pt_file: Path,
        dbname: str,
        output_dir: Path,
        min_candidates_per_query: int,
) -> dict[str, Any]:
    """Convert encode.pt payload into files consumed by run_reqo_train.py."""
    if min_candidates_per_query <= 0:
        raise ValueError("--min-candidates-per-query must be positive.")

    payload = load_pt_payload(pt_file)
    data_list = payload["data_list"]
    if not data_list:
        raise RuntimeError(f"No data records found in: {pt_file}")

    grouped: dict[Any, list[Any]] = {}
    for data in data_list:
        query_group_id = get_query_group_id(data)
        if query_group_id is None:
            raise ValueError("Found a data item without query_group_id.")
        grouped.setdefault(query_group_id, []).append(data)

    dataset_rows: list[list[Any]] = []
    query_index: list[Any] = []
    query_metadata: list[dict[str, Any]] = []
    query_plans_index: list[list[Any]] = []
    query_plans_index_num: list[int] = []
    query_plans_postgres_cost: list[list[float]] = []
    dropped_groups = 0
    dropped_plans = 0

    for query_group_id in sorted(grouped, key=query_sort_key):
        group = grouped[query_group_id]
        if len(group) < min_candidates_per_query:
            dropped_groups += 1
            dropped_plans += len(group)
            continue

        start_len = len(dataset_rows)
        plan_ids: list[Any] = []
        postgres_costs: list[float] = []

        for plan_idx, data in enumerate(group):
            metadata = get_data_attr(data, "metadata", {}) or {}
            runtime = get_runtime_ms(data, metadata)
            if runtime is None:
                dropped_plans += 1
                continue

            x = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
            edge_list_e_by_2 = build_edge_list_e_by_2(data.edge_index, query_group_id)
            dataset_rows.append([x, edge_list_e_by_2, float(runtime)])

            plan_ids.append(get_candidate_id(data, metadata))
            postgres_costs.append(
                float(metadata.get("postgres_total_cost", np.nan))
            )

        kept = len(dataset_rows) - start_len
        if kept >= min_candidates_per_query:
            query_index.append(query_group_id)
            query_metadata.append(query_metadata_from_group(query_group_id, group))
            query_plans_index.append(plan_ids)
            query_plans_index_num.append(kept)
            query_plans_postgres_cost.append(postgres_costs)
        else:
            dataset_rows = dataset_rows[:start_len]
            dropped_groups += 1
            dropped_plans += len(group)

    if not dataset_rows:
        raise RuntimeError(
            "No rows written. Make sure encode.pt contains runtime labels "
            "and enough candidates per query."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / f"postgresql_{dbname}_executed_query"

    save_object_rows(Path(f"{prefix}_plans_dataset.npy"), dataset_rows)
    save_object_vector(Path(f"{prefix}_index.npy"), query_index)
    save_object_vector(Path(f"{prefix}_metadata.npy"), query_metadata)
    save_object_vector(Path(f"{prefix}_plans_index.npy"), query_plans_index)
    save_object_vector(
        Path(f"{prefix}_plans_index_num.npy"),
        query_plans_index_num,
    )
    save_object_vector(
        Path(f"{prefix}_plans_postgres_cost.npy"),
        query_plans_postgres_cost,
    )
    metadata_csv_path = output_dir / f"postgresql_{dbname}_executed_query_metadata.csv"
    write_query_metadata_csv(metadata_csv_path, query_metadata)

    summary = {
        "pt_file": str(pt_file.resolve()),
        "reqo_dataset_dir": str(output_dir.resolve()),
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

    summary_path = output_dir / (
        f"postgresql_{dbname}_executed_query_conversion_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=4, default=str),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (Path("Data") / args.dbname / "datasets").resolve()
    )

    summary = convert_pt_to_reqo_npy(
        pt_file=args.pt_file.expanduser().resolve(),
        dbname=args.dbname,
        output_dir=output_dir,
        min_candidates_per_query=args.min_candidates_per_query,
    )
    print("Saved original Reqo .npy dataset files:")
    print(json.dumps(summary, indent=4, default=str))


if __name__ == "__main__":
    main()

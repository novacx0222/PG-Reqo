"""Materialize fold train/test SQL CSVs from existing fold membership.

Use this after candidate hint SQL CSVs have been generated with a fold-specific
error profile. The script does not create a new random split. It reads
folds/<fold-source>_fold_<id>.csv and slices each SOURCE=CSV into:

  fold_sql/<source>/fold_<id>/train.csv
  fold_sql/<source>/fold_<id>/test.csv

This keeps fold membership stable while allowing each fold to have candidate
plans generated under its own error profile.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_imdb_fold_splits import (
    FOLD_FIELDNAMES,
    load_source_rows,
    parse_source_csv_arg,
    write_sql_csv,
)

QueryKey = tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fold_sql train/test CSVs from an existing fold split."
    )
    parser.add_argument(
        "--source-csv",
        action="append",
        required=True,
        metavar="SOURCE=CSV",
        help="Candidate source and SQL CSV path. Repeat for multiple sources.",
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        required=True,
        help="Directory containing <fold-source>_fold_<fold-id>.csv.",
    )
    parser.add_argument(
        "--fold-source",
        default="original",
        help="Fold membership source prefix. Default: original.",
    )
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root output directory. Writes fold_sql/ below it.",
    )
    parser.add_argument(
        "--min-candidates-per-query",
        type=int,
        default=2,
        help="Drop source query groups with fewer candidates. Default: 2.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a source is missing any fold membership query key.",
    )
    return parser.parse_args()


def as_int(value: str, column_name: str, csv_path: Path) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid integer value for {column_name!r} in {csv_path}: {value!r}"
        ) from exc


def load_fold_keys(
        folds_dir: Path,
        fold_source: str,
        fold_id: int,
) -> dict[str, list[QueryKey]]:
    fold_csv = (
        folds_dir.expanduser().resolve()
        / f"{fold_source}_fold_{fold_id}.csv"
    )
    if not fold_csv.is_file():
        raise ValueError(f"Fold membership CSV does not exist: {fold_csv}")

    keys_by_split: dict[str, list[QueryKey]] = {
        "train": [],
        "test": [],
    }
    with fold_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing_columns = sorted(set(FOLD_FIELDNAMES) - set(reader.fieldnames or []))
        if missing_columns:
            raise ValueError(
                f"{fold_csv} is missing required columns: {missing_columns}"
            )

        for row in reader:
            split = row["split"]
            if split not in keys_by_split:
                raise ValueError(f"Unexpected split {split!r} in {fold_csv}.")
            keys_by_split[split].append((
                as_int(row["template_id"], "template_id", fold_csv),
                as_int(row["original_query_id"], "original_query_id", fold_csv),
            ))

    if not keys_by_split["train"] or not keys_by_split["test"]:
        raise ValueError(f"{fold_csv} must contain both train and test rows.")
    return keys_by_split


def available_keys(
        keys: list[QueryKey],
        grouped_rows: dict[QueryKey, list[dict[str, str]]],
) -> tuple[list[QueryKey], int]:
    kept_keys = [key for key in keys if key in grouped_rows]
    return kept_keys, len(keys) - len(kept_keys)


def main() -> None:
    args = parse_args()
    if args.fold_id <= 0:
        raise ValueError("--fold-id must be positive.")
    if args.min_candidates_per_query <= 0:
        raise ValueError("--min-candidates-per-query must be positive.")

    keys_by_split = load_fold_keys(
        folds_dir=args.folds_dir,
        fold_source=args.fold_source,
        fold_id=args.fold_id,
    )
    fold_sql_root = args.output_root.expanduser().resolve() / "fold_sql"

    for source_arg in args.source_csv:
        source_name, csv_path = parse_source_csv_arg(source_arg)
        grouped_rows = load_source_rows(
            csv_path=csv_path,
            min_candidates_per_query=args.min_candidates_per_query,
        )

        for split in ("train", "test"):
            split_keys, missing_count = available_keys(
                keys=keys_by_split[split],
                grouped_rows=grouped_rows,
            )
            if missing_count:
                message = (
                    f"{source_name} fold {args.fold_id} {split}: "
                    f"missing {missing_count} query groups after candidate filtering."
                )
                if args.strict:
                    raise ValueError(message)
                print(f"WARNING: {message}")

            output_csv = (
                fold_sql_root
                / source_name
                / f"fold_{args.fold_id}"
                / f"{split}.csv"
            )
            group_count, row_count = write_sql_csv(
                output_csv=output_csv,
                keys=split_keys,
                grouped_rows=grouped_rows,
            )
            print(
                f"{source_name} fold {args.fold_id} {split}: "
                f"query_groups={group_count}, rows={row_count}, csv={output_csv}"
            )

    print(f"Fold SQL CSVs: {fold_sql_root}")


if __name__ == "__main__":
    main()

"""Materialize fold train/test SQL CSVs from existing fold membership.

Use this after candidate hint SQL CSVs have been generated with a fold-specific
error profile. The script does not create a new random split. It reads
folds/<fold-source>_fold_<id>.csv ownership files and slices each SOURCE=CSV into:

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
LEGACY_FOLD_COLUMNS = {"eval_fold_id", "train_fold_id", "split", "query_fold_id"}


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
    folds_dir = folds_dir.expanduser().resolve()
    fold_csvs = sorted(folds_dir.glob(f"{fold_source}_fold_*.csv"))
    if len(fold_csvs) != 2:
        raise ValueError(
            "This separate-error-profile branch expects exactly 2 fold files "
            f"for source={fold_source!r}; found {len(fold_csvs)} in {folds_dir}."
        )
    def read_owned_keys(owner_fold_id: int) -> list[QueryKey]:
        fold_csv = folds_dir / f"{fold_source}_fold_{owner_fold_id}.csv"
        if not fold_csv.is_file():
            raise ValueError(f"Fold membership CSV does not exist: {fold_csv}")

        keys: list[QueryKey] = []
        with fold_csv.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            missing_columns = sorted(set(FOLD_FIELDNAMES) - set(reader.fieldnames or []))
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
                row_fold_id = as_int(row["fold_id"], "fold_id", fold_csv)
                if row_fold_id != owner_fold_id:
                    raise ValueError(
                        f"{fold_csv} contains row for fold_id={row_fold_id}; "
                        f"expected {owner_fold_id}."
                    )
                keys.append((
                    as_int(row["template_id"], "template_id", fold_csv),
                    as_int(row["original_query_id"], "original_query_id", fold_csv),
                ))
        if not keys:
            raise ValueError(f"{fold_csv} contains no query keys.")
        return keys

    train_fold_id = 3 - fold_id
    return {
        "train": read_owned_keys(train_fold_id),
        "test": read_owned_keys(fold_id),
    }


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
    if args.fold_id not in {1, 2}:
        raise ValueError(
            "This separate-error-profile workflow expects --fold-id to be 1 or 2; "
            f"got {args.fold_id}."
        )
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

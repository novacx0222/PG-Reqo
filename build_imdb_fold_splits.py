"""Build shared IMDb folds for multiple hint-candidate sources.

By default this script splits query groups by (template_id, original_query_id).
With --split-granularity template, it splits by template_id and keeps every
query from the same template in the same fold. It writes:

  - folds/<source>_fold_<k>.csv
  - fold_sql/<source>/fold_<k>/train.csv
  - fold_sql/<source>/fold_<k>/test.csv

The fold membership CSV keeps the same schema as train.py split outputs.
The train/test SQL CSVs keep the encoder schema consumed by reqo_encode_sql.py.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from imdb_workload_common import load_sql_groups


SQL_FIELDNAMES = [
    "query_group_id",
    "template_id",
    "original_query_id",
    "candidate_id",
    "sql_text",
]

FOLD_FIELDNAMES = [
    "fold_id",
    "split",
    "global_query_idx",
    "fold_query_idx",
    "query_group_id",
    "template_id",
    "original_query_id",
    "candidate_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared k-fold splits for IMDb hint SQL CSVs."
    )
    parser.add_argument(
        "--source-csv",
        action="append",
        default=[],
        metavar="SOURCE=CSV",
        help=(
            "Candidate source and SQL CSV path. Repeat for sources such as "
            "robdp_last_level_8x1__0x0=/path/8x1__0x0.csv and "
            "reqo_guc=/path/reqo_guc.csv. If omitted, folds are built from "
            "the original SQL workload only."
        ),
    )
    parser.add_argument(
        "--sqls-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing {template_id}-0_{workload_name} SQLs. "
            "Required when --source-csv is omitted."
        ),
    )
    parser.add_argument(
        "--workload-name",
        default=None,
        help="Workload name used with --sqls-dir when --source-csv is omitted.",
    )
    parser.add_argument(
        "--skip-template-id-vals",
        type=int,
        nargs="+",
        default=[],
        help="Template IDs to skip when --source-csv is omitted. Default: none.",
    )
    parser.add_argument(
        "--query-id-limit",
        type=int,
        default=None,
        help="Keep original SQL query IDs in [0, limit). Default: keep all.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root output directory. Writes folds/ and fold_sql/ below it.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="Number of folds.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed used before assigning queries to folds. Default: 0.",
    )
    parser.add_argument(
        "--split-granularity",
        choices=("query", "template"),
        default="query",
        help=(
            "Split unit. 'query' assigns each (template_id, original_query_id) "
            "independently. 'template' assigns whole template_id groups to "
            "folds. Default: query."
        ),
    )
    parser.add_argument(
        "--warn-if-not-query-granularity",
        action="store_true",
        help=(
            "Print a warning when --split-granularity is not query. This is "
            "useful for separate-error-profile experiments, where query-level "
            "folds avoid holding out whole templates from profile generation."
        ),
    )
    parser.add_argument(
        "--baseline-source",
        action="append",
        default=["original", "robdp"],
        help=(
            "Fold-membership-only source to write under folds/. Repeat to add "
            "more baselines. Default: original and robdp."
        ),
    )
    parser.add_argument(
        "--key-policy",
        choices=("intersection", "union"),
        default="intersection",
        help=(
            "How to choose canonical query keys across sources. Default: "
            "intersection, which guarantees every trainable source has the "
            "same fold query set."
        ),
    )
    parser.add_argument(
        "--min-candidates-per-query",
        type=int,
        default=2,
        help=(
            "Drop source query groups with fewer candidates before building "
            "canonical folds. Default: 2."
        ),
    )
    return parser.parse_args()


def sanitize_source_name(name: str) -> str:
    clean = name.replace("/", "__")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", clean):
        raise ValueError(
            f"Invalid source name {name!r}; use letters, numbers, '_', '.', or '-'."
        )
    return clean


def parse_source_csv_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"--source-csv must be SOURCE=CSV, got: {value}")
    source_name, csv_path = value.split("=", 1)
    source_name = sanitize_source_name(source_name.strip())
    if not source_name:
        raise ValueError("--source-csv source name cannot be empty.")
    return source_name, Path(csv_path).expanduser().resolve()


def as_int(value: Any) -> int:
    return int(float(str(value)))


def candidate_sort_key(row: dict[str, str]) -> tuple[int, str]:
    try:
        return as_int(row["candidate_id"]), row["sql_text"]
    except (KeyError, ValueError):
        return 0, row.get("sql_text", "")


def load_source_rows(
        csv_path: Path,
        min_candidates_per_query: int,
) -> OrderedDict[tuple[int, int], list[dict[str, str]]]:
    if not csv_path.is_file():
        raise ValueError(f"Source CSV does not exist: {csv_path}")

    grouped: OrderedDict[tuple[int, int], list[dict[str, str]]] = OrderedDict()
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(set(SQL_FIELDNAMES) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {missing}")
        for row in reader:
            key = (as_int(row["template_id"]), as_int(row["original_query_id"]))
            grouped.setdefault(key, []).append(row)

    filtered: OrderedDict[tuple[int, int], list[dict[str, str]]] = OrderedDict()
    for key, rows in grouped.items():
        if len(rows) < min_candidates_per_query:
            continue
        filtered[key] = sorted(rows, key=candidate_sort_key)
    return filtered


def build_canonical_keys(
        source_rows: dict[str, OrderedDict[tuple[int, int], list[dict[str, str]]]],
        key_policy: str,
) -> list[tuple[int, int]]:
    key_sets = [set(rows.keys()) for rows in source_rows.values()]
    if not key_sets:
        raise ValueError("No source rows loaded.")
    if key_policy == "intersection":
        keys = set.intersection(*key_sets)
    else:
        keys = set.union(*key_sets)
    if not keys:
        raise ValueError("No canonical query keys remain after applying key policy.")
    return sorted(keys)


def build_canonical_keys_from_sql_workload(
        sqls_dir: Path,
        workload_name: str,
        skip_template_id_vals: list[int],
        query_id_limit: int | None,
) -> list[tuple[int, int]]:
    """Build fold keys directly from original IMDb SQL files."""
    sql_groups = load_sql_groups(
        sqls_dir=sqls_dir,
        workload_name=workload_name,
        skip_template_id_vals=skip_template_id_vals,
        query_id_limit=query_id_limit,
    )
    keys = [
        (template_id, original_query_id)
        for template_id in sorted(sql_groups)
        for original_query_id in sorted(sql_groups[template_id])
    ]
    if not keys:
        raise ValueError("No SQL query keys were loaded from the workload.")
    return keys


def assign_test_folds(
        keys: list[tuple[int, int]],
        fold_count: int,
        split_seed: int,
) -> dict[tuple[int, int], int]:
    if fold_count <= 1:
        raise ValueError("--fold must be greater than 1.")
    if fold_count > len(keys):
        raise ValueError(
            f"--fold={fold_count} is larger than query count {len(keys)}."
        )

    shuffled = list(keys)
    random.Random(split_seed).shuffle(shuffled)
    assignments = {}
    for idx, key in enumerate(shuffled):
        assignments[key] = idx % fold_count + 1
    return assignments


def assign_template_test_folds(
        keys: list[tuple[int, int]],
        fold_count: int,
        split_seed: int,
) -> dict[tuple[int, int], int]:
    if fold_count <= 1:
        raise ValueError("--fold must be greater than 1.")

    template_ids = sorted({template_id for template_id, _ in keys})
    if fold_count > len(template_ids):
        raise ValueError(
            f"--fold={fold_count} is larger than template count {len(template_ids)}."
        )

    shuffled_templates = list(template_ids)
    random.Random(split_seed).shuffle(shuffled_templates)
    template_assignments = {
        template_id: idx % fold_count + 1
        for idx, template_id in enumerate(shuffled_templates)
    }
    return {
        key: template_assignments[key[0]]
        for key in keys
    }


def build_test_fold_assignments(
        keys: list[tuple[int, int]],
        fold_count: int,
        split_seed: int,
        split_granularity: str,
) -> dict[tuple[int, int], int]:
    if split_granularity == "query":
        return assign_test_folds(
            keys=keys,
            fold_count=fold_count,
            split_seed=split_seed,
        )
    if split_granularity == "template":
        return assign_template_test_folds(
            keys=keys,
            fold_count=fold_count,
            split_seed=split_seed,
        )
    raise ValueError(f"Unknown split granularity: {split_granularity}")


def write_fold_membership_csv(
        output_csv: Path,
        fold_id: int,
        keys: list[tuple[int, int]],
        test_assignments: dict[tuple[int, int], int],
        candidate_counts: dict[tuple[int, int], int],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    test_idx = 0
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FOLD_FIELDNAMES)
        writer.writeheader()
        for global_query_idx, (template_id, original_query_id) in enumerate(keys):
            split = "test" if test_assignments[(template_id, original_query_id)] == fold_id else "train"
            fold_query_idx: int | str
            if split == "test":
                fold_query_idx = test_idx
                test_idx += 1
            else:
                fold_query_idx = ""
            writer.writerow({
                "fold_id": fold_id,
                "split": split,
                "global_query_idx": global_query_idx,
                "fold_query_idx": fold_query_idx,
                "query_group_id": global_query_idx,
                "template_id": template_id,
                "original_query_id": original_query_id,
                "candidate_count": candidate_counts.get(
                    (template_id, original_query_id),
                    1,
                ),
            })


def write_sql_csv(
        output_csv: Path,
        keys: list[tuple[int, int]],
        grouped_rows: OrderedDict[tuple[int, int], list[dict[str, str]]],
) -> tuple[int, int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SQL_FIELDNAMES)
        writer.writeheader()
        for query_group_id, key in enumerate(keys):
            rows = grouped_rows.get(key, [])
            for row in rows:
                writer.writerow({
                    "query_group_id": query_group_id,
                    "template_id": key[0],
                    "original_query_id": key[1],
                    "candidate_id": row["candidate_id"],
                    "sql_text": row["sql_text"],
                })
                row_count += 1
    return len(keys), row_count


def main() -> None:
    args = parse_args()
    if args.min_candidates_per_query <= 0:
        raise ValueError("--min-candidates-per-query must be positive.")
    if args.query_id_limit is not None and args.query_id_limit < 0:
        raise ValueError("--query-id-limit must be non-negative.")
    if args.warn_if_not_query_granularity and args.split_granularity != "query":
        print(
            "WARNING: separate-error-profile experiments should normally use "
            "--split-granularity query so every fold profile still sees all "
            "templates through its train split."
        )

    source_specs = [parse_source_csv_arg(value) for value in args.source_csv]
    if source_specs:
        source_rows = {
            source_name: load_source_rows(csv_path, args.min_candidates_per_query)
            for source_name, csv_path in source_specs
        }
        canonical_keys = build_canonical_keys(source_rows, args.key_policy)
    else:
        if args.sqls_dir is None or args.workload_name is None:
            raise ValueError(
                "--sqls-dir and --workload-name are required when "
                "--source-csv is omitted."
            )
        source_rows = {}
        canonical_keys = build_canonical_keys_from_sql_workload(
            sqls_dir=args.sqls_dir,
            workload_name=args.workload_name,
            skip_template_id_vals=args.skip_template_id_vals,
            query_id_limit=args.query_id_limit,
        )
        if args.key_policy != "intersection":
            print("WARNING: --key-policy is ignored without --source-csv.")

    test_assignments = build_test_fold_assignments(
        keys=canonical_keys,
        fold_count=args.fold,
        split_seed=args.split_seed,
        split_granularity=args.split_granularity,
    )

    output_root = args.output_root.expanduser().resolve()
    folds_dir = output_root / "folds"
    fold_sql_dir = output_root / "fold_sql"

    print(f"Canonical query keys: {len(canonical_keys)}")
    print(f"Canonical templates: {len({template_id for template_id, _ in canonical_keys})}")
    print(f"Folds: {args.fold}")
    print(f"Key policy: {args.key_policy}")
    print(f"Split granularity: {args.split_granularity}")

    baseline_sources = [
        sanitize_source_name(source_name)
        for source_name in (args.baseline_source or [])
    ]
    source_names = list(source_rows)
    for fold_id in range(1, args.fold + 1):
        test_keys = [
            key for key in canonical_keys
            if test_assignments[key] == fold_id
        ]
        train_keys = [
            key for key in canonical_keys
            if test_assignments[key] != fold_id
        ]

        for baseline_source in baseline_sources:
            write_fold_membership_csv(
                output_csv=folds_dir / f"{baseline_source}_fold_{fold_id}.csv",
                fold_id=fold_id,
                keys=canonical_keys,
                test_assignments=test_assignments,
                candidate_counts={},
            )

        for source_name in source_names:
            grouped_rows = source_rows[source_name]
            candidate_counts = {
                key: len(rows)
                for key, rows in grouped_rows.items()
            }
            write_fold_membership_csv(
                output_csv=folds_dir / f"{source_name}_fold_{fold_id}.csv",
                fold_id=fold_id,
                keys=canonical_keys,
                test_assignments=test_assignments,
                candidate_counts=candidate_counts,
            )
            train_csv = fold_sql_dir / source_name / f"fold_{fold_id}" / "train.csv"
            test_csv = fold_sql_dir / source_name / f"fold_{fold_id}" / "test.csv"
            train_groups, train_rows = write_sql_csv(
                output_csv=train_csv,
                keys=train_keys,
                grouped_rows=grouped_rows,
            )
            test_groups, test_rows = write_sql_csv(
                output_csv=test_csv,
                keys=test_keys,
                grouped_rows=grouped_rows,
            )
            print(
                f"{source_name} fold {fold_id}: "
                f"train_groups={train_groups}, train_rows={train_rows}, "
                f"test_groups={test_groups}, test_rows={test_rows}"
            )

    print(f"Fold membership CSVs: {folds_dir}")
    if source_names:
        print(f"Fold SQL CSVs: {fold_sql_dir}")
    else:
        print("Fold SQL CSVs: not written because no --source-csv was provided.")


if __name__ == "__main__":
    main()

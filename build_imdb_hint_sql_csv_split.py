"""Build hint-augmented SQL CSVs from exported plan hints.

Each parameter group under ``--results-path`` is converted into train/test
CSVs. ``query_group_id`` is the sequential Reqo query-group id, while
``template_id`` and ``original_query_id`` preserve the original IMDb identity.
``candidate_id`` is the hint/plan id within one query group.
"""

import argparse
import csv
import random
import re
from pathlib import Path

from imdb_workload_common import load_sql_groups

HINT_SOURCE_CHOICES = ("robdp", "robdp-with-partial", "reqo")
CSV_FIELDNAMES = [
    "query_group_id",
    "template_id",
    "original_query_id",
    "candidate_id",
    "sql_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hint-augmented IMDb SQL CSVs from exported hints."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="RobDP results directory, such as /opt/results.",
    )
    parser.add_argument(
        "--sqls-dir",
        type=Path,
        required=True,
        help="Directory containing {template_id}-0_{workload_name} SQLs.",
    )
    parser.add_argument(
        "--workload-name",
        required=True,
        help="Workload name used to load original SQLs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where train/test CSVs per parameter group will be written.",
    )
    parser.add_argument(
        "--parameter-groups",
        nargs="+",
        default=None,
        help="Optional parameter groups to process, e.g. 1x0/0x0.",
    )
    parser.add_argument(
        "--query-id-limit",
        type=int,
        default=None,
        help="Keep original SQL query IDs in [0, limit). Default: keep all.",
    )
    parser.add_argument(
        "--hint-source",
        choices=HINT_SOURCE_CHOICES,
        default="robdp",
        help=(
            "Hint source mode: robdp reads only normal RobDP hints; "
            "robdp-with-partial reads normal and partial RobDP hints; "
            "reqo reads only Reqo GUC hints. Default: robdp."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of query groups assigned to train. Default: 0.9.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Random seed for query-group train/test splitting. Default: 0.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate split-related arguments."""
    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if args.query_id_limit is not None and args.query_id_limit < 0:
        raise ValueError("--query-id-limit must be non-negative.")


def discover_parameter_group_dirs(
        results_path: Path,
        parameter_groups: list[str] | None,
        hint_source: str,
) -> list[Path]:
    """Return parameter-group directories like results_path/1x0/0x0."""
    if parameter_groups is not None:
        return [
            results_path / parameter_group
            for parameter_group in parameter_groups
        ]

    group_pattern = re.compile(r"^\d+x\d+$")
    group_dirs = []
    for add_group_dir in results_path.iterdir():
        if add_group_dir.is_dir() and add_group_dir.name == "reqo_guc":
            if hint_source == "reqo":
                group_dirs.append(add_group_dir)
            continue
        if hint_source == "reqo":
            continue
        if (
                not add_group_dir.is_dir()
                or not group_pattern.fullmatch(add_group_dir.name)
        ):
            continue
        for retain_group_dir in add_group_dir.iterdir():
            if (
                    retain_group_dir.is_dir()
                    and group_pattern.fullmatch(retain_group_dir.name)
            ):
                group_dirs.append(retain_group_dir)
    return sorted(group_dirs, key=lambda path: path.relative_to(results_path).as_posix())


def discover_query_dirs(parameter_group_dir: Path) -> list[tuple[int, int, Path]]:
    """Return sorted (template_id, original_query_id, query_dir) triples."""
    query_dirs = []
    for template_dir in parameter_group_dir.iterdir():
        if not template_dir.is_dir() or not template_dir.name.isdigit():
            continue
        template_id = int(template_dir.name)
        for query_dir in template_dir.iterdir():
            if query_dir.is_dir() and query_dir.name.isdigit():
                query_dirs.append((template_id, int(query_dir.name), query_dir))
    return sorted(query_dirs, key=lambda item: (item[0], item[1]))


def list_hint_files(query_dir: Path, hint_source: str) -> list[Path]:
    """List hint files in deterministic round order."""
    if hint_source == "reqo":
        hint_files = list(query_dir.glob("reqo_hints_[0-9]*.txt"))
    else:
        hint_files = list(query_dir.glob("last_level_hints_[0-9]*.txt"))
    if hint_source == "robdp-with-partial":
        hint_files.extend(
            query_dir.glob("last_level_partial_hints_[0-9]*.txt")
        )
    round_pattern = re.compile(r"_(\d+)\.txt$")
    return sorted(
        hint_files,
        key=lambda path: (
            "partial" in path.name,
            int(round_pattern.search(path.name).group(1)),
            path.name,
        ),
    )


def load_deduped_hints(
        query_dir: Path,
        hint_source: str,
) -> list[str]:
    """Load non-empty hint lines, deduplicated within one query directory."""
    hints = []
    seen_hints = set()
    for hint_file in list_hint_files(query_dir, hint_source):
        with hint_file.open("r", encoding="utf-8") as file:
            for raw_line in file:
                hint = raw_line.strip()
                if not hint or hint in seen_hints:
                    continue
                seen_hints.add(hint)
                hints.append(hint)
    return hints


def output_filename_for_group(
        output_dir: Path,
        results_path: Path,
        parameter_group_dir: Path,
        split_name: str,
) -> Path:
    """Build a stable CSV filename for one parameter-group split."""
    group_name = parameter_group_dir.relative_to(results_path).as_posix()
    return output_dir / f"{group_name.replace('/', '__')}__{split_name}.csv"


def split_query_dirs(
        query_dirs: list[tuple[int, int, Path]],
        train_ratio: float,
        split_seed: int,
) -> tuple[list[tuple[int, int, Path]], list[tuple[int, int, Path]]]:
    """Split query directories while keeping all plans for a query together."""
    if len(query_dirs) <= 1:
        return query_dirs, []

    shuffled_query_dirs = list(query_dirs)
    random.Random(split_seed).shuffle(shuffled_query_dirs)
    train_count = int(len(shuffled_query_dirs) * train_ratio + 0.5)
    train_count = min(max(train_count, 1), len(shuffled_query_dirs) - 1)

    train_keys = {
        (template_id, original_query_id)
        for template_id, original_query_id, _ in shuffled_query_dirs[:train_count]
    }
    train_query_dirs = [
        item
        for item in query_dirs
        if (item[0], item[1]) in train_keys
    ]
    test_query_dirs = [
        item
        for item in query_dirs
        if (item[0], item[1]) not in train_keys
    ]
    return train_query_dirs, test_query_dirs


def write_query_dirs_csv(
        output_csv: Path,
        query_dirs: list[tuple[int, int, Path]],
        sql_groups: dict[int, dict[int, str]],
        hint_source: str,
) -> tuple[int, int]:
    """Write one split CSV and return query-group and row counts."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDNAMES,
        )
        writer.writeheader()

        for query_group_id, (template_id, original_query_id, query_dir) in enumerate(query_dirs):
            try:
                original_sql = sql_groups[template_id][original_query_id]
            except KeyError as exc:
                raise KeyError(
                    "Cannot find original SQL for "
                    f"template_id={template_id}, original_query_id={original_query_id}"
                ) from exc

            hints = load_deduped_hints(
                query_dir=query_dir,
                hint_source=hint_source,
            )
            for candidate_id, hint in enumerate(hints):
                writer.writerow({
                    "query_group_id": query_group_id,
                    "template_id": template_id,
                    "original_query_id": original_query_id,
                    "candidate_id": candidate_id,
                    "sql_text": f"{hint}\n{original_sql}",
                })
                row_count += 1

    return len(query_dirs), row_count


def build_group_split_csvs(
        parameter_group_dir: Path,
        results_path: Path,
        output_dir: Path,
        sql_groups: dict[int, dict[int, str]],
        hint_source: str,
        train_ratio: float,
        split_seed: int,
) -> tuple[Path, Path, int, int, int, int]:
    """Write one parameter group's train/test hint-augmented SQL CSVs."""
    query_dirs = discover_query_dirs(parameter_group_dir)
    train_query_dirs, test_query_dirs = split_query_dirs(
        query_dirs=query_dirs,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    train_csv = output_filename_for_group(
        output_dir=output_dir,
        results_path=results_path,
        parameter_group_dir=parameter_group_dir,
        split_name="train",
    )
    test_csv = output_filename_for_group(
        output_dir=output_dir,
        results_path=results_path,
        parameter_group_dir=parameter_group_dir,
        split_name="test",
    )
    train_group_count, train_row_count = write_query_dirs_csv(
        output_csv=train_csv,
        query_dirs=train_query_dirs,
        sql_groups=sql_groups,
        hint_source=hint_source,
    )
    test_group_count, test_row_count = write_query_dirs_csv(
        output_csv=test_csv,
        query_dirs=test_query_dirs,
        sql_groups=sql_groups,
        hint_source=hint_source,
    )
    return (
        train_csv,
        test_csv,
        train_group_count,
        test_group_count,
        train_row_count,
        test_row_count,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    results_path = args.results_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    sql_groups = load_sql_groups(
        sqls_dir=args.sqls_dir,
        workload_name=args.workload_name,
        skip_template_id_vals=[],
        query_id_limit=args.query_id_limit,
    )
    parameter_group_dirs = discover_parameter_group_dirs(
        results_path=results_path,
        parameter_groups=args.parameter_groups,
        hint_source=args.hint_source,
    )

    print(f"Parameter groups: {len(parameter_group_dirs)}")
    print(f"Train ratio: {args.train_ratio}")
    total_train_groups = 0
    total_test_groups = 0
    total_train_rows = 0
    total_test_rows = 0
    for parameter_group_dir in parameter_group_dirs:
        (
            train_csv,
            test_csv,
            train_group_count,
            test_group_count,
            train_row_count,
            test_row_count,
        ) = build_group_split_csvs(
            parameter_group_dir=parameter_group_dir,
            results_path=results_path,
            output_dir=output_dir,
            sql_groups=sql_groups,
            hint_source=args.hint_source,
            train_ratio=args.train_ratio,
            split_seed=args.split_seed,
        )
        total_train_groups += train_group_count
        total_test_groups += test_group_count
        total_train_rows += train_row_count
        total_test_rows += test_row_count
        print(
            f"{parameter_group_dir.relative_to(results_path)}: "
            f"train_groups={train_group_count}, train_rows={train_row_count}, "
            f"test_groups={test_group_count}, test_rows={test_row_count}"
        )
        print(f"  train_csv={train_csv}")
        print(f"  test_csv={test_csv}")

    print(f"Total train groups: {total_train_groups}")
    print(f"Total test groups: {total_test_groups}")
    print(f"Total train CSV rows: {total_train_rows}")
    print(f"Total test CSV rows: {total_test_rows}")


if __name__ == "__main__":
    main()

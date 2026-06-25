"""Build hint-augmented SQL CSVs from RobDP final-level plan hints.

Each parameter group under ``--results-path`` is converted into one CSV with
columns ``sql_id,query_id,sql_text``. Within one parameter group, each
``template_id/query_id`` directory becomes one sequential ``sql_id``. Each
deduplicated hint line under that directory becomes one ``query_id`` and is
prepended to the original SQL text loaded from ``--sqls-dir``.
"""

import argparse
import csv
import re
from pathlib import Path

from imdb_workload_common import load_sql_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hint-augmented IMDb SQL CSVs from RobDP results."
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
        help="Directory where one CSV per parameter group will be written.",
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
        "--include-parameterized",
        action="store_false",
        help="Also read last_level_plans_parameterized_*.txt files.",
    )
    return parser.parse_args()


def discover_parameter_group_dirs(
        results_path: Path,
        parameter_groups: list[str] | None,
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


def list_hint_files(query_dir: Path, include_parameterized: bool) -> list[Path]:
    """List hint files in deterministic round order."""
    hint_files = list(query_dir.glob("last_level_plans_[0-9]*.txt"))
    if include_parameterized:
        hint_files.extend(
            query_dir.glob("last_level_plans_parameterized_[0-9]*.txt")
        )
    round_pattern = re.compile(r"_(\d+)\.txt$")
    return sorted(
        hint_files,
        key=lambda path: (
            "parameterized" in path.name,
            int(round_pattern.search(path.name).group(1)),
            path.name,
        ),
    )


def load_deduped_hints(
        query_dir: Path,
        include_parameterized: bool,
) -> list[str]:
    """Load non-empty hint lines, deduplicated within one query directory."""
    hints = []
    seen_hints = set()
    for hint_file in list_hint_files(query_dir, include_parameterized):
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
) -> Path:
    """Build a stable CSV filename for one parameter group."""
    group_name = parameter_group_dir.relative_to(results_path).as_posix()
    return output_dir / f"{group_name.replace('/', '__')}.csv"


def build_group_csv(
        parameter_group_dir: Path,
        results_path: Path,
        output_dir: Path,
        sql_groups: dict[int, dict[int, str]],
        include_parameterized: bool,
) -> tuple[Path, int, int]:
    """Write one parameter group's hint-augmented SQL CSV."""
    query_dirs = discover_query_dirs(parameter_group_dir)
    output_csv = output_filename_for_group(
        output_dir=output_dir,
        results_path=results_path,
        parameter_group_dir=parameter_group_dir,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sql_id", "query_id", "sql_text"],
        )
        writer.writeheader()

        for sql_id, (template_id, original_query_id, query_dir) in enumerate(query_dirs):
            try:
                original_sql = sql_groups[template_id][original_query_id]
            except KeyError as exc:
                raise KeyError(
                    "Cannot find original SQL for "
                    f"template_id={template_id}, query_id={original_query_id}"
                ) from exc

            hints = load_deduped_hints(
                query_dir=query_dir,
                include_parameterized=include_parameterized,
            )
            for hint_query_id, hint in enumerate(hints):
                writer.writerow({
                    "sql_id": sql_id,
                    "query_id": hint_query_id,
                    "sql_text": f"{hint}\n{original_sql}",
                })
                row_count += 1

    return output_csv, len(query_dirs), row_count


def main() -> None:
    args = parse_args()
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
    )

    print(f"Parameter groups: {len(parameter_group_dirs)}")
    total_small_groups = 0
    total_rows = 0
    for parameter_group_dir in parameter_group_dirs:
        output_csv, small_group_count, row_count = build_group_csv(
            parameter_group_dir=parameter_group_dir,
            results_path=results_path,
            output_dir=output_dir,
            sql_groups=sql_groups,
            include_parameterized=args.include_parameterized,
        )
        total_small_groups += small_group_count
        total_rows += row_count
        print(
            f"{parameter_group_dir.relative_to(results_path)}: "
            f"small_groups={small_group_count}, rows={row_count}, "
            f"csv={output_csv}"
        )

    print(f"Total small groups: {total_small_groups}")
    print(f"Total CSV rows: {total_rows}")


if __name__ == "__main__":
    main()

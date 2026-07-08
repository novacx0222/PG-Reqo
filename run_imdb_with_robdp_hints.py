"""Run the IMDb workload with RobDP and final-level plan hint exports.

This runner behaves like ``run_imdb_with_robdp.py`` and additionally writes
normal and partial final-DP-level plan hints beside each score file. The
connected PostgreSQL user must be allowed to set the two PGC_SUSET hint-path
GUCs.
"""

import argparse
from pathlib import Path

from imdb_workload_common import (
    GUCDict,
    load_sql_groups,
    print_sql_group_statistics,
    validate_common_args,
)
from run_imdb_with_robdp import (
    generate_additional_guc_dict_list,
    generate_base_guc_dict,
    parse_args,
    run_workload,
)


def build_hint_path_guc_dict(
    query_output_dir: Path,
    round_number: int,
) -> GUCDict:
    """Build the two final-level plan hint filenames for one query round."""
    return {
        "last_level_hint_filename": str(
            query_output_dir / f"last_level_hints_{round_number}.txt"
        ),
        "last_level_partial_hint_filename": str(
            query_output_dir
            / f"last_level_partial_hints_{round_number}.txt"
        ),
    }


def add_hint_args(parser: argparse.ArgumentParser) -> None:
    """Add options that are only meaningful for final-level hint export."""
    parser.add_argument(
        "--final-level-path-limit",
        type=int,
        default=13,
        help="RobDP final-level path limit used while exporting hints. Default: 13.",
    )


def add_final_level_path_limit(
    additional_guc_dict_list: list[GUCDict],
    final_level_path_limit: int,
) -> list[GUCDict]:
    """Attach final_level_path_limit to every RobDP parameter group."""
    if final_level_path_limit < 0:
        raise ValueError("--final-level-path-limit must be non-negative.")
    return [
        {
            **additional_guc_dict,
            "final_level_path_limit": final_level_path_limit,
        }
        for additional_guc_dict in additional_guc_dict_list
    ]


def parse_hint_args() -> argparse.Namespace:
    """Parse common RobDP options plus hint-export-only options."""
    return parse_args(
        description=(
            "Run the IMDb workload against RobDP and export final-level "
            "plan hints."
        ),
        add_extra_args=add_hint_args,
    )


def main() -> None:
    """Load the workload and execute RobDP while exporting plan hints."""
    args = parse_hint_args()
    validate_common_args(args)

    base_guc_dict = generate_base_guc_dict(args)
    additional_guc_dict_list = add_final_level_path_limit(
        additional_guc_dict_list=generate_additional_guc_dict_list(
            main_objective_id_vals=args.main_objective_id_vals,
            retain_strategy_id_vals=args.retain_strategy_id_vals,
            path_limit=args.path_limit,
        ),
        final_level_path_limit=args.final_level_path_limit,
    )
    sql_groups = load_sql_groups(
        sqls_dir=args.sqls_dir,
        workload_name=args.workload_name,
        skip_template_id_vals=args.skip_template_id_vals,
        query_id_limit=args.query_id_limit,
    )
    print_sql_group_statistics(sql_groups, args.workload_name)
    run_workload(
        args=args,
        sql_groups=sql_groups,
        base_guc_dict=base_guc_dict,
        additional_guc_dict_list=additional_guc_dict_list,
        pre_profile_guc_builder=build_hint_path_guc_dict,
    )


if __name__ == "__main__":
    main()

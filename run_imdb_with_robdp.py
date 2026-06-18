"""Run the IMDb workload against the RobDP PostgreSQL backend.

For every RobDP parameter group, template, query, and round, this script sets
the required GUCs, selects the template-specific error profiles, executes the
chosen EXPLAIN mode, and stores both RobDP scores and returned query results.
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import psycopg2

from imdb_workload_common import (
    GUCDict,
    SQLGroups,
    create_argument_parser,
    execute_query,
    load_sql_groups,
    open_connection,
    print_sql_group_statistics,
    save_query_results,
    set_guc_dict,
    validate_common_args,
)


def generate_additional_guc_dict_list(
        main_objective_id_vals: list[int],
        retain_strategy_id_vals: list[int],
        path_limit: int,
) -> list[GUCDict]:
    """Generate the RobDP parameter groups evaluated by the workload."""
    additional_guc_dict_list = []

    for main_objective_id in main_objective_id_vals:
        assert 0 <= main_objective_id <= 16

        # Basic: no retention, e.g., E[Penalty] * 1.
        additional_guc_dict_list.append({
            "main_objective_id": main_objective_id,
            "retain_strategy_id": 0,
            "final_score_id": main_objective_id,
            "add_path_limit": 1,
            "retain_path_limit": 0,
        })
        # Local objective only, e.g., E[Penalty] * 8.
        additional_guc_dict_list.append({
            "main_objective_id": main_objective_id,
            "retain_strategy_id": 0,
            "final_score_id": main_objective_id,
            "add_path_limit": path_limit,
            "retain_path_limit": 0,
        })

        for retain_strategy_id in retain_strategy_id_vals:
            assert 0 <= retain_strategy_id <= 16
            if retain_strategy_id == main_objective_id:
                continue

            # Local objective plus retained paths from another objective.
            additional_guc_dict_list.append({
                "main_objective_id": main_objective_id,
                "retain_strategy_id": retain_strategy_id,
                "final_score_id": main_objective_id,
                "add_path_limit": 1,
                "retain_path_limit": path_limit,
            })

    return additional_guc_dict_list


def parse_args(
    description: str = "Run the IMDb workload against the RobDP backend."
) -> argparse.Namespace:
    """Parse common workload options and RobDP-specific GUC options."""
    parser = create_argument_parser(description=description)
    parser.add_argument(
        "--enable-rows-dist",
        choices=["on", "off"],
        default="on",
        help="Enable row-distribution estimation. Default: on.",
    )
    parser.add_argument(
        "--error-sample-count",
        type=int,
        default=20,
        help="Error sample count. Default: 20.",
    )
    parser.add_argument(
        "--error-bin-count",
        type=int,
        default=8,
        help="Error bin count. Default: 8.",
    )
    parser.add_argument(
        "--error-sample-kde-bandwidth",
        type=float,
        default=0.1,
        help="Error-sample KDE bandwidth. Default: 0.1.",
    )
    parser.add_argument(
        "--main-objective-id-vals",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Main objective IDs. Default: 0 1.",
    )
    parser.add_argument(
        "--retain-strategy-id-vals",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Retain strategy IDs. Default: 0 1.",
    )
    parser.add_argument(
        "--path-limit",
        type=int,
        default=8,
        help="Path limit. Default: 8.",
    )
    return parser.parse_args()


def generate_base_guc_dict(args: argparse.Namespace) -> GUCDict:
    """Build the shared RobDP GUC settings from command-line arguments."""
    if args.error_sample_count < 0:
        raise ValueError("--error-sample-count must be non-negative.")
    if args.error_bin_count < 0:
        raise ValueError("--error-bin-count must be non-negative.")
    if args.error_sample_kde_bandwidth < 0:
        raise ValueError("--error-sample-kde-bandwidth must be non-negative.")

    return {
        "statement_timeout": args.statement_timeout,
        "enable_rows_dist": args.enable_rows_dist,
        "error_sample_count": args.error_sample_count,
        "error_bin_count": args.error_bin_count,
        "error_sample_kde_bandwidth": args.error_sample_kde_bandwidth,
    }


def build_error_profile_path(
        sqls_dir: Path,
        template_id: int,
        workload_name: str,
) -> Path:
    """Build the absolute error-profile directory for one template."""
    return (
            sqls_dir.expanduser().resolve()
            / f"{template_id}-0_{workload_name}"
            / "error_profile"
    )


def build_parameter_group_path(additional_guc_dict: GUCDict) -> Path:
    """Build the directory-style name for one RobDP parameter group."""
    return (
        Path(
            f"{additional_guc_dict['add_path_limit']}x"
            f"{additional_guc_dict['main_objective_id']}"
        )
        / (
            f"{additional_guc_dict['retain_path_limit']}x"
            f"{additional_guc_dict['retain_strategy_id']}"
        )
    )


def build_output_paths(
        results_path: Path,
        additional_guc_dict: GUCDict,
        template_id: int,
        query_id: int,
        round_number: int,
) -> tuple[Path, Path]:
    """Build RobDP score and result paths for one query execution."""
    query_output_dir = (
            results_path.expanduser().resolve()
            / build_parameter_group_path(additional_guc_dict)
            / str(template_id)
            / str(query_id)
    )
    query_output_dir.mkdir(parents=True, exist_ok=True)
    return (
        query_output_dir / f"score_{round_number}.txt",
        query_output_dir / f"results_{round_number}.txt",
    )


def run_single_sql(
        cursor: psycopg2.extensions.cursor,
        sql_string: str,
        base_guc_dict: GUCDict,
        additional_guc_dict: GUCDict,
        score_filename: Path,
    error_profile_path: Path,
    results_filename: Path,
    run_mode: str,
    pre_profile_guc_dict: GUCDict | None = None,
) -> None:
    """Configure RobDP, execute one SQL query, and save its returned rows."""
    # Keep this order: shared GUCs, parameter GUCs, output GUCs, profiles, query.
    set_guc_dict(cursor, base_guc_dict)
    set_guc_dict(cursor, additional_guc_dict)
    set_guc_dict(cursor, {"score_filename": str(score_filename)})
    if pre_profile_guc_dict is not None:
        set_guc_dict(cursor, pre_profile_guc_dict)
    set_guc_dict(cursor, {"error_profile_path": str(error_profile_path)})

    result_text = execute_query(cursor, sql_string, run_mode)
    save_query_results(
        results_filename=results_filename,
        header_lines=[
            f"Parameter group: {build_parameter_group_path(additional_guc_dict)}",
            f"Run mode: {run_mode}",
            f"Additional GUCs: {additional_guc_dict}",
        ],
        result_text=result_text,
    )


def run_workload(
    args: argparse.Namespace,
    sql_groups: SQLGroups,
    base_guc_dict: GUCDict,
    additional_guc_dict_list: list[GUCDict],
    pre_profile_guc_builder: Callable[[Path, int], GUCDict] | None = None,
) -> None:
    """Execute all parameter groups, templates, queries, and rounds."""
    with open_connection(
            dbname=args.dbname,
            host=args.host,
            port=str(args.port),
            user=args.user,
            password=args.password,
    ) as conn:
        with conn.cursor() as cursor:
            for group_index, additional_guc_dict in enumerate(
                    additional_guc_dict_list
            ):
                print(
                    f"Running parameter group {group_index + 1}/"
                    f"{len(additional_guc_dict_list)}: "
                    f"{additional_guc_dict}"
                )

                for template_id in sorted(sql_groups):
                    sql_by_query_id = sql_groups[template_id]
                    error_profile_path = build_error_profile_path(
                        sqls_dir=args.sqls_dir,
                        template_id=template_id,
                        workload_name=args.workload_name,
                    )
                    for query_id in sorted(sql_by_query_id):
                        sql_string = sql_by_query_id[query_id]
                        for round_number in range(1, args.rounds + 1):
                            score_filename, results_filename = build_output_paths(
                                results_path=args.results_path,
                                additional_guc_dict=additional_guc_dict,
                                template_id=template_id,
                                query_id=query_id,
                                round_number=round_number,
                            )
                            pre_profile_guc_dict = (
                                pre_profile_guc_builder(
                                    score_filename.parent,
                                    round_number,
                                )
                                if pre_profile_guc_builder is not None
                                else None
                            )
                            print(
                                "Executing SQL: "
                                f"template_id={template_id}, "
                                f"query_id={query_id}, "
                                f"round={round_number}/{args.rounds}"
                            )
                            run_single_sql(
                                cursor=cursor,
                                sql_string=sql_string,
                                base_guc_dict=base_guc_dict,
                                additional_guc_dict=additional_guc_dict,
                                score_filename=score_filename,
                                error_profile_path=error_profile_path,
                                results_filename=results_filename,
                                run_mode=args.run_mode,
                                pre_profile_guc_dict=pre_profile_guc_dict,
                            )


def main() -> None:
    """Load the workload, prepare RobDP configurations, and execute all runs."""
    args = parse_args()
    validate_common_args(args)

    base_guc_dict = generate_base_guc_dict(args)
    additional_guc_dict_list = generate_additional_guc_dict_list(
        main_objective_id_vals=args.main_objective_id_vals,
        retain_strategy_id_vals=args.retain_strategy_id_vals,
        path_limit=args.path_limit,
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
    )


if __name__ == "__main__":
    main()

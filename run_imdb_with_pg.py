"""Run the IMDb workload against an unmodified PostgreSQL backend.

This baseline runner sets only standard PostgreSQL options, executes every
template/query/round in deterministic order, and stores returned EXPLAIN or
query rows. It does not configure RobDP scores or error profiles.
"""

import argparse
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


def parse_args() -> argparse.Namespace:
    """Parse options for the original PostgreSQL baseline."""
    parser = create_argument_parser(
        description="Run the IMDb workload against original PostgreSQL."
    )
    return parser.parse_args()


def generate_base_guc_dict(args: argparse.Namespace) -> GUCDict:
    """Build the standard PostgreSQL settings used by the baseline."""
    return {
        "statement_timeout": args.statement_timeout
    }


def build_results_filename(
        results_path: Path,
        template_id: int,
        query_id: int,
        round_number: int,
) -> Path:
    """Build the baseline result filename for one query execution."""
    query_output_dir = (
            results_path.expanduser().resolve()
            / "original"
            / str(template_id)
            / str(query_id)
    )
    query_output_dir.mkdir(parents=True, exist_ok=True)
    return query_output_dir / f"results_{round_number}.txt"


def run_single_sql(
        cursor: psycopg2.extensions.cursor,
        sql_string: str,
        base_guc_dict: GUCDict,
        results_filename: Path,
        run_mode: str,
) -> None:
    """Set standard GUCs, execute one query, and save its returned rows."""
    set_guc_dict(cursor, base_guc_dict)
    result_text = execute_query(cursor, sql_string, run_mode)
    save_query_results(
        results_filename=results_filename,
        header_lines=[
            'Parameter group: original',
            f"Run mode: {run_mode}",
        ],
        result_text=result_text,
    )


def run_workload(
        args: argparse.Namespace,
        sql_groups: SQLGroups,
        base_guc_dict: GUCDict,
) -> None:
    """Execute every template, query, and round on original PostgreSQL."""
    with open_connection(
            dbname=args.dbname,
            host=args.host,
            port=str(args.port),
            user=args.user,
            password=args.password,
    ) as conn:
        with conn.cursor() as cursor:
            for template_id in sorted(sql_groups):
                sql_by_query_id = sql_groups[template_id]
                for query_id in sorted(sql_by_query_id):
                    sql_string = sql_by_query_id[query_id]
                    for round_number in range(1, args.rounds + 1):
                        results_filename = build_results_filename(
                            results_path=args.results_path,
                            template_id=template_id,
                            query_id=query_id,
                            round_number=round_number,
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
                            results_filename=results_filename,
                            run_mode=args.run_mode,
                        )


def main() -> None:
    """Load the IMDb workload and execute it on original PostgreSQL."""
    args = parse_args()
    validate_common_args(args)

    base_guc_dict = generate_base_guc_dict(args)
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
    )


if __name__ == "__main__":
    main()

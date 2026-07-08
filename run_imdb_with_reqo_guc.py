"""Run IMDb queries with the original Reqo 13 PostgreSQL GUC candidates.

For every template/query/round, this runner executes the query under the
default PostgreSQL optimizer plus the 12 Reqo ``enable_* = off`` combinations.
It saves the returned EXPLAIN rows and derives one pg_hint_plan-style hint line
from each JSON plan into ``reqo_hints_{round}.txt``.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from imdb_workload_common import (
    GUCDict,
    SQLGroups,
    build_executable_sql,
    create_argument_parser,
    format_query_results,
    load_sql_groups,
    open_connection,
    print_sql_group_statistics,
    save_query_results,
    set_guc_dict,
    validate_common_args,
)


REQO_GUC_DEFAULTS: GUCDict = {
    "enable_nestloop": "on",
    "enable_hashjoin": "on",
    "enable_mergejoin": "on",
    "enable_indexscan": "on",
}

REQO_GUC_CANDIDATES: list[GUCDict] = [
    {},
    {"enable_nestloop": "off"},
    {"enable_nestloop": "off", "enable_indexscan": "off"},
    {"enable_hashjoin": "off"},
    {"enable_hashjoin": "off", "enable_indexscan": "off"},
    {"enable_mergejoin": "off"},
    {"enable_mergejoin": "off", "enable_indexscan": "off"},
    {"enable_nestloop": "off", "enable_mergejoin": "off"},
    {
        "enable_nestloop": "off",
        "enable_mergejoin": "off",
        "enable_indexscan": "off",
    },
    {"enable_nestloop": "off", "enable_hashjoin": "off"},
    {
        "enable_nestloop": "off",
        "enable_hashjoin": "off",
        "enable_indexscan": "off",
    },
    {"enable_mergejoin": "off", "enable_hashjoin": "off"},
    {
        "enable_mergejoin": "off",
        "enable_hashjoin": "off",
        "enable_indexscan": "off",
    },
]

JOIN_HINTS = {
    "Hash Join": "HashJoin",
    "Merge Join": "MergeJoin",
    "Nested Loop": "NestLoop",
}

SCAN_HINTS = {
    "Seq Scan": "SeqScan",
    "Index Scan": "IndexScan",
    "Index Only Scan": "IndexOnlyScan",
    "Bitmap Heap Scan": "BitmapScan",
    "Tid Scan": "TidScan",
    "TID Range Scan": "TidScan",
}


def parse_args() -> argparse.Namespace:
    """Parse options for the Reqo-GUC PostgreSQL runner."""
    parser = create_argument_parser(
        description=(
            "Run IMDb queries with the original Reqo 13 GUC candidates and "
            "export pg_hint_plan hints derived from EXPLAIN JSON."
        )
    )
    parser.set_defaults(run_mode="explain-analyze-json")
    for action in parser._actions:
        if action.dest == "run_mode":
            action.help = (
                "SQL execution mode. Default: explain-analyze-json. Choices: "
                "explain-json or explain-analyze-json."
            )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate common options plus the JSON requirement for hint export."""
    validate_common_args(args)
    if args.run_mode not in {"explain-json", "explain-analyze-json"}:
        raise ValueError(
            "Reqo hint export requires --run-mode explain-json or "
            "explain-analyze-json."
        )


def generate_base_guc_dict(args: argparse.Namespace) -> GUCDict:
    """Build standard PostgreSQL settings used by all 13 candidates."""
    return {
        "statement_timeout": args.statement_timeout,
    }


def build_query_output_dir(
        results_path: Path,
        template_id: int,
        query_id: int,
) -> Path:
    """Build the output directory for one template/query pair."""
    query_output_dir = (
        results_path.expanduser().resolve()
        / "reqo_guc"
        / str(template_id)
        / str(query_id)
    )
    query_output_dir.mkdir(parents=True, exist_ok=True)
    return query_output_dir


def build_effective_reqo_guc_dict(candidate_gucs: GUCDict) -> GUCDict:
    """Reset all Reqo-controlled optimizer GUCs, then apply one candidate."""
    return {
        **REQO_GUC_DEFAULTS,
        **candidate_gucs,
    }


def quote_hint_identifier(identifier: str) -> str:
    """Quote a table alias or index name when pg_hint_plan needs it."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        return identifier
    return '"' + identifier.replace('"', '""') + '"'


def plan_children(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return child plan nodes."""
    return plan.get("Plans", []) or []


def scan_alias(plan: dict[str, Any]) -> str | None:
    """Return the hintable alias for a scan node."""
    alias = plan.get("Alias") or plan.get("Relation Name")
    if isinstance(alias, str) and alias:
        return quote_hint_identifier(alias)
    return None


def collect_leaf_aliases(plan: dict[str, Any]) -> list[str]:
    """Collect scan aliases below a plan node, preserving plan-tree order."""
    alias = scan_alias(plan)
    if alias is not None:
        return [alias]

    aliases: list[str] = []
    for child in plan_children(plan):
        aliases.extend(collect_leaf_aliases(child))
    return aliases


def leading_expression(plan: dict[str, Any]) -> str:
    """Build the nested Leading() expression for a plan tree."""
    node_type = plan.get("Node Type")
    children = plan_children(plan)

    if node_type in JOIN_HINTS and len(children) >= 2:
        exprs = [leading_expression(child) for child in children]
        exprs = [expr for expr in exprs if expr]
        if len(exprs) == 1:
            return exprs[0]
        if len(exprs) >= 2:
            return f"({' '.join(exprs)})"

    alias = scan_alias(plan)
    if alias is not None:
        return alias

    exprs = [leading_expression(child) for child in children]
    exprs = [expr for expr in exprs if expr]
    if len(exprs) == 1:
        return exprs[0]
    if len(exprs) >= 2:
        return f"({' '.join(exprs)})"
    return ""


def collect_bitmap_index_names(plan: dict[str, Any]) -> list[str]:
    """Collect index names from Bitmap Index Scan descendants."""
    index_names: list[str] = []
    if plan.get("Node Type") == "Bitmap Index Scan":
        index_name = plan.get("Index Name")
        if isinstance(index_name, str) and index_name:
            index_names.append(quote_hint_identifier(index_name))
    for child in plan_children(plan):
        index_names.extend(collect_bitmap_index_names(child))
    return index_names


def append_join_and_scan_hints(plan: dict[str, Any], hints: list[str]) -> None:
    """Append Join and Scan hints from one plan node and its children."""
    node_type = plan.get("Node Type")

    join_hint = JOIN_HINTS.get(node_type)
    if join_hint is not None:
        aliases = collect_leaf_aliases(plan)
        if aliases:
            hints.append(f"{join_hint}({' '.join(aliases)})")

    scan_hint = SCAN_HINTS.get(node_type)
    alias = scan_alias(plan)
    if scan_hint is not None and alias is not None:
        args = [alias]
        if node_type in {"Index Scan", "Index Only Scan"}:
            index_name = plan.get("Index Name")
            if isinstance(index_name, str) and index_name:
                args.append(quote_hint_identifier(index_name))
        elif node_type == "Bitmap Heap Scan":
            args.extend(collect_bitmap_index_names(plan))
        hints.append(f"{scan_hint}({' '.join(args)})")

    for child in plan_children(plan):
        append_join_and_scan_hints(child, hints)


def plan_to_hint(plan: dict[str, Any]) -> str:
    """Convert one EXPLAIN JSON Plan object into a pg_hint_plan hint line."""
    hint_parts: list[str] = []
    leading = leading_expression(plan)
    if leading:
        hint_parts.append(f"Leading({leading})")
    append_join_and_scan_hints(plan, hint_parts)
    return f"/*+ {' '.join(hint_parts)} */"


def extract_plan_from_explain_rows(rows: list[tuple]) -> dict[str, Any]:
    """Extract the top Plan object from EXPLAIN (FORMAT JSON) rows."""
    if len(rows) != 1 or len(rows[0]) != 1:
        raise ValueError("Expected a single EXPLAIN JSON row and column.")

    explain_value = rows[0][0]
    if isinstance(explain_value, str):
        explain_value = json.loads(explain_value)
    if not isinstance(explain_value, list) or not explain_value:
        raise ValueError("Expected EXPLAIN JSON to return a non-empty list.")

    explain_doc = explain_value[0]
    if not isinstance(explain_doc, dict) or "Plan" not in explain_doc:
        raise ValueError("Expected EXPLAIN JSON document to contain Plan.")
    return explain_doc["Plan"]


def execute_explain_json(
        cursor: Any,
        sql_string: str,
        run_mode: str,
) -> tuple[str, dict[str, Any]]:
    """Execute one SQL in JSON EXPLAIN mode and return text plus Plan."""
    cursor.execute(build_executable_sql(sql_string, run_mode))
    rows = cursor.fetchall() if cursor.description is not None else []
    result_text = format_query_results(rows)
    return result_text, extract_plan_from_explain_rows(rows)


def run_single_candidate(
        cursor: Any,
        sql_string: str,
        base_guc_dict: GUCDict,
        candidate_index: int,
        candidate_gucs: GUCDict,
        results_filename: Path,
        hint_file,
        run_mode: str,
) -> None:
    """Execute one Reqo GUC candidate and append result plus hint."""
    effective_candidate_gucs = build_effective_reqo_guc_dict(candidate_gucs)
    set_guc_dict(cursor, base_guc_dict)
    set_guc_dict(cursor, effective_candidate_gucs)

    result_text, plan = execute_explain_json(cursor, sql_string, run_mode)
    hint_file.write(plan_to_hint(plan))
    hint_file.write("\n")
    hint_file.flush()

    save_query_results(
        results_filename=results_filename,
        header_lines=[
            "Parameter group: reqo_guc",
            f"Run mode: {run_mode}",
            f"Reqo candidate: {candidate_index}",
            f"Reqo GUCs: {effective_candidate_gucs}",
        ],
        result_text=result_text,
    )


def run_workload(
        args: argparse.Namespace,
        sql_groups: SQLGroups,
        base_guc_dict: GUCDict,
) -> None:
    """Execute all SQLs under the 13 Reqo GUC candidates."""
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
                    query_output_dir = build_query_output_dir(
                        results_path=args.results_path,
                        template_id=template_id,
                        query_id=query_id,
                    )
                    for round_number in range(1, args.rounds + 1):
                        results_filename = (
                            query_output_dir / f"results_{round_number}.txt"
                        )
                        hints_filename = (
                            query_output_dir / f"reqo_hints_{round_number}.txt"
                        )
                        print(
                            "Executing Reqo GUC candidates: "
                            f"template_id={template_id}, "
                            f"query_id={query_id}, "
                            f"round={round_number}/{args.rounds}"
                        )
                        with hints_filename.open("w", encoding="utf-8") as hint_file:
                            for candidate_index, candidate_gucs in enumerate(
                                REQO_GUC_CANDIDATES
                            ):
                                try:
                                    run_single_candidate(
                                        cursor=cursor,
                                        sql_string=sql_string,
                                        base_guc_dict=base_guc_dict,
                                        candidate_index=candidate_index,
                                        candidate_gucs=candidate_gucs,
                                        results_filename=results_filename,
                                        hint_file=hint_file,
                                        run_mode=args.run_mode,
                                    )
                                except Exception as exc:
                                    conn.rollback()
                                    save_query_results(
                                        results_filename=results_filename,
                                        header_lines=[
                                            "Parameter group: reqo_guc",
                                            f"Run mode: {args.run_mode}",
                                            f"Reqo candidate: {candidate_index}",
                                            "Status: failed",
                                            f"Reqo GUCs: {build_effective_reqo_guc_dict(candidate_gucs)}",
                                        ],
                                        result_text=repr(exc),
                                    )


def main() -> None:
    """Load the IMDb workload and execute Reqo GUC candidates."""
    args = parse_args()
    validate_args(args)

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

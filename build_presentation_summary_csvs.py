"""Build presentation-friendly CSVs from summarize_all_groups.py outputs.

The heavy summary step already writes normalized CSVs such as
combined_fold_summary.csv, combined_overall_summary.csv, method_win_counts.csv,
and specific_comparisons.csv. This script reshapes those files into compact
tables that are easier to paste into slides.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create presentation CSVs from a summarize_all_groups output dir."
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        required=True,
        help="Directory containing summarize_all_groups.py outputs.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Objective/parameter groups, e.g. 1x1__0x0 8x1__0x0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: --summary-dir.",
    )
    parser.add_argument(
        "--float-digits",
        type=int,
        default=6,
        help="Digits after decimal for floating point output.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required input CSV does not exist: {path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def format_value(value: Any, digits: int) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "" if value is None else str(value)
    return f"{parsed:.{digits}f}"


def group_prefix(group: str) -> str:
    return group.replace("/", "__")


def group_label(group: str) -> str:
    return group.replace("__", "/")


def scope_label(scope: str) -> str:
    if scope == "overall":
        return "Overall"
    if scope.startswith("fold_"):
        return f"Fold {scope.split('_', 1)[1]}"
    return str(scope)


def fold_scope(row: dict[str, str]) -> str:
    return f"fold_{row['fold_id']}"


def rows_by_scope(
        fold_rows: list[dict[str, str]],
        overall_rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    out = {fold_scope(row): row for row in fold_rows}
    for row in overall_rows:
        out[row["scope"]] = row
    return out


def ordered_scopes(fold_rows: list[dict[str, str]]) -> list[str]:
    folds = sorted(
        (fold_scope(row) for row in fold_rows),
        key=lambda value: int(value.split("_", 1)[1]),
    )
    return folds + ["overall"]


def runtime_methods(groups: list[str]) -> list[tuple[str, str]]:
    methods = [
        ("Original PG", "original_avg_ms"),
        ("Original Reqo Selected", "reqo_guc_reqo_avg_ms"),
    ]
    for group in groups:
        label = group_label(group)
        prefix = group_prefix(group)
        methods.extend([
            (f"{label} RobDP Direct", f"{prefix}_robdp_avg_ms"),
            (
                f"{label} Last-Level Reqo Selected",
                f"{prefix}_robdp_last_level_reqo_avg_ms",
            ),
        ])
    return methods


def build_overall_runtime(
        rows_by_name: dict[str, dict[str, str]],
        scopes: list[str],
        groups: list[str],
        digits: int,
) -> list[dict[str, Any]]:
    original_overall = as_float(rows_by_name["overall"].get("original_avg_ms"))
    output = []
    for label, column in runtime_methods(groups):
        row: dict[str, Any] = {"method": label}
        for scope in scopes:
            row[f"{scope}_avg_ms"] = format_value(
                rows_by_name[scope].get(column),
                digits,
            )
        overall_value = as_float(rows_by_name["overall"].get(column))
        vs_original = ratio(overall_value, original_overall)
        speedup = ratio(original_overall, overall_value)
        row["vs_original_pg_ratio"] = format_value(vs_original, digits)
        row["speedup_vs_original_pg"] = format_value(speedup, digits)
        output.append(row)
    return output


def build_oracle_min_cost_reqo(
        rows_by_name: dict[str, dict[str, str]],
        scopes: list[str],
        groups: list[str],
        digits: int,
) -> list[dict[str, Any]]:
    candidate_sets = [(
        "Original Reqo-GUC",
        "reqo_guc_oracle_avg_ms",
        "reqo_guc_min_cost_avg_ms",
        "reqo_guc_reqo_avg_ms",
    )]
    for group in groups:
        prefix = group_prefix(group)
        candidate_sets.append((
            f"{group_label(group)} RobDP Last-Level",
            f"{prefix}_robdp_last_level_oracle_avg_ms",
            f"{prefix}_robdp_last_level_min_cost_avg_ms",
            f"{prefix}_robdp_last_level_reqo_avg_ms",
        ))

    output = []
    for label, oracle_col, min_cost_col, reqo_col in candidate_sets:
        row: dict[str, Any] = {"candidate_set": label}
        for scope in scopes:
            scoped = rows_by_name[scope]
            row[f"{scope}_oracle_avg_ms"] = format_value(scoped.get(oracle_col), digits)
            row[f"{scope}_min_cost_avg_ms"] = format_value(
                scoped.get(min_cost_col),
                digits,
            )
            row[f"{scope}_reqo_selected_avg_ms"] = format_value(
                scoped.get(reqo_col),
                digits,
            )

        overall = rows_by_name["overall"]
        oracle = as_float(overall.get(oracle_col))
        min_cost = as_float(overall.get(min_cost_col))
        reqo = as_float(overall.get(reqo_col))
        row["overall_reqo_vs_min_cost_ratio"] = format_value(
            ratio(reqo, min_cost),
            digits,
        )
        row["overall_reqo_vs_oracle_ratio"] = format_value(ratio(reqo, oracle), digits)
        row["overall_min_cost_vs_oracle_ratio"] = format_value(
            ratio(min_cost, oracle),
            digits,
        )
        output.append(row)
    return output


def label_for_method(method: str, groups: list[str]) -> str:
    labels = {
        "original_pg": "Original PG",
        "original_reqo_selected": "Original Reqo Selected",
    }
    for group in groups:
        prefix = group_prefix(group)
        label = group_label(group)
        labels[f"{prefix}_robdp_direct"] = f"{label} RobDP Direct"
        labels[f"{prefix}_robdp_reqo_selected"] = (
            f"{label} Last-Level Reqo Selected"
        )
    return labels.get(method, method)


def pivot_metric_rows(
        rows: list[dict[str, str]],
        key_column: str,
        scopes: list[str],
) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        key = row[key_column]
        out.setdefault(key, {})[row["scope"]] = row
    for key, scoped_rows in out.items():
        missing = [scope for scope in scopes if scope not in scoped_rows]
        if missing:
            raise ValueError(f"{key_column}={key} is missing scopes: {missing}")
    return out


def build_winner_count(
        rows: list[dict[str, str]],
        scopes: list[str],
        groups: list[str],
        digits: int,
) -> list[dict[str, Any]]:
    by_method = pivot_metric_rows(rows, "method", scopes)
    output = []
    for method, scoped_rows in by_method.items():
        out: dict[str, Any] = {"method": label_for_method(method, groups)}
        for scope in scopes:
            row = scoped_rows[scope]
            out[f"{scope}_query_count"] = row.get("query_count", "")
            out[f"{scope}_win_count_including_ties"] = row.get("win_count", "")
            out[f"{scope}_win_rate_including_ties"] = format_value(
                row.get("win_rate"),
                digits,
            )
            out[f"{scope}_unique_win_count"] = row.get("unique_win_count", "")
            out[f"{scope}_unique_win_rate"] = format_value(
                row.get("unique_win_rate"),
                digits,
            )
            out[f"{scope}_avg_runtime_ms"] = format_value(
                row.get("avg_runtime_ms"),
                digits,
            )
        output.append(out)
    return output


def comparison_label(comparison: str, groups: list[str]) -> str:
    labels = {
        "original_reqo_selected_vs_original_pg": (
            "Original Reqo Selected < Original PG"
        ),
    }
    for group in groups:
        prefix = group_prefix(group)
        labels[f"{prefix}_robdp_reqo_selected_vs_robdp_direct"] = (
            f"{group_label(group)} Reqo Selected < {group_label(group)} RobDP Direct"
        )
    return labels.get(comparison, comparison)


def build_specific_comparisons(
        rows: list[dict[str, str]],
        scopes: list[str],
        groups: list[str],
        digits: int,
) -> list[dict[str, Any]]:
    by_comparison = pivot_metric_rows(rows, "comparison", scopes)
    output = []
    for comparison, scoped_rows in by_comparison.items():
        out: dict[str, Any] = {"comparison": comparison_label(comparison, groups)}
        for scope in scopes:
            row = scoped_rows[scope]
            out[f"{scope}_compared_query_count"] = row.get("compared_query_count", "")
            out[f"{scope}_left_better_count"] = row.get("left_better_count", "")
            out[f"{scope}_left_better_rate"] = format_value(
                row.get("left_better_rate"),
                digits,
            )
            out[f"{scope}_right_better_count"] = row.get("right_better_count", "")
            out[f"{scope}_tie_count"] = row.get("tie_count", "")
        output.append(out)
    return output


def has_planning_columns(rows_by_name: dict[str, dict[str, str]], groups: list[str]) -> bool:
    columns = set(rows_by_name["overall"])
    if "original_planning_avg_ms" not in columns:
        return False
    return any(
        f"{group_prefix(group)}_robdp_planning_avg_ms" in columns
        for group in groups
    )


def build_planning_time(
        rows_by_name: dict[str, dict[str, str]],
        scopes: list[str],
        groups: list[str],
        digits: int,
) -> list[dict[str, Any]]:
    methods = [("Original PG", "original_avg_ms", "original_planning_avg_ms")]
    for group in groups:
        prefix = group_prefix(group)
        methods.append((
            f"{group_label(group)} RobDP Direct",
            f"{prefix}_robdp_avg_ms",
            f"{prefix}_robdp_planning_avg_ms",
        ))

    original_total = None
    original_overall = rows_by_name["overall"]
    original_exec = as_float(original_overall.get("original_avg_ms"))
    original_plan = as_float(original_overall.get("original_planning_avg_ms"))
    if original_exec is not None and original_plan is not None:
        original_total = original_exec + original_plan

    output = []
    for label, exec_col, planning_col in methods:
        out: dict[str, Any] = {"method": label}
        for scope in scopes:
            row = rows_by_name[scope]
            execution = as_float(row.get(exec_col))
            planning = as_float(row.get(planning_col))
            total = (
                execution + planning
                if execution is not None and planning is not None
                else None
            )
            out[f"{scope}_execution_avg_ms"] = format_value(execution, digits)
            out[f"{scope}_planning_avg_ms"] = format_value(planning, digits)
            out[f"{scope}_execution_plus_planning_avg_ms"] = format_value(
                total,
                digits,
            )
        method_total = as_float(out.get("overall_execution_plus_planning_avg_ms"))
        out["execution_plus_planning_vs_original_pg_ratio"] = format_value(
            ratio(method_total, original_total),
            digits,
        )
        output.append(out)
    return output


def weighted_average(
        fold_rows: list[dict[str, str]],
        column: str,
) -> float | None:
    total_weight = 0
    total_value = 0.0
    for row in fold_rows:
        value = as_float(row.get(column))
        count = as_float(row.get("query_count"))
        if value is None or count is None:
            return None
        total_weight += int(count)
        total_value += value * int(count)
    if total_weight == 0:
        return None
    return total_value / total_weight


def build_data_checks(
        fold_rows: list[dict[str, str]],
        overall_rows: list[dict[str, str]],
        method_win_rows: list[dict[str, str]],
        query_method_rows: list[dict[str, str]],
        groups: list[str],
        digits: int,
) -> list[dict[str, str]]:
    overall = overall_rows[0] if overall_rows else {}
    checks: list[dict[str, str]] = []

    metric_columns = [
        "original_avg_ms",
        "reqo_guc_reqo_avg_ms",
    ]
    for group in groups:
        prefix = group_prefix(group)
        metric_columns.extend([
            f"{prefix}_robdp_avg_ms",
            f"{prefix}_robdp_last_level_reqo_avg_ms",
        ])
    for column in metric_columns:
        weighted = weighted_average(fold_rows, column)
        overall_value = as_float(overall.get(column))
        ok = (
            weighted is not None
            and overall_value is not None
            and abs(weighted - overall_value) <= 1e-6
        )
        checks.append({
            "check": f"weighted_overall_{column}",
            "status": "ok" if ok else "warning",
            "detail": (
                f"weighted={format_value(weighted, digits)}, "
                f"overall={format_value(overall_value, digits)}"
            ),
        })

    query_count_by_scope: dict[str, int] = {}
    for row in method_win_rows:
        if row.get("method") == "original_pg":
            query_count_by_scope[row["scope"]] = int(float(row["query_count"]))
    tied_queries_by_scope: dict[str, int] = {}
    for row in query_method_rows:
        methods = [m for m in row.get("tied_best_methods", "").split(";") if m]
        extra = max(0, len(methods) - 1)
        scope = f"fold_{row['fold_id']}"
        tied_queries_by_scope[scope] = tied_queries_by_scope.get(scope, 0) + (
            1 if extra else 0
        )
        tied_queries_by_scope["overall"] = tied_queries_by_scope.get("overall", 0) + (
            1 if extra else 0
        )

    for scope, query_count in sorted(query_count_by_scope.items()):
        scoped = [row for row in method_win_rows if row["scope"] == scope]
        unique_wins = sum(int(float(row["unique_win_count"])) for row in scoped)
        inclusive_wins = sum(int(float(row["win_count"])) for row in scoped)
        tied_queries = tied_queries_by_scope.get(scope, 0)
        checks.append({
            "check": f"unique_winner_counts_sum_{scope}",
            "status": "ok" if unique_wins + tied_queries == query_count else "warning",
            "detail": (
                f"unique_wins={unique_wins}, query_count={query_count}, "
                f"tie_queries={tied_queries}"
            ),
        })
        checks.append({
            "check": f"inclusive_winner_counts_sum_{scope}",
            "status": "ok" if inclusive_wins >= query_count else "warning",
            "detail": (
                f"inclusive_wins={inclusive_wins}, query_count={query_count}, "
                f"extra_tie_memberships={inclusive_wins - query_count}"
            ),
        })
    return checks


def fieldnames_for_scoped_table(
        first_column: str,
        scopes: list[str],
        suffixes: list[str],
        trailing: list[str] | None = None,
) -> list[str]:
    fields = [first_column]
    for scope in scopes:
        fields.extend([f"{scope}_{suffix}" for suffix in suffixes])
    if trailing:
        fields.extend(trailing)
    return fields


def main() -> None:
    args = parse_args()
    summary_dir = args.summary_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else summary_dir
    )

    fold_rows = read_csv(summary_dir / "combined_fold_summary.csv")
    overall_rows = read_csv(summary_dir / "combined_overall_summary.csv")
    method_win_rows = read_csv(summary_dir / "method_win_counts.csv")
    specific_rows = read_csv(summary_dir / "specific_comparisons.csv")
    query_method_rows = read_csv(summary_dir / "query_method_runtimes.csv")

    scopes = ordered_scopes(fold_rows)
    by_scope = rows_by_scope(fold_rows, overall_rows)

    overall_runtime_rows = build_overall_runtime(
        by_scope,
        scopes,
        args.groups,
        args.float_digits,
    )
    write_csv(
        output_dir / "presentation_overall_runtime.csv",
        fieldnames_for_scoped_table(
            "method",
            scopes,
            ["avg_ms"],
            ["vs_original_pg_ratio", "speedup_vs_original_pg"],
        ),
        overall_runtime_rows,
    )

    oracle_rows = build_oracle_min_cost_reqo(
        by_scope,
        scopes,
        args.groups,
        args.float_digits,
    )
    write_csv(
        output_dir / "presentation_oracle_min_cost_reqo.csv",
        fieldnames_for_scoped_table(
            "candidate_set",
            scopes,
            ["oracle_avg_ms", "min_cost_avg_ms", "reqo_selected_avg_ms"],
            [
                "overall_reqo_vs_min_cost_ratio",
                "overall_reqo_vs_oracle_ratio",
                "overall_min_cost_vs_oracle_ratio",
            ],
        ),
        oracle_rows,
    )

    winner_rows = build_winner_count(
        method_win_rows,
        scopes,
        args.groups,
        args.float_digits,
    )
    write_csv(
        output_dir / "presentation_per_query_winner_count.csv",
        fieldnames_for_scoped_table(
            "method",
            scopes,
            [
                "query_count",
                "win_count_including_ties",
                "win_rate_including_ties",
                "unique_win_count",
                "unique_win_rate",
                "avg_runtime_ms",
            ],
        ),
        winner_rows,
    )

    comparison_rows = build_specific_comparisons(
        specific_rows,
        scopes,
        args.groups,
        args.float_digits,
    )
    write_csv(
        output_dir / "presentation_specific_comparisons.csv",
        fieldnames_for_scoped_table(
            "comparison",
            scopes,
            [
                "compared_query_count",
                "left_better_count",
                "left_better_rate",
                "right_better_count",
                "tie_count",
            ],
        ),
        comparison_rows,
    )

    if has_planning_columns(by_scope, args.groups):
        planning_rows = build_planning_time(
            by_scope,
            scopes,
            args.groups,
            args.float_digits,
        )
        write_csv(
            output_dir / "presentation_planning_time.csv",
            fieldnames_for_scoped_table(
                "method",
                scopes,
                [
                    "execution_avg_ms",
                    "planning_avg_ms",
                    "execution_plus_planning_avg_ms",
                ],
                ["execution_plus_planning_vs_original_pg_ratio"],
            ),
            planning_rows,
        )

    check_rows = build_data_checks(
        fold_rows,
        overall_rows,
        method_win_rows,
        query_method_rows,
        args.groups,
        args.float_digits,
    )
    write_csv(
        output_dir / "presentation_data_checks.csv",
        ["check", "status", "detail"],
        check_rows,
    )

    print(f"Wrote presentation CSVs to: {output_dir}")


if __name__ == "__main__":
    main()

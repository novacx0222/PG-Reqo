"""Run all per-group summaries and build cross-group win analyses.

This is the Step 5 driver for the pre-split IMDb workflow. It keeps the
single-group summarize_fold_runtimes.py behavior, then combines all requested
parameter groups and compares non-oracle methods query by query.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Any

import combine_group_summaries as combine
import summarize_fold_runtimes as fold_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all RobDP parameter groups in one command."
    )
    parser.add_argument("--folds-dir", type=Path, required=True)
    parser.add_argument("--fold-source", default="original")
    parser.add_argument("--results-path", type=Path, required=True)
    parser.add_argument("--original-results-dir", type=Path, default=None)
    parser.add_argument(
        "--robdp-runtime-results-root",
        type=Path,
        default=None,
        help=(
            "Root containing RobDP direct runtime results as <root>/<main>/<retain>. "
            "Default: <results-path>/robdp."
        ),
    )
    parser.add_argument("--reqo-guc-results-dir", type=Path, default=None)
    parser.add_argument(
        "--train-results-root",
        type=Path,
        required=True,
        help=(
            "Root containing trained outputs, for example "
            "Results/imdbloadbase/presplit-0710."
        ),
    )
    parser.add_argument("--robdp-trained-prefix", default="robdp_last_level_")
    parser.add_argument("--reqo-guc-trained-results-dir", type=Path, default=None)
    parser.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Parameter groups such as 1x1__0x0 8x1__0x0.",
    )
    parser.add_argument("--runtime-agg", choices=("mean", "min", "median"), default="mean")
    parser.add_argument(
        "--reqo-guc-runner-agg",
        choices=("min", "mean", "median"),
        default="min",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tie-tolerance-ms",
        type=float,
        default=1e-9,
        help="Runtime tolerance for considering two methods tied. Default: 1e-9.",
    )
    return parser.parse_args()


def group_to_param_dir(group: str) -> Path:
    return Path(group.replace("__", "/", 1))


def group_prefix(group: str) -> str:
    return combine.group_prefix(group)


def as_float(value: Any) -> float | None:
    return fold_summary.as_float_or_none(value)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_one_group(
        args: argparse.Namespace,
        group: str,
        output_dir: Path,
) -> list[dict[str, Any]]:
    results_path = args.results_path.expanduser().resolve()
    folds_dir = args.folds_dir.expanduser().resolve()
    train_results_root = args.train_results_root.expanduser().resolve()
    original_dir = (
        args.original_results_dir.expanduser().resolve()
        if args.original_results_dir is not None
        else results_path / "original"
    )
    robdp_runtime_root = (
        args.robdp_runtime_results_root.expanduser().resolve()
        if args.robdp_runtime_results_root is not None
        else results_path / "robdp"
    )
    reqo_guc_dir = (
        args.reqo_guc_results_dir.expanduser().resolve()
        if args.reqo_guc_results_dir is not None
        else results_path / "reqo_guc"
    )
    robdp_dir = robdp_runtime_root / group_to_param_dir(group)
    robdp_trained_dir = train_results_root / f"{args.robdp_trained_prefix}{group}"
    reqo_guc_trained_dir = (
        args.reqo_guc_trained_results_dir.expanduser().resolve()
        if args.reqo_guc_trained_results_dir is not None
        else train_results_root / "reqo_guc"
    )

    fold_summary.require_dir(folds_dir, "folds")
    fold_summary.require_dir(original_dir, "original results")
    fold_summary.require_dir(robdp_dir, f"RobDP runtime results for {group}")
    fold_summary.require_dir(reqo_guc_dir, "Reqo-GUC runner results")
    fold_summary.require_dir(robdp_trained_dir, f"RobDP trained results for {group}")
    fold_summary.require_dir(reqo_guc_trained_dir, "Reqo-GUC trained results")

    fold_rows = fold_summary.load_fold_rows(folds_dir, args.fold_source)
    robdp_selection = fold_summary.load_selection_rows(robdp_trained_dir)
    reqo_guc_selection = fold_summary.load_selection_rows(reqo_guc_trained_dir)

    missing_robdp = sorted(set(fold_rows) - set(robdp_selection))
    missing_reqo_guc = sorted(set(fold_rows) - set(reqo_guc_selection))
    if missing_robdp:
        print(f"Warning: {len(missing_robdp)} test queries lack {group} selection rows")
    if missing_reqo_guc:
        print(f"Warning: {len(missing_reqo_guc)} test queries lack Reqo-GUC selection rows")

    fold_query_rows = fold_summary.build_fold_query_summary(
        fold_rows=fold_rows,
        original_dir=original_dir,
        robdp_dir=robdp_dir,
        reqo_guc_dir=reqo_guc_dir,
        robdp_selection=robdp_selection,
        reqo_guc_selection=reqo_guc_selection,
        runtime_agg=args.runtime_agg,
        reqo_guc_runner_agg=args.reqo_guc_runner_agg,
    )
    fold_rows_summary = fold_summary.build_group_summary(fold_query_rows, "fold_id")
    overall_rows = fold_summary.build_overall_summary(fold_query_rows)

    group_output = output_dir / group
    fold_summary.write_csv(
        group_output / "fold_query_summary.csv",
        fold_summary.FOLD_QUERY_FIELDS,
        fold_query_rows,
    )
    fold_summary.write_csv(
        group_output / "fold_summary.csv",
        ["fold_id"] + fold_summary.SUMMARY_FIELDS,
        fold_rows_summary,
    )
    fold_summary.write_csv(
        group_output / "overall_summary.csv",
        ["scope"] + fold_summary.SUMMARY_FIELDS,
        overall_rows,
    )

    print(f"Wrote {group} summaries to: {group_output}")
    return fold_query_rows


def query_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["fold_id"]),
        int(row["template_id"]),
        int(row["original_query_id"]),
    )


def method_names(groups: list[str]) -> list[str]:
    methods = [
        "original_pg",
        "original_reqo_selected",
    ]
    for group in groups:
        prefix = group_prefix(group)
        methods.extend([
            f"{prefix}_robdp_direct",
            f"{prefix}_robdp_reqo_selected",
        ])
    return methods


def build_query_method_rows(
        group_rows_by_group: dict[str, list[dict[str, Any]]],
        groups: list[str],
        tie_tolerance_ms: float,
) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}

    for group_idx, group in enumerate(groups):
        prefix = group_prefix(group)
        for row in group_rows_by_group[group]:
            key = query_key(row)
            out = rows_by_key.setdefault(key, {
                "fold_id": key[0],
                "template_id": key[1],
                "original_query_id": key[2],
            })
            if group_idx == 0:
                out["original_pg_runtime_ms"] = row.get("original_runtime_ms", "")
                out["original_reqo_selected_runtime_ms"] = row.get(
                    "reqo_guc_reqo_runtime_ms",
                    "",
                )
            out[f"{prefix}_robdp_direct_runtime_ms"] = row.get("robdp_runtime_ms", "")
            out[f"{prefix}_robdp_reqo_selected_runtime_ms"] = row.get(
                "robdp_last_level_reqo_runtime_ms",
                "",
            )

    method_order = method_names(groups)
    for row in rows_by_key.values():
        runtimes = {
            method: as_float(row.get(f"{method}_runtime_ms"))
            for method in method_order
        }
        present = {
            method: runtime
            for method, runtime in runtimes.items()
            if runtime is not None
        }
        if not present:
            row["best_method"] = ""
            row["best_runtime_ms"] = ""
            row["tied_best_methods"] = ""
            continue
        best_runtime = min(present.values())
        tied = [
            method
            for method, runtime in present.items()
            if abs(runtime - best_runtime) <= tie_tolerance_ms
        ]
        row["best_method"] = tied[0] if len(tied) == 1 else "tie"
        row["best_runtime_ms"] = best_runtime
        row["tied_best_methods"] = ";".join(tied)

    return [
        rows_by_key[key]
        for key in sorted(rows_by_key)
    ]


def rows_for_scope(
        rows: list[dict[str, Any]],
        scope: str,
) -> list[dict[str, Any]]:
    if scope == "overall":
        return rows
    fold_id = int(scope.split("_", 1)[1])
    return [row for row in rows if int(row["fold_id"]) == fold_id]


def scope_names(rows: list[dict[str, Any]]) -> list[str]:
    folds = sorted({int(row["fold_id"]) for row in rows})
    return [f"fold_{fold_id}" for fold_id in folds] + ["overall"]


def build_method_win_counts(
        rows: list[dict[str, Any]],
        methods: list[str],
) -> list[dict[str, Any]]:
    output_rows = []
    for scope in scope_names(rows):
        scoped_rows = rows_for_scope(rows, scope)
        query_count = len(scoped_rows)
        for method in methods:
            present_values = [
                as_float(row.get(f"{method}_runtime_ms"))
                for row in scoped_rows
            ]
            present_values = [value for value in present_values if value is not None]
            win_count = sum(
                1
                for row in scoped_rows
                if method in str(row.get("tied_best_methods", "")).split(";")
            )
            unique_win_count = sum(
                1
                for row in scoped_rows
                if row.get("best_method") == method
            )
            available = len(present_values)
            output_rows.append({
                "scope": scope,
                "method": method,
                "query_count": query_count,
                "available_query_count": available,
                "win_count": win_count,
                "win_rate": win_count / query_count if query_count else "",
                "unique_win_count": unique_win_count,
                "unique_win_rate": (
                    unique_win_count / query_count
                    if query_count
                    else ""
                ),
                "avg_runtime_ms": mean(present_values) if present_values else "",
            })
    return output_rows


def compare_pair(
        rows: list[dict[str, Any]],
        left_method: str,
        right_method: str,
        tie_tolerance_ms: float,
) -> dict[str, int]:
    counts = {
        "compared_query_count": 0,
        "left_better_count": 0,
        "right_better_count": 0,
        "tie_count": 0,
    }
    for row in rows:
        left = as_float(row.get(f"{left_method}_runtime_ms"))
        right = as_float(row.get(f"{right_method}_runtime_ms"))
        if left is None or right is None:
            continue
        counts["compared_query_count"] += 1
        delta = left - right
        if abs(delta) <= tie_tolerance_ms:
            counts["tie_count"] += 1
        elif delta < 0:
            counts["left_better_count"] += 1
        else:
            counts["right_better_count"] += 1
    return counts


def build_pairwise_rows(
        rows: list[dict[str, Any]],
        methods: list[str],
        tie_tolerance_ms: float,
) -> list[dict[str, Any]]:
    output_rows = []
    for scope in scope_names(rows):
        scoped_rows = rows_for_scope(rows, scope)
        for left in methods:
            for right in methods:
                if left == right:
                    compared = sum(
                        1
                        for row in scoped_rows
                        if as_float(row.get(f"{left}_runtime_ms")) is not None
                    )
                    output_rows.append({
                        "scope": scope,
                        "method": left,
                        "beats_method": right,
                        "compared_query_count": compared,
                        "win_count": "",
                        "win_rate": "",
                        "tie_count": compared,
                        "loss_count": "",
                    })
                    continue
                counts = compare_pair(scoped_rows, left, right, tie_tolerance_ms)
                compared = counts["compared_query_count"]
                output_rows.append({
                    "scope": scope,
                    "method": left,
                    "beats_method": right,
                    "compared_query_count": compared,
                    "win_count": counts["left_better_count"],
                    "win_rate": (
                        counts["left_better_count"] / compared
                        if compared
                        else ""
                    ),
                    "tie_count": counts["tie_count"],
                    "loss_count": counts["right_better_count"],
                })
    return output_rows


def build_pairwise_matrices(
        pairwise_rows: list[dict[str, Any]],
        methods: list[str],
        value_column: str,
) -> list[dict[str, Any]]:
    by_scope_method = {
        (row["scope"], row["method"], row["beats_method"]): row
        for row in pairwise_rows
    }
    matrix_rows = []
    scopes = list(dict.fromkeys(row["scope"] for row in pairwise_rows))
    for scope in scopes:
        for method in methods:
            out = {"scope": scope, "method": method}
            for beats_method in methods:
                out[beats_method] = by_scope_method[
                    (scope, method, beats_method)
                ].get(value_column, "")
            matrix_rows.append(out)
    return matrix_rows


def build_specific_comparisons(
        rows: list[dict[str, Any]],
        groups: list[str],
        tie_tolerance_ms: float,
) -> list[dict[str, Any]]:
    comparisons = [
        (
            "original_reqo_selected_vs_original_pg",
            "original_reqo_selected",
            "original_pg",
        ),
    ]
    for group in groups:
        prefix = group_prefix(group)
        comparisons.append((
            f"{prefix}_robdp_reqo_selected_vs_robdp_direct",
            f"{prefix}_robdp_reqo_selected",
            f"{prefix}_robdp_direct",
        ))

    output_rows = []
    for scope in scope_names(rows):
        scoped_rows = rows_for_scope(rows, scope)
        for comparison_name, left, right in comparisons:
            counts = compare_pair(scoped_rows, left, right, tie_tolerance_ms)
            compared = counts["compared_query_count"]
            output_rows.append({
                "scope": scope,
                "comparison": comparison_name,
                "left_method": left,
                "right_method": right,
                "compared_query_count": compared,
                "left_better_count": counts["left_better_count"],
                "left_better_rate": (
                    counts["left_better_count"] / compared
                    if compared
                    else ""
                ),
                "right_better_count": counts["right_better_count"],
                "right_better_rate": (
                    counts["right_better_count"] / compared
                    if compared
                    else ""
                ),
                "tie_count": counts["tie_count"],
                "tie_rate": counts["tie_count"] / compared if compared else "",
            })
    return output_rows


def write_combined_summaries(
        output_dir: Path,
        groups: list[str],
) -> None:
    fold_rows = combine.combine_summary_kind(
        summary_root=output_dir,
        groups=groups,
        filename="fold_summary.csv",
        key_column="fold_id",
    )
    overall_rows = combine.combine_summary_kind(
        summary_root=output_dir,
        groups=groups,
        filename="overall_summary.csv",
        key_column="scope",
    )
    combine.write_csv(
        output_dir / "combined_fold_summary.csv",
        combine.combined_fieldnames("fold_id", groups),
        fold_rows,
    )
    combine.write_csv(
        output_dir / "combined_overall_summary.csv",
        combine.combined_fieldnames("scope", groups),
        overall_rows,
    )


def write_win_analysis(
        output_dir: Path,
        group_rows_by_group: dict[str, list[dict[str, Any]]],
        groups: list[str],
        tie_tolerance_ms: float,
) -> None:
    methods = method_names(groups)
    query_rows = build_query_method_rows(
        group_rows_by_group=group_rows_by_group,
        groups=groups,
        tie_tolerance_ms=tie_tolerance_ms,
    )
    pairwise_rows = build_pairwise_rows(query_rows, methods, tie_tolerance_ms)
    win_count_rows = build_method_win_counts(query_rows, methods)
    specific_rows = build_specific_comparisons(query_rows, groups, tie_tolerance_ms)

    query_fields = [
        "fold_id",
        "template_id",
        "original_query_id",
        "best_method",
        "best_runtime_ms",
        "tied_best_methods",
    ]
    for method in methods:
        query_fields.append(f"{method}_runtime_ms")

    write_csv(output_dir / "query_method_runtimes.csv", query_fields, query_rows)
    write_csv(
        output_dir / "method_win_counts.csv",
        [
            "scope",
            "method",
            "query_count",
            "available_query_count",
            "win_count",
            "win_rate",
            "unique_win_count",
            "unique_win_rate",
            "avg_runtime_ms",
        ],
        win_count_rows,
    )
    write_csv(
        output_dir / "pairwise_win_counts.csv",
        [
            "scope",
            "method",
            "beats_method",
            "compared_query_count",
            "win_count",
            "win_rate",
            "tie_count",
            "loss_count",
        ],
        pairwise_rows,
    )
    write_csv(
        output_dir / "pairwise_win_rate_matrix.csv",
        ["scope", "method"] + methods,
        build_pairwise_matrices(pairwise_rows, methods, "win_rate"),
    )
    write_csv(
        output_dir / "pairwise_win_count_matrix.csv",
        ["scope", "method"] + methods,
        build_pairwise_matrices(pairwise_rows, methods, "win_count"),
    )
    write_csv(
        output_dir / "specific_comparisons.csv",
        [
            "scope",
            "comparison",
            "left_method",
            "right_method",
            "compared_query_count",
            "left_better_count",
            "left_better_rate",
            "right_better_count",
            "right_better_rate",
            "tie_count",
            "tie_rate",
        ],
        specific_rows,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    group_rows_by_group = {}
    for group in args.groups:
        print(f"===== Summarizing {group} =====")
        group_rows_by_group[group] = summarize_one_group(args, group, output_dir)

    write_combined_summaries(output_dir, args.groups)
    write_win_analysis(
        output_dir=output_dir,
        group_rows_by_group=group_rows_by_group,
        groups=args.groups,
        tie_tolerance_ms=args.tie_tolerance_ms,
    )

    print(f"Wrote combined summaries and win analysis to: {output_dir}")


if __name__ == "__main__":
    main()

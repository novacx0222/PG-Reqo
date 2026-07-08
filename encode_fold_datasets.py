"""Encode one fold's train/test SQL CSVs with shared train normalization.

For each source/fold, this script runs:

  1. train.csv -> train/encode.pt
  2. train/encode.pt -> train Reqo .npy dataset
  3. test.csv -> test/encode.pt, using train/norm_stats.json
  4. test/encode.pt -> test Reqo .npy dataset

The test encoder always uses train/norm_stats.json so train/test features live
in the same normalized space.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode one pre-split IMDb fold for Reqo training."
    )
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument(
        "--fold-sql-root",
        type=Path,
        required=True,
        help="Root containing <source>/fold_<id>/{train,test}.csv.",
    )
    parser.add_argument(
        "--encoding-root",
        type=Path,
        required=True,
        help="Root for encode.pt outputs, e.g. /data/robdp/encoding.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Root for Reqo .npy datasets, e.g. "
            "/data/robdp/Reqo-PG/Data/imdbloadbase/datasets."
        ),
    )
    parser.add_argument("--dbname", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="5432")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--stats-dir", type=Path, required=True)
    parser.add_argument("--statement-timeout-ms", type=int, default=0)
    parser.add_argument("--min-candidates-per-query", type=int, default=2)
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Reqo repository root. Default: directory containing this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Required file does not exist: {path}")


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(printable)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def build_common_encode_cmd(args: argparse.Namespace, sql_file: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.repo_root / "Utils" / "reqo_encode_sql_save_pt.py"),
        "--sql-file",
        str(sql_file),
        "--dbname",
        args.dbname,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--user",
        args.user,
        "--stats-dir",
        str(args.stats_dir),
        "--output-dir",
        str(output_dir),
        "--analyze",
        "--min-candidates-per-query",
        str(args.min_candidates_per_query),
    ]
    if args.password is not None:
        cmd.extend(["--password", args.password])
    if args.statement_timeout_ms > 0:
        cmd.extend(["--statement-timeout-ms", str(args.statement_timeout_ms)])
    if args.skip_errors:
        cmd.append("--skip-errors")
    return cmd


def build_convert_cmd(args: argparse.Namespace, pt_file: Path, dataset_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(args.repo_root / "Utils" / "reqo_pt_to_npy.py"),
        "--pt-file",
        str(pt_file),
        "--dbname",
        args.dbname,
        "--output-dir",
        str(dataset_dir),
        "--min-candidates-per-query",
        str(args.min_candidates_per_query),
    ]


def main() -> None:
    args = parse_args()
    if args.fold_id <= 0:
        raise ValueError("--fold-id must be positive.")
    if args.min_candidates_per_query <= 0:
        raise ValueError("--min-candidates-per-query must be positive.")

    repo_root = args.repo_root.expanduser().resolve()
    args.repo_root = repo_root
    fold_dir = (
        args.fold_sql_root.expanduser().resolve()
        / args.source_name
        / f"fold_{args.fold_id}"
    )
    train_csv = fold_dir / "train.csv"
    test_csv = fold_dir / "test.csv"
    require_file(train_csv)
    require_file(test_csv)

    encoding_base = (
        args.encoding_root.expanduser().resolve()
        / args.source_name
        / f"fold_{args.fold_id}"
    )
    dataset_base = (
        args.dataset_root.expanduser().resolve()
        / args.source_name
        / f"fold_{args.fold_id}"
    )
    train_encoding_dir = encoding_base / "train"
    test_encoding_dir = encoding_base / "test"
    train_dataset_dir = dataset_base / "train"
    test_dataset_dir = dataset_base / "test"
    norm_stats_path = train_encoding_dir / "norm_stats.json"

    train_cmd = build_common_encode_cmd(
        args=args,
        sql_file=train_csv,
        output_dir=train_encoding_dir,
    )
    train_cmd.extend(["--norm-stats-output", str(norm_stats_path)])
    train_convert_cmd = build_convert_cmd(
        args=args,
        pt_file=train_encoding_dir / "encode.pt",
        dataset_dir=train_dataset_dir,
    )

    test_cmd = build_common_encode_cmd(
        args=args,
        sql_file=test_csv,
        output_dir=test_encoding_dir,
    )
    test_cmd.extend(["--norm-stats", str(norm_stats_path)])
    test_convert_cmd = build_convert_cmd(
        args=args,
        pt_file=test_encoding_dir / "encode.pt",
        dataset_dir=test_dataset_dir,
    )

    run_command(train_cmd, cwd=repo_root, dry_run=args.dry_run)
    run_command(train_convert_cmd, cwd=repo_root, dry_run=args.dry_run)
    run_command(test_cmd, cwd=repo_root, dry_run=args.dry_run)
    run_command(test_convert_cmd, cwd=repo_root, dry_run=args.dry_run)

    print(f"Train encoding: {train_encoding_dir}")
    print(f"Test encoding: {test_encoding_dir}")
    print(f"Train dataset: {train_dataset_dir}")
    print(f"Test dataset: {test_dataset_dir}")


if __name__ == "__main__":
    main()

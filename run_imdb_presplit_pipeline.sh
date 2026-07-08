#!/usr/bin/env bash
set -euo pipefail

# IMDb pre-split Reqo/RobDP pipeline.
#
# Defaults are hard-coded for the 2026-07-08 IMDb experiment. Edit the
# configuration block below if you want a new experiment directory or a
# different workload.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON="${PYTHON:-python3}"
REPO_ROOT="/data/robdp/Reqo-PG"

DBNAME="imdbloadbase"
DB_USER="hx68"
DB_HOST="localhost"
DB_PORT="5432"

SQLS_DIR="/data/robdp/imdb-error-profile-0612"
WORKLOAD_NAME="cardinality"
STATS_DIR="${REPO_ROOT}/Data/${DBNAME}/database_statistics"
DATABASE_STATISTICS_DIR="${STATS_DIR}"

EXPERIMENT_ROOT="/data/robdp/imdb-presplit-0708"
RUNNER_RESULTS_PATH="${EXPERIMENT_ROOT}/runner_outputs"
HINT_SQL_CSV_DIR="${EXPERIMENT_ROOT}/hint-sql-csv"
FOLDS_DIR="${EXPERIMENT_ROOT}/folds"
FOLD_SQL_ROOT="${EXPERIMENT_ROOT}/fold_sql"
ENCODING_ROOT="${EXPERIMENT_ROOT}/encoding"
DATASET_ROOT="${REPO_ROOT}/Data/${DBNAME}/datasets-presplit-0708"
TRAIN_RESULTS_ROOT="${REPO_ROOT}/Results/${DBNAME}/presplit-0708"
SUMMARY_ROOT="${EXPERIMENT_ROOT}/summary"

SKIP_TEMPLATE_IDS=(29)
QUERY_ID_LIMIT="100"
STATEMENT_TIMEOUT="60s"
ROUNDS="1"
FOLD_COUNT=2
SPLIT_SEED=0
MIN_CANDIDATES_PER_QUERY=2
SAVE_MODEL=1

MAIN_OBJECTIVE_IDS=(1 3)
RETAIN_STRATEGY_IDS=(1 3)
FINAL_LEVEL_PATH_LIMIT="13"

GROUPS=(
  "1x1__0x0"
  "1x1__8x3"
  "1x3__0x0"
  "1x3__8x1"
  "8x1__0x0"
  "8x3__0x0"
)

REQO_GUC_SOURCE="reqo_guc"

DRY_RUN=0
RERUN_EXISTING=0
STAGES="runner,csv,split,encode,train,summary"
STEP_INDEX=0
TOTAL_STEPS=0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage:
  ./run_imdb_presplit_pipeline.sh [--dry-run] [--rerun-existing] [--stages LIST]

Options:
  --dry-run         Print checks and commands without executing commands.
  --rerun-existing  Run even when outputs already exist. This never deletes
                    files; runner result files may append.
  --stages LIST     Comma-separated stages to run.
                    Choices: runner,csv,split,encode,train,summary
                    Default: runner,csv,split,encode,train,summary
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --rerun-existing)
      RERUN_EXISTING=1
      shift
      ;;
    --stages)
      STAGES="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

stage_enabled() {
  local stage="$1"
  [[ ",${STAGES}," == *",${stage},"* ]]
}

parameter_dir() {
  local group="$1"
  echo "${group//__//}"
}

robdp_source_name() {
  local group="$1"
  echo "robdp_last_level_${group}"
}

has_content() {
  local path="$1"
  if [[ -f "$path" ]]; then
    [[ -s "$path" ]]
  elif [[ -d "$path" ]]; then
    [[ -n "$(find "$path" -mindepth 1 -print -quit 2>/dev/null)" ]]
  else
    return 1
  fi
}

quote_command() {
  printf '  '
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

progress_line() {
  local index="$1"
  local total="$2"
  local stage="$3"
  local name="$4"
  local width=30
  local filled=$(( index * width / total ))
  local empty=$(( width - filled ))
  local bar_done bar_empty

  printf -v bar_done '%*s' "$filled" ''
  printf -v bar_empty '%*s' "$empty" ''
  bar_done="${bar_done// /#}"
  bar_empty="${bar_empty// /-}"
  printf '\n[%s%s] %d/%d %s: %s\n' \
    "$bar_done" "$bar_empty" "$index" "$total" "$stage" "$name"
}

print_configuration() {
  cat <<EOF
IMDb pre-split Reqo/RobDP pipeline
Repository:       ${REPO_ROOT}
Database:         ${DBNAME}
SQLs:             ${SQLS_DIR}
Experiment root:  ${EXPERIMENT_ROOT}
Runner results:   ${RUNNER_RESULTS_PATH}
Hint SQL CSVs:    ${HINT_SQL_CSV_DIR}
Folds:            ${FOLDS_DIR}
Fold SQL:         ${FOLD_SQL_ROOT}
Encoding:         ${ENCODING_ROOT}
Datasets:         ${DATASET_ROOT}
Train results:    ${TRAIN_RESULTS_ROOT}
Summary:          ${SUMMARY_ROOT}
Groups:           ${GROUPS[*]}
Fold count:       ${FOLD_COUNT}
Stages:           ${STAGES}
EOF
}

count_total_steps() {
  TOTAL_STEPS=0
  if stage_enabled "runner"; then
    TOTAL_STEPS=$((TOTAL_STEPS + 3))
  fi
  if stage_enabled "csv"; then
    TOTAL_STEPS=$((TOTAL_STEPS + 2))
  fi
  if stage_enabled "split"; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
  fi
  if stage_enabled "encode"; then
    TOTAL_STEPS=$((TOTAL_STEPS + ( ${#GROUPS[@]} + 1 ) * FOLD_COUNT))
  fi
  if stage_enabled "train"; then
    TOTAL_STEPS=$((TOTAL_STEPS + ( ${#GROUPS[@]} + 1 ) * FOLD_COUNT))
  fi
  if stage_enabled "summary"; then
    TOTAL_STEPS=$((TOTAL_STEPS + ${#GROUPS[@]}))
  fi
  if [[ "$TOTAL_STEPS" -eq 0 ]]; then
    echo "No stages selected." >&2
    exit 1
  fi
}

run_step() {
  local stage="$1"
  local name="$2"
  local missing_inputs=()
  local completed_outputs=()
  local missing_outputs=()
  local path

  if ! stage_enabled "$stage"; then
    return
  fi

  STEP_INDEX=$((STEP_INDEX + 1))
  progress_line "$STEP_INDEX" "$TOTAL_STEPS" "$stage" "$name"
  echo "Working directory: ${REPO_ROOT}"
  echo "Command:"
  quote_command

  for path in "${INPUTS[@]}"; do
    if [[ ! -e "$path" ]]; then
      missing_inputs+=("$path")
    fi
  done

  if [[ "${#missing_inputs[@]}" -gt 0 ]]; then
    echo "Status: FAILED precheck"
    echo "Missing inputs:"
    printf '  - %s\n' "${missing_inputs[@]}"
    exit 1
  fi

  for path in "${OUTPUTS[@]}"; do
    if has_content "$path"; then
      completed_outputs+=("$path")
    else
      missing_outputs+=("$path")
    fi
  done

  if [[ "${#completed_outputs[@]}" -gt 0 ]]; then
    echo "Existing non-empty outputs:"
    printf '  - %s\n' "${completed_outputs[@]:0:10}"
    if [[ "${#completed_outputs[@]}" -gt 10 ]]; then
      echo "  ... $(( ${#completed_outputs[@]} - 10 )) more"
    fi
  fi

  if [[ "${#completed_outputs[@]}" -gt 0 && "${#missing_outputs[@]}" -eq 0 && "$RERUN_EXISTING" -eq 0 ]]; then
    echo "Status: SKIPPED, outputs already exist"
    return
  fi

  if [[ "${#completed_outputs[@]}" -gt 0 && "${#missing_outputs[@]}" -gt 0 && "$RERUN_EXISTING" -eq 0 ]]; then
    echo "Status: FAILED precheck"
    echo "Partial outputs already exist. Use --rerun-existing to run anyway,"
    echo "or clean/use a new experiment directory."
    echo "Missing/empty outputs:"
    printf '  - %s\n' "${missing_outputs[@]:0:10}"
    if [[ "${#missing_outputs[@]}" -gt 10 ]]; then
      echo "  ... $(( ${#missing_outputs[@]} - 10 )) more"
    fi
    exit 1
  fi

  if [[ "${#completed_outputs[@]}" -gt 0 && "$RERUN_EXISTING" -eq 1 ]]; then
    echo "Status: RUNNING despite existing outputs"
    echo "Warning: this script does not delete files; runner outputs may append."
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Status: DRY-RUN"
    return
  fi

  (cd "$REPO_ROOT" && "${CMD[@]}")
  echo "Status: DONE"
}

common_workload_args=(
  --dbname "$DBNAME"
  --host "$DB_HOST"
  --port "$DB_PORT"
  --user "$DB_USER"
  --sqls-dir "$SQLS_DIR"
  --workload-name "$WORKLOAD_NAME"
  --skip-template-id-vals "${SKIP_TEMPLATE_IDS[@]}"
  --query-id-limit "$QUERY_ID_LIMIT"
  --results-path "$RUNNER_RESULTS_PATH"
  --statement-timeout "$STATEMENT_TIMEOUT"
  --rounds "$ROUNDS"
  --run-mode explain-analyze-json
)

all_robdp_runner_dirs=()
for group in "${GROUPS[@]}"; do
  all_robdp_runner_dirs+=("${RUNNER_RESULTS_PATH}/$(parameter_dir "$group")")
done

print_configuration
count_total_steps


# ---------------------------------------------------------------------------
# 0. Run raw workload outputs
# ---------------------------------------------------------------------------

CMD=(
  "$PYTHON" "${REPO_ROOT}/run_imdb_with_pg.py"
  "${common_workload_args[@]}"
)
INPUTS=("${REPO_ROOT}/run_imdb_with_pg.py" "$SQLS_DIR")
OUTPUTS=("${RUNNER_RESULTS_PATH}/original")
run_step "runner" "Run original PostgreSQL baseline"

CMD=(
  "$PYTHON" "${REPO_ROOT}/run_imdb_with_robdp_hints.py"
  "${common_workload_args[@]}"
  --main-objective-id-vals "${MAIN_OBJECTIVE_IDS[@]}"
  --retain-strategy-id-vals "${RETAIN_STRATEGY_IDS[@]}"
  --final-level-path-limit "$FINAL_LEVEL_PATH_LIMIT"
)
INPUTS=("${REPO_ROOT}/run_imdb_with_robdp_hints.py" "$SQLS_DIR")
OUTPUTS=("${all_robdp_runner_dirs[@]}")
run_step "runner" "Run RobDP last-level hint export and RobDP runtime"

CMD=(
  "$PYTHON" "${REPO_ROOT}/run_imdb_with_reqo_guc.py"
  "${common_workload_args[@]}"
)
INPUTS=("${REPO_ROOT}/run_imdb_with_reqo_guc.py" "$SQLS_DIR")
OUTPUTS=("${RUNNER_RESULTS_PATH}/reqo_guc")
run_step "runner" "Run Reqo-GUC candidates and raw Reqo-GUC runtime"


# ---------------------------------------------------------------------------
# 1. Convert exported hints to hint SQL CSVs
# ---------------------------------------------------------------------------

parameter_groups=()
for group in "${GROUPS[@]}"; do
  parameter_groups+=("$(parameter_dir "$group")")
done

robdp_hint_csv_outputs=()
for group in "${GROUPS[@]}"; do
  robdp_hint_csv_outputs+=("${HINT_SQL_CSV_DIR}/${group}.csv")
done

CMD=(
  "$PYTHON" "${REPO_ROOT}/build_imdb_hint_sql_csv.py"
  --results-path "$RUNNER_RESULTS_PATH"
  --sqls-dir "$SQLS_DIR"
  --workload-name "$WORKLOAD_NAME"
  --output-dir "$HINT_SQL_CSV_DIR"
  --query-id-limit "$QUERY_ID_LIMIT"
  --hint-source robdp
  --parameter-groups "${parameter_groups[@]}"
)
INPUTS=("${REPO_ROOT}/build_imdb_hint_sql_csv.py" "$SQLS_DIR" "${all_robdp_runner_dirs[@]}")
OUTPUTS=("${robdp_hint_csv_outputs[@]}")
run_step "csv" "Build RobDP last-level hint SQL CSVs"

CMD=(
  "$PYTHON" "${REPO_ROOT}/build_imdb_hint_sql_csv.py"
  --results-path "$RUNNER_RESULTS_PATH"
  --sqls-dir "$SQLS_DIR"
  --workload-name "$WORKLOAD_NAME"
  --output-dir "$HINT_SQL_CSV_DIR"
  --query-id-limit "$QUERY_ID_LIMIT"
  --hint-source reqo
  --parameter-groups reqo_guc
)
INPUTS=("${REPO_ROOT}/build_imdb_hint_sql_csv.py" "$SQLS_DIR" "${RUNNER_RESULTS_PATH}/reqo_guc")
OUTPUTS=("${HINT_SQL_CSV_DIR}/reqo_guc.csv")
run_step "csv" "Build Reqo-GUC hint SQL CSV"


# ---------------------------------------------------------------------------
# 2. Build shared folds and fold-specific train/test SQL CSVs
# ---------------------------------------------------------------------------

source_csv_args=()
split_inputs=("${REPO_ROOT}/build_imdb_fold_splits.py")
split_outputs=()

for group in "${GROUPS[@]}"; do
  source_csv_args+=(
    --source-csv "$(robdp_source_name "$group")=${HINT_SQL_CSV_DIR}/${group}.csv"
  )
  split_inputs+=("${HINT_SQL_CSV_DIR}/${group}.csv")
done
source_csv_args+=(--source-csv "${REQO_GUC_SOURCE}=${HINT_SQL_CSV_DIR}/reqo_guc.csv")
split_inputs+=("${HINT_SQL_CSV_DIR}/reqo_guc.csv")

for ((fold_id = 1; fold_id <= FOLD_COUNT; fold_id++)); do
  split_outputs+=("${FOLDS_DIR}/original_fold_${fold_id}.csv")
  split_outputs+=("${FOLDS_DIR}/robdp_fold_${fold_id}.csv")
  split_outputs+=("${FOLDS_DIR}/${REQO_GUC_SOURCE}_fold_${fold_id}.csv")
  split_outputs+=("${FOLD_SQL_ROOT}/${REQO_GUC_SOURCE}/fold_${fold_id}/train.csv")
  split_outputs+=("${FOLD_SQL_ROOT}/${REQO_GUC_SOURCE}/fold_${fold_id}/test.csv")

  for group in "${GROUPS[@]}"; do
    source_name="$(robdp_source_name "$group")"
    split_outputs+=("${FOLDS_DIR}/${source_name}_fold_${fold_id}.csv")
    split_outputs+=("${FOLD_SQL_ROOT}/${source_name}/fold_${fold_id}/train.csv")
    split_outputs+=("${FOLD_SQL_ROOT}/${source_name}/fold_${fold_id}/test.csv")
  done
done

CMD=(
  "$PYTHON" "${REPO_ROOT}/build_imdb_fold_splits.py"
  "${source_csv_args[@]}"
  --output-root "$EXPERIMENT_ROOT"
  --fold "$FOLD_COUNT"
  --split-seed "$SPLIT_SEED"
  --min-candidates-per-query "$MIN_CANDIDATES_PER_QUERY"
)
INPUTS=("${split_inputs[@]}")
OUTPUTS=("${split_outputs[@]}")
run_step "split" "Build shared fold splits"


# ---------------------------------------------------------------------------
# 3. Encode each source/fold
# ---------------------------------------------------------------------------

trainable_sources=()
for group in "${GROUPS[@]}"; do
  trainable_sources+=("$(robdp_source_name "$group")")
done
trainable_sources+=("$REQO_GUC_SOURCE")

for source_name in "${trainable_sources[@]}"; do
  for ((fold_id = 1; fold_id <= FOLD_COUNT; fold_id++)); do
    CMD=(
      "$PYTHON" "${REPO_ROOT}/encode_fold_datasets.py"
      --source-name "$source_name"
      --fold-id "$fold_id"
      --fold-sql-root "$FOLD_SQL_ROOT"
      --encoding-root "$ENCODING_ROOT"
      --dataset-root "$DATASET_ROOT"
      --dbname "$DBNAME"
      --host "$DB_HOST"
      --port "$DB_PORT"
      --user "$DB_USER"
      --stats-dir "$STATS_DIR"
      --statement-timeout-ms 60000
      --min-candidates-per-query "$MIN_CANDIDATES_PER_QUERY"
      --repo-root "$REPO_ROOT"
    )
    INPUTS=(
      "${REPO_ROOT}/encode_fold_datasets.py"
      "${REPO_ROOT}/Utils/reqo_encode_sql.py"
      "$STATS_DIR"
      "${FOLD_SQL_ROOT}/${source_name}/fold_${fold_id}/train.csv"
      "${FOLD_SQL_ROOT}/${source_name}/fold_${fold_id}/test.csv"
    )
    OUTPUTS=(
      "${ENCODING_ROOT}/${source_name}/fold_${fold_id}/train/encode.pt"
      "${ENCODING_ROOT}/${source_name}/fold_${fold_id}/train/norm_stats.json"
      "${ENCODING_ROOT}/${source_name}/fold_${fold_id}/test/encode.pt"
      "${DATASET_ROOT}/${source_name}/fold_${fold_id}/train/postgresql_${DBNAME}_executed_query_plans_dataset.npy"
      "${DATASET_ROOT}/${source_name}/fold_${fold_id}/test/postgresql_${DBNAME}_executed_query_plans_dataset.npy"
    )
    run_step "encode" "Encode ${source_name} fold ${fold_id}"
  done
done


# ---------------------------------------------------------------------------
# 4. Train on already split datasets
# ---------------------------------------------------------------------------

for source_name in "${trainable_sources[@]}"; do
  for ((fold_id = 1; fold_id <= FOLD_COUNT; fold_id++)); do
    CMD=(
      "$PYTHON" "${REPO_ROOT}/train_no_split.py"
      --dbname "$DBNAME"
      --fold-id "$fold_id"
      --train-dataset-dir "${DATASET_ROOT}/${source_name}/fold_${fold_id}/train"
      --test-dataset-dir "${DATASET_ROOT}/${source_name}/fold_${fold_id}/test"
      --output-dir "${TRAIN_RESULTS_ROOT}/${source_name}/fold_${fold_id}"
      --database-statistics-dir "$DATABASE_STATISTICS_DIR"
    )
    if [[ "$SAVE_MODEL" -eq 1 ]]; then
      CMD+=(--save-model)
    fi

    INPUTS=(
      "${REPO_ROOT}/train_no_split.py"
      "${DATASET_ROOT}/${source_name}/fold_${fold_id}/train/postgresql_${DBNAME}_executed_query_plans_dataset.npy"
      "${DATASET_ROOT}/${source_name}/fold_${fold_id}/test/postgresql_${DBNAME}_executed_query_plans_dataset.npy"
      "$DATABASE_STATISTICS_DIR"
    )
    OUTPUTS=(
      "${TRAIN_RESULTS_ROOT}/${source_name}/fold_${fold_id}/reqo_fold_${fold_id}_query_selection.csv"
      "${TRAIN_RESULTS_ROOT}/${source_name}/fold_${fold_id}/reqo_fold_${fold_id}_candidate_scores.csv"
      "${TRAIN_RESULTS_ROOT}/${source_name}/fold_${fold_id}/reqo_fold_${fold_id}_results.txt"
    )
    if [[ "$SAVE_MODEL" -eq 1 ]]; then
      OUTPUTS+=("${TRAIN_RESULTS_ROOT}/${source_name}/fold_${fold_id}/reqo_fold_${fold_id}_model.pth")
    fi
    run_step "train" "Train ${source_name} fold ${fold_id}"
  done
done


# ---------------------------------------------------------------------------
# 5. Summarize each RobDP parameter group
# ---------------------------------------------------------------------------

for group in "${GROUPS[@]}"; do
  source_name="$(robdp_source_name "$group")"
  CMD=(
    "$PYTHON" "${REPO_ROOT}/summarize_fold_runtimes.py"
    --folds-dir "$FOLDS_DIR"
    --results-path "$RUNNER_RESULTS_PATH"
    --experiment-name "$group"
    --robdp-trained-results-dir "${TRAIN_RESULTS_ROOT}/${source_name}"
    --reqo-guc-trained-results-dir "${TRAIN_RESULTS_ROOT}/${REQO_GUC_SOURCE}"
    --output-dir "${SUMMARY_ROOT}/${group}"
  )
  INPUTS=(
    "${REPO_ROOT}/summarize_fold_runtimes.py"
    "$FOLDS_DIR"
    "${RUNNER_RESULTS_PATH}/original"
    "${RUNNER_RESULTS_PATH}/$(parameter_dir "$group")"
    "${RUNNER_RESULTS_PATH}/reqo_guc"
    "${TRAIN_RESULTS_ROOT}/${source_name}"
    "${TRAIN_RESULTS_ROOT}/${REQO_GUC_SOURCE}"
  )
  OUTPUTS=(
    "${SUMMARY_ROOT}/${group}/fold_query_summary.csv"
    "${SUMMARY_ROOT}/${group}/fold_summary.csv"
    "${SUMMARY_ROOT}/${group}/overall_summary.csv"
  )
  run_step "summary" "Summarize ${group}"
done

echo
echo "Pipeline finished."

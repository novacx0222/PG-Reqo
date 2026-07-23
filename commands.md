```zsh
ln -s /winhomes/xc241/imdbloadbase /opt/dbs/imdbload
```

```zsh
/opt/pgsql/16.2.0000/bin/pg_ctl -D /opt/dbs/imdbload start
/opt/pgsql/16.2.0000/bin/psql -U hx68 -d imdbloadbase
```

```zsh
python3.12 /winhomes/xc241/robdp/PG-Reqo/run_imdb_with_pg.py --dbname imdbloadbase --user hx68 --sqls-dir /winhomes/xc241/robdp/error-profile-temp --workload-name cardinality --skip-template-id-vals 1 2 3 4 5 29 --query-id-limit 100 --results-path /opt/results --statement-timeout 120s

python3.12 /winhomes/xc241/robdp/PG-Reqo/run_imdb_with_pg.py --dbname imdbloadbase --user hx68 --sqls-dir /winhomes/xc241/robdp/error-profile-temp --workload-name cardinality --skip-template-id-vals 1 2 3 4 5 6 29 --query-id-limit 100 --results-path /winhomes/xc241/robdp/results --statement-timeout 120s --rounds 2
```

```zsh
/opt/pgsql/16.2.0000/bin/pg_ctl -D /winhomes/hx68/imdbloadbase stop

python3 /winhomes/hx68/robdp/Reqo-PG/run_imdb_with_pg.py --dbname imdbloadbase --user hx68 --sqls-dir /winhomes/hx68/robdp/error-profile-temp --workload-name cardinality --skip-template-id-vals 29 --query-id-limit 100 --results-path /winhomes/hx68/robdp/results --statement-timeout 60s --rounds 1

python3 /winhomes/hx68/robdp/Reqo-PG/run_imdb_with_robdp_hints.py --dbname imdbloadbase --user hx68 --sqls-dir /winhomes/hx68/robdp/imdb-error-profile-0612 --workload-name cardinality --skip-template-id-vals 29 --query-id-limit 100 --results-path /winhomes/hx68/robdp/results --statement-timeout 60s --rounds 3 --main-objective-id-vals 1 --retain-strategy-id-vals 0 1 3 16

python3 /winhomes/hx68/robdp/Reqo-PG/Utils/gen_db_stats.py --dbname imdbloadbase --user hx68 --host localhost --port 5432


python3 /winhomes/hx68/robdp/Reqo-PG/Utils/reqo_encode_sql.py --sql-file /winhomes/hx68/robdp/Reqo-PG/Utils/gen_db_stats.py --dbname imdbload --user hx68 --stats-dir /winhomes/hx68/robdp/Reqo-PG/Data/imdbloadbase/database_statistics --output-dir /winhomes/hx68/robdp/Reqo-PG/Data/imdbloadbase/encoding --analyze

python3 /winhomes/hx68/robdp/Reqo-PG/build_imdb_hint_sql_csv.py --results-path /winhomes/hx68/robdp/results --sqls-dir /winhomes/hx68/robdp/imdb-error-profile-0612 --workload-name cardinality --output-dir /winhomes/hx68/robdp/results/hint-sql-csv

```

/winhomes/hx68/robdp/results/hint-sql-csv/1x1__0x0_sample_1pct.csv

```text
0. caught a cold
1. GPU
2. pokka: query slow
3. willow: should be ok but no GPU
4. all lot of hints for a given query
```

```zsh
python3 /data/robdp/Reqo-PG/run_imdb_with_robdp_hints.py --dbname imdbloadbase --user hx68 --sqls-dir /data/robdp/imdb-error-profile-0612 --workload-name cardinality --skip-template-id-vals 29 --query-id-limit 100 --results-path /data/robdp/results --statement-timeout 60s --rounds 1 --main-objective-id-vals 1 3 --retain-strategy-id-vals 1 3 --run-mode explain-json

python3 /data/robdp/Reqo-PG/Utils/gen_db_stats.py --dbname imdbloadbase --user hx68 --host localhost --port 5432

python3 /data/robdp/Reqo-PG/build_imdb_hint_sql_csv.py --results-path /data/robdp/results --sqls-dir /data/robdp/imdb-error-profile-0612 --workload-name cardinality --output-dir /data/robdp/results/hint-sql-csv

python3 /data/robdp/Reqo-PG/Utils/reqo_encode_sql.py --sql-file /data/robdp/results/hint-sql-csv/1x1__0x0.csv --dbname imdbloadbase --user hx68 --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics --output-dir /data/robdp/Reqo-PG/Data/imdbloadbase/encoding-0626 --analyze

python3 /data/robdp/Reqo-PG/Utils/reqo_encode_sql.py --sql-file /data/robdp/results/hint-sql-csv/1x1__0x0_top1000.csv --dbname imdbloadbase --user hx68 --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics --output-dir /data/robdp/Reqo-PG/Data/imdbloadbase/encoding-0626 --analyze

python3 run_reqo_train.py --dbname imdbloadbase --k 2 --save-model
```

```bash
python3 /data/robdp/Reqo-PG/Utils/reqo_encode_sql.py --sql-file /data/robdp/results/hint-sql-csv/1x1__0x0.csv --dbname imdbloadbase --user hx68 --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics --output-dir /data/robdp/Reqo-PG/Data/imdbloadbase/encoding-0626 --analyze && mv /data/robdp/Data/imdbloadbase/datasets /data/robdp/Reqo-PG/Data/imdbloadbase/datasets && python3 run_reqo_train.py --dbname imdbloadbase --k 2 --save-model

python3 /data/robdp/Reqo-PG/Utils/reqo_encode_sql.py --sql-file /data/robdp/results/hint-sql-csv/8x1__0x0.csv --dbname imdbloadbase --user hx68 --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics --output-dir /data/robdp/Reqo-PG/Data/imdbloadbase/encoding-0626-8x1__0x0 --analyze && mv /data/robdp/Data/imdbloadbase/datasets /data/robdp/Reqo-PG/Data/imdbloadbase/datasets && python3 run_reqo_train.py --dbname imdbloadbase --k 2 --save-model


python3 Utils/reqo_pt_to_npy.py \
  --pt-file /data/robdp/Reqo-PG/Data/imdbloadbase/encoding-0626-8x1__0x0/encode.pt \
  --dbname imdbloadbase \
  --output-dir /data/robdp/Reqo-PG/Data/imdbloadbase/datasets \
  --min-candidates-per-query 2
```

```bash
python3 run_plain_queries_from_folds.py \
  --fold-results-dir /data/robdp/Reqo-PG/Results/imdbloadbase \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --dbname imdbloadbase \
  --host localhost \
  --port 5432 \
  --user hx68 \
  --system-name robdp \
  --output-csv /data/robdp/Reqo-PG/Results/imdbloadbase/robdp_plain_runtime_8x1__0x0.csv \
  --rounds 1 \
  --statement-timeout 60s \
  --parameter-group-dir /data/robdp/results/8x1/0x0

python3 run_plain_queries_from_folds.py \
  --fold-results-dir /data/robdp/Reqo-PG/Results/imdbloadbase \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --dbname imdbloadbase \
  --host localhost \
  --port 5432 \
  --user hx68 \
  --system-name original \
  --output-csv /data/robdp/Reqo-PG/Results/imdbloadbase/original_plain_runtime_8x1__0x0.csv \
  --rounds 1 \
  --statement-timeout 60s \
  --parameter-group-dir /data/robdp/results/8x1/0x0
```

2026-07-06

```zsh
python3 /data/robdp/Reqo-PG/run_imdb_with_robdp_hints.py --dbname imdbloadbase --user hx68 --sqls-dir /data/robdp/imdb-error-profile-0612 --workload-name cardinality --skip-template-id-vals 29 --query-id-limit 100 --results-path /data/robdp/results --statement-timeout 60s --rounds 1 --main-objective-id-vals 1 3 --retain-strategy-id-vals 1 3 --run-mode explain-json

# [ble: elapsed 2h12m46s (CPU 2.4%)] python3 /data/robdp/Reqo-PG/run_imdb_with_rob


python3 /data/robdp/Reqo-PG/build_imdb_hint_sql_csv.py --results-path /data/robdp/results --sqls-dir /data/robdp/imdb-error-profile-0612 --workload-name cardinality --output-dir /data/robdp/results/hint-sql-csv

# Parameter groups: 6
# 1x1/0x0: query_groups=3200, rows=33368, csv=/data/robdp/results/hint-sql-csv/1x1__0x0.csv
# 1x1/8x3: query_groups=3200, rows=29544, csv=/data/robdp/results/hint-sql-csv/1x1__8x3.csv

# 1x3/0x0: query_groups=3200, rows=32990, csv=/data/robdp/results/hint-sql-csv/1x3__0x0.csv
# 1x3/8x1: query_groups=3200, rows=30346, csv=/data/robdp/results/hint-sql-csv/1x3__8x1.csv
# 8x1/0x0: query_groups=3200, rows=29187, csv=/data/robdp/results/hint-sql-csv/8x1__0x0.csv
# 8x3/0x0: query_groups=3200, rows=30327, csv=/data/robdp/results/hint-sql-csv/8x3__0x0.csv
# Total query groups: 19200
# Total CSV rows: 185762

for group in 1x1__0x0 1x1__8x3 1x3__0x0 1x3__8x1 8x1__0x0 8x3__0x0; do
  echo "===== Encoding ${group} ====="

  python3 /data/robdp/Reqo-PG/Utils/reqo_encode_sql.py \
    --sql-file "/data/robdp/results/hint-sql-csv/${group}.csv" \
    --dbname imdbloadbase \
    --user hx68 \
    --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics \
    --output-dir "/data/robdp/Reqo-PG/Data/imdbloadbase/encoding-${group}" \
    --analyze \
    || echo "WARNING: ${group} failed, continuing..."
done

```

```bash
for group in 1x1__0x0 1x1__8x3 1x3__0x0 1x3__8x1 8x1__0x0 8x3__0x0; do
  echo "===== Converting ${group} ====="

  python3 Utils/reqo_pt_to_npy.py \
    --pt-file /data/robdp/Reqo-PG/Data/imdbloadbase/encoding-${group}/encode.pt \
    --dbname imdbloadbase \
    --output-dir /data/robdp/Reqo-PG/Data/imdbloadbase/datasets-${group} \
    --min-candidates-per-query 2 \
    || echo "WARNING: ${group} failed, continuing..."
done
```

```bash
for d in datasets-1*; do
  [ -e "$d" ] || continue
  mv -- "$d" "datasets/${d#datasets-}"
done

for d in encoding-1*; do
  [ -e "$d" ] || continue
  mv -- "$d" "encoding/${d#encoding-}"
done

tar cvf datasets-0710.tar datasets
tar cvf encoding-0710.tar encoding
```

```bash
for group in 1x1__0x0 1x1__8x3 1x3__0x0 1x3__8x1 8x1__0x0 8x3__0x0; do
  echo "===== Training ${group} ====="

  python3 run_reqo_train.py --dbname imdbloadbase --k 2 --save-model --experiment_name ${group} \
    || echo "WARNING: ${group} failed, continuing..."
done
```

```bash
groups=(
  "1x1__0x0"
  "1x1__8x3"
  "1x3__0x0"
  "1x3__8x1"
  "8x1__0x0"
  "8x3__0x0"
)

for group in "${groups[@]}"; do
  param_dir="${group/__//}"

  echo "===== Running ${group} | parameter dir: ${param_dir} ====="

  python3 run_plain_queries_from_folds.py \
    --fold-results-dir /data/robdp/Reqo-PG/Results/imdbloadbase/${group} \
    --sqls-dir /data/robdp/imdb-error-profile-0612 \
    --workload-name cardinality \
    --dbname imdbloadbase \
    --host localhost \
    --port 5432 \
    --user hx68 \
    --system-name robdp \
    --output-csv "/data/robdp/Reqo-PG/Results/imdbloadbase/${group}/robdp_plain_runtime_${group}.csv" \
    --rounds 1 \
    --statement-timeout 60s \
    --parameter-group-dir "/data/robdp/results/${param_dir}" \
    || echo "WARNING: ${group} failed, continuing..."
done
```

2026-07-10

```bash
# 0. Original PostgreSQL runtime: keep ANALYZE
/opt/pgsql/16.2.0000/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ start

python3 /data/robdp/Reqo-PG/run_imdb_with_pg.py \
  --dbname imdbloadbase \
  --host localhost \
  --port 5432 \
  --user hx68 \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --skip-template-id-vals 29 \
  --query-id-limit 100 \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs/original \
  --statement-timeout 60s \
  --rounds 1 \
  --run-mode explain-analyze-json

/opt/pgsql/16.2.0000/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ stop
```

```bash
# 0. Original RobDP runtime: keep ANALYZE
/opt/pgsql/16.2.0524/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ start

python3 /data/robdp/Reqo-PG/run_imdb_with_robdp.py \
  --dbname imdbloadbase \
  --host localhost \
  --port 5432 \
  --user hx68 \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --skip-template-id-vals 29 \
  --query-id-limit 100 \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs/robdp \
  --statement-timeout 60s \
  --rounds 1 \
  --run-mode explain-analyze-json \
  --main-objective-id-vals 1 \
  --retain-strategy-id-vals 1

/opt/pgsql/16.2.0524/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ stop
```

```bash
# 0. RobDP last-level hints: no ANALYZE
/opt/pgsql/16.2.0626/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ start

python3 /data/robdp/Reqo-PG/run_imdb_with_robdp_hints.py \
  --dbname imdbloadbase \
  --host localhost \
  --port 5432 \
  --user hx68 \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --skip-template-id-vals 29 \
  --query-id-limit 100 \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs/robdp_last_level \
  --statement-timeout 60s \
  --rounds 1 \
  --run-mode explain-json \
  --main-objective-id-vals 1 \
  --retain-strategy-id-vals 1 \
  --final-level-path-limit 13

/opt/pgsql/16.2.0626/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ stop
```

```bash
# 0. Reqo-GUC hints: no ANALYZE
/opt/pgsql/16.2.0000/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ start

python3 /data/robdp/Reqo-PG/run_imdb_with_reqo_guc.py \
  --dbname imdbloadbase \
  --host localhost \
  --port 5432 \
  --user hx68 \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --skip-template-id-vals 29 \
  --query-id-limit 100 \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs/reqo_guc \
  --statement-timeout 60s \
  --rounds 1 \
  --run-mode explain-json

/opt/pgsql/16.2.0000/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ stop
```

```bash
# 1. Build RobDP hint SQL CSVs
python3 /data/robdp/Reqo-PG/build_imdb_hint_sql_csv.py \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs/robdp_last_level \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --output-dir /data/robdp/imdb-presplit-0710/hint-sql-csv \
  --query-id-limit 100 \
  --hint-source robdp \
  --parameter-groups 1x1/0x0 8x1/0x0
```

```bash
# 1. Build Reqo-GUC hint SQL CSV
python3 /data/robdp/Reqo-PG/build_imdb_hint_sql_csv.py \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs \
  --sqls-dir /data/robdp/imdb-error-profile-0612 \
  --workload-name cardinality \
  --output-dir /data/robdp/imdb-presplit-0710/hint-sql-csv \
  --query-id-limit 100 \
  --hint-source reqo \
  --parameter-groups reqo_guc
```

```bash
# 2. Build shared folds
python3 /data/robdp/Reqo-PG/build_imdb_fold_splits.py \
  --source-csv robdp_last_level_1x1__0x0=/data/robdp/imdb-presplit-0710/hint-sql-csv/1x1__0x0.csv \
  --source-csv robdp_last_level_8x1__0x0=/data/robdp/imdb-presplit-0710/hint-sql-csv/8x1__0x0.csv \
  --source-csv reqo_guc=/data/robdp/imdb-presplit-0710/hint-sql-csv/reqo_guc.csv \
  --output-root /data/robdp/imdb-presplit-0710 \
  --fold 2 \
  --split-seed 0 \
  --min-candidates-per-query 2
```

```bash
# 3A. Run EXPLAIN ANALYZE once per source and save raw plan caches
/opt/pgsql/16.2.hint/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ start

# for source_and_csv in \
#   "reqo_guc:/data/robdp/imdb-presplit-0710/hint-sql-csv/reqo_guc.csv" \
#   "robdp_last_level_1x1__0x0:/data/robdp/imdb-presplit-0710/hint-sql-csv/1x1__0x0.csv" \
#   "robdp_last_level_8x1__0x0:/data/robdp/imdb-presplit-0710/hint-sql-csv/8x1__0x0.csv"
# do

for source_and_csv in \
  "reqo_guc:/data/robdp/imdb-presplit-0710/hint-sql-csv/reqo_guc.csv"
do
  source="${source_and_csv%%:*}"
  sql_file="${source_and_csv#*:}"
  cache_dir="/data/robdp/imdb-presplit-0710/encoding/${source}/full"
  cache_file="${cache_dir}/plans_cache.json"

  echo "===== Collecting raw plan cache for ${source} ====="

  python3 /data/robdp/Reqo-PG/Utils/reqo_encode_sql_save_pt.py \
    --sql-file "${sql_file}" \
    --dbname imdbloadbase \
    --host localhost \
    --port 5432 \
    --user hx68 \
    --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics \
    --output-dir "${cache_dir}" \
    --analyze \
    --plans-cache-output "${cache_file}" \
    --plans-cache-only \
    --statement-timeout-ms 60000 \
    --min-candidates-per-query 2 \
    || echo "WARNING: ${source} cache collection failed, continuing..."
done

/opt/pgsql/16.2.hint/bin/pg_ctl -D /winhomes/hx68/imdbloadbase/ stop

# 3B. Encode each fold from the raw plan cache:
#     train computes norm_stats.json; test reuses the train stats.
for source in \
  reqo_guc \
  robdp_last_level_1x1__0x0 \
  robdp_last_level_8x1__0x0
do
  cache_file="/data/robdp/imdb-presplit-0710/encoding/${source}/full/plans_cache.json"

  for fold in 1 2; do
    echo "===== Encoding ${source} fold ${fold} from cache ====="

    python3 /data/robdp/Reqo-PG/encode_fold_datasets.py \
      --source-name "${source}" \
      --fold-id "${fold}" \
      --fold-sql-root /data/robdp/imdb-presplit-0710/fold_sql \
      --encoding-root /data/robdp/imdb-presplit-0710/encoding \
      --dataset-root /data/robdp/Reqo-PG/Data/imdbloadbase/datasets-presplit-0710 \
      --dbname imdbloadbase \
      --host localhost \
      --port 5432 \
      --user hx68 \
      --stats-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics \
      --plans-cache-input "${cache_file}" \
      --statement-timeout-ms 60000 \
      --min-candidates-per-query 2 \
      --repo-root /data/robdp/Reqo-PG \
      || echo "WARNING: ${source} fold ${fold} failed, continuing..."
  done
done
```

```bash
# 4. Train all sources/folds
for source in \
  robdp_last_level_1x1__0x0 \
  robdp_last_level_8x1__0x0 \
  reqo_guc
do
  for fold in 1 2; do
    echo "===== Training ${source} fold ${fold} ====="

    python3 /data/robdp/Reqo-PG/train_no_split.py \
      --dbname imdbloadbase \
      --fold-id "${fold}" \
      --train-dataset-dir "/data/robdp/Reqo-PG/Data/imdbloadbase/datasets-presplit-0710/${source}/fold_${fold}/train" \
      --test-dataset-dir "/data/robdp/Reqo-PG/Data/imdbloadbase/datasets-presplit-0710/${source}/fold_${fold}/test" \
      --output-dir "/data/robdp/Reqo-PG/Results/imdbloadbase/presplit-0710/${source}/fold_${fold}" \
      --database-statistics-dir /data/robdp/Reqo-PG/Data/imdbloadbase/database_statistics \
      --save-model \
      || echo "WARNING: ${source} fold ${fold} failed, continuing..."
  done
done
```

```bash
# 5. Summarize all groups
python3 /data/robdp/Reqo-PG/summarize_all_groups.py \
  --folds-dir /data/robdp/imdb-presplit-0710/folds \
  --results-path /data/robdp/imdb-presplit-0710/runner_outputs \
  --robdp-runtime-results-root /data/robdp/imdb-presplit-0710/runner_outputs/robdp \
  --train-results-root /data/robdp/Reqo-PG/Results/imdbloadbase/presplit-0710 \
  --groups 1x1__0x0 8x1__0x0 \
  --output-dir /data/robdp/imdb-presplit-0710/summary
```

```text
runner_outputs/original
runner_outputs/robdp/{parameter_group}
runner_outputs/robdp_last_level/{parameter_group}
runner_outputs/reqo_guc
Results/imdbloadbase/presplit-0710/robdp_last_level_{group} [train]
Results/imdbloadbase/presplit-0710/reqo_guc [train]
```

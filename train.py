import os
import csv
import numpy as np
from numpy import nanmean
from torch import optim
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from Models.reqo_model import Reqo
from Utils.loss import *
from Utils.evaluate import get_qerror_and_spearman, get_plansubop_and_runtime, write_results_to_file, plot_runtimes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset, batch_size, shuffle_or_not):
    labels = [row[2] for row in dataset]
    data = [Data(x=torch.FloatTensor(row[0]),
                 edge_index=sort_edge_index(torch.LongTensor(row[1]).t()),
                 y=row[2]) for row in dataset]
    dataset_loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=shuffle_or_not)
    return dataset_loader, max(labels), min(labels)


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def _get_candidate_id(query_candidate_ids_i, local_query_idx, candidate_idx):
    if query_candidate_ids_i is None:
        return candidate_idx
    candidate_ids = query_candidate_ids_i[local_query_idx]
    if candidate_idx >= len(candidate_ids):
        return candidate_idx
    candidate_id = candidate_ids[candidate_idx]
    if candidate_id is None or candidate_id == "":
        return candidate_idx
    return candidate_id


def _blank_if_none(value):
    return "" if value is None else value


def _default_query_metadata(query_group_id):
    return {
        "query_group_id": query_group_id,
        "template_id": None,
        "original_query_id": None,
    }


def _normalize_query_metadata(metadata, fallback_query_group_id):
    if metadata is None:
        return _default_query_metadata(fallback_query_group_id)
    if hasattr(metadata, "item") and not isinstance(metadata, dict):
        try:
            metadata = metadata.item()
        except ValueError:
            pass
    if not isinstance(metadata, dict):
        return _default_query_metadata(fallback_query_group_id)

    query_group_id = metadata.get("query_group_id", fallback_query_group_id)
    return {
        "query_group_id": query_group_id,
        "template_id": metadata.get("template_id"),
        "original_query_id": metadata.get("original_query_id"),
    }


def _metadata_for_index(query_metadata, query_index, idx):
    if query_index is None:
        query_index = list(range(idx + 1))
    query_group_id = query_index[idx]
    if query_metadata is None or idx >= len(query_metadata):
        return _default_query_metadata(query_group_id)
    return _normalize_query_metadata(query_metadata[idx], query_group_id)


def load_query_metadata(dbname, query_index):
    """Load optional query metadata aligned with executed_query_index.npy."""
    metadata_path = f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_metadata.npy'
    if not os.path.exists(metadata_path):
        return [
            _default_query_metadata(query_group_id)
            for query_group_id in query_index
        ]

    raw_metadata = np.load(metadata_path, allow_pickle=True)
    normalized_metadata = [
        _normalize_query_metadata(raw_metadata[idx], query_index[idx])
        if idx < len(raw_metadata)
        else _default_query_metadata(query_index[idx])
        for idx in range(len(query_index))
    ]
    return np.array(normalized_metadata, dtype=object)


def write_fold_split_details(
        filename,
        fold_id,
        query_index,
        query_metadata,
        query_plans_index_num,
        sample_q_num1,
        sample_q_num2,
):
    """Write the train/test query split used by one fold."""
    fieldnames = [
        "fold_id",
        "split",
        "global_query_idx",
        "fold_query_idx",
        "query_group_id",
        "template_id",
        "original_query_id",
        "candidate_count",
    ]
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        fold_test_idx = 0
        for global_query_idx, query_group_id in enumerate(query_index):
            metadata = _metadata_for_index(
                query_metadata,
                query_index,
                global_query_idx,
            )
            split = (
                "test"
                if sample_q_num1 <= global_query_idx < sample_q_num2
                else "train"
            )
            if split == "test":
                fold_query_idx = fold_test_idx
                fold_test_idx += 1
            else:
                fold_query_idx = ""

            writer.writerow({
                "fold_id": fold_id,
                "split": split,
                "global_query_idx": global_query_idx,
                "fold_query_idx": fold_query_idx,
                "query_group_id": metadata["query_group_id"],
                "template_id": _blank_if_none(metadata["template_id"]),
                "original_query_id": _blank_if_none(metadata["original_query_id"]),
                "candidate_count": int(query_plans_index_num[global_query_idx]),
            })


def write_candidate_score_details(
        filename,
        fold_id,
        query_index_i,
        query_metadata_i,
        query_plans_index_i,
        query_plans_index_num_i,
        query_postgres_cost_i,
        pred_iv,
        actual_latency,
):
    """Write every test candidate's score, runtime, and selection flags."""
    fieldnames = [
        "fold_id",
        "fold_query_idx",
        "query_group_id",
        "template_id",
        "original_query_id",
        "candidate_idx",
        "candidate_id",
        "actual_runtime_ms",
        "postgres_cost",
        "model_score",
        "is_postgres_choice",
        "is_model_choice",
        "is_optimal_choice",
        "runtime_vs_optimal_ratio",
    ]
    p_n = 0
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for local_query_idx, plan_count in enumerate(query_plans_index_num_i):
            plan_count = int(plan_count)
            query_actual_set = actual_latency[p_n:p_n + plan_count]
            query_pred_set = pred_iv[p_n:p_n + plan_count]
            query_postgres_cost_set = query_postgres_cost_i[local_query_idx]

            postgres_select_idx = int(np.argmin(query_postgres_cost_set))
            model_select_idx = int(np.argmin(query_pred_set))
            optimal_select_idx = int(np.argmin(query_actual_set))
            optimal_runtime = float(query_actual_set[optimal_select_idx])
            query_group_id = (
                query_index_i[local_query_idx]
                if query_index_i is not None
                else local_query_idx
            )
            metadata = _metadata_for_index(
                query_metadata_i,
                query_index_i,
                local_query_idx,
            )

            for candidate_idx in range(plan_count):
                actual_runtime = float(query_actual_set[candidate_idx])
                writer.writerow({
                    "fold_id": fold_id,
                    "fold_query_idx": local_query_idx,
                    "query_group_id": metadata["query_group_id"],
                    "template_id": _blank_if_none(metadata["template_id"]),
                    "original_query_id": _blank_if_none(metadata["original_query_id"]),
                    "candidate_idx": candidate_idx,
                    "candidate_id": _get_candidate_id(
                        query_plans_index_i,
                        local_query_idx,
                        candidate_idx,
                    ),
                    "actual_runtime_ms": actual_runtime,
                    "postgres_cost": float(query_postgres_cost_set[candidate_idx]),
                    "model_score": float(query_pred_set[candidate_idx]),
                    "is_postgres_choice": candidate_idx == postgres_select_idx,
                    "is_model_choice": candidate_idx == model_select_idx,
                    "is_optimal_choice": candidate_idx == optimal_select_idx,
                    "runtime_vs_optimal_ratio": _safe_ratio(
                        actual_runtime,
                        optimal_runtime,
                    ),
                })

            p_n += plan_count


def write_query_selection_details(
        filename,
        fold_id,
        query_index_i,
        query_metadata_i,
        query_plans_index_i,
        query_plans_index_num_i,
        query_postgres_cost_i,
        pred_iv,
        actual_latency,
):
    """Write per-query plan choices for the best epoch in one fold."""
    fieldnames = [
        "fold_id",
        "fold_query_idx",
        "query_group_id",
        "template_id",
        "original_query_id",
        "candidate_count",
        "postgres_candidate_idx",
        "postgres_candidate_id",
        "postgres_cost",
        "postgres_runtime_ms",
        "model_candidate_idx",
        "model_candidate_id",
        "model_score",
        "model_runtime_ms",
        "optimal_candidate_idx",
        "optimal_candidate_id",
        "optimal_runtime_ms",
        "model_vs_postgres_runtime_ratio",
        "model_vs_optimal_runtime_ratio",
        "outcome_vs_postgres",
    ]
    p_n = 0
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for local_query_idx, plan_count in enumerate(query_plans_index_num_i):
            plan_count = int(plan_count)
            query_actual_set = actual_latency[p_n:p_n + plan_count]
            query_pred_set = pred_iv[p_n:p_n + plan_count]
            query_postgres_cost_set = query_postgres_cost_i[local_query_idx]

            postgres_select_idx = int(np.argmin(query_postgres_cost_set))
            model_select_idx = int(np.argmin(query_pred_set))
            optimal_select_idx = int(np.argmin(query_actual_set))

            postgres_runtime = float(query_actual_set[postgres_select_idx])
            model_runtime = float(query_actual_set[model_select_idx])
            optimal_runtime = float(query_actual_set[optimal_select_idx])

            if model_runtime < postgres_runtime:
                outcome = "improved"
            elif model_runtime > postgres_runtime:
                outcome = "regressed"
            else:
                outcome = "tie"

            query_group_id = (
                query_index_i[local_query_idx]
                if query_index_i is not None
                else local_query_idx
            )
            metadata = _metadata_for_index(
                query_metadata_i,
                query_index_i,
                local_query_idx,
            )
            writer.writerow({
                "fold_id": fold_id,
                "fold_query_idx": local_query_idx,
                "query_group_id": metadata["query_group_id"],
                "template_id": _blank_if_none(metadata["template_id"]),
                "original_query_id": _blank_if_none(metadata["original_query_id"]),
                "candidate_count": plan_count,
                "postgres_candidate_idx": postgres_select_idx,
                "postgres_candidate_id": _get_candidate_id(
                    query_plans_index_i,
                    local_query_idx,
                    postgres_select_idx,
                ),
                "postgres_cost": float(query_postgres_cost_set[postgres_select_idx]),
                "postgres_runtime_ms": postgres_runtime,
                "model_candidate_idx": model_select_idx,
                "model_candidate_id": _get_candidate_id(
                    query_plans_index_i,
                    local_query_idx,
                    model_select_idx,
                ),
                "model_score": float(query_pred_set[model_select_idx]),
                "model_runtime_ms": model_runtime,
                "optimal_candidate_idx": optimal_select_idx,
                "optimal_candidate_id": _get_candidate_id(
                    query_plans_index_i,
                    local_query_idx,
                    optimal_select_idx,
                ),
                "optimal_runtime_ms": optimal_runtime,
                "model_vs_postgres_runtime_ratio": _safe_ratio(
                    model_runtime,
                    postgres_runtime,
                ),
                "model_vs_optimal_runtime_ratio": _safe_ratio(
                    model_runtime,
                    optimal_runtime,
                ),
                "outcome_vs_postgres": outcome,
            })
            p_n += plan_count


def train(dbname, reqo_config, k_i, trainset, testset, save_path, query_plans_index_num_i, query_postgres_cost_i,
          save_model, query_index_i=None, query_metadata_i=None, query_plans_index_i=None):
    batch_size = reqo_config["batch_size"]
    table_columns_number = np.load(f'Data/{dbname}/database_statistics/table_columns_number.npy')

    train_loader, max_sup_train_label, min_sup_train_label = load_dataset(trainset, batch_size, True)
    test_loader, max_sup_test_label, min_sup_test_label = load_dataset(testset, batch_size, False)
    testset_size = len(testset)
    max_label_log = math.log(max(max_sup_train_label, max_sup_test_label) + 1)
    min_label_log = math.log(min(min_sup_train_label, min_sup_test_label) + 1)

    # load model
    encoder_params = {k: v for k, v in reqo_config.items() if k.startswith("encoder_")}
    estimator_params = {k: v for k, v in reqo_config.items() if k.startswith("estimator_")}
    encoder_params["encoder_table_num"] = len(table_columns_number)
    encoder_params["encoder_column_num"] = sum(table_columns_number)

    model = Reqo(encoder_params=encoder_params, estimator_params=estimator_params)
    model = model.to(device)

    # criteon_logmse = LogMSELoss()
    criteon_data_uncertainty = DataUncertaintyLoss()
    criteon_ranking = PairRankingLoss(margin=reqo_config["pairrankingloss_margin"])
    optimizer = optim.Adam(model.parameters(), lr=reqo_config["learning_rate"])

    epochs = 100
    early_stop = 0
    best_test_perf = float('inf')
    best_pred_iv = None
    best_actual_latency = None
    for epoch in range(epochs):
        if early_stop >= 20:
            break
        # Train
        model.train()
        step = 0
        train_loss_all = 0
        with tqdm(train_loader, desc=f'Fold {k_i} Epoch: {epoch + 1} Training ', unit='batch') as train_loader:
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                batch_train_pred, batch_train_va, batch_train_iv = model(batch, table_columns_number)
                train_data_uncertainty_loss = criteon_data_uncertainty(batch_train_pred, batch_train_va,
                                                                       batch.y.float(), max_label_log, min_label_log)
                train_ranking_loss = criteon_ranking(batch_train_iv, batch.y.float(), max_label_log, min_label_log)
                train_loss = train_data_uncertainty_loss + train_ranking_loss
                train_loss.backward()
                optimizer.step()

                batch_graph_num = batch.num_graphs
                train_loss_all += train_loss.item() * batch_graph_num
                step += batch_graph_num
            avg_train_loss = float(train_loss_all / step)

        # Evaluate
        model.eval()
        step = 0
        test_loss_all = 0
        actual_latency = torch.zeros(testset_size, device=device)
        pred_ev = torch.zeros(testset_size, device=device)
        pred_va = torch.zeros(testset_size, device=device)
        pred_iv = torch.zeros(testset_size, device=device)
        with tqdm(test_loader, desc=f'Fold {k_i} Epoch: {epoch + 1} Testing ', unit='batch') as test_loader:
            for batch in test_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    batch_test_pred, batch_test_va, batch_test_iv = model(batch, table_columns_number)
                test_data_uncertainty = criteon_data_uncertainty(
                    batch_test_pred, batch_test_va, batch.y.float(), max_label_log, min_label_log
                )
                test_ranking_loss = criteon_ranking(batch_test_iv, batch.y.float(), max_label_log, min_label_log)
                test_loss = test_data_uncertainty + test_ranking_loss
                batch_graph_num = batch.num_graphs
                test_loss_all += test_loss.item() * batch_graph_num
                end = step + batch_graph_num
                pred_ev[step:end] = batch_test_pred.view(-1)
                pred_va[step:end] = batch_test_va.view(-1)
                pred_iv[step:end] = batch_test_iv.view(-1)
                actual_latency[step:end] = batch.y.view(-1)
                step = end
            avg_test_loss = float(test_loss_all / step)

        pred_ev = pred_ev.cpu().numpy()
        pred_iv = pred_iv.cpu().numpy()
        actual_latency = actual_latency.cpu().numpy()

        cost_estimation_results = get_qerror_and_spearman(pred_ev, actual_latency, max_label_log, min_label_log)
        robustness_results, runtime_per_query = get_plansubop_and_runtime(pred_iv, actual_latency,
                                                                          query_postgres_cost_i,
                                                                          query_plans_index_num_i)
        print(
            f'Fold {k_i} Epoch {epoch + 1}: train_loss: {avg_train_loss}, test_loss: {avg_test_loss}, spearmancorrelation: {cost_estimation_results[-1]}, optimal_runtime_ratio: {robustness_results[11]}')

        # # Early stop based on test loss
        # if avg_test_loss < best_test_perf:
        #     best_test_perf = avg_test_loss
        #     best_model = model.state_dict()
        #     best_cost_estimation_results = cost_estimation_results
        #     best_robustness_results = robustness_results
        #     best_runtime_per_query = runtime_per_query
        #     early_stop = 0
        # else:
        #     early_stop += 1

        # Early stop based on optimal runtime ratio
        if best_pred_iv is None or robustness_results[11] < best_test_perf:
            best_test_perf = robustness_results[11]
            best_model = model.state_dict()
            best_cost_estimation_results = cost_estimation_results
            best_robustness_results = robustness_results
            best_runtime_per_query = runtime_per_query
            best_pred_iv = pred_iv.copy()
            best_actual_latency = actual_latency.copy()
            early_stop = 0
        else:
            early_stop += 1

    cost_estimation_results = best_cost_estimation_results
    robustness_results = best_robustness_results
    runtime_per_query = best_runtime_per_query

    print(
        f'Fold {k_i}: test results: qerror_median: {cost_estimation_results[4]}, '
        f'qerror_top99mean: {cost_estimation_results[2]}, '
        f'spearman_correlation: {cost_estimation_results[-1]}, '
        f'subop_median: {robustness_results[4]}, '
        f'subop_top99mean: {robustness_results[2]}, '
        f'model_to_postgresql_runtime_ratio: {robustness_results[10]}, '
        f'model_to_optimal_runtime_ratio: {robustness_results[11]}'
    )
    os.makedirs(save_path, exist_ok=True)
    write_results_to_file(cost_estimation_results + robustness_results, expl_or_not=False,
                          filename=save_path + 'reqo_fold_' + str(k_i) + '_results.txt')
    write_query_selection_details(
        filename=save_path + 'reqo_fold_' + str(k_i) + '_query_selection.csv',
        fold_id=k_i,
        query_index_i=query_index_i,
        query_metadata_i=query_metadata_i,
        query_plans_index_i=query_plans_index_i,
        query_plans_index_num_i=query_plans_index_num_i,
        query_postgres_cost_i=query_postgres_cost_i,
        pred_iv=best_pred_iv,
        actual_latency=best_actual_latency,
    )
    write_candidate_score_details(
        filename=save_path + 'reqo_fold_' + str(k_i) + '_candidate_scores.csv',
        fold_id=k_i,
        query_index_i=query_index_i,
        query_metadata_i=query_metadata_i,
        query_plans_index_i=query_plans_index_i,
        query_plans_index_num_i=query_plans_index_num_i,
        query_postgres_cost_i=query_postgres_cost_i,
        pred_iv=best_pred_iv,
        actual_latency=best_actual_latency,
    )
    if save_model:
        torch.save(best_model, save_path + 'reqo_fold_' + str(k_i) + '_model.pth')
    return cost_estimation_results + robustness_results, runtime_per_query


def k_fold_train(dbname, reqo_config, k=10, save_model=False):
    save_path = f'Results/{dbname}/'
    os.makedirs(save_path, exist_ok=True)
    dataset = np.load(
        f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_dataset.npy', allow_pickle=True
    )
    query_plans_index_num = np.load(
        f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_index_num.npy', allow_pickle=True
    )
    query_postgres_cost = np.load(
        f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_postgres_cost.npy', allow_pickle=True
    )
    query_index = np.load(
        f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_index.npy', allow_pickle=True
    )
    query_metadata = load_query_metadata(dbname, query_index)
    query_plans_index = np.load(
        f'Data/{dbname}/datasets/postgresql_{dbname}_executed_query_plans_index.npy', allow_pickle=True
    )
    k_sample_num = round(len(query_plans_index_num) / k)

    all_results = []
    all_postgres_runtimes, all_reqo_runtimes, all_optimal_runtimes = [], [], []
    for k_i in range(0, k):
        sample_q_num1 = k_sample_num * k_i
        sample_q_num2 = k_sample_num * (k_i + 1)
        sample_p_num1 = sum(query_plans_index_num[:sample_q_num1])
        sample_p_num2 = sample_p_num1 + sum(query_plans_index_num[sample_q_num1:sample_q_num2])

        trainset = np.concatenate((dataset[:sample_p_num1], dataset[sample_p_num2:]), axis=0)
        testset = dataset[sample_p_num1:sample_p_num2]

        query_plans_index_num_i = query_plans_index_num[sample_q_num1:sample_q_num2]
        query_postgres_cost_i = query_postgres_cost[sample_q_num1:sample_q_num2]
        query_index_i = query_index[sample_q_num1:sample_q_num2]
        query_metadata_i = query_metadata[sample_q_num1:sample_q_num2]
        query_plans_index_i = query_plans_index[sample_q_num1:sample_q_num2]
        fold_save_path = save_path + 'fold_' + str(k_i + 1) + '/'
        os.makedirs(fold_save_path, exist_ok=True)
        write_fold_split_details(
            filename=fold_save_path + 'reqo_fold_' + str(k_i + 1) + '_split.csv',
            fold_id=k_i + 1,
            query_index=query_index,
            query_metadata=query_metadata,
            query_plans_index_num=query_plans_index_num,
            sample_q_num1=sample_q_num1,
            sample_q_num2=sample_q_num2,
        )

        results, runtime_per_query = train(
            dbname, reqo_config, k_i + 1, trainset, testset, fold_save_path,
            query_plans_index_num_i, query_postgres_cost_i, save_model,
            query_index_i, query_metadata_i, query_plans_index_i
        )
        all_results.append(results)
        all_postgres_runtimes.extend(runtime_per_query[0])
        all_reqo_runtimes.extend(runtime_per_query[1])
        all_optimal_runtimes.extend(runtime_per_query[2])
    write_results_to_file(nanmean(np.array(all_results), axis=0), expl_or_not=False,
                          filename=save_path + 'reqo_avg_results.txt')
    plot_runtimes(all_postgres_runtimes, all_reqo_runtimes, all_optimal_runtimes,
                  save_path + 'reqo_runtime_performance.png')


if __name__ == '__main__':
    dbname = 'stats'
    reqo_config = {
        'batch_size': 256,
        'learning_rate': 0.001,
        'encoder_attention_heads': 8,
        'encoder_conv_layers': 4,
        'encoder_gnn_embedding_dim': 256,
        'encoder_gnn_dropout_rate': 0.1,
        'encoder_dirgnn_alpha': 0.3,
        'encoder_node_type_embedding_dim': 16,
        'encoder_column_embedding_dim': 8,
        'explainer_fcn_layers': 4,
        'explainer_explanation_embedding_dim': 512,
        'explainer_fcn_dropout_rate': 0.1,
        'estimator_fcn_layers': 4,
        'estimator_estimation_embedding_dim': 512,
        'estimator_fcn_dropout_rate': 0.1
    }
    k_fold_train(dbname, reqo_config, k=10)

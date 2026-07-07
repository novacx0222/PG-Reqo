import argparse
from typing import Any

from Utils.postgresql_database_statistic_generator import generate_postgresql_database_statistic
from Utils.postgresql_workload_generator import generate_postgresql_workload_with_hints
from Utils.query_plan_feature_extraction import generate_dataset, generate_dataset_with_explanation
from train import k_fold_train
from train_with_explanation import k_fold_train_with_explanation


def main(
        dbname: str, user: str, password: str, host: str, port: str,
        query_file_path: str, explain_or_not: bool, save_model: bool, experiment_name: str, reqo_config: dict[str, Any]
):
    db_params: dict[str, str] = {
        "dbname": dbname,
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        # "options": "-c statement_timeout=600000"
    }

    # Generate the database statistic for the given database
    generate_postgresql_database_statistic(db_params)

    # Execute the queries with hints and store the generated query plans and labels
    generate_postgresql_workload_with_hints(db_params, query_file_path)

    # Encode query plans and generate the datasets
    if not explain_or_not:
        generate_dataset(dbname)
    else:
        generate_dataset_with_explanation(dbname)

    # Train and evaluate the model with k-fold cross validation
    if explain_or_not:
        k_fold_train(dbname, experiment_name, reqo_config, k=10, save_model=save_model)
    else:
        k_fold_train_with_explanation(dbname, experiment_name, reqo_config, k=10, save_model=save_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Database connection parameters'
    )
    parser.add_argument(
        '--dbname', type=str, required=True, help='Database name'
    )
    parser.add_argument(
        '--user', type=str, default='postgres', required=True, help='PostgreSQL user name'
    )
    parser.add_argument(
        '--password', type=str, required=True, help='PostgreSQL user password'
    )
    parser.add_argument(
        '--host', type=str, default='localhost', required=True, help='PostgreSQL server host'
    )
    parser.add_argument(
        '--port', type=str, default='5432', required=True, help='PostgreSQL server port'
    )
    parser.add_argument(
        '--query_file_path', type=str, required=True, help='Path to the query file'
    )
    parser.add_argument(
        '--experiment_name', type=str, required=True,
        help='Experiment name used under Data/<dbname>/datasets and Results/<dbname>.'
    )
    parser.add_argument(
        '--explain_or_not', type=bool, default=False, required=True,
        help='Whether to train the explainer for cost estimation'
    )
    parser.add_argument(
        '--save_model', type=bool, default=False, required=False, help='Whether to save the trained model'
    )

    args = parser.parse_args()
    reqo_config = {
        'batch_size': 128,
        'learning_rate': 0.001,
        'encoder_attention_heads': 1,
        'encoder_conv_layers': 3,
        'encoder_gnn_embedding_dim': 256,
        'encoder_gnn_dropout_rate': 0.1,
        'encoder_dirgnn_alpha': 0.3,
        'encoder_node_type_embedding_dim': 16,
        'encoder_column_embedding_dim': 8,
        'explainer_fcn_layers': 4,
        'explainer_explanation_embedding_size': 512,
        'explainer_fcn_dropout_rate': 0.1,
        'estimator_fcn_layers': 4,
        'estimator_estimation_embedding_size': 512,
        'estimator_fcn_dropout_rate': 0.1,
        'pairrankingloss_margin': 3
    }
    main(
        args.dbname, args.user, args.password, args.host, args.port,
        args.query_file_path, args.explain_or_not, args.save_model, args.experiment_name, reqo_config
    )

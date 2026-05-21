#!/usr/bin/env python3
"""
Run original Reqo training with custom dbname/config.

This file only calls the author's train.k_fold_train.
It does not redefine the model, loss, or training loop.
"""

import argparse
from train import k_fold_train


def build_reqo_config(args):
    """Build the config expected by the original train.py."""
    return {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "pairrankingloss_margin": args.pairrankingloss_margin,

        "encoder_attention_heads": args.encoder_attention_heads,
        "encoder_conv_layers": args.encoder_conv_layers,
        "encoder_gnn_embedding_dim": args.encoder_gnn_embedding_dim,
        "encoder_gnn_dropout_rate": args.encoder_gnn_dropout_rate,
        "encoder_dirgnn_alpha": args.encoder_dirgnn_alpha,
        "encoder_node_type_embedding_dim": args.encoder_node_type_embedding_dim,
        "encoder_column_embedding_dim": args.encoder_column_embedding_dim,

        "estimator_fcn_layers": args.estimator_fcn_layers,
        "estimator_estimation_embedding_dim": args.estimator_estimation_embedding_dim,
        "estimator_fcn_dropout_rate": args.estimator_fcn_dropout_rate,

        # Kept for compatibility. train.py without explanation does not use these directly.
        "explainer_fcn_layers": args.explainer_fcn_layers,
        "explainer_explanation_embedding_dim": args.explainer_explanation_embedding_dim,
        "explainer_fcn_dropout_rate": args.explainer_fcn_dropout_rate,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dbname", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--save-model", action="store_true")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--pairrankingloss-margin", type=float, default=0.0)

    parser.add_argument("--encoder-attention-heads", type=int, default=8)
    parser.add_argument("--encoder-conv-layers", type=int, default=4)
    parser.add_argument("--encoder-gnn-embedding-dim", type=int, default=256)
    parser.add_argument("--encoder-gnn-dropout-rate", type=float, default=0.1)
    parser.add_argument("--encoder-dirgnn-alpha", type=float, default=0.3)
    parser.add_argument("--encoder-node-type-embedding-dim", type=int, default=16)
    parser.add_argument("--encoder-column-embedding-dim", type=int, default=8)

    parser.add_argument("--estimator-fcn-layers", type=int, default=4)
    parser.add_argument("--estimator-estimation-embedding-dim", type=int, default=512)
    parser.add_argument("--estimator-fcn-dropout-rate", type=float, default=0.1)

    parser.add_argument("--explainer-fcn-layers", type=int, default=4)
    parser.add_argument("--explainer-explanation-embedding-dim", type=int, default=512)
    parser.add_argument("--explainer-fcn-dropout-rate", type=float, default=0.1)

    args = parser.parse_args()
    reqo_config = build_reqo_config(args)

    k_fold_train(
        dbname=args.dbname,
        reqo_config=reqo_config,
        k=args.k,
        save_model=args.save_model,
    )


if __name__ == "__main__":
    main()

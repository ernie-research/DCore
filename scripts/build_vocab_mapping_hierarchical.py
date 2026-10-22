#!/usr/bin/env python3
"""Build hierarchical vocab mapping: in-vocab d2t/t2d + OOV k-means clusters.

Usage:
    python scripts/build_vocab_mapping_hierarchical.py \
        --vocab-mapping cache/models/Qwen3-8B/vocab_mapping.pt \
        --lm-head-weight cache/models/Qwen3-8B/model.safetensors \
        --lm-head-key lm_head.weight \
        --num-clusters 256 \
        --output cache/models/Qwen3-8B/vocab_mapping_hierarchical.pt
"""

import argparse
import torch
import numpy as np
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-mapping", type=str, required=True)
    parser.add_argument("--lm-head-weight", type=str, required=True,
                        help="Path to safetensors/pt file containing lm_head.weight")
    parser.add_argument("--lm-head-key", type=str, default="lm_head.weight")
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load existing vocab mapping
    mapping = torch.load(args.vocab_mapping, map_location="cpu", weights_only=True)
    d2t = mapping["d2t"]  # [V_s]
    t2d = mapping["t2d"]  # [V_total]
    V_total = t2d.shape[0]
    V_s = d2t.shape[0]
    print(f"In-vocab size V_s={V_s}, total vocab V={V_total}")

    # Load lm_head weight
    if args.lm_head_weight.endswith(".safetensors"):
        with safe_open(args.lm_head_weight, framework="pt", device="cpu") as f:
            lm_head_weight = f.get_tensor(args.lm_head_key)
    else:
        state = torch.load(args.lm_head_weight, map_location="cpu", weights_only=True)
        lm_head_weight = state[args.lm_head_key] if isinstance(state, dict) else state
    print(f"lm_head_weight shape: {lm_head_weight.shape}")

    # Identify OOV tokens (t2d == -1)
    oov_mask = t2d < 0
    oov_token_ids = oov_mask.nonzero(as_tuple=True)[0]  # [num_oov]
    num_oov = oov_token_ids.shape[0]
    print(f"OOV tokens: {num_oov}")

    # Extract OOV embeddings and run k-means
    oov_embeddings = lm_head_weight[oov_token_ids].float().numpy()
    C = args.num_clusters

    from sklearn.cluster import MiniBatchKMeans
    print(f"Running MiniBatchKMeans with C={C}...")
    kmeans = MiniBatchKMeans(n_clusters=C, random_state=args.seed, batch_size=4096, max_iter=300)
    labels = kmeans.fit_predict(oov_embeddings)  # [num_oov]
    print("Clustering done.")

    # Build oov_cluster_ids: [V_total], in-vocab=-1, OOV=0..C-1
    oov_cluster_ids = torch.full((V_total,), -1, dtype=torch.long)
    oov_cluster_ids[oov_token_ids] = torch.from_numpy(labels.astype(np.int64))

    # Build cluster_offsets and cluster_token_ids (sorted by cluster)
    sorted_indices = np.argsort(labels)
    cluster_token_ids = oov_token_ids[sorted_indices]  # [num_oov], sorted by cluster
    # offsets: cluster i spans [offsets[i], offsets[i+1])
    cluster_offsets = torch.zeros(C + 1, dtype=torch.long)
    counts = np.bincount(labels, minlength=C)
    cluster_offsets[1:] = torch.from_numpy(np.cumsum(counts).astype(np.int64))

    # Save
    output = {
        "d2t": d2t,
        "t2d": t2d,
        "oov_cluster_ids": oov_cluster_ids,
        "cluster_offsets": cluster_offsets,
        "cluster_token_ids": cluster_token_ids,
    }
    torch.save(output, args.output)
    print(f"Saved hierarchical vocab mapping to {args.output}")
    print(f"  d2t: {d2t.shape}, t2d: {t2d.shape}")
    print(f"  oov_cluster_ids: {oov_cluster_ids.shape} (non -1: {(oov_cluster_ids >= 0).sum()})")
    print(f"  cluster_offsets: {cluster_offsets.shape}, cluster_token_ids: {cluster_token_ids.shape}")


if __name__ == "__main__":
    main()

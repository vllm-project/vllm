#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precompute SQuat calibration matrices for a given model.

Outputs a .pt file containing per-layer M_update correction matrices
used by the SQuat attention backend (--kv-cache-dtype squat).

Reference: Wang et al., "SQuat: Subspace-orthogonal KV Cache Quantization",
COLM 2025; preprint arXiv:2503.24358.

Usage:
    python tools/calibrate_squat.py \\
        --model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --subspace-dim 60 --lamb 0.001 \\
        --output rotations/model.pt
"""

import argparse
import math
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def build_hadamard(d: int) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction)."""
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def generate_At_inv(
    quant_group_size: int,
    Q_hat: torch.Tensor,
    lamb: float = 0.001,
    tol: float = 1e-7,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Generate correction matrices for block-wise SQuat quantization."""
    num_kv_heads, subspace_dim, head_dim = Q_hat.shape
    T = (head_dim + quant_group_size - 1) // quant_group_size
    device = Q_hat.device
    dtype = Q_hat.dtype

    eye = torch.eye(head_dim, device=device, dtype=dtype)
    A_T = eye.unsqueeze(0).expand(num_kv_heads, -1, -1) + lamb * torch.matmul(
        Q_hat.transpose(-1, -2), Q_hat
    )

    matrices = [None] * T
    matrices[T - 1] = A_T.clone()

    for t in range(T - 1, 0, -1):
        current_dim = t * quant_group_size
        M_t1 = A_T[:, :current_dim, :current_dim]
        N_t1 = A_T[:, current_dim : current_dim + quant_group_size, :current_dim]
        O_t1 = A_T[
            :,
            current_dim : current_dim + quant_group_size,
            current_dim : current_dim + quant_group_size,
        ]

        I_mat = torch.eye(quant_group_size, device=device, dtype=dtype)
        O_t1_inv = torch.inverse(
            O_t1 + tol * I_mat.unsqueeze(0).expand(num_kv_heads, -1, -1)
        )
        A_t = M_t1 - torch.matmul(N_t1.transpose(-1, -2), torch.matmul(O_t1_inv, N_t1))
        matrices[t - 1] = A_t[:, :, -quant_group_size:]
        A_T = A_t

    P_inv = torch.inverse(matrices[-1])
    return matrices, P_inv


def calibrate(
    model_name: str,
    subspace_dim: int = 60,
    lamb: float = 0.001,
    quant_group_size: int = 64,
    device: str = "cuda",
) -> dict:
    """Compute per-layer SQuat calibration matrices."""
    config = AutoConfig.from_pretrained(
        model_name, token=os.environ.get("HF_TOKEN"), trust_remote_code=True
    )
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    num_q_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_q_heads)
    hidden_size = config.hidden_size

    print(f"Model: {model_name}")
    print(
        f"  layers={num_layers}, kv_heads={num_kv_heads}, "
        f"q_heads={num_q_heads}, head_dim={head_dim}"
    )
    print(
        f"  subspace_dim={subspace_dim}, lambda={lamb}, "
        f"quant_group_size={quant_group_size}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        token=os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )

    H = build_hadamard(head_dim).to(device)
    rotations = {}
    heads_per_kv = num_q_heads // num_kv_heads

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        W_q = layer.self_attn.q_proj.weight.data.float().to(device)
        W_q_heads = W_q.view(num_q_heads, head_dim, hidden_size)

        Q_hat_all = []
        for kv_idx in range(num_kv_heads):
            start = kv_idx * heads_per_kv
            W_group = W_q_heads[start : start + heads_per_kv]

            cov = torch.zeros(head_dim, head_dim, dtype=torch.float64, device=device)
            for h in range(heads_per_kv):
                W_h = W_group[h].double()
                cov += W_h @ W_h.T

            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            r = min(subspace_dim, head_dim)
            top_eigenvalues = eigenvalues[-r:]
            top_eigenvectors = eigenvectors[:, -r:]

            Q_hat_kv = (
                top_eigenvalues.sqrt().unsqueeze(-1) * top_eigenvectors.T
            ) @ H.double()

            Q_hat_all.append(Q_hat_kv)

        Q_hat_stacked = torch.stack(Q_hat_all, dim=0)
        Ainv_t, P_inv = generate_At_inv(quant_group_size, Q_hat_stacked, lamb=lamb)

        g = quant_group_size
        T = (head_dim + g - 1) // g
        M_update = None
        if T == 2:
            H_t = Ainv_t[0]
            B_t = P_inv[:, g:, :g]
            M_update = torch.matmul(H_t.transpose(-2, -1), B_t.transpose(-2, -1))

        rotations[f"layer_{layer_idx}"] = {
            **({"M_update": M_update.float().cpu()} if M_update is not None else {}),
        }

        if (layer_idx + 1) % 8 == 0 or layer_idx == num_layers - 1:
            print(f"  Calibrated {layer_idx + 1}/{num_layers} layers")

    del model
    torch.accelerator.empty_cache()
    return rotations


def main():
    parser = argparse.ArgumentParser(description="Calibrate SQuat rotation matrices")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--subspace-dim", type=int, default=60)
    parser.add_argument("--lamb", type=float, default=0.001)
    parser.add_argument("--quant-group-size", type=int, default=64)
    parser.add_argument("--output", required=True, help="Output .pt file")
    args = parser.parse_args()

    rotations = calibrate(
        model_name=args.model,
        subspace_dim=args.subspace_dim,
        lamb=args.lamb,
        quant_group_size=args.quant_group_size,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(rotations, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

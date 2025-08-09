# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.linear_attn import LinearAttentionBackend
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionBackend
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionBackend


def get_mamba_attn_backend(mamba_type: str) -> type[AttentionBackend]:
    if mamba_type == "mamba1":
        return Mamba1AttentionBackend
    if mamba_type == "mamba2":
        return Mamba2AttentionBackend
    if mamba_type == "linear_attention":
        return LinearAttentionBackend

    raise NotImplementedError(f"Mamba Attention type {mamba_type} is not "
                              "supported yet.")

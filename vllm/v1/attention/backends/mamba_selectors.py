# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.attention.backends.abstract import AttentionBackend
from vllm.v1.attention.backends.mamba_attn import Mamba2AttentionBackend


def get_mamba_attn_backend(mamba_type: str) -> type[AttentionBackend]:
    if mamba_type in ("mamba2", "short_conv"):
        return Mamba2AttentionBackend

    raise NotImplementedError(f"Mamba Attention type {mamba_type} is not "
                              "supported yet.")

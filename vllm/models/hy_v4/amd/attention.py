# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention as BaseAttention

from .rocm import HYV4ROCMAiterMLASparseBackend


class HYV4MLAAttention(BaseAttention):
    """HY V4 MLA attention using the sink-capable ROCm AITER backend."""

    def _resolve_sink_backend(self, kv_cache_dtype: str):
        return HYV4ROCMAiterMLASparseBackend

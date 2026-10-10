# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve the ROCm import path for the shared Triton decode implementation."""

from vllm.models.inkling.common.ops.triton_rel_attention_decode import (
    decode_split_count as decode_split_count,
)
from vllm.models.inkling.common.ops.triton_rel_attention_decode import (
    inkling_rel_attention_split_kv_decode as inkling_rel_attention_split_kv_decode,
)
from vllm.models.inkling.common.ops.triton_rel_attention_decode import (
    use_split_kv_decode as use_split_kv_decode,
)

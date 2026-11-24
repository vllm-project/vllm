# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.models.hy_v4.common.dfly_speculator import (
    HYV4DFlySpeculator as _HYV4DFlySpeculator,
)

from .attention import HYV4MLAAttention
from .dfly import HYV4DFlyForCausalLM


class HYV4DFlySpeculator(_HYV4DFlySpeculator):
    draft_model_cls = HYV4DFlyForCausalLM  # type: ignore[assignment]
    attention_cls = HYV4MLAAttention

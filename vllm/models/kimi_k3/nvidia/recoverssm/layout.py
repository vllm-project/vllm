# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KDA Mamba page layout with RecoverSSM.

Each KDA layer's Mamba page holds the conv state and the SSM state (one slot per
request with RecoverSSM). Unless the instance is prefill-only, RecoverSSM's
verify records follow them in the same page: per head and verified token, the
FP32 correction ``u_t`` [V] and the activation-dtype key and raw gate [2K]. A
verify step writes them and the commit after sampling reads them.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import CacheConfig
from vllm.config.model import ModelDType
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.interfaces import MambaStateShapes


@dataclass(frozen=True)
class RecoverSSMRecords:
    """One KDA layer's verify records, views of its Mamba pages."""

    correction: torch.Tensor  # [slots, H, Q, V] float32
    key_gate: torch.Tensor  # [slots, H, Q, 2K] activation dtype


def recoverssm_records_in_page(cache_config: CacheConfig) -> bool:
    """Whether the KDA Mamba pages hold RecoverSSM's verify records."""
    return (
        cache_config.use_kda_recoverssm and not cache_config.kda_recoverssm_prefill_only
    )


def kda_state_dtypes(
    model_dtype: ModelDType | torch.dtype, cache_config: CacheConfig
) -> tuple[torch.dtype, ...]:
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(
        model_dtype,
        cache_config.mamba_cache_dtype,
        cache_config.mamba_ssm_cache_dtype,
    )
    if recoverssm_records_in_page(cache_config):
        dtypes = MambaStateDtypeCalculator.append_kda_recoverssm_record(
            dtypes, model_dtype
        )
    return dtypes


def kda_state_shapes(
    cache_config: CacheConfig,
    *,
    tp_size: int,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
    num_spec: int,
) -> "MambaStateShapes":
    shapes = MambaStateShapeCalculator.kda_state_shape(
        tp_size,
        num_heads,
        head_dim,
        conv_kernel_size=conv_kernel_size,
        num_spec=num_spec,
    )
    if not recoverssm_records_in_page(cache_config):
        return shapes
    return MambaStateShapeCalculator.append_kda_recoverssm_record(
        shapes,
        num_heads,
        head_dim,
        tp_world_size=tp_size,
        spec_query_len=1 + num_spec,
    )

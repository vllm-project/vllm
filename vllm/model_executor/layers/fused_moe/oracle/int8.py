# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)

logger = init_logger(__name__)


def select_int8_moe_backend(
    config: FusedMoEConfig,
) -> type[mk.FusedMoEExperts]:
    from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

    supported, reason = TritonExperts.is_supported_config(
        TritonExperts,
        config,
        None,
        None,
        mk.FusedMoEActivationFormat.Standard,
    )
    if not supported:
        raise ValueError(
            f"INT8 Triton MoE backend does not support the "
            f"deployment configuration: {reason}"
        )

    logger.info_once("Using Triton INT8 MoE backend", scope="local")
    return TritonExperts


def make_int8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FusedMoEQuantConfig:
    return int8_w8a16_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=None,
        w2_zp=None,
    )


def make_int8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: SharedExperts | None = None,
) -> mk.FusedMoEKernel:
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
    )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=shared_experts,
        inplace=not moe_config.disable_inplace,
    )

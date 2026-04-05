# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import Enum

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


class Int8MoeBackend(Enum):
    TRITON = "TRITON"


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> type[mk.FusedMoEExperts]:
    if backend == Int8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts

        return TritonExperts

    raise ValueError(f"Unknown INT8 MoE backend: {backend.value}")


def select_int8_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """Select INT8 W8A16 MoE backend."""
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    backend = Int8MoeBackend.TRITON
    experts_cls = backend_to_kernel_cls(backend)

    # INT8 passes (None, None) for quant keys — TritonExperts accepts this.
    supported, reason = experts_cls.is_supported_config(
        experts_cls, config, None, None, activation_format
    )
    if not supported:
        raise ValueError(
            f"INT8 MoE backend {backend.value} does not support the "
            f"deployment configuration: {reason}"
        )

    logger.info_once("Using %s INT8 MoE backend", backend.value, scope="local")
    return backend, experts_cls


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
    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    # Create Experts
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=shared_experts,
        inplace=not moe_config.disable_inplace and not is_monolithic,
    )

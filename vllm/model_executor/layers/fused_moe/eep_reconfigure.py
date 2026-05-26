# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import signature
from typing import TYPE_CHECKING, Any

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def _make_eep_experts(
    quant_method: FusedMoEMethodBase,
    source_experts: FusedMoEExpertsModular,
    prepare_finalize: FusedMoEPrepareAndFinalizeModular,
    moe_config: FusedMoEConfig,
) -> FusedMoEExpertsModular:
    experts_cls = source_experts.__class__
    assert quant_method.moe_quant_config is not None
    experts_kwargs: dict[str, Any] = {
        "moe_config": moe_config,
        "quant_config": quant_method.moe_quant_config,
    }
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts_kwargs.update(
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )

    # Expert kernels with extra init params need explicit EEP support.
    generic_arg_names = set(signature(mk.FusedMoEExperts.__init__).parameters)
    ctor_arg_names = set(signature(experts_cls.__init__).parameters)
    unsupported_args = ctor_arg_names - generic_arg_names
    missing_args = set(experts_kwargs) - ctor_arg_names
    if unsupported_args or missing_args:
        raise NotImplementedError(
            f"{experts_cls.__name__} experts do not support Elastic EP."
        )

    return experts_cls(**experts_kwargs)


def make_eep_staged_quant_method(
    module: "FusedMoE",
    moe_config: FusedMoEConfig,
) -> FusedMoEMethodBase | None:
    quant_method = module.quant_method
    if not quant_method.supports_internal_mk:
        return None
    if getattr(quant_method, "wraps_legacy_quant_method", False):
        return None

    old_batched_format = (
        module.moe_config.moe_parallel_config.use_batched_activation_format
    )
    new_batched_format = moe_config.moe_parallel_config.use_batched_activation_format
    assert old_batched_format == new_batched_format

    moe_kernel = quant_method.moe_kernel
    if moe_kernel is None:
        return None
    if moe_kernel.is_monolithic:
        raise NotImplementedError(
            "Elastic EP full modular-kernel staging is not supported for "
            "monolithic fused MoE kernels."
        )
    if quant_method.moe_quant_config is None:
        raise ValueError(
            "Elastic EP full modular-kernel staging requires initialized "
            "MoE quant config."
        )

    prepare_finalize = maybe_make_prepare_finalize(
        moe_config,
        quant_method.moe_quant_config,
        routing_tables=None,
        allow_new_interface=True,
        use_monolithic=quant_method.is_monolithic,
        eep_stage=True,
    )
    assert prepare_finalize is not None
    assert isinstance(prepare_finalize, FusedMoEPrepareAndFinalizeModular)

    source_experts = moe_kernel.fused_experts
    assert isinstance(source_experts, FusedMoEExpertsModular)

    experts = _make_eep_experts(
        quant_method,
        source_experts,
        prepare_finalize,
        moe_config,
    )

    if isinstance(quant_method, FusedMoEModularMethod):
        base_quant_method = quant_method.old_quant_method
    else:
        base_quant_method = quant_method

    return FusedMoEModularMethod(
        base_quant_method,
        mk.FusedMoEKernel(
            prepare_finalize,
            experts,
            inplace=moe_kernel.inplace,
        ),
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)

logger = init_logger(__name__)


class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    HUMMING = "HUMMING"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[Int8MoeBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    return [Int8MoeBackend.TRITON, Int8MoeBackend.HUMMING]


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Int8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    elif backend == Int8MoeBackend.HUMMING:
        import vllm.model_executor.layers.fused_moe.fused_humming_moe as humming_moe

        return [
            humming_moe.BatchedHummingGroupedExperts,
            humming_moe.HummingGroupedExperts,
            humming_moe.HummingIndexedExperts,
        ]

    else:
        raise ValueError(f"Unknown Int8 MoE backend: {backend.value}")


def map_int8_backend(runner_backend: MoEBackend) -> Int8MoeBackend:
    """Map user's MoEBackend to Int8MoeBackend."""
    mapping = {
        "triton": Int8MoeBackend.TRITON,
        "humming": Int8MoeBackend.HUMMING,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for Int8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt8StaticChannelSym,
    activation_key: QuantKey | None = kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the primary Int8 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    if config.is_lora_enabled:
        return Int8MoeBackend.TRITON, backend_to_kernel_cls(Int8MoeBackend.TRITON)[0]

    AVAILABLE_BACKENDS = _get_priority_backends(config)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Int8MoeBackend) -> str:
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Int8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Int8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Int8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"Int8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: Int8MoeBackend,
    ) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_int8_backend(runner_backend)
        return _return_or_raise(requested_backend)

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Int8 MoE backend supports the deployment configuration."
    )


def make_int8_moe_quant_config(
    int8_backend: Int8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    layer: torch.nn.Module | None = None,
) -> FusedMoEQuantConfig:
    assert (a1_scale is None and a2_scale is None) or (
        a1_scale is not None and a2_scale is not None
    ), "a1_scale and a2_scale must both be provided or both be None"

    if int8_backend == Int8MoeBackend.HUMMING:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            get_humming_moe_quant_config,
        )

        assert isinstance(layer, FusedMoE)
        return get_humming_moe_quant_config(layer)

    assert int8_backend == Int8MoeBackend.TRITON

    if a1_scale is None or a2_scale is None:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
        )

    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        per_act_token_quant=per_act_token_quant,
    )


def convert_to_int8_moe_kernel_format(
    int8_backend: Int8MoeBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_input_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int8_backend == Int8MoeBackend.HUMMING:
        from vllm.model_executor.layers.quantization.utils.humming_utils import (
            prepare_humming_moe_layer,
        )

        quant_method_name = layer.quant_method.__class__.__name__
        if "CompressedTensors" in quant_method_name:
            from compressed_tensors.quantization import QuantizationArgs

            weight_quant = getattr(layer.quant_method, "weight_quant", None)
            assert isinstance(weight_quant, QuantizationArgs)
            quant_config = weight_quant.model_dump()
            quant_config["quant_method"] = "compressed-tensors"
            quant_config["format"] = "int-quantized"
        else:
            assert "Int8Online" in quant_method_name
            replace_parameter(layer, "w13_weight", (w13 + 128).view(torch.int32))
            replace_parameter(layer, "w2_weight", (w2 + 128).view(torch.int32))
            w13_scale = w13_scale.to(layer.params_dtype).unsqueeze(-1)
            w2_scale = w2_scale.to(layer.params_dtype).unsqueeze(-1)
            layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
            quant_config = {"quant_method": "humming", "dtype": "int8"}

        prepare_humming_moe_layer(layer, quant_config)

        return (
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )

    return w13, w2, w13_scale, w2_scale


def make_int8_moe_kernel(
    int8_backend: Int8MoeBackend,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: SharedExperts | None = None,
    layer: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    extra_kwargs = {}
    if int8_backend == Int8MoeBackend.HUMMING:
        assert layer is not None
        extra_kwargs = {"layer": layer}

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
            **extra_kwargs,
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            **extra_kwargs,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=shared_experts,
        inplace=not moe_config.disable_inplace,
    )

    return kernel

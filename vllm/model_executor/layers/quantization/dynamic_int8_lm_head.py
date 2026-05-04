# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic INT8 quantization for the lm_head layer.

When enabled via ``--dynamic-lm-head-quantization int8``, the lm_head weights
are quantized to per-channel symmetric INT8 at model-load time and dispatched
through the kernel selected by ``choose_mp_linear_kernel``.  The original
FP16/BF16 weights are kept for embedding lookups in weight-tied models.
"""

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.layer_utils import (
    replace_parameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types


def should_use_dynamic_int8_lm_head(embedding_dim: int) -> bool:
    """Return True when the dynamic INT8 lm_head path should be used."""
    import logging

    from vllm.config import get_current_vllm_config

    logger = logging.getLogger(__name__)

    model_config = get_current_vllm_config().model_config
    if model_config.dynamic_lm_head_quantization != "int8":
        return False

    # Probe whether any kernel can handle signed int8 per-channel for this
    # embedding_dim.  Use a dummy output dim since can_implement only checks
    # K (partition_weight_shape[0]).
    probe_config = MPLinearLayerConfig(
        full_weight_shape=(embedding_dim, 1024),
        partition_weight_shape=(embedding_dim, 1024),
        weight_type=scalar_types.int8,
        act_type=torch.float16,
        group_size=-1,
        zero_points=False,
        has_g_idx=False,
    )
    try:
        kernel_cls = choose_mp_linear_kernel(probe_config)
    except (ValueError, KeyError):
        logger.warning(
            "dynamic_int8_lm_head: requested but no kernel available "
            "for embedding_dim=%d on this platform; "
            "falling back to default lm_head",
            embedding_dim,
        )
        return False

    logger.info(
        "dynamic_int8_lm_head: ENABLED for embedding_dim=%d (kernel: %s)",
        embedding_dim,
        kernel_cls.__name__,
    )
    return True


class DynamicInt8LMHeadMethod(QuantizeMethodBase):
    """Quantize the lm_head to INT8 at load time, dispatch via MPLinearKernel."""

    _w_orig: torch.Tensor

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [M, K] in FP16/BF16
        act_dtype = weight.dtype

        # Keep original weights for fallback and embedding
        self._w_orig = weight.contiguous()

        # Per-channel symmetric INT8 quantization (absmax scaling).
        # This is the simplest correct scheme: scale = max(|w|) / 127 per
        # output channel.  More sophisticated approaches (percentile
        # clipping, learned scales) could improve accuracy for outlier-heavy
        # distributions but would add complexity and latency at load time
        # with marginal benefit — the lm_head is applied once per token.
        scales = weight.abs().amax(dim=1).clamp(min=1e-10) / 127.0
        q = (weight / scales.unsqueeze(1)).round().clamp(-128, 127).to(torch.int8)

        replace_parameter(layer, "weight", q)
        layer.register_parameter(
            "weight_scale",
            Parameter(scales.to(act_dtype), requires_grad=False),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Ensure the _rocm_skinny_w8 custom op is registered.
        import vllm.model_executor.kernels.linear.mixed_precision.hip_w8a16  # noqa: F401
        from vllm.utils.platform_utils import num_compute_units

        w_q = layer.weight
        w_s = layer.weight_scale
        x_2d = x.reshape(-1, x.shape[-1])
        M = w_q.shape[0]
        out_shape = x.shape[:-1] + (M,)

        N = x_2d.shape[0]
        K = x_2d.shape[1]
        ctx = (
            nullcontext()
            if torch.compiler.is_compiling()
            else torch.profiler.record_function(f"dynamic_int8_lm_head {N}x{M}x{K}")
        )
        with ctx:
            cu_count = num_compute_units()
            output = torch.ops._rocm_skinny_w8.w8a16_apply(
                x_2d,
                w_q,
                w_s,
                self._w_orig,
                bias,
                cu_count,
            )
        return output.reshape(out_shape)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, self._w_orig)

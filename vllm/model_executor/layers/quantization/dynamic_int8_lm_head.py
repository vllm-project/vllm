# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dynamic INT8 quantization for the lm_head layer.

When enabled via ``--dynamic-lm-head-quantization int8`` the lm_head weights
are quantized to per-channel symmetric INT8 at model-load time and dispatched
through the kernel selected by ``choose_mp_linear_kernel``.  The original
FP16/BF16 weights are kept for embedding lookups in weight-tied models.

Pass ``--dynamic-lm-head-quantization int8:gN`` (e.g. ``int8:g32``,
``int8:g64``, ``int8:g128``) to switch to per-group-N symmetric INT8
quantization along K.  The grouped path uses the ``wvSplitK_int8`` ROCm
kernel extended to absorb a 2-D ``[M, K/G]`` scale tensor inside the K
reduction loop.

Debug fallback ``VLLM_DYNAMIC_INT8_LM_HEAD_SIM=1`` dispatches the grouped
weight via bf16 ``dispatch_unquantized_gemm`` after a quantize-dequantize
round-trip (perf-neutral, accuracy-equivalent — used to validate quant
schemes before kernel work).
"""

import os
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


def _parse_int8_group_size(spec: str | None) -> int | None:
    """Parse the ``--dynamic-lm-head-quantization`` value.

    Accepted INT8 forms:
      * ``"int8"`` → ``-1`` (per-channel, production kernel path)
      * ``"int8:gN"`` (e.g. ``int8:g32``) → positive int N (per-group along K)

    Returns ``None`` if ``spec`` is not in the int8 family (caller should
    treat this as "do not engage the dynamic int8 lm_head path"). Raises
    ``ValueError`` for malformed ``int8:*`` strings.
    """
    if spec is None or spec == "":
        return None
    if spec == "int8":
        return -1
    if not spec.startswith("int8:"):
        return None
    suffix = spec[len("int8:") :]
    if not suffix.startswith("g"):
        raise ValueError(
            f"Unsupported --dynamic-lm-head-quantization spec '{spec}'. "
            f"Use 'int8' for per-channel or 'int8:gN' for per-group "
            f"(e.g. int8:g32, int8:g64, int8:g128)."
        )
    try:
        g = int(suffix[1:])
    except ValueError as exc:
        raise ValueError(
            f"Cannot parse group size from '{spec}'. "
            f"Expected 'int8:gN' where N is a positive integer."
        ) from exc
    if g <= 0:
        raise ValueError(
            f"--dynamic-lm-head-quantization '{spec}' has non-positive "
            f"group size {g}; must be > 0."
        )
    return g


def _resolve_group_size(spec: str | None) -> int:
    """Resolve the effective group size from the CLI spec.

    Returns ``-1`` for per-channel INT8 (when spec is ``"int8"`` or absent),
    or a positive int (e.g. 32, 64, 128) for per-group INT8.
    """
    parsed = _parse_int8_group_size(spec)
    return parsed if parsed is not None else -1


def _use_simulation_fallback() -> bool:
    """Phase-1 quantize-dequantize-bf16 fallback.

    Set ``VLLM_DYNAMIC_INT8_LM_HEAD_SIM=1`` to dispatch via bf16
    ``dispatch_unquantized_gemm`` (no kernel call, perf-neutral but exercises
    exactly the same numerics as the Phase-1 accuracy-simulation gate).
    """
    return os.environ.get("VLLM_DYNAMIC_INT8_LM_HEAD_SIM", "0").strip() == "1"


def should_use_dynamic_int8_lm_head(embedding_dim: int) -> bool:
    """Return True when the dynamic INT8 lm_head path should be used."""
    import logging

    from vllm.config import get_current_vllm_config

    logger = logging.getLogger(__name__)

    model_config = get_current_vllm_config().model_config
    spec = model_config.dynamic_lm_head_quantization
    if _parse_int8_group_size(spec) is None:
        # Spec is not in the int8 family.
        return False

    group_size = _resolve_group_size(spec)
    sim_mode = _use_simulation_fallback()

    if group_size > 0 and embedding_dim % group_size != 0:
        logger.warning(
            "dynamic_int8_lm_head: group_size=%d does not divide "
            "embedding_dim=%d; falling back to default lm_head",
            group_size,
            embedding_dim,
        )
        return False

    if group_size > 0 and sim_mode:
        # Phase-1 accuracy-simulation fallback: weight is quantize-dequantized
        # in process_weights_after_loading and dispatched via bf16 GEMM.
        logger.info(
            "dynamic_int8_lm_head: ENABLED for embedding_dim=%d "
            "(group_size=%d, INT8 ACCURACY-SIMULATION mode — "
            "uses bf16 GEMM, no perf benefit)",
            embedding_dim,
            group_size,
        )
        return True

    # Probe whether any kernel can handle int8 (per-channel or per-group) for
    # this embedding_dim.  Use a dummy output dim since can_implement only
    # checks K (partition_weight_shape[0]).
    probe_config = MPLinearLayerConfig(
        full_weight_shape=(embedding_dim, 1024),
        partition_weight_shape=(embedding_dim, 1024),
        weight_type=scalar_types.int8,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )
    try:
        kernel_cls = choose_mp_linear_kernel(probe_config)
    except (ValueError, KeyError):
        logger.warning(
            "dynamic_int8_lm_head: requested but no kernel available "
            "for embedding_dim=%d group_size=%d on this platform; "
            "falling back to default lm_head",
            embedding_dim,
            group_size,
        )
        return False

    logger.info(
        "dynamic_int8_lm_head: ENABLED for embedding_dim=%d "
        "(group_size=%d, kernel: %s)",
        embedding_dim,
        group_size,
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
        from vllm.config import get_current_vllm_config

        weight = layer.weight.data  # [M, K] in FP16/BF16
        act_dtype = weight.dtype

        # Keep original weights for fallback and embedding
        self._w_orig = weight.contiguous()

        vllm_config = get_current_vllm_config()
        spec = (
            vllm_config.model_config.dynamic_lm_head_quantization
            if vllm_config.model_config
            else None
        )
        group_size = _resolve_group_size(spec)
        sim_mode = _use_simulation_fallback()
        self._group_size = group_size
        self._sim_mode = sim_mode

        if group_size > 0 and sim_mode:
            # Phase-1 accuracy-simulation: per-group INT8 quant-dequant back
            # to bf16; apply path dispatches through dispatch_unquantized_gemm.
            M, K = weight.shape
            assert K % group_size == 0, (
                f"K={K} not divisible by group_size={group_size}"
            )
            w_g = weight.reshape(M, K // group_size, group_size)
            scales = w_g.abs().amax(dim=2).clamp(min=1e-10) / 127.0  # [M, K/G]
            q = (w_g / scales.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
            w_dequant = (
                (q.to(act_dtype) * scales.unsqueeze(-1).to(act_dtype))
                .reshape(M, K)
                .contiguous()
            )
            replace_parameter(layer, "weight", w_dequant)
            layer.register_parameter(
                "weight_scale",
                Parameter(
                    scales.to(act_dtype).reshape(M, K // group_size),
                    requires_grad=False,
                ),
            )
            import logging

            logging.getLogger(__name__).info(
                "dynamic_int8_lm_head SIM: INT8 per-group-%d round-trip "
                "— bf16 dequant weight stored",
                group_size,
            )
            return

        if group_size > 0:
            # Per-group INT8 quant: store INT8 weight and 2-D [M, K/G] scale.
            # Apply path dispatches via the wvSplitK_int8 kernel extended in
            # META3-5-ALT-PHASE2 to absorb the 2-D scale tensor.
            M, K = weight.shape
            assert K % group_size == 0, (
                f"K={K} not divisible by group_size={group_size}"
            )
            w_g = weight.reshape(M, K // group_size, group_size)
            scales = w_g.abs().amax(dim=2).clamp(min=1e-10) / 127.0  # [M, K/G]
            q = (
                (w_g / scales.unsqueeze(-1))
                .round()
                .clamp(-128, 127)
                .to(torch.int8)
                .reshape(M, K)
                .contiguous()
            )
            replace_parameter(layer, "weight", q)
            layer.register_parameter(
                "weight_scale",
                Parameter(scales.to(act_dtype).contiguous(), requires_grad=False),
            )
            return

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
        # Phase-1 simulation fallback: weight is already dequantized bf16/fp16.
        # Dispatch via the same path the unquantized lm_head uses so we get
        # the ROCm-optimized custom op (rocm_unquantized_gemm) under
        # torch.compile + cudagraph instead of plain F.linear.
        if getattr(self, "_sim_mode", False):
            from vllm.model_executor.layers.utils import dispatch_unquantized_gemm

            return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)

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
        group_size = getattr(self, "_group_size", -1)
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
                group_size,
            )
        return output.reshape(out_shape)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, self._w_orig)

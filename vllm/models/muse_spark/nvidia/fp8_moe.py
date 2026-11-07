# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x online FP8 routed experts (``fp8_per_channel``) on Triton."""

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoeWeightScaleSupported,
    RoutedExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    make_fp8_moe_kernel,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PtpcOnlineMoEMethod,
    _fp8_channel_scale,
)
from vllm.model_executor.layers.quantization.online.moe_base import (
    OnlineMoEMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    weight_amax,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MusePostExpertFinalize,
    MuseSparkUnquantizedFusedMoE,
    _attach_post_expert_finalize,
    _validate_route_major_deployment,
)
from vllm.platforms import current_platform


def quantize_fp8_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 values and fp32 scales, one scale per row (output channel).

    fp8_per_channel's arithmetic without its in-place chunk copies: the
    layerwise reload counts what a loader stores by its ``copy_`` calls.
    """
    scale = _fp8_channel_scale(weight_amax(weight, dim=-1, keepdim=True))
    fp8_min, fp8_max = get_fp8_min_max()
    values = (weight / scale).clamp_(fp8_min, fp8_max)
    return values.to(current_platform.fp8_dtype()), scale


def online_quant_method(
    quant_config: OnlineQuantizationConfig,
    layer_cls: type[RoutedExperts],
    prefix: str,
    moe_config: FusedMoEConfig,
) -> FusedMoEMethodBase:
    """The experts' method under an online config: FP8, or BF16 if ignored."""
    resolved = quant_config.resolve_quant_method_cls(layer_cls, prefix)
    if resolved is None:
        return MuseSparkUnquantizedFusedMoE(moe_config)
    source, quant_key_str, target_pattern, _, method_cls = resolved
    if method_cls is not Fp8PtpcOnlineMoEMethod:
        raise NotImplementedError(
            "MuseSpark1x online MoE quantization supports "
            f"fp8_per_channel only; got {quant_key_str}."
        )
    # Counted in the online quantization log, as get_quant_method does.
    quant_config.quantized_layers[prefix] = (
        source.value,
        quant_key_str,
        target_pattern,
    )
    return MuseSparkFp8FusedMoE(moe_config)


class MuseSparkFp8FusedMoE(Fp8PtpcOnlineMoEMethod):
    """Online FP8 routed experts on Triton (``fp8_per_channel``): per-output-
    channel weight scales, dynamic per-token activation scales.

    Unlike the generic online method, which loads BF16 onto the meta device and
    quantizes each layer once, every expert is quantized as it loads, straight
    into FP8 parameters allocated at startup (see ``quantize_on_load``). Any
    later ``load_weights``, an RL weight update direct or through the layerwise
    reload, re-quantizes in place: no BF16 copy of the experts and no new
    tensors under captured CUDA graphs.
    """

    uses_meta_device = False
    muse_post_expert_finalize: MusePostExpertFinalize

    def __init__(self, moe: FusedMoEConfig) -> None:
        # The post-expert norm needs route-major GEMM2 rows, which the Triton
        # experts hand back; the oracle would pick CUTLASS on Hopper.
        _validate_route_major_deployment(moe)
        supported, reason = TritonExperts.is_supported_config(
            TritonExperts,
            moe,
            kFp8StaticChannelSym,
            kFp8DynamicTokenSym,
            FusedMoEActivationFormat.Standard,
        )
        if not supported:
            raise NotImplementedError(
                f"MuseSpark FP8 MoE runs on Triton experts, which reject this "
                f"deployment: {reason}"
            )
        OnlineMoEMethodBase.__init__(self, moe)
        self.weight_block_size = None
        self.block_quant = False
        self.weight_scale_name = "weight_scale"
        self.fp8_backend = Fp8MoeBackend.TRITON
        self.experts_cls = TritonExperts

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        if self.moe.has_bias:
            raise NotImplementedError("MuseSpark FP8 MoE has no expert biases.")
        w13_rows = self.moe.w13_num_shards * intermediate_size_per_partition
        for name, shape in (
            ("w13", (num_experts, w13_rows, hidden_size)),
            ("w2", (num_experts, hidden_size, intermediate_size_per_partition)),
        ):
            weight = torch.nn.Parameter(
                torch.empty(shape, dtype=current_platform.fp8_dtype()),
                requires_grad=False,
            )
            layer.register_parameter(f"{name}_weight", weight)
            set_weight_attrs(weight, extra_weight_attrs)
            scale = torch.nn.Parameter(
                torch.ones(*shape[:2], 1, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter(f"{name}_weight_scale", scale)
            set_weight_attrs(
                scale,
                {
                    **extra_weight_attrs,
                    "quant_method": FusedMoeWeightScaleSupported.CHANNEL.value,
                },
            )
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def quantize_on_load(
        self,
        layer: RoutedExperts,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
    ) -> bool | None:
        """Quantize one BF16 expert weight (w1, w3 or w2) into this rank's FP8
        parameter and scales. Returns whether the expert is local, or ``None``
        for any other weight, which the default loader takes."""
        if (
            weight_name not in ("w13_weight", "w2_weight")
            or loaded_weight.dtype == param.dtype
        ):
            return None
        local_id = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local_id == -1:
            return False
        # Cut this rank's TP slice before moving it to the device, as
        # RoutedExperts._load_w13 / _load_w2 slice it: w1/w3 rows (column
        # parallel), w2 input columns (row parallel). A w2 row's scale thus
        # spans this rank's slice, as when fp8_per_channel quantizes a
        # RowParallelLinear shard.
        shard_dim = 1 if shard_id == "w2" else 0
        tp_size = layer.moe_config.moe_parallel_config.tp_size
        size = loaded_weight.shape[shard_dim] // tp_size
        local = loaded_weight.narrow(shard_dim, size * layer.moe_config.tp_rank, size)
        values, scale = quantize_fp8_rows(local.to(param.device))
        # Already sliced: the base loaders only place w1/w3 in their half.
        scale_param = getattr(layer, f"{weight_name}_scale")
        for dst, src in ((param, values), (scale_param, scale)):
            if shard_id == "w2":
                layer._load_w2(
                    expert_data=dst.data[local_id],
                    shard_dim=shard_dim,
                    loaded_weight=src,
                    tp_rank=layer.moe_config.tp_rank,
                    load_full=True,
                )
            else:
                layer._load_w13(
                    expert_data=dst.data[local_id],
                    shard_dim=shard_dim,
                    shard_id=shard_id,
                    loaded_weight=src,
                    tp_rank=layer.moe_config.tp_rank,
                    load_full=True,
                )
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The weights were quantized in place, in the layout the Triton experts
        # read, so the kernel is built once and kept across weight updates.
        if self.moe_kernel is not None:
            return
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_fp8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=TritonExperts,
            fp8_backend=self.fp8_backend,
            routing_tables=layer._expert_routing_tables(),
        )
        _attach_post_expert_finalize(self)

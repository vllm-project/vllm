# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp routed experts on DeepGEMM MegaMoE.

Routed experts load in the checkpoint's format (BF16, 128x128 block-FP8, MXFP8
or MXFP4); block-FP8 experts run as stored on SM90 and as MXFP8 on SM100.
`--quantization` / `--quantization-config` can force MXFP8 or MXFP4, and the
experts are then requantized as they load. The MTP drafter's config does not
carry the online config, so its experts take the default format. The
sigmoid-gated shared expert stays unfused and runs on an aux stream next to the
router.
"""

from collections.abc import Iterable

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightFormat,
    mega_moe_weight_format,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.config_utils import (
    get_checkpoint_quant_keys,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    break_fp4_bytes,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kMxfp4Static,
    kMxfp8Static,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)

# Per-expert tensors of one projection, by checkpoint expert format. Formats
# MegaMoE runs share the name of their MEGA_MOE_* format; "fp8_block" is the
# 128x128 block-FP8 layout of the official FP8 checkpoint.
_SHARD_TENSORS = {
    "bf16": ("weight",),
    "fp8_block": ("weight", "weight_scale_inv"),
    "mxfp8": ("weight", "weight_scale"),
    "mxfp4": ("weight", "weight_scale"),
}
_ONLINE_FORMATS = {kMxfp8Static: "mxfp8", kMxfp4Static: "mxfp4"}


def _requantize(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    checkpoint_format: str,
    weight_format: MegaMoEWeightFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize `[N, K]` checkpoint expert weights to MXFP8 or MXFP4.

    Dequantizes to FP32, rounds as DeepGEMM's own casts and returns the scales
    as `[N, K // 32]` UE8M0 exponents.
    """
    if checkpoint_format == "mxfp4":
        x = break_fp4_bytes(weight.view(torch.uint8), torch.float32)
    else:
        x = weight.float()
    if scale is not None:
        if checkpoint_format != "fp8_block":
            # UE8M0 exponents.
            scale = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
        rows, cols = x.shape
        scale = scale.float().repeat_interleave(rows // scale.shape[0], 0)
        x = x * scale.repeat_interleave(cols // scale.shape[1], 1)
    deep_gemm = _import_deep_gemm()
    cast = (
        deep_gemm.per_token_cast_to_fp4
        if weight_format.name == "mxfp4"
        else deep_gemm.per_token_cast_to_fp8
    )
    values, sf = cast(x, use_ue8m0=True, gran_k=32)
    # Power-of-two FP32 scales: the exponent bits are the UE8M0 value.
    return values, (sf.view(torch.int32) >> 23).to(torch.uint8)


def _checkpoint_expert_format(
    quant_config: QuantizationConfig | None, prefix: str
) -> str:
    """The format the checkpoint stores the routed experts at `prefix` in."""
    if quant_config is None or isinstance(quant_config, OnlineQuantizationConfig):
        return "bf16"
    if isinstance(quant_config, Fp8Config):
        if is_layer_skipped(
            prefix,
            quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
            match_mode=quant_config.ignored_layers_match_mode,
        ):
            return "bf16"
        if (
            quant_config.weight_block_size == [128, 128]
            and quant_config.store_dtype is None
        ):
            return "fp8_block"
        raise NotImplementedError(
            "Qwen4Exp MegaMoE reads FP8 experts with 128x128 block scales only."
        )
    formats = {None: "bf16", kMxfp8Static: "mxfp8", kMxfp4Static: "mxfp4"}
    try:
        weight_key, _ = get_checkpoint_quant_keys(quant_config, prefix, RoutedExperts)
        return formats[weight_key]
    except (NotImplementedError, KeyError):
        raise NotImplementedError(
            "Qwen4Exp MegaMoE reads BF16, 128x128 block-FP8, MXFP8 or MXFP4 "
            f"experts; {quant_config.get_name()} stores {prefix} otherwise."
        ) from None


def mega_moe_expert_formats(
    vllm_config: VllmConfig, prefix: str
) -> tuple[str, MegaMoEWeightFormat]:
    """The checkpoint format of the routed experts and the format they run in.

    A format forced through the online quantization config wins. Otherwise the
    experts run in the checkpoint format.
    """
    quant_config = vllm_config.quant_config
    checkpoint_format = _checkpoint_expert_format(quant_config, prefix)
    online_config = (
        quant_config
        if isinstance(quant_config, OnlineQuantizationConfig)
        else getattr(quant_config, "online_quantization_config", None)
    )
    if (
        online_config is not None
        and (resolved := online_config.resolve_quant_method_cls(RoutedExperts, prefix))
        is not None
    ):
        source, quant_key_str, target_pattern, spec, _ = resolved
        online_format = _ONLINE_FORMATS.get(spec.weight) if spec.weight else None
        if online_format is None:
            raise NotImplementedError(
                f"Qwen4Exp MegaMoE runs routed experts as mxfp8 or mxfp4; "
                f"quantization_config asks for {quant_key_str} at {prefix}."
            )
        if online_format != checkpoint_format:
            # Counted in the online quantization summary like a requantized layer.
            online_config.quantized_layers[prefix] = (
                source.value,
                quant_key_str,
                target_pattern,
            )
        return checkpoint_format, mega_moe_weight_format(online_format)
    return checkpoint_format, mega_moe_weight_format(checkpoint_format)


class Qwen4ExpMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts loaded from per-expert or fused checkpoint tensors.

    Experts stored in `weight_format` load through the generic MegaMoE loader.
    Others are requantized one expert projection at a time, once its weight
    and scale have both arrived.
    """

    def __init__(
        self, vllm_config: VllmConfig, checkpoint_format: str, **kwargs
    ) -> None:
        super().__init__(vllm_config, activation=MegaMoEActivation(), **kwargs)
        self.checkpoint_format = checkpoint_format
        self._shard_tensors = _SHARD_TENSORS[checkpoint_format]
        # Parameters may sit on meta (weight-cache IPC loading, layerwise
        # reload), so requantize on the current GPU.
        self._device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        # Tensors of expert shards still missing their weight or scale; a
        # chunked weight update may split the two across load_weights calls.
        self._pending_shards: dict[tuple[int, str], dict[str, torch.Tensor]] = {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the experts' checkpoint tensors; return the parameters written,
        so the loader still flags a parameter nothing reached."""
        if self.checkpoint_format == self.weight_format.name:
            return set(super().load_weights(weights))
        loaded: set[str] = set()
        for name, tensor in weights:
            for expert_id, shard_id, kind, shard in self._expert_shards(name, tensor):
                kind, shard = self._checkpoint_tensor(kind, shard_id, shard)
                if kind not in self._shard_tensors:
                    raise ValueError(
                        f"{self.prefix}: unexpected {self.checkpoint_format} expert "
                        f"tensor {name!r}; expected {', '.join(self._shard_tensors)}."
                    )
                if not self._map_global_expert_id(expert_id):
                    continue
                pending = self._pending_shards.setdefault((expert_id, shard_id), {})
                pending[kind] = shard
                if len(pending) < len(self._shard_tensors):
                    continue
                del self._pending_shards[expert_id, shard_id]
                scale = None
                if self.checkpoint_format != "bf16":
                    scale = pending[self._shard_tensors[1]].to(self._device)
                weight, scale = _requantize(
                    pending["weight"].to(self._device),
                    scale,
                    self.checkpoint_format,
                    self.weight_format,
                )
                prefix = "w2" if shard_id == "w2" else "w13"
                for suffix, value in (("weight", weight), ("weight_scale", scale)):
                    param_name = f"{prefix}_{suffix}"
                    param = getattr(self, param_name)
                    # The parameter's own loader: layerwise reload wraps it.
                    param.weight_loader(
                        param, value, param_name, shard_id=shard_id, expert_id=expert_id
                    )
                    loaded.add(param_name)
        return loaded

    def process_weights_after_loading(self) -> None:
        if self._pending_shards:
            raise ValueError(
                f"{self.prefix}: expert shards without both weight and scale: "
                f"{sorted(self._pending_shards)[:4]}"
            )
        super().process_weights_after_loading()


class Qwen4ExpMegaMoeBlock(nn.Module):
    """Qwen3Next-style MoE whose routed experts run as one MegaMoE launch.

    Takes this rank's tokens and returns their routed plus shared-expert
    output; no collective follows.
    """

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        if not parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE requires expert parallel. Enable it with "
                "--enable-expert-parallel."
            )
        placement = parallel_config.expert_placement_strategy
        if placement != "linear":
            # The EP weight filter would hand this rank other experts.
            raise NotImplementedError(
                "Qwen4Exp MegaMoE places experts linearly; got "
                f"--expert-placement-strategy {placement}."
            )
        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > 1:
            raise NotImplementedError(
                "Qwen4Exp MegaMoE supports tensor parallel size 1; scale out "
                "with --data-parallel-size instead."
            )
        # Read by the DeepGEMM MegaMoE warmup.
        self.use_sequence_parallel = False
        self.top_k = config.num_experts_per_tok
        self.renormalize = config.norm_topk_prob

        ep_group = get_ep_group()
        self.n_routed_experts = config.num_experts
        self.n_shared_experts = 1
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        if self.n_physical_experts % ep_group.world_size:
            raise ValueError(
                f"{self.n_physical_experts} experts do not divide evenly over "
                f"{ep_group.world_size} EP ranks."
            )
        self.n_local_physical_experts = self.n_physical_experts // ep_group.world_size

        self.gate = GateLinear(
            config.hidden_size, config.num_experts, prefix=f"{prefix}.gate"
        )
        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )
        self.shared_expert = Qwen3MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=vllm_config.quant_config,
            reduce_results=False,
            expert_gate=self.shared_expert_gate,
            prefix=f"{prefix}.shared_expert",
        )
        # The shared MLP overlaps the router on the aux stream and joins before
        # the MegaMoE launch, which needs every SM.
        self.shared_stream = (
            None if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM else aux_stream()
        )
        self.shared_events = (torch.cuda.Event(), torch.cuda.Event())
        checkpoint_format, weight_format = mega_moe_expert_formats(
            vllm_config, f"{prefix}.experts"
        )
        logger.info_once(
            "Qwen4Exp MegaMoE runs %s routed experts (checkpoint: %s).",
            weight_format.name,
            checkpoint_format,
        )
        self.experts = Qwen4ExpMegaMoEExperts(
            vllm_config,
            checkpoint_format=checkpoint_format,
            weight_format=weight_format,
            num_experts=self.n_physical_experts,
            num_local_experts=self.n_local_physical_experts,
            experts_start_idx=ep_group.rank_in_group * self.n_local_physical_experts,
            num_logical_experts=self.n_logical_experts,
            top_k=self.top_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        def route():
            router_logits, _ = self.gate(hidden_states)
            return fused_topk(
                hidden_states, router_logits, self.top_k, self.renormalize
            )

        (topk_weights, topk_ids, _), shared_out = maybe_execute_in_parallel(
            route,
            lambda: self.shared_expert(hidden_states),
            *self.shared_events,
            self.shared_stream,
        )
        y = self.experts(hidden_states, topk_weights, topk_ids)
        y += shared_out
        return y

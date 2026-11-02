# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp routed experts on DeepGEMM MegaMoE.

Block-FP8 checkpoint experts are requantized as they load, to MXFP8 by default
or to MXFP4 through ``--quantization-config``. The MTP drafter's config does not
carry it, so its experts stay MXFP8. The sigmoid-gated shared expert stays
unfused and runs on an aux stream next to the router.
"""

from collections.abc import Iterable
from importlib import import_module

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    MEGA_MOE_MXFP8,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightFormat,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
    kMxfp4Static,
    kMxfp8Dynamic,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm
from vllm.utils.torch_utils import aux_stream, current_stream

logger = init_logger(__name__)

# K values per UE8M0 scale of the MX formats.
_MX_GROUP = 32
_CHECKPOINT_SHARDS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
# Bounds the FP32 temporaries of one requantize call to a few GiB: at EP1,
# one projection batches every routed expert.
_REQUANTIZE_CHUNK_ELEMENTS = 1 << 27


def requantize_fp8_block(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    weight_format: MegaMoEWeightFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize ``[N, K]`` block-FP8 weights to MXFP8 or MXFP4.

    Rounds as DeepGEMM's ``per_token_cast_to_fp8`` / ``per_token_cast_to_fp4``
    and returns the scales as ``[N, K // 32]`` UE8M0 exponents. Rows are
    independent, so experts may be stacked along dim 0.
    """
    rows, cols = weight.shape
    cast = (
        "per_token_cast_to_fp4"
        if weight_format is MEGA_MOE_MXFP4
        else "per_token_cast_to_fp8"
    )
    cast = getattr(import_module(f"{_import_deep_gemm().__name__}.utils"), cast)
    scale = weight_scale_inv.float()
    scale = scale.repeat_interleave(rows // scale.shape[0], 0)
    scale = scale.repeat_interleave(cols // scale.shape[1], 1)
    values, sf = cast(weight.float() * scale, use_ue8m0=True, gran_k=_MX_GROUP)
    # Power-of-two FP32 scales: the exponent bits are the UE8M0 value.
    return values, (sf.view(torch.int32) >> 23).to(torch.uint8)


def mega_moe_weight_format(vllm_config: VllmConfig, prefix: str) -> MegaMoEWeightFormat:
    """MXFP8 unless ``--quantization-config`` resolves MXFP4 for ``prefix``."""
    quant_config = vllm_config.quant_config
    if not (
        isinstance(quant_config, Fp8Config)
        and quant_config.weight_block_size == [128, 128]
    ):
        name = None if quant_config is None else quant_config.get_name()
        raise NotImplementedError(
            "Qwen4Exp MegaMoE requires a 128x128 block-FP8 checkpoint; "
            f"got quantization {name!r}."
        )
    if is_layer_skipped(
        prefix,
        quant_config.ignored_layers,
        fused_mapping=quant_config.packed_modules_mapping,
        match_mode=quant_config.ignored_layers_match_mode,
    ):
        raise NotImplementedError(
            f"Qwen4Exp MegaMoE requires block-FP8 experts; the checkpoint leaves "
            f"{prefix} unquantized."
        )
    online_config = quant_config.online_quantization_config
    if online_config is None:
        return MEGA_MOE_MXFP8
    resolved = online_config.resolve_quant_method_cls(RoutedExperts, prefix)
    if resolved is None:
        return MEGA_MOE_MXFP8
    source, quant_key_str, target_pattern, spec, _ = resolved
    if spec.weight == kMxfp4Static:
        weight_format = MEGA_MOE_MXFP4
    elif spec.weight == kMxfp8Dynamic:
        weight_format = MEGA_MOE_MXFP8
    else:
        raise NotImplementedError(
            f"Qwen4Exp MegaMoE runs routed experts as mxfp8 or mxfp4; "
            f"quantization_config asks for {quant_key_str} at {prefix}."
        )
    # Counted in the online quantization summary like a requantized layer.
    online_config.quantized_layers[prefix] = (
        source.value,
        quant_key_str,
        target_pattern,
    )
    return weight_format


def flush_mega_moe_requantize(model: nn.Module) -> None:
    """Requantize the complete expert shards buffered by ``load_weights``.

    Half pairs stay buffered: a chunked weight update may pass an expert's
    weight and its scale in different calls.
    """
    for module in model.modules():
        if isinstance(module, Qwen4ExpMegaMoEExperts):
            module.flush_requantize()


class _FinishRequantize(QuantizeMethodBase):
    """Flushes the experts; fails on shards missing their weight or scale."""

    requires_device_loading = False
    supports_pre_processed_weights = True

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise NotImplementedError("MegaMoE experts allocate weights at construction")

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("MegaMoE experts run through their forward")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        assert isinstance(layer, Qwen4ExpMegaMoEExperts)
        layer.flush_requantize(strict=True)


class Qwen4ExpMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts loaded from per-expert block-FP8 checkpoint tensors.

    Each EP rank buffers the complete (weight, scale) pairs of its own experts
    and requantizes them per projection in a few batched calls.
    """

    def __init__(self, vllm_config: VllmConfig, **kwargs) -> None:
        super().__init__(vllm_config, activation=MegaMoEActivation(), **kwargs)
        self.quant_method = _FinishRequantize()
        # Parameters may sit on meta (weight-cache IPC loading, layerwise
        # reload), so stage and requantize on the current GPU.
        self._requantize_device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        self._pending_shards: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
        # Complete (expert, weight, block scale) shards per shard id.
        self._ready_shards: dict[str, list[tuple[int, torch.Tensor, torch.Tensor]]] = {}
        # Logical experts this rank holds (redundant replicas share one).
        self._num_owned_experts = len(
            {
                p % self.num_logical_experts
                for p in range(self.experts_start_idx, self.experts_end_idx)
            }
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load ``{expert}.{gate,up,down}_proj.{weight,weight_scale_inv}``."""
        loaded: set[str] = set()
        for name, tensor in weights:
            expert, proj, kind = name.split(".")
            expert_id, shard_id = int(expert), _CHECKPOINT_SHARDS[proj]
            prefix = "w2" if shard_id == "w2" else "w13"
            loaded.update((f"{prefix}_weight", f"{prefix}_weight_scale"))
            if not self._map_global_expert_id(expert_id):
                continue
            pending = self._pending_shards.setdefault((expert_id, shard_id), {})
            # Stage on the GPU as it arrives: no CPU-side torch ops, which stall
            # when several engines load on one host.
            pending[kind] = tensor.to(self._requantize_device)
            if len(pending) < 2:
                continue
            del self._pending_shards[(expert_id, shard_id)]
            ready = self._ready_shards.setdefault(shard_id, [])
            ready.append((expert_id, pending["weight"], pending["weight_scale_inv"]))
            if len(ready) == self._num_owned_experts:
                self._requantize_ready(shard_id)
        return loaded

    def flush_requantize(self, strict: bool = False) -> None:
        for shard_id in list(self._ready_shards):
            self._requantize_ready(shard_id)
        if strict and self._pending_shards:
            raise ValueError(
                f"{self.prefix}: expert shards without both weight and scale: "
                f"{sorted(self._pending_shards)[:4]}"
            )

    def _requantize_ready(self, shard_id: str) -> None:
        ready = self._ready_shards.pop(shard_id)
        rows = ready[0][1].shape[0]
        if any(w.shape != ready[0][1].shape for _, w, _ in ready):
            raise ValueError(f"{self.prefix}: {shard_id} shards differ in shape.")
        prefix = "w2" if shard_id == "w2" else "w13"
        chunk = max(1, _REQUANTIZE_CHUNK_ELEMENTS // ready[0][1].numel())
        for start in range(0, len(ready), chunk):
            batch = ready[start : start + chunk]
            weight, scale = requantize_fp8_block(
                torch.cat([w for _, w, _ in batch]),
                torch.cat([s for _, _, s in batch]),
                self.weight_format,
            )
            for i, (expert_id, _, _) in enumerate(batch):
                for suffix, value in (("weight", weight), ("weight_scale", scale)):
                    param_name = f"{prefix}_{suffix}"
                    # The parameter's own loader: layerwise reload wraps it.
                    param = getattr(self, param_name)
                    param.weight_loader(
                        param,
                        value[i * rows : (i + 1) * rows],
                        param_name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )


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
        self.shared_expert = Qwen2MoeMLP(
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
        if self.shared_stream is not None:
            self.shared_input_ready = torch.cuda.Event()
            self.shared_output_ready = torch.cuda.Event()
        weight_format = mega_moe_weight_format(vllm_config, f"{prefix}.experts")
        logger.info_once("Qwen4Exp MegaMoE runs %s routed experts.", weight_format.name)
        self.experts = Qwen4ExpMegaMoEExperts(
            vllm_config,
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
        if self.shared_stream is None:
            shared_out = self.shared_expert(hidden_states)
        else:
            self.shared_input_ready.record(current_stream())
            with torch.cuda.stream(self.shared_stream):
                self.shared_input_ready.wait(self.shared_stream)
                shared_out = self.shared_expert(hidden_states)
                self.shared_output_ready.record(self.shared_stream)
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids, _ = fused_topk(
            hidden_states, router_logits, self.top_k, self.renormalize
        )
        if self.shared_stream is not None:
            self.shared_output_ready.wait(current_stream())
        y = self.experts(hidden_states, topk_weights, topk_ids)
        y += shared_out
        return y

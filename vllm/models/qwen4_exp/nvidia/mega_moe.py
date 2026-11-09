# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp routed experts on DeepGEMM MegaMoE.

Routed experts stored as MXFP8, MXFP4 or BF16 load straight into the kernel
layout. `--quantization` / `--quantization-config` can force another MegaMoE
format, and the experts are then requantized as they load. Block-FP8 experts,
which MegaMoE cannot run, are requantized to MXFP8 unless a format is forced.
The MTP drafter's config does not carry the online config, so its experts
take the default format.

The shared expert is fused into the same launch, weighted per token by its
sigmoid gate (DeepGEMM's `shared_expert_weights`), when the installed DeepGEMM
supports it and its checkpoint tensors can be requantized like the routed
experts'. Otherwise it stays unfused and runs on an aux stream next to the
router.
"""

import inspect
from collections.abc import Iterable

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
    MEGA_MOE_WEIGHT_FORMATS,
    DeepGemmMegaMoEExperts,
    MegaMoEActivation,
    MegaMoEWeightFormat,
    MegaMoEWeightSpec,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
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
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)

_CHECKPOINT_SHARDS = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}
# Per-expert tensors of one projection, by checkpoint expert format. Formats
# MegaMoE runs share the name of their MEGA_MOE_* format; "fp8_block" is the
# 128x128 block-FP8 layout of the official FP8 checkpoint.
_SHARD_TENSORS = {
    "bf16": ("weight",),
    "fp8_block": ("weight", "weight_scale_inv"),
    "mxfp8": ("weight", "weight_scale"),
    "mxfp4": ("weight", "weight_scale"),
}
_ONLINE_FORMATS = {kMxfp8Static: MEGA_MOE_MXFP8, kMxfp4Static: MEGA_MOE_MXFP4}


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
        if weight_format is MEGA_MOE_MXFP4
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
    experts run as stored, and block-FP8 experts are requantized to MXFP8.
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
        weight_format = _ONLINE_FORMATS.get(spec.weight) if spec.weight else None
        if weight_format is None:
            raise NotImplementedError(
                f"Qwen4Exp MegaMoE runs routed experts as mxfp8 or mxfp4; "
                f"quantization_config asks for {quant_key_str} at {prefix}."
            )
        if weight_format.name != checkpoint_format:
            # Counted in the online quantization summary like a requantized layer.
            online_config.quantized_layers[prefix] = (
                source.value,
                quant_key_str,
                target_pattern,
            )
        return checkpoint_format, weight_format
    if checkpoint_format == "fp8_block":
        logger.warning_once(
            "DeepGEMM MegaMoE cannot run the 128x128 block-FP8 routed experts of "
            "this checkpoint, so they are requantized to MXFP8 at load time. Pass "
            '--quantization-config \'{"moe": "mxfp8"}\' (or "mxfp4") to choose '
            "the format explicitly."
        )
        return checkpoint_format, MEGA_MOE_MXFP8
    return checkpoint_format, MEGA_MOE_WEIGHT_FORMATS[checkpoint_format]


def _checkpoint_shared_expert_format(
    quant_config: QuantizationConfig | None, prefix: str
) -> str | None:
    """The format the checkpoint stores the shared expert at `prefix` in, or
    None if the fused loader cannot read it.

    Checkpoints may keep it unquantized next to quantized routed experts: the
    official FP8 checkpoint lists it in `modules_to_not_convert`.
    """
    if quant_config is None or isinstance(quant_config, OnlineQuantizationConfig):
        return "bf16"
    if not isinstance(quant_config, Fp8Config):
        return None
    # The checkpoint's own projection names, as `modules_to_not_convert` lists
    # them: `gate_up_proj` would depend on the model's packed-modules mapping.
    skipped = {
        is_layer_skipped(
            f"{prefix}.{proj}",
            quant_config.ignored_layers,
            match_mode=quant_config.ignored_layers_match_mode,
        )
        for proj in _CHECKPOINT_SHARDS
    }
    if skipped == {True}:
        return "bf16"
    if (
        skipped == {False}
        and quant_config.weight_block_size == [128, 128]
        and quant_config.store_dtype is None
    ):
        return "fp8_block"
    return None


def _fuses_gated_shared_expert(
    shared_checkpoint_format: str | None, weight_format: MegaMoEWeightFormat
) -> bool:
    """Whether the gated shared expert can run inside the MegaMoE launch."""
    if weight_format.uses_nvfp4_api or shared_checkpoint_format is None:
        # No per-token shared weights in nvfp4_mega_moe, or a shared-expert
        # checkpoint format the fused loader does not read.
        return False
    kernel = getattr(_import_deep_gemm(), weight_format.kernel, None)
    return (
        kernel is not None
        and "shared_expert_weights" in inspect.signature(kernel).parameters
    )


class _CheckLoaded(QuantizeMethodBase):
    """Run by vLLM after loading: fails if an expert shard got its weight but
    not its scale, or the reverse, instead of leaving that expert zeroed."""

    requires_device_loading = False
    supports_pre_processed_weights = True

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise NotImplementedError("MegaMoE experts allocate weights at construction")

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("MegaMoE experts run through their forward")

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        assert isinstance(layer, Qwen4ExpMegaMoEExperts)
        if layer.pending_shards:
            raise ValueError(
                f"{layer.prefix}: expert shards without both weight and scale: "
                f"{sorted(layer.pending_shards, key=str)[:4]}"
            )


class Qwen4ExpMegaMoEExperts(DeepGemmMegaMoEExperts):
    """MegaMoE experts loaded from per-expert or fused checkpoint tensors.

    Experts stored in `weight_format` are written straight into the kernel
    layout. Others are requantized one expert projection at a time, once its
    weight and scale have both arrived.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        checkpoint_format: str,
        shared_expert_intermediate_size: int = 0,
        shared_checkpoint_format: str = "bf16",
        **kwargs,
    ) -> None:
        # Read by `_fusable_shared_experts` during the base constructor.
        self._shared_expert_intermediate_size = shared_expert_intermediate_size
        super().__init__(vllm_config, activation=MegaMoEActivation(), **kwargs)
        self.quant_method = _CheckLoaded()
        self.checkpoint_format = checkpoint_format
        self._shard_tensors = _SHARD_TENSORS[checkpoint_format]
        self.shared_checkpoint_format = shared_checkpoint_format
        # Parameters may sit on meta (weight-cache IPC loading, layerwise
        # reload), so requantize on the current GPU.
        self._device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        # Tensors of expert shards still missing their weight or scale; a
        # chunked weight update may split the two across load_weights calls.
        # Key (None, shard) is the fused shared expert.
        self.pending_shards: dict[tuple[int | None, str], dict[str, torch.Tensor]] = {}

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, MegaMoEWeightSpec | None]:
        """Fuse the shared expert at the precision of a routed expert slice.

        Its checkpoint tensors are requantized as they load, like the routed
        experts', so only its size and the installed DeepGEMM decide.
        """
        size = self._shared_expert_intermediate_size
        if size == 0:
            return 0, None
        if size % self.intermediate_size:
            raise ValueError(
                f"Shared expert intermediate size {size} is not a multiple of "
                f"the routed {self.intermediate_size}."
            )
        spec = self.weight_format.shared[0]
        if not self._kernel_fuses_shared_experts(spec):
            raise ValueError("The installed DeepGEMM cannot fuse shared experts.")
        return size // self.intermediate_size, spec

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load `{expert}.{gate,up,down}_proj.{weight,<scale>}` or the fused
        `gate_up_proj` / `down_proj` BF16 tensors of every expert, and the
        fused shared expert's tensors as `shared.{gate,up,down}_proj.*`."""
        loaded: set[str] = set()
        for name, tensor in weights:
            if (
                name in ("gate_up_proj", "down_proj")
                and self.checkpoint_format == "bf16"
            ):
                # Fused [E, N, K] BF16 experts; gate rows precede up rows.
                for expert_id in range(self.num_logical_experts):
                    if not self._map_global_expert_id(expert_id):
                        continue
                    if name == "down_proj":
                        shards = [("w2", tensor[expert_id])]
                    else:
                        shards = list(zip(("w1", "w3"), tensor[expert_id].chunk(2)))
                    for shard_id, shard in shards:
                        loaded |= self._load_shard(expert_id, shard_id, "weight", shard)
                continue
            parts = name.split(".")
            shared = parts[0] == "shared" and self.has_fused_shared_experts
            fmt = self.shared_checkpoint_format if shared else self.checkpoint_format
            if (
                len(parts) != 3
                or parts[1] not in _CHECKPOINT_SHARDS
                or parts[2] not in _SHARD_TENSORS[fmt]
            ):
                raise ValueError(
                    f"{self.prefix}: unexpected {fmt} expert tensor {name!r}; "
                    "expected {expert,shared}.{gate,up,down}_proj."
                    f"{{{','.join(_SHARD_TENSORS[fmt])}}}."
                )
            expert, proj, kind = parts
            if shared:
                loaded |= self._load_shard(None, _CHECKPOINT_SHARDS[proj], kind, tensor)
            elif self._map_global_expert_id(int(expert)):
                loaded |= self._load_shard(
                    int(expert), _CHECKPOINT_SHARDS[proj], kind, tensor
                )
        return loaded

    def _load_shard(
        self, expert_id: int | None, shard_id: str, kind: str, tensor: torch.Tensor
    ) -> set[str]:
        """Write or requantize one checkpoint tensor of a routed expert, or of
        the fused shared expert when `expert_id` is None; return the parameters
        written, so the loader still flags a parameter nothing reached."""
        target, checkpoint_format = self.weight_format, self.checkpoint_format
        if expert_id is None:
            # The shared expert runs at `shared_spec`: MXFP8 next to MX experts.
            assert self.shared_spec is not None
            if self.shared_spec.scale_group_k:
                target = MEGA_MOE_MXFP8
            checkpoint_format = self.shared_checkpoint_format
        if checkpoint_format == target.name:
            suffix = "weight" if kind == "weight" else "weight_scale"
            return {self._write(expert_id, shard_id, suffix, tensor)}
        shard_tensors = _SHARD_TENSORS[checkpoint_format]
        pending = self.pending_shards.setdefault((expert_id, shard_id), {})
        pending[kind] = tensor
        if len(pending) < len(shard_tensors):
            return set()
        del self.pending_shards[(expert_id, shard_id)]
        scale = None
        if checkpoint_format != "bf16":
            scale = pending[shard_tensors[1]].to(self._device)
        weight, scale = _requantize(
            pending["weight"].to(self._device), scale, checkpoint_format, target
        )
        return {
            self._write(expert_id, shard_id, "weight", weight),
            self._write(expert_id, shard_id, "weight_scale", scale),
        }

    def _write(
        self, expert_id: int | None, shard_id: str, suffix: str, tensor: torch.Tensor
    ) -> str:
        if expert_id is None:
            name = f"shared_{'l2' if shard_id == 'w2' else 'l1'}_{suffix}"
            param = getattr(self, name)
            # The shared loader reads the shard and kind from the name.
            param.weight_loader(param, tensor, f"shared_experts.{shard_id}.{suffix}")
            return name
        name = f"{'w2' if shard_id == 'w2' else 'w13'}_{suffix}"
        param = getattr(self, name)
        # The parameter's own loader: layerwise reload wraps it.
        param.weight_loader(param, tensor, name, shard_id=shard_id, expert_id=expert_id)
        return name


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
        checkpoint_format, weight_format = mega_moe_expert_formats(
            vllm_config, f"{prefix}.experts"
        )
        shared_checkpoint_format = _checkpoint_shared_expert_format(
            vllm_config.quant_config, f"{prefix}.shared_expert"
        )
        fuse_shared = _fuses_gated_shared_expert(
            shared_checkpoint_format, weight_format
        )
        logger.info_once(
            "Qwen4Exp MegaMoE runs %s routed experts (checkpoint: %s), shared "
            "expert %s (checkpoint: %s).",
            weight_format.name,
            checkpoint_format,
            "fused" if fuse_shared else "unfused",
            shared_checkpoint_format,
        )
        # A fused shared expert lives in `self.experts`; its gate stays here.
        self.shared_expert: Qwen2MoeMLP | None = None
        if not fuse_shared:
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        # The unfused shared MLP (or the fused one's gate) overlaps the router
        # on the aux stream and joins before the MegaMoE launch, which needs
        # every SM.
        self.shared_stream = (
            None if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM else aux_stream()
        )
        self.shared_events = (torch.cuda.Event(), torch.cuda.Event())
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
            shared_expert_intermediate_size=(
                config.shared_expert_intermediate_size if fuse_shared else 0
            ),
            # Read only when the shared expert is fused.
            shared_checkpoint_format=shared_checkpoint_format or "bf16",
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        def route():
            router_logits, _ = self.gate(hidden_states)
            return fused_topk(
                hidden_states, router_logits, self.top_k, self.renormalize
            )

        shared_expert = self.shared_expert
        if shared_expert is not None:
            (topk_weights, topk_ids, _), shared_out = maybe_execute_in_parallel(
                route,
                lambda: shared_expert(hidden_states),
                *self.shared_events,
                self.shared_stream,
            )
            y = self.experts(hidden_states, topk_weights, topk_ids)
            y += shared_out
            return y

        def shared_gate() -> torch.Tensor:
            gate_logits, _ = self.shared_expert_gate(hidden_states)
            return torch.sigmoid(gate_logits.view(-1).float())

        (topk_weights, topk_ids, _), shared_weights = maybe_execute_in_parallel(
            route, shared_gate, *self.shared_events, self.shared_stream
        )
        return self.experts(
            hidden_states, topk_weights, topk_ids, shared_expert_weights=shared_weights
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route a fused shared expert's checkpoint tensors to the experts as
        `experts.shared.{gate,up,down}_proj.*`."""

        def rename(name: str, tensor: torch.Tensor) -> str:
            if self.shared_expert is None and name.startswith("shared_expert."):
                proj, _, kind = name.removeprefix("shared_expert.").partition(".")
                if proj == "gate_up_proj":
                    # Stacked by the model's weights mapper; undo it.
                    proj = ("gate_proj", "up_proj")[tensor.shard_id]
                return f"experts.shared.{proj}.{kind}"
            return name

        return AutoWeightsLoader(self).load_weights(
            (rename(name, tensor), tensor) for name, tensor in weights
        )

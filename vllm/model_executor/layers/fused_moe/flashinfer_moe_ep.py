# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import torch

from vllm.config import get_current_vllm_config
from vllm.config.kernel import (
    FLASHINFER_MOE_EP_BACKENDS,
    FLASHINFER_MOE_EP_CUTEDSL,
    FLASHINFER_MOE_EP_DEEP_GEMM,
)
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
    kMxfp4Static,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm"]
FlashInferMoeEpWeightFormat = Literal["nvfp4", "mxfp4", "fp8_block"]
_WEIGHT_FORMATS: dict[QuantKey | None, FlashInferMoeEpWeightFormat] = {
    kNvfp4Static: "nvfp4",
    kMxfp4Static: "mxfp4",
    kFp8Static128BlockSym: "fp8_block",
}

# FP4 weights run on the Blackwell megakernels, block FP8 weights on the
# Hopper FP8 pull-style CuTeDSL megakernel.
_BLACKWELL_CAPABILITIES = frozenset({(10, 0), (10, 3)})
_FORMAT_CAPABILITIES: dict[FlashInferMoeEpWeightFormat, frozenset[tuple[int, int]]] = {
    "nvfp4": _BLACKWELL_CAPABILITIES,
    "mxfp4": _BLACKWELL_CAPABILITIES,
    "fp8_block": frozenset({(9, 0)}),
}
_SUPPORTED_CAPABILITIES = frozenset().union(*_FORMAT_CAPABILITIES.values())


def _current_capability() -> tuple[int, int] | None:
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    return capability.major, capability.minor


def supports_current_device() -> bool:
    return _current_capability() in _SUPPORTED_CAPABILITIES


_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@dataclass(frozen=True)
class FlashInferMoeEpBackendSpec:
    kernel: FlashInferMoeEpKernel
    weight_formats: frozenset[FlashInferMoeEpWeightFormat]


@dataclass(frozen=True)
class FlashInferMoeEpWeights:
    w13: torch.Tensor
    w2: torch.Tensor
    w13_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


@dataclass(frozen=True)
class FlashInferMoeEpEpilogue:
    input_norm_const: float = 1.0
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    fc1_norm_const: torch.Tensor | None = None


_BACKEND_SPECS = {
    FLASHINFER_MOE_EP_CUTEDSL: FlashInferMoeEpBackendSpec(
        kernel="cutedsl",
        weight_formats=frozenset({"nvfp4", "mxfp4", "fp8_block"}),
    ),
    FLASHINFER_MOE_EP_DEEP_GEMM: FlashInferMoeEpBackendSpec(
        kernel="deep_gemm",
        weight_formats=frozenset({"mxfp4"}),
    ),
}

assert frozenset(_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS


def is_flashinfer_moe_ep_backend(moe_backend: str) -> bool:
    return moe_backend in _BACKEND_SPECS


def flashinfer_moe_ep_backend_spec(
    moe_backend: str,
) -> FlashInferMoeEpBackendSpec:
    try:
        return _BACKEND_SPECS[moe_backend]
    except KeyError:
        raise ValueError(
            f"{moe_backend!r} is not a FlashInfer MoE-EP backend; expected "
            f"one of {sorted(_BACKEND_SPECS)}"
        ) from None


def flashinfer_moe_ep_unsupported_reasons(
    moe: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[str, ...]:
    """Deployment constraints of the megakernel beyond the framework's checks.

    ``activation_key`` is ``None`` for MXFP4 checkpoints (the kernel quantizes
    bf16 activations itself) but marks a W4A16 recipe for NVFP4 weights.
    """
    spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
    unsupported: list[str] = []
    weight_format = _WEIGHT_FORMATS.get(weight_key)
    if weight_format is None:
        unsupported.append(f"weight scheme {weight_key}")
    elif weight_format not in spec.weight_formats:
        unsupported.append(f"{weight_format.upper()} weights")
    elif _current_capability() not in _FORMAT_CAPABILITIES[weight_format]:
        unsupported.append(f"{weight_format.upper()} weights on this GPU")
    if weight_key == kNvfp4Static and activation_key is None:
        unsupported.append("A16 activations with NVFP4 weights")
    if weight_key == kFp8Static128BlockSym and activation_key != kFp8Dynamic128Sym:
        unsupported.append(f"activation scheme {activation_key} with block FP8")
    if moe.skip_final_all_reduce:
        unsupported.append("skip_final_all_reduce")
    if moe.in_dtype != torch.bfloat16:
        unsupported.append(f"activation dtype {moe.in_dtype}")
    if moe.has_bias:
        unsupported.append("expert bias")
    if moe.swiglu_alpha is not None or moe.swiglu_beta is not None:
        unsupported.append("custom SwiGLU alpha or beta")
    if (
        spec.kernel == "deep_gemm"
        and moe.routing_method is not RoutingMethodType.DeepseekV4
    ):
        unsupported.append(f"routing method {moe.routing_method.name}")

    vllm_config = get_current_vllm_config()
    if vllm_config.weight_transfer_config is not None:
        unsupported.append("runtime weight transfer")
    if vllm_config.parallel_config.enable_dbo:
        unsupported.append("dual batch overlap")
    if spec.kernel == "deep_gemm" and vllm_config.parallel_config.enable_eplb:
        unsupported.append("EPLB")
    return tuple(unsupported)


def validate_flashinfer_moe_ep_layer(layer: RoutedExperts) -> None:
    if layer.expert_map_manager.placement_strategy != "linear":
        raise ValueError(
            f"{layer.moe_config.moe_backend} requires contiguous linear "
            "expert placement"
        )


def apply_topk_in_fc1(
    moe: FusedMoEConfig,
    *,
    apply_router_weight_on_input: bool = False,
) -> bool:
    return (
        apply_router_weight_on_input
        or moe.routing_method is RoutingMethodType.DeepseekV4
    )


def _dequant_mxfp4_ue8m0_gran32(
    packed: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    raw = packed.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    values = torch.empty(
        raw.shape[0],
        raw.shape[1] * 2,
        dtype=torch.float32,
        device=raw.device,
    )
    values[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    values[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    encoded_scale = (
        scale
        if scale.dtype == torch.float8_e8m0fnu
        else scale.view(torch.float8_e8m0fnu)
    )
    decoded_scale = encoded_scale.float()
    return (values * decoded_scale.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _dequant_mxfp4_experts(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    num_experts, output_size, packed_input_size = weight.shape
    result = torch.empty(
        num_experts,
        output_size,
        packed_input_size * 2,
        dtype=torch.bfloat16,
        device=weight.device,
    )
    for expert_id in range(num_experts):
        result[expert_id] = _dequant_mxfp4_ue8m0_gran32(
            weight[expert_id], scale[expert_id]
        )
    return result


def mxfp4_moe_ep_weights(
    moe_backend: str,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FlashInferMoeEpWeights:
    spec = flashinfer_moe_ep_backend_spec(moe_backend)
    if spec.kernel == "deep_gemm":
        return FlashInferMoeEpWeights(w13, w2, w13_scale, w2_scale)
    return FlashInferMoeEpWeights(
        w13=_dequant_mxfp4_experts(w13, w13_scale),
        w2=_dequant_mxfp4_experts(w2, w2_scale),
    )


# The SM90 FP8 megakernel folds SwiGLU over FC1 rows ordered
# [pair, {gate, up}, _SM90_GATE_UP_INTERLEAVE].
_SM90_GATE_UP_INTERLEAVE = 8
_FP8_BLOCK = 128
# The SM90 FP8 megakernel runs its whole token buffer, with a launch geometry
# picked for that size, so small batches get small buffers of their own.
_SM90_WORKSPACE_CAPACITIES = (64, 256, 1024)


def sm90_fp8_block_moe_ep_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder block-FP8 ``w13`` (gate rows, then up rows) for the SM90 kernel.

    After the gate/up interleave a 128-row scale block holds 64 gate and 64 up
    rows from two checkpoint blocks. Both take the larger of the two scales and
    the other half is rescaled. Power-of-two scales make that an exponent
    shift, so only values that drop out of the E4M3 normal range round.

    Args:
        w13: ``(E, 2 * I, H)`` E4M3 weights.
        w13_scale: ``(E, 2 * I / 128, H / 128)`` float32 dequantization scales.

    Returns:
        The ``(E, 2 * I, H)`` weights in kernel row order and their
        ``(E, 2 * I / 128, H / 128)`` scales.

    """
    num_experts, two_i, hidden = w13.shape
    intermediate = two_i // 2
    num_blocks = intermediate // _FP8_BLOCK
    assert intermediate % _FP8_BLOCK == 0 and hidden % _FP8_BLOCK == 0, w13.shape
    assert w13_scale.shape == (num_experts, 2 * num_blocks, hidden // _FP8_BLOCK)
    gate_scale, up_scale = w13_scale.split(num_blocks, dim=1)
    shared_scale = torch.maximum(gate_scale, up_scale)
    out = torch.empty_like(w13)
    # One expert at a time bounds the float32 temporaries.
    for expert in range(num_experts):
        scale = torch.stack((gate_scale[expert], up_scale[expert]))
        ratio = (scale / shared_scale[expert]).repeat_interleave(_FP8_BLOCK, 1)
        ratio = ratio.repeat_interleave(_FP8_BLOCK, 2)
        halves = w13[expert].view(2, intermediate, hidden).float() * ratio
        out[expert] = (
            halves.to(w13.dtype)
            .view(2, intermediate // _SM90_GATE_UP_INTERLEAVE, -1, hidden)
            .transpose(0, 1)
            .reshape(two_i, hidden)
        )
    return out, shared_scale.repeat_interleave(2, dim=1).contiguous()


class _FlashInferMoeEpApi:
    def __init__(self) -> None:
        try:
            from flashinfer.moe_ep import (
                BootstrapConfig,
                DeepGemmMegaMoeConfig,
                FleetParams,
                MegaConfig,
                MoEEpMegaLayer,
                MoEEpTensors,
                MoEWeightPack,
                Nvfp4CutedslMegaMoeConfig,
                Sm90PullFp8MegaMoeConfig,
            )
        except ImportError as err:
            raise RuntimeError(
                "FlashInfer MoE-EP requires the public API shipped with "
                "FlashInfer 0.7.2 or newer"
            ) from err

        self.BootstrapConfig = BootstrapConfig
        self.DeepGemmMegaMoeConfig = DeepGemmMegaMoeConfig
        self.FleetParams = FleetParams
        self.MegaConfig = MegaConfig
        self.MoEEpMegaLayer = MoEEpMegaLayer
        self.MoEEpTensors = MoEEpTensors
        self.MoEWeightPack = MoEWeightPack
        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig
        self.Sm90PullFp8MegaMoeConfig = Sm90PullFp8MegaMoeConfig


def _load_flashinfer_moe_ep_api() -> _FlashInferMoeEpApi:
    return _FlashInferMoeEpApi()


def _expose_deep_gemm_to_flashinfer() -> None:
    deep_gemm = _import_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError(
            "FlashInfer MoE-EP DeepGEMM requires a usable DeepGEMM installation"
        )
    sys.modules.setdefault("deep_gemm", deep_gemm)


class FlashInferMoeEp:
    def __init__(
        self,
        moe: FusedMoEConfig,
        weights: FlashInferMoeEpWeights,
        epilogue: FlashInferMoeEpEpilogue | None = None,
        *,
        apply_topk_in_fc1: bool,
        block_fp8: bool = False,
    ) -> None:
        spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
        if spec.kernel == "deep_gemm":
            _expose_deep_gemm_to_flashinfer()
        api = _load_flashinfer_moe_ep_api()
        if epilogue is None:
            epilogue = FlashInferMoeEpEpilogue()
        ep_group = get_ep_group()
        bootstrap = api.BootstrapConfig(
            world_size=ep_group.world_size,
            rank=ep_group.rank_in_group,
            process_group=ep_group.device_group,
            device=torch.accelerator.current_device_index(),
        )
        fleet_params = api.FleetParams(
            num_experts=moe.num_experts,
            max_tokens_per_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
        )
        if block_fp8:
            megakernel = api.Sm90PullFp8MegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                fp8_scale_mode="blockwise",
                gate_up_clamp=moe.swiglu_limit,
            )
        elif spec.kernel == "cutedsl":
            # In-kernel fc2 reduce keeps its default (off); FlashInfer 0.7.1rc1
            # renamed in_kernel_fc2_reduce to enable_in_kernel_fc2_reduce.
            megakernel = api.Nvfp4CutedslMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                gate_up_clamp=moe.swiglu_limit,
                fast_math=True,
                apply_topk_in_fc1=apply_topk_in_fc1,
                combine_dtype="bf16",
                input_norm_const=epilogue.input_norm_const,
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                knobs=None,
            )
        else:
            megakernel = api.DeepGemmMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                activation_clamp=moe.swiglu_limit,
                fast_math=True,
            )
        if block_fp8:
            # The oracle already put w13 in the kernel's row order
            # (sm90_fp8_block_moe_ep_weights); the kernel reads (E, K, N) views
            # with K contiguous.
            assert weights.w13_scale is not None and weights.w2_scale is not None
            weight_pack = None
            backend = api.MegaConfig(
                megakernel=megakernel,
                quantize_input=True,
                preprocess_weights=False,
                transformed_weights=(
                    (weights.w13.transpose(1, 2), weights.w13_scale, None, None),
                    (weights.w2.transpose(1, 2), weights.w2_scale, None, None),
                ),
            )
        else:
            weight_pack = api.MoEWeightPack(
                w13=weights.w13,
                w2=weights.w2,
                w13_scale=weights.w13_scale,
                w2_scale=weights.w2_scale,
            )
            backend = api.MegaConfig(
                megakernel=megakernel,
                quantize_input=True,
                preprocess_weights=True,
            )

        self._moe_ep_tensors_cls = api.MoEEpTensors
        self._mega_layer: Any | None = api.MoEEpMegaLayer(
            bootstrap,
            fleet_params,
            weight_pack,
            backend,
        )
        # Created collectively, in the same order on every EP rank.
        self._workspaces: list[tuple[int, Any]] = []
        if block_fp8:
            self._workspaces = [
                (capacity, self._mega_layer.create_workspace(capacity))
                for capacity in _SM90_WORKSPACE_CAPACITIES
                if capacity < moe.max_num_tokens
            ]
        self._device = weights.w13.device
        self._hidden_size = moe.hidden_dim
        self._max_num_tokens = moe.max_num_tokens
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self._epilogue: FlashInferMoeEpEpilogue | None = epilogue
        self._return_workspace_view = moe.routing_method is RoutingMethodType.DeepseekV4
        self._block_fp8 = block_fp8

    @property
    def can_overlap_shared_experts(self) -> bool:
        return False

    @property
    def output_is_reduced(self) -> bool:
        return True

    @property
    def topk_indices_dtype(self) -> torch.dtype:
        return torch.int32

    @property
    def is_monolithic(self) -> bool:
        return False

    def _tensors(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Any:
        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)
        return self._moe_ep_tensors_cls(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_alpha=epilogue.fc1_alpha,
            fc2_alpha=epilogue.fc2_alpha,
            fc1_norm_const=epilogue.fc1_norm_const,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        if hidden_states.shape[0] > self._max_num_tokens:
            raise ValueError(
                f"FlashInfer MoE-EP got {hidden_states.shape[0]} tokens, but "
                f"the workspace supports at most {self._max_num_tokens}"
            )
        tensors = self._tensors(hidden_states, topk_ids, topk_weights)
        if self._return_workspace_view and getattr(
            self._mega_layer, "supports_output_view", False
        ):
            return self._mega_layer.forward(tensors, return_workspace_view=True)
        return self._mega_layer.forward(tensors, workspace=self._workspace())

    def _workspace(self) -> Any | None:
        """The smallest workspace that fits the batch on every EP rank.

        All ranks must use the same buffer, so without the DP-wide token
        counts this falls back to the default workspace.
        """
        if not self._workspaces or not is_forward_context_available():
            return None
        dp_metadata = get_forward_context().dp_metadata
        if dp_metadata is None:
            return None
        num_tokens = int(dp_metadata.num_tokens_across_dp_cpu.max())
        for capacity, workspace in self._workspaces:
            if num_tokens <= capacity:
                return workspace
        return None

    @torch.inference_mode()
    def warmup(self) -> None:
        if self._mega_layer is None:
            return
        hidden_states = torch.zeros(
            1,
            self._hidden_size,
            dtype=torch.bfloat16,
            device=self._device,
        )
        topk_ids = (
            torch.arange(self._top_k, dtype=torch.int32, device=self._device)
            .mul_(self._num_experts // self._top_k)
            .view(1, self._top_k)
        )
        topk_weights = torch.full(
            (1, self._top_k),
            1.0 / self._top_k,
            dtype=torch.float32,
            device=self._device,
        )
        tensors = self._tensors(hidden_states, topk_ids, topk_weights)
        self._mega_layer.warmup(tensors)
        for _, workspace in self._workspaces:
            self._mega_layer.warmup(tensors, workspace=workspace)

    def kernel_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel-resident fc1 weight, fc1 scales, fc2 weight, fc2 scales.

        Contiguous per-expert tensors that alias the memory the megakernel
        reads, so a caller may permute experts in place (EPLB). One-byte FP4
        data and scales come back as uint8; block-FP8 weights keep FP8.
        """
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        # The SM90 FP8 legs carry two (unused in blockwise mode) dequant scales.
        (fc1_weight, fc1_sf, *_), (fc2_weight, fc2_sf, *_) = (
            self._mega_layer._transformed
        )
        # Block-FP8 weights keep their dtype, so the layer holds the kernel's
        # FP8 runtime layout that preprocessed-weight loading rebuilds from.
        as_bytes = not self._block_fp8
        return (
            _expert_major_bytes(fc1_weight, as_bytes),
            _expert_major_bytes(fc1_sf, as_bytes),
            _expert_major_bytes(fc2_weight, as_bytes),
            _expert_major_bytes(fc2_sf, as_bytes),
        )

    def expert_weight_views(self) -> tuple[torch.Tensor, ...]:
        """Every per-expert tensor the kernel reads: weights plus epilogue."""
        epilogue = cast(FlashInferMoeEpEpilogue, self._epilogue)
        vectors = (epilogue.fc1_alpha, epilogue.fc2_alpha, epilogue.fc1_norm_const)
        return (*self.kernel_weights(), *(v for v in vectors if v is not None))

    def destroy(self) -> None:
        mega_layer = self._mega_layer
        if mega_layer is None:
            return
        mega_layer.destroy()
        self._mega_layer = None
        self._workspaces = []
        self._epilogue = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


def _expert_major_bytes(tensor: torch.Tensor, as_bytes: bool = True) -> torch.Tensor:
    """Contiguous storage of a per-expert kernel tensor, 1-byte dtypes as uint8
    unless ``as_bytes`` is False.

    The CuTeDSL kernel keeps (E, K, N) transpose views over (E, N, K) storage.
    """
    if not tensor.is_contiguous():
        tensor = tensor.transpose(1, 2)
    if not tensor.is_contiguous():
        raise ValueError(f"expert tensor is not expert-major: {tensor.stride()}")
    if as_bytes and tensor.element_size() == 1:
        return tensor.view(torch.uint8)
    return tensor

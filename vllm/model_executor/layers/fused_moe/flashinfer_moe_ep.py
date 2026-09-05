# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import sys
import weakref
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
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import import_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm"]
FlashInferMoeEpWeightFormat = Literal["nvfp4", "mxfp4"]

_SUPPORTED_CAPABILITIES = frozenset({(10, 0), (10, 3)})
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
        weight_formats=frozenset({"nvfp4", "mxfp4"}),
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


def validate_flashinfer_moe_ep_config(
    moe: FusedMoEConfig,
    weight_format: FlashInferMoeEpWeightFormat,
    *,
    use_a16: bool = False,
) -> None:
    if not is_flashinfer_moe_ep_backend(moe.moe_backend):
        return

    spec = flashinfer_moe_ep_backend_spec(moe.moe_backend)
    unsupported: list[str] = []
    if weight_format not in spec.weight_formats:
        unsupported.append(f"{weight_format.upper()} weights")
    if use_a16:
        unsupported.append("A16 activations with NVFP4 weights")
    if not moe.moe_parallel_config.use_ep:
        unsupported.append("expert parallel disabled")
    if moe.is_lora_enabled:
        unsupported.append("LoRA")
    if moe.skip_final_all_reduce:
        unsupported.append("skip_final_all_reduce")
    if moe.in_dtype != torch.bfloat16:
        unsupported.append(f"activation dtype {moe.in_dtype}")
    if moe.activation is not MoEActivation.SILU:
        unsupported.append(f"activation {moe.activation.value}")
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
    if vllm_config.parallel_config.enable_eplb:
        unsupported.append("EPLB")

    capability = current_platform.get_device_capability()
    if capability is not None:
        value = (capability.major, capability.minor)
        if value not in _SUPPORTED_CAPABILITIES:
            unsupported.append(f"compute capability {value[0]}.{value[1]}")

    if unsupported:
        raise ValueError(
            f"{moe.moe_backend} does not support: {', '.join(unsupported)}"
        )


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


def _require_finite_positive(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all() or not (value > 0).all():
        raise ValueError(f"{name} must contain finite positive scales")


def modelopt_nvfp4_moe_ep_data(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2_scale_2: torch.Tensor,
    *,
    intermediate_size: int,
) -> tuple[FlashInferMoeEpWeights, FlashInferMoeEpEpilogue]:
    _require_finite_positive("w13_weight_scale_2", w13_scale_2)
    _require_finite_positive("w2_weight_scale_2", w2_scale_2)

    gate_scale = w13_scale_2[:, 0].float()
    up_scale = w13_scale_2[:, 1].float()
    folded_w13_scale = w13_scale
    if not torch.equal(gate_scale, up_scale):
        ratio = up_scale / gate_scale
        up_block_scale = w13_scale[:, intermediate_size:, :].float()
        folded = up_block_scale * ratio[:, None, None]
        folded_e4m3 = folded.to(torch.float8_e4m3fn)
        if not torch.equal(folded_e4m3.float(), folded):
            raise ValueError(
                "NVFP4 MoE cannot merge the gate and up weight_scale_2 "
                "values because their ratio is not exactly representable in "
                "float8_e4m3fn"
            )
        folded_w13_scale = w13_scale.clone()
        folded_w13_scale[:, intermediate_size:, :] = folded_e4m3

    weights = FlashInferMoeEpWeights(
        w13=w13,
        w2=w2,
        w13_scale=folded_w13_scale,
        w2_scale=w2_scale,
    )
    epilogue = FlashInferMoeEpEpilogue(
        fc1_alpha=gate_scale.clone().contiguous(),
        fc2_alpha=w2_scale_2.float().clone().contiguous(),
    )
    return weights, epilogue


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
            )
        except ImportError as err:
            raise RuntimeError(
                "FlashInfer MoE-EP requires the public API shipped with "
                "FlashInfer 0.6.18 or newer"
            ) from err

        self.BootstrapConfig = BootstrapConfig
        self.DeepGemmMegaMoeConfig = DeepGemmMegaMoeConfig
        self.FleetParams = FleetParams
        self.MegaConfig = MegaConfig
        self.MoEEpMegaLayer = MoEEpMegaLayer
        self.MoEEpTensors = MoEEpTensors
        self.MoEWeightPack = MoEWeightPack
        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig


def _load_flashinfer_moe_ep_api() -> _FlashInferMoeEpApi:
    return _FlashInferMoeEpApi()


def _expose_deep_gemm_to_flashinfer() -> None:
    deep_gemm = import_deep_gemm()
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
        weight_pack = api.MoEWeightPack(
            w13=weights.w13,
            w2=weights.w2,
            w13_scale=weights.w13_scale,
            w2_scale=weights.w2_scale,
        )
        if spec.kernel == "cutedsl":
            megakernel = api.Nvfp4CutedslMegaMoeConfig(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                gate_up_clamp=moe.swiglu_limit,
                fast_math=True,
                apply_topk_in_fc1=apply_topk_in_fc1,
                in_kernel_fc2_reduce=False,
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
        self._device = weights.w13.device
        self._hidden_size = moe.hidden_dim
        self._max_num_tokens = moe.max_num_tokens
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self._epilogue: FlashInferMoeEpEpilogue | None = epilogue
        self._return_workspace_view = moe.routing_method is RoutingMethodType.DeepseekV4
        _register_flashinfer_moe_ep(self)

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

    @property
    def destroyed(self) -> bool:
        return self._mega_layer is None

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
        return self._mega_layer(tensors)

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
        self._mega_layer.warmup(self._tensors(hidden_states, topk_ids, topk_weights))

    def destroy(self) -> None:
        mega_layer = self._mega_layer
        if mega_layer is None:
            return
        mega_layer.destroy()
        self._mega_layer = None
        self._epilogue = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


def make_flashinfer_moe_ep(
    moe: FusedMoEConfig,
    layer: RoutedExperts,
    weights: FlashInferMoeEpWeights,
    epilogue: FlashInferMoeEpEpilogue | None = None,
) -> FlashInferMoeEp:
    validate_flashinfer_moe_ep_layer(layer)
    return FlashInferMoeEp(
        moe,
        weights,
        epilogue,
        apply_topk_in_fc1=apply_topk_in_fc1(
            moe,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
        ),
    )


def make_mxfp4_flashinfer_moe_ep(
    moe: FusedMoEConfig,
    layer: RoutedExperts,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> FlashInferMoeEp:
    weights = mxfp4_moe_ep_weights(
        moe.moe_backend,
        w13,
        w2,
        w13_scale,
        w2_scale,
    )
    return make_flashinfer_moe_ep(moe, layer, weights)


_FLASHINFER_MOE_EP_ADAPTERS: list[weakref.ReferenceType[FlashInferMoeEp]] = []


def _live_flashinfer_moe_ep() -> list[FlashInferMoeEp]:
    live: list[FlashInferMoeEp] = []
    refs: list[weakref.ReferenceType[FlashInferMoeEp]] = []
    for ref in _FLASHINFER_MOE_EP_ADAPTERS:
        adapter = ref()
        if adapter is not None:
            refs.append(ref)
            if not adapter.destroyed:
                live.append(adapter)
    _FLASHINFER_MOE_EP_ADAPTERS[:] = refs
    return live


def _register_flashinfer_moe_ep(adapter: FlashInferMoeEp) -> None:
    _FLASHINFER_MOE_EP_ADAPTERS.append(weakref.ref(adapter))


def has_flashinfer_moe_ep() -> bool:
    return bool(_live_flashinfer_moe_ep())


def warmup_flashinfer_moe_ep() -> None:
    for adapter in _live_flashinfer_moe_ep():
        adapter.warmup()


def destroy_flashinfer_moe_ep() -> None:
    for adapter in _live_flashinfer_moe_ep():
        adapter.destroy()
    _FLASHINFER_MOE_EP_ADAPTERS.clear()

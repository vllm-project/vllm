# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import functools
import inspect
import os
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
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _import_deep_gemm

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

logger = init_logger(__name__)

FlashInferMoeEpKernel = Literal["cutedsl", "deep_gemm"]
FlashInferMoeEpWeightFormat = Literal["nvfp4", "mxfp4"]
_WEIGHT_FORMATS: dict[QuantKey | None, FlashInferMoeEpWeightFormat] = {
    kNvfp4Static: "nvfp4",
    kMxfp4Static: "mxfp4",
}

_BLACKWELL_CAPABILITIES = frozenset({(10, 0), (10, 3)})
_SM107_CAPABILITY = (10, 7)
_FORMAT_CAPABILITIES: dict[FlashInferMoeEpWeightFormat, frozenset[tuple[int, int]]] = {
    "nvfp4": _BLACKWELL_CAPABILITIES | {_SM107_CAPABILITY},
    "mxfp4": _BLACKWELL_CAPABILITIES,
}
_SUPPORTED_CAPABILITIES = frozenset().union(*_FORMAT_CAPABILITIES.values())

_SM107_NVFP4_CONFIG = "Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig"
# FlashInfer 0.7.3 renamed apply_topk_in_fc1 to this field; its presence marks 0.7.3,
# which also brings fused staging and switchable route checks to the SM107 config.
_ROUTING_WEIGHTS_FIELD = "apply_routing_weights_before_fc2"
_SM107_ROUTE_CHECK_ENV = "FLASHINFER_SM107_MEGA_CHECK_ROUTES"


def _current_capability() -> tuple[int, int] | None:
    capability = current_platform.get_device_capability()
    if capability is None:
        return None
    return capability.major, capability.minor


def supports_current_device() -> bool:
    return _current_capability() in _SUPPORTED_CAPABILITIES


def is_sm107_device() -> bool:
    return _current_capability() == _SM107_CAPABILITY


@functools.cache
def flashinfer_sm107_nvfp4_gap() -> str | None:
    """Why the installed FlashInfer cannot run the SM107 NVFP4 MegaMoE of
    FlashInfer 0.7.3 or newer, or None if it can."""
    try:
        import flashinfer
        from flashinfer import moe_ep
    except (ImportError, RuntimeError) as err:
        return f"SM107 without a usable flashinfer.moe_ep ({type(err).__name__}: {err})"
    config_cls = getattr(moe_ep, _SM107_NVFP4_CONFIG, None)
    if (
        config_cls is not None
        and _ROUTING_WEIGHTS_FIELD in inspect.signature(config_cls).parameters
    ):
        return None
    return (
        f"SM107 with FlashInfer {getattr(flashinfer, '__version__', 'unknown')} "
        "(requires flashinfer-python>=0.7.3, then `vllm download-kernels`)"
    )


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
    elif is_sm107_device() and (gap := flashinfer_sm107_nvfp4_gap()):
        unsupported.append(gap)
    if weight_key == kNvfp4Static and activation_key is None:
        unsupported.append("A16 activations with NVFP4 weights")
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
    elif is_sm107_device() and vllm_config.parallel_config.enable_eplb:
        unsupported.append("EPLB on SM107")
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


class _FlashInferMoeEpApi:
    def __init__(self) -> None:
        try:
            import flashinfer
            import flashinfer.moe_ep as moe_ep
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
                "FlashInfer 0.7.0 or newer"
            ) from err

        self.BootstrapConfig = BootstrapConfig
        self.DeepGemmMegaMoeConfig = DeepGemmMegaMoeConfig
        self.FleetParams = FleetParams
        self.MegaConfig = MegaConfig
        self.MoEEpMegaLayer = MoEEpMegaLayer
        self.MoEEpTensors = MoEEpTensors
        self.MoEWeightPack = MoEWeightPack
        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig
        self.version: str = getattr(flashinfer, "__version__", "unknown")
        self.Sm107Nvfp4MegaMoeConfig: Any = getattr(moe_ep, _SM107_NVFP4_CONFIG, None)


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
        on_sm107 = spec.kernel == "cutedsl" and is_sm107_device()
        if on_sm107:
            config_cls = api.Sm107Nvfp4MegaMoeConfig
            if config_cls is None:
                raise RuntimeError(
                    f"{moe.moe_backend} on SM107 requires flashinfer.moe_ep."
                    f"{_SM107_NVFP4_CONFIG} (flashinfer-python>=0.7.3)"
                )
            # FlashInfer 0.7.3 asserts on an invalid SM107 route, killing the worker;
            # default the checks off unless set.
            route_checks = os.environ.setdefault(_SM107_ROUTE_CHECK_ENV, "0")
            megakernel = config_cls(
                intermediate_size=moe.intermediate_size,
                top_k=moe.experts_per_token,
                gate_up_clamp=moe.swiglu_limit,
                fast_math=True,
                apply_topk_in_fc1=apply_topk_in_fc1,
                combine_dtype="bf16",
                input_norm_const=epilogue.input_norm_const,
                knobs=None,
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
        if on_sm107:
            logger.info_once(
                "FlashInfer MoE-EP megakernel %s from FlashInfer %s: knobs=%s, "
                "kernel_variant=%s, max_tokens_per_rank=%d, epilogue from "
                "checkpoint input scales=%s, %s=%s",
                type(megakernel).__name__,
                api.version,
                megakernel.knobs,
                getattr(megakernel, "kernel_variant", "inference"),
                moe.max_num_tokens,
                epilogue.fc1_norm_const is not None,
                _SM107_ROUTE_CHECK_ENV,
                route_checks,
                scope="process",
            )
        self._device = weights.w13.device
        self._hidden_size = moe.hidden_dim
        self._max_num_tokens = moe.max_num_tokens
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self._epilogue: FlashInferMoeEpEpilogue | None = epilogue
        self._return_workspace_view = (
            moe.routing_method is RoutingMethodType.DeepseekV4 and not on_sm107
        )

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

    def kernel_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Kernel-resident fc1 weight, fc1 scales, fc2 weight, fc2 scales.

        Contiguous per-expert byte tensors that alias the memory the megakernel
        reads, so a caller may permute experts in place (EPLB).
        """
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE-EP layer was destroyed")
        (fc1_weight, fc1_sf), (fc2_weight, fc2_sf) = self._mega_layer._transformed
        return (
            _expert_major_bytes(fc1_weight),
            _expert_major_bytes(fc1_sf),
            _expert_major_bytes(fc2_weight),
            _expert_major_bytes(fc2_sf),
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
        self._epilogue = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


def _expert_major_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Contiguous storage of a per-expert kernel tensor, 1-byte dtypes as uint8.

    The CuTeDSL kernel keeps (E, K, N) transpose views over (E, N, K) storage.
    """
    if not tensor.is_contiguous():
        tensor = tensor.transpose(1, 2)
    if not tensor.is_contiguous():
        raise ValueError(f"expert tensor is not expert-major: {tensor.stride()}")
    return tensor.view(torch.uint8) if tensor.element_size() == 1 else tensor

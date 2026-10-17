# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepGEMM MegaMoE experts: fused EP dispatch, expert GEMMs and combine.

Each ``MEGA_MOE_*`` format is one row of the SM100 feature matrix. MX weights
and activations carry one UE8M0 scale per 1x32 block. NVFP4 weights carry one
E4M3 scale per 1x16 block plus a per-expert FP32 global scale; NVFP4 dispatch
scales each token by its own FP32 scale.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from inspect import signature
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.prepare_megamoe import (
    prepare_megamoe_inputs,
    prepare_megamoe_nvfp4_inputs,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    eplb_map_to_physical_and_record,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kNvfp4DynamicToken,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

logger = init_logger(__name__)


@dataclass(frozen=True)
class MegaMoEActivation:
    """Gated activation fused into the MegaMoE L1 epilogue.

    ``swiglu`` computes ``gate * sigmoid(alpha * gate) * (up + beta)`` after
    clamping gate from above and up symmetrically to ``clamp``; this covers
    plain, DeepSeek-V4 (``clamp``) and GPT-OSS style (``alpha``/``beta``)
    experts. ``situ`` is Kimi K3's SiTU with gate tanh scale ``alpha`` and
    linear tanh scale ``beta``; ``beta=0`` leaves the linear branch unscaled.
    """

    name: Literal["swiglu", "situ"] = "swiglu"
    clamp: float | None = None
    alpha: float = 1.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in ("swiglu", "situ"):
            raise ValueError(f"Unsupported MegaMoE activation {self.name!r}.")
        if not (math.isfinite(self.alpha) and math.isfinite(self.beta)):
            raise ValueError(f"MegaMoE activation scales must be finite: {self}.")
        if self.clamp is not None and not self.clamp >= 0:
            raise ValueError(f"MegaMoE activation clamp must be >= 0: {self}.")
        if self.name == "situ" and (
            self.clamp is not None or self.alpha <= 0 or self.beta < 0
        ):
            raise ValueError(
                "SiTU takes no clamp, a positive alpha and a non-negative beta; "
                f"got {self}."
            )

    def kernel_kwargs(self) -> dict[str, Any]:
        # Defaults stay implicit so DeepGEMM builds without alpha/beta still work.
        kwargs: dict[str, Any] = {"activation_clamp": self.clamp}
        if self.name != "swiglu":
            kwargs["activation"] = self.name
        if self.alpha != 1.0:
            kwargs["activation_alpha"] = self.alpha
        if self.beta != 0.0:
            kwargs["activation_beta"] = self.beta
        return kwargs


@dataclass(frozen=True)
class MegaMoEWeightSpec:
    """Storage of one MegaMoE weight in the kernel layout."""

    dtype: torch.dtype
    # Weight values packed per storage element along K (2 for FP4).
    values_per_element: int = 1
    # K values per block scale; None for unscaled weights.
    scale_group_k: int | None = None
    # Block scale dtype: UE8M0 (MX) or E4M3 (NVFP4).
    scale_dtype: torch.dtype | None = None

    @property
    def has_global_scale(self) -> bool:
        """E4M3 block scales pair with an FP32 per-tensor scale."""
        return self.scale_dtype == torch.float8_e4m3fn


_MXFP4_WEIGHT = MegaMoEWeightSpec(
    torch.int8,
    values_per_element=2,
    scale_group_k=32,
    scale_dtype=torch.float8_e8m0fnu,
)
_MXFP8_WEIGHT = MegaMoEWeightSpec(
    torch.float8_e4m3fn, scale_group_k=32, scale_dtype=torch.float8_e8m0fnu
)
_NVFP4_WEIGHT = MegaMoEWeightSpec(
    torch.int8,
    values_per_element=2,
    scale_group_k=16,
    scale_dtype=torch.float8_e4m3fn,
)
_BF16_WEIGHT = MegaMoEWeightSpec(torch.bfloat16)


@dataclass(frozen=True)
class MegaMoEWeightFormat:
    """One routed-expert precision of the MegaMoE feature matrix."""

    name: str
    # DeepGEMM MMA tag; it also keys the symmetric-buffer layout.
    mma_type: str
    kernel: str
    routed: MegaMoEWeightSpec
    # Precisions a shared expert may have to run inside the same launch.
    shared: tuple[MegaMoEWeightSpec, ...]
    # Quantization of the dispatched hidden states; None dispatches BF16.
    hidden_quant: QuantKey | None
    activations: frozenset[str]

    @property
    def uses_nvfp4_api(self) -> bool:
        """``nvfp4_mega_moe`` has its own symmetric buffer, BLOCK_M and scales."""
        return self.kernel == "nvfp4_mega_moe"


MEGA_MOE_MXFP4 = MegaMoEWeightFormat(
    name="mxfp4",
    mma_type="fp8xfp4",
    kernel="fp8_fp4_mega_moe",
    routed=_MXFP4_WEIGHT,
    shared=(_MXFP8_WEIGHT,),
    hidden_quant=kMxfp8Dynamic,
    activations=frozenset({"swiglu", "situ"}),
)
MEGA_MOE_MXFP8 = MegaMoEWeightFormat(
    name="mxfp8",
    mma_type="fp8xfp8",
    kernel="fp8_fp4_mega_moe",
    routed=_MXFP8_WEIGHT,
    shared=(_MXFP8_WEIGHT,),
    hidden_quant=kMxfp8Dynamic,
    activations=frozenset({"swiglu", "situ"}),
)
MEGA_MOE_NVFP4 = MegaMoEWeightFormat(
    name="nvfp4",
    mma_type="nvfp4",
    kernel="nvfp4_mega_moe",
    routed=_NVFP4_WEIGHT,
    shared=(_MXFP8_WEIGHT, _BF16_WEIGHT),
    hidden_quant=kNvfp4DynamicToken,
    activations=frozenset({"swiglu"}),
)
MEGA_MOE_BF16 = MegaMoEWeightFormat(
    name="bf16",
    mma_type="bf16xbf16",
    kernel="bf16_mega_moe",
    routed=_BF16_WEIGHT,
    shared=(_BF16_WEIGHT,),
    hidden_quant=None,
    activations=frozenset({"swiglu"}),
)
MEGA_MOE_WEIGHT_FORMATS = {
    fmt.name: fmt
    for fmt in (MEGA_MOE_MXFP4, MEGA_MOE_MXFP8, MEGA_MOE_NVFP4, MEGA_MOE_BF16)
}


def check_deep_gemm_mega_moe_supported(
    device: torch.device, hidden_size: int, intermediate_size: int
) -> None:
    """Raise unless DeepGEMM MegaMoE can run on ``device`` with these shapes.

    Raises:
        NotImplementedError: If the device architecture is unsupported.
        ValueError: If the shapes are unsupported.

    """
    if not current_platform.is_device_capability_family(
        100, device_id=device.index or 0
    ):
        raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
    if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
        raise ValueError(
            "DeepGEMM MegaMoE requires hidden and intermediate sizes "
            "to be multiples of 128."
        )


def _linear_weight_scale(linear: nn.Module) -> torch.Tensor | None:
    scale = getattr(linear, "weight_scale", None)
    return getattr(linear, "weight_scale_inv", None) if scale is None else scale


def _matches_weight_spec(linear: nn.Module, spec: MegaMoEWeightSpec) -> bool:
    """Whether a checkpoint linear can be written into a ``spec`` weight."""
    weight, scale = linear.weight, _linear_weight_scale(linear)
    if weight.dtype != spec.dtype or weight.dim() != 2:
        return False
    if spec.scale_group_k is None:
        return scale is None
    if scale is None or scale.dim() != 2:
        return False
    (rows, cols), (scale_rows, scale_cols) = weight.shape, scale.shape
    scale_dtypes = {spec.scale_dtype}
    if spec.scale_dtype == torch.float8_e8m0fnu:
        scale_dtypes.add(torch.uint8)
    return (
        scale.dtype in scale_dtypes
        and rows % scale_rows == 0
        and cols % scale_cols == 0
        and (cols // scale_cols) % spec.scale_group_k == 0
    )


class DeepGemmMegaMoEExperts(nn.Module):
    """Routed experts, and optionally the shared expert, as one MegaMoE launch.

    Parameters are registered in the kernel layout and every checkpoint shard is
    written straight into it, so loading (and reloading) needs no post-load
    transform. Whether the shared expert runs inside the launch is decided here,
    before any weight loads; a fused layer owns the shared-expert weights and
    the caller drops its shared MLP.

    NVFP4 experts also take per-expert ``weight_scale_2`` (``w1`` and ``w3``
    must match: one L1 scale per expert) and ``w2`` ``input_scale``; the kernel
    quantizes every L2 input with the max of the latter over all loaded
    experts. Dispatch uses per-token scales, so ``w1``/``w3`` ``input_scale``
    are not loaded.

    Args:
        num_experts: Physical experts across EP ranks, redundant included.
        activation: Activation fused into the L1 epilogue.
        weight_format: Routed-expert precision, a row of the feature matrix.
        shared_experts: Shared MLP to fuse if it is unsharded and in one of
            the ``weight_format.shared`` precisions; ``None`` never fuses.

    """

    _symm_buffers: ClassVar[dict[tuple, Any]] = {}
    _synchronized_ep_groups: ClassVar[set[tuple[int, int]]] = set()

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        activation: MegaMoEActivation,
        weight_format: MegaMoEWeightFormat,
        shared_experts: nn.Module | None = None,
        num_logical_experts: int | None = None,
        fast_math: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        if activation.name not in weight_format.activations:
            raise NotImplementedError(
                f"DeepGEMM MegaMoE {weight_format.name} experts support "
                f"{sorted(weight_format.activations)}, not {activation.name!r}."
            )
        device = torch.get_default_device()
        if device.type != current_platform.device_type:  # meta under ipc_cache
            device = torch.device(
                current_platform.device_type, torch.accelerator.current_device_index()
            )
        check_deep_gemm_mega_moe_supported(device, hidden_size, intermediate_size)

        self.prefix = prefix
        self.capture_fn: Callable[[torch.Tensor], None] | None = None
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.num_logical_experts = (
            num_logical_experts if num_logical_experts is not None else num_experts
        )
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.weight_format = weight_format
        self.activation = activation
        self.fast_math = fast_math
        self._kernel_kwargs = activation.kernel_kwargs()
        self.eplb_state = EplbLayerState()
        # L2 input scale of every logical expert; the NVFP4 kernel takes the max.
        self._w2_input_scales = torch.zeros(self.num_logical_experts, device="cpu")
        self.l2_activation_scale = 1.0
        # The first of each expert's w1/w3 weight_scale_2 until the other loads.
        self._pending_l1_scales: dict[int, tuple[str, float]] = {}

        self.num_shared_experts, self.shared_spec = self._fusable_shared_experts(
            shared_experts
        )
        self._create_weights()
        for name, param in self.named_parameters(recurse=False):
            shared = name.startswith("shared_")
            loader = self.shared_weight_loader if shared else self.weight_loader
            set_weight_attrs(param, {"weight_loader": loader})
            if not shared and name.endswith("weight_scale"):
                param.quant_method = "block"

        # Register in the static forward context so the custom-op wrapper
        # can look up this module by name from within a torch.compile graph.
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @property
    def has_fused_shared_experts(self) -> bool:
        return self.num_shared_experts > 0

    def _fusable_shared_experts(
        self, shared_experts: nn.Module | None
    ) -> tuple[int, MegaMoEWeightSpec | None]:
        """How many shared experts the launch fuses, and in which precision."""
        if shared_experts is None:
            return 0, None
        gate_up, down = shared_experts.gate_up_proj, shared_experts.down_proj
        hidden, inter = self.hidden_size, self.intermediate_size
        shared_size = down.weight.shape[-1]
        spec = next(
            (
                spec
                for spec in self.weight_format.shared
                if all(_matches_weight_spec(lin, spec) for lin in (gate_up, down))
            ),
            None,
        )
        if (
            tuple(gate_up.weight.shape) != (2 * shared_size, hidden)
            or tuple(down.weight.shape) != (hidden, shared_size)
            or shared_size % inter
        ):
            reason = (
                f"needs an unsharded shared MLP over hidden size {hidden} with a "
                f"multiple of {inter} intermediate; got {tuple(gate_up.weight.shape)}"
                f" and {tuple(down.weight.shape)}"
            )
        elif spec is None:
            scale = _linear_weight_scale(gate_up)
            supported = ", ".join(
                f"{s.dtype}{' with 1x32 UE8M0 scales' if s.scale_group_k else ''}"
                for s in self.weight_format.shared
            )
            reason = (
                f"{self.weight_format.name} experts fuse {supported} shared "
                f"experts; got weight {gate_up.weight.dtype} and scale "
                f"{None if scale is None else (tuple(scale.shape), scale.dtype)}"
            )
        elif not self._kernel_fuses_shared_experts(spec):
            reason = (
                "the installed DeepGEMM Python API is older than the vLLM source. "
                "Rebuild the vendored _deep_gemm_C extension to enable it"
            )
        else:
            return shared_size // inter, spec
        logger.warning_once("Disabling MegaMoE shared-expert fusion: %s.", reason)
        return 0, None

    def _kernel_fuses_shared_experts(self, spec: MegaMoEWeightSpec) -> bool:
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        fmt = self.weight_format
        buffer_api = "get_symm_buffer_for_mega_moe"
        block_m_api = "get_block_m_for_mega_moe"
        if fmt.uses_nvfp4_api:
            buffer_api = "NVFP4SymmBuffer"
            block_m_api = "get_block_m_for_nvfp4_mega_moe"
        try:
            api = {
                *signature(getattr(deep_gemm, buffer_api)).parameters,
                *signature(getattr(deep_gemm, fmt.kernel)).parameters,
            }
        except (AttributeError, TypeError, ValueError):
            return False
        # Scaled shared inputs are laid out by the runtime BLOCK_M.
        return {
            "num_shared_experts",
            "shared_l1_weights",
            "shared_l2_weights",
        } <= api and (spec.scale_group_k is None or hasattr(deep_gemm, block_m_api))

    def _create_weights(self) -> None:
        """Register the routed and fused shared-expert parameters.

        Routed parameters are indexed by local expert on dim 0, which EPLB moves.
        """
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size
        fmt = self.weight_format
        self._register_weight("w13", (e, 2 * i, h), fmt.routed)
        self._register_weight("w2", (e, h, i), fmt.routed)
        if fmt.routed.has_global_scale:
            # w13_weight_scale_2 is the kernel's L1 alpha (per-token dispatch).
            for name in ("w13_weight_scale_2", "w2_weight_scale_2", "w2_input_scale"):
                scale = torch.zeros(e, dtype=torch.float32)
                self.register_parameter(name, nn.Parameter(scale, requires_grad=False))
            # w2 global weight scale times the L2 activation scale, per launch.
            l2_alpha = torch.zeros(e, dtype=torch.float32)
            self.register_buffer("_l2_alpha", l2_alpha, persistent=False)
        if self.shared_spec is not None:
            # The shared expert has the layout of one routed expert slice.
            s = i * self.num_shared_experts
            self._register_weight("shared_l1", (2 * s, h), self.shared_spec)
            self._register_weight("shared_l2", (h, s), self.shared_spec)

    def _register_weight(
        self, name: str, shape: tuple[int, ...], spec: MegaMoEWeightSpec
    ) -> None:
        def register(name: str, data: torch.Tensor) -> None:
            self.register_parameter(name, nn.Parameter(data, requires_grad=False))

        *outer, n, k = shape
        register(
            f"{name}_weight",
            torch.zeros(*outer, n, k // spec.values_per_element, dtype=spec.dtype),
        )
        if spec.scale_group_k is not None:
            # Block scales, MN-major, four per int32 along K.
            scale_k = k // (4 * spec.scale_group_k)
            scale = torch.zeros(*outer, scale_k, n, dtype=torch.int32)
            register(f"{name}_weight_scale", scale.transpose(-1, -2))

    def _kernel_weights(
        self, name: str
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(self, f"{name}_weight")
        scale = getattr(self, f"{name}_weight_scale", None)
        return weight if scale is None else (weight, scale)

    @property
    def l1_weights(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_weights("w13")

    @property
    def l2_weights(self) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._kernel_weights("w2")

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Local slots that logical expert ``expert_id`` occupies on this rank."""
        physical_ids: list[int] = []
        for p in range(self.experts_start_idx, self.experts_end_idx):
            if p % self.num_logical_experts == expert_id:
                physical_ids.append(p - self.experts_start_idx)
        return physical_ids

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not loaded_weight.is_meta:
            self._record_global_scale(weight_name, loaded_weight, shard_id, expert_id)
        local_expert_ids = self._map_global_expert_id(expert_id)
        param_prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_expert_ids or param_prefix not in weight_name:
            return False if return_success else None
        is_global_scale = weight_name.endswith(("weight_scale_2", "input_scale"))
        is_scale = weight_name.endswith("weight_scale")
        for local_expert_id in local_expert_ids:
            if is_global_scale:
                param.data[local_expert_id].copy_(loaded_weight.reshape(()))
            else:
                self._load_weight(
                    param.data[local_expert_id], loaded_weight, shard_id, is_scale
                )
        return True if return_success else None

    def _record_global_scale(
        self, weight_name: str, scale: torch.Tensor, shard_id: str, expert_id: int
    ) -> None:
        """Track NVFP4 global scales the kernel takes one of per launch or expert.

        Every rank sees the scales of all experts, so ranks agree and EPLB can
        move experts.
        """
        if weight_name.endswith("w2_input_scale"):
            self._w2_input_scales[expert_id] = scale.max()
            self.l2_activation_scale = float(self._w2_input_scales.max()) or 1.0
        elif weight_name.endswith("w13_weight_scale_2"):
            # Compare w1 and w3 of the same load, not the live (maybe stale) slot.
            prev = self._pending_l1_scales.pop(expert_id, None)
            value = float(scale)
            if prev is None or prev[0] == shard_id:
                self._pending_l1_scales[expert_id] = (shard_id, value)
            elif not math.isclose(value, prev[1], rel_tol=1e-5):
                raise ValueError(
                    f"MegaMoE NVFP4 expert {expert_id}: {prev[0]} weight_scale_2 "
                    f"{prev[1]} != {shard_id} weight_scale_2 {value}; the kernel "
                    "takes one L1 scale per expert."
                )

    def load_shared_expert_weight(self, name: str, loaded_weight: torch.Tensor) -> str:
        """Load one shared-expert checkpoint tensor; return the parameter name."""
        if ".w2." in name or ".down_proj." in name:
            param_name = "shared_l2_weight"
        elif ".w1." in name or ".w3." in name:
            param_name = "shared_l1_weight"
        else:
            raise ValueError(f"Unexpected shared-expert tensor: {name}")
        if "weight_scale" in name:
            param_name += "_scale"
        param = getattr(self, param_name)
        param.weight_loader(param, loaded_weight, name)
        return f"{name.partition('shared_experts.')[0]}experts.{param_name}"

    def shared_weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str
    ) -> None:
        shard_id = (
            "w1" if ".w1." in weight_name else "w3" if ".w3." in weight_name else "w2"
        )
        is_scale = "weight_scale" in weight_name
        self._load_weight(param.data, loaded_weight, shard_id, is_scale)

    def _load_weight(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        shard_id: str,
        is_scale: bool,
    ) -> None:
        """Write checkpoint shard ``src`` into ``dst`` in the kernel layout.

        ``dst`` is one routed expert's slice or a shared-expert parameter.
        """
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        rows = dst.shape[0] // (1 if shard_id == "w2" else 2)
        if is_scale:
            src = self._pack_scale(src, rows, dst.shape[1] * 4)
        else:
            src = src.view(dst.dtype)
            if tuple(src.shape) != (rows, dst.shape[1]):
                raise ValueError(
                    f"MegaMoE {shard_id} weight shape {tuple(src.shape)} does not "
                    f"match the kernel shard ({rows}, {dst.shape[1]})"
                )
        dst, src = self._kernel_layout_shard(dst, src, shard_id, is_scale)
        dst.copy_(src)

    @staticmethod
    def _pack_scale(sf: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
        """Expand block scales to the kernel granularity, four per int32 along K."""
        sf = sf.view(torch.uint8)
        if rows % sf.shape[0] or cols % sf.shape[1]:
            raise ValueError(
                f"Scale shape {tuple(sf.shape)} does not tile ({rows}, {cols})"
            )
        sf = sf.repeat_interleave(rows // sf.shape[0], 0)
        sf = sf.repeat_interleave(cols // sf.shape[1], 1)
        return sf.contiguous().clone().view(torch.int32)

    @staticmethod
    def _kernel_layout_shard(
        dst: torch.Tensor,
        src: torch.Tensor,
        shard_id: str,
        is_scale: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pair a checkpoint shard with its destination in the MegaMoE layout."""
        k = src.shape[-1]
        if shard_id == "w2":
            if not is_scale:
                return dst, src
            # UTCCP: scale row 128*b + 32*j + i -> 128*b + 4*i + j.
            return dst.view(-1, 32, 4, k).transpose(1, 2), src.reshape(-1, 4, 32, k)
        half = 0 if shard_id == "w1" else 1
        if not is_scale:
            # Interleave gate/up by 8 rows: r -> (r // 8) * 16 + 8 * half + r % 8.
            return dst.view(-1, 2, 8, k)[:, half], src.reshape(-1, 8, k)
        # Interleave + UTCCP: 64b + 16j + 8q + m -> 128b + 64q + 32half + 4m + j.
        dst = dst.view(-1, 2, 2, 8, 4, k)[:, :, half].permute(0, 3, 1, 2, 4)
        return dst, src.reshape(-1, 4, 2, 8, k)

    def get_symm_buffer(self):
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
            None if self.shared_spec is None else self.shared_spec.dtype,
            self.weight_format.mma_type,
            self.activation.name,
        )
        symm_buffer = self._symm_buffers.get(key)
        if symm_buffer is None:
            args = (
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
            )
            if self.weight_format.uses_nvfp4_api:
                symm_buffer = deep_gemm.NVFP4SymmBuffer(
                    *args,
                    num_shared_experts=self.num_shared_experts,
                    shared_dtype=torch.float8_e4m3fn
                    if self.shared_spec is None
                    else self.shared_spec.dtype,
                )
            else:
                symm_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
                    *args,
                    num_shared_experts=self.num_shared_experts,
                    mma_type=self.weight_format.mma_type,
                    activation=self.activation.name,
                )
            self._symm_buffers[key] = symm_buffer
        return symm_buffer

    def get_block_m(self, num_tokens: int) -> int:
        """BLOCK_M the MegaMoE scheduler runs ``num_tokens`` local tokens with."""
        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        args = (
            get_ep_group().world_size,
            self.num_experts,
            self.get_symm_buffer().num_max_tokens_per_rank,
            num_tokens,
            self.top_k,
        )
        if self.weight_format.uses_nvfp4_api:
            return deep_gemm.get_block_m_for_nvfp4_mega_moe(*args)
        return deep_gemm.get_block_m_for_mega_moe(*args, self.weight_format.mma_type)

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        self.eplb_state.set_layer_state(
            moe_layer_idx,
            expert_load_view,
            logical_to_physical_map,
            logical_replica_count,
        )

    def get_expert_weights(self) -> list[torch.Tensor]:
        def _to_eplb_view(name: str, t: torch.Tensor) -> torch.Tensor:
            """Return a (num_local_experts, -1) view with contiguous memory layout."""
            assert t.shape[0] == self.num_local_experts
            if t.is_contiguous():
                return t.view(self.num_local_experts, -1)
            elif t.dim() == 3 and t.stride(1) == 1 and t.stride(2) == t.shape[1]:
                # scales have shape (E, M, N) with memory layout (E, N, M)
                back = torch.transpose(t, 1, 2)
                assert back.is_contiguous()
                return back.view(self.num_local_experts, -1)

            raise AssertionError(
                f"MegaMoE EPLB {name}: non-contiguous expert tensor with "
                f"unexpected layout shape={tuple(t.shape)} "
                f"stride={tuple(t.stride())} dtype={t.dtype}"
            )

        return [
            _to_eplb_view(name, param)
            for name, param in self.named_parameters(recurse=False)
            if not name.startswith("shared_")
        ]

    def update_expert_map(self) -> None:
        pass

    @property
    def layer_id(self) -> int:
        return extract_layer_index(self.prefix)

    def _synchronize_first_launch(self) -> None:
        ep_group = get_ep_group()
        key = (id(ep_group.cpu_group), torch.accelerator.current_device_index())
        if key in self._synchronized_ep_groups:
            return
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=ep_group.cpu_group)
        self._synchronized_ep_groups.add(key)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        """Run the routed and shared experts on this rank's tokens.

        Args:
            hidden_states: BF16 ``[num_tokens, hidden_size]`` activations.
            topk_weights: ``[num_tokens, top_k]`` routing weights.
            topk_ids: ``[num_tokens, top_k]`` logical expert ids.
            shared_experts: Unfused shared MLP, run on ``hidden_states`` and
                added to the output. Must be ``None`` when this layer fused the
                shared expert at construction.

        Returns:
            BF16 ``[num_tokens, hidden_size]`` routed plus shared expert output.

        """
        if shared_experts is not None and self.has_fused_shared_experts:
            raise ValueError(
                f"{self.prefix} fuses the shared expert into MegaMoE; "
                "pass shared_experts=None."
            )
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"MegaMoE got {num_tokens} tokens, but the symmetric buffer was "
                f"sized for {self.max_num_tokens}."
            )
        self._synchronize_first_launch()

        is_padding = None
        if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
            is_padding = get_forward_context().is_padding
            if is_padding is not None:
                is_padding = is_padding[:num_tokens]

        if self.capture_fn is not None:
            self.capture_fn(topk_ids)

        # EPLB: map logical expert IDs to physical replicas and record load.
        eplb_state = self.eplb_state
        if eplb_state.logical_to_physical_map is not None:
            assert eplb_state.expert_load_view is not None
            assert eplb_state.logical_replica_count is not None
            assert eplb_state.should_record_tensor is not None
            if is_padding is not None:
                topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=eplb_state.expert_load_view,
                logical_to_physical_map=eplb_state.logical_to_physical_map,
                logical_replica_count=eplb_state.logical_replica_count,
                record_enabled=eplb_state.should_record_tensor,
                num_unpadded_tokens=eplb_state.num_unpadded_tokens_tensors[
                    dbo_current_ubatch_id()
                ]
                if eplb_state.num_unpadded_tokens_tensors is not None
                else None,
                layer_state=eplb_state,
                topk_weights=topk_weights,
            )

        from vllm.utils.deep_gemm import _import_deep_gemm

        deep_gemm = _import_deep_gemm()
        fmt = self.weight_format
        symm_buffer = self.get_symm_buffer()
        shared_x_sf = None
        shared_block_m = None
        if self.shared_spec is not None and self.shared_spec.scale_group_k:
            shared_x_sf = symm_buffer.shared_l1_acts_sf
            shared_block_m = self.get_block_m(num_tokens)
        kwargs = dict(self._kernel_kwargs)
        if fmt.uses_nvfp4_api:
            l2_activation_scale = self.l2_activation_scale
            prepare_megamoe_nvfp4_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                symm_buffer.x[:num_tokens],
                symm_buffer.x_sf[:num_tokens],
                symm_buffer.x_scales[:num_tokens],
                symm_buffer.topk_idx[:num_tokens],
                symm_buffer.topk_weights[:num_tokens],
                is_padding=is_padding,
                shared_x=symm_buffer.shared_l1_acts[:num_tokens]
                if self.has_fused_shared_experts
                else None,
                shared_x_sf=shared_x_sf,
                shared_block_m=shared_block_m,
                w2_global_scale=self.w2_weight_scale_2,
                l2_alpha=self._l2_alpha,
                l2_activation_scale=l2_activation_scale,
            )
            kwargs.update(
                l1_alpha=self.w13_weight_scale_2,
                l2_alpha=self._l2_alpha,
                l2_activation_scale=l2_activation_scale,
                use_x_scales=True,
            )
        else:
            quantized = fmt.hidden_quant is not None
            prepare_megamoe_inputs(
                hidden_states,
                topk_weights,
                topk_ids,
                symm_buffer.x[:num_tokens],
                symm_buffer.x_sf[:num_tokens] if quantized else None,
                symm_buffer.topk_idx[:num_tokens],
                symm_buffer.topk_weights[:num_tokens],
                is_padding=is_padding,
                shared_x_sf=shared_x_sf,
                shared_block_m=shared_block_m,
                hidden_quant=fmt.hidden_quant,
            )

        if self.has_fused_shared_experts:
            kwargs["shared_l1_weights"] = self._kernel_weights("shared_l1")
            kwargs["shared_l2_weights"] = self._kernel_weights("shared_l2")
        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        getattr(deep_gemm, fmt.kernel)(
            y,
            self.l1_weights,
            self.l2_weights,
            symm_buffer,
            fast_math=self.fast_math,
            **kwargs,
        )
        if shared_experts is not None:
            y += shared_experts(hidden_states)
        return y


DeepGemmMegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` helpers for DeepSeek V4 vLLM integration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.distributed import get_ep_group
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from flashinfer.moe_ep import BootstrapConfig, MoEEpMegaLayer

    from vllm.config import VllmConfig


# Every mega kernel is Blackwell-only, so the arch is a property of the family
# rather than of a particular backend: validate it against the live device
# instead of encoding it in the backend name.
FI_MOE_EP_MIN_CAPABILITY = (10, 0)


@dataclass(frozen=True)
class FiMoeEpBackendSpec:
    """Static properties of one ``flashinfer_moe_ep_*`` backend string.

    The backend names the kernel *family*; the arch comes from the device and
    the weight handling from the checkpoint, so this only has to carry which
    megakernel to build and whether it needs NVSHMEM in the runtime set.
    """

    megakernel: str
    needs_nvshmem: bool


FI_MOE_EP_BACKENDS: dict[str, FiMoeEpBackendSpec] = {
    # Consumes an MXFP4 checkpoint verbatim (e2m1 weights + E8M0 per-32
    # scales) -- the same recipe the native deep_gemm mega path uses.
    "flashinfer_moe_ep_mega_deep_gemm": FiMoeEpBackendSpec(
        megakernel="deep_gemm_mega",
        needs_nvshmem=False,
    ),
    # The checkpoint picks the weight path, not the kernel: an NVFP4
    # checkpoint is consumed prequantized (no round trip), while MXFP4 weights
    # are dequantized to bf16 and requantized. See
    # models/deepseek_v4/nvidia/fi_moe.py:ckpt_uses_nvfp4_experts.
    "flashinfer_moe_ep_mega_cutedsl": FiMoeEpBackendSpec(
        megakernel="nvfp4_cutedsl",
        needs_nvshmem=True,
    ),
}

_FI_RUNTIME_HANDLE: Any = None
_FI_MOE_EP_RUNTIME_AVAILABLE: bool | None = None


def _has_fi_moe_ep_runtime() -> bool:
    """True when the installed flashinfer exposes the moe_ep runtime helpers."""
    global _FI_MOE_EP_RUNTIME_AVAILABLE
    if _FI_MOE_EP_RUNTIME_AVAILABLE is not None:
        return _FI_MOE_EP_RUNTIME_AVAILABLE
    try:
        from flashinfer.moe_ep import (  # noqa: F401
            bootstrap_moe_ep_runtime,
            ensure_moe_ep_cuda_device,
            finalize_moe_ep_runtime,
        )
    except ImportError:
        _FI_MOE_EP_RUNTIME_AVAILABLE = False
    else:
        _FI_MOE_EP_RUNTIME_AVAILABLE = True
    return _FI_MOE_EP_RUNTIME_AVAILABLE


def is_fi_moe_ep_backend(moe_backend: str) -> bool:
    if moe_backend not in FI_MOE_EP_BACKENDS:
        return False
    if not _has_fi_moe_ep_runtime():
        raise ImportError(
            f"moe_backend={moe_backend!r} requires the flashinfer.moe_ep "
            "runtime, which the installed flashinfer does not provide. "
            "Install a flashinfer build with moe_ep support, or use "
            "moe_backend=deep_gemm_mega_moe for the native mega path."
        )
    return True


def fi_moe_ep_backend_spec(moe_backend: str) -> FiMoeEpBackendSpec:
    try:
        return FI_MOE_EP_BACKENDS[moe_backend]
    except KeyError:
        raise ValueError(
            f"{moe_backend!r} is not a flashinfer moe_ep backend; expected "
            f"one of {sorted(FI_MOE_EP_BACKENDS)}"
        ) from None


def validate_fi_moe_ep_config(vllm_config: VllmConfig) -> None:
    """Config-time checks for the mega-MoE backends, native and flashinfer."""
    moe_backend = vllm_config.kernel_config.moe_backend
    if not is_fi_moe_ep_backend(moe_backend):
        return

    # flashinfer validates the arch too, but not until the layer constructor
    # runs during weight load; check here so the error names the flag the user
    # actually typed.
    capability = current_platform.get_device_capability()
    if capability is not None:
        cc = (capability.major, capability.minor)
        if cc < FI_MOE_EP_MIN_CAPABILITY:
            want = FI_MOE_EP_MIN_CAPABILITY
            raise ValueError(
                f"moe_backend={moe_backend!r} requires compute capability "
                f"{want[0]}.{want[1]}+ (the mega kernels are Blackwell-only), "
                f"but this device is {cc[0]}.{cc[1]}."
            )

    if vllm_config.parallel_config.enable_eplb:
        raise NotImplementedError(
            f"EPLB is not supported with moe_backend={moe_backend!r}: the "
            "flashinfer moe_ep experts neither apply the logical-to-physical "
            "expert map nor report per-expert load, so rebalancing would move "
            "weights without moving routing. Use "
            "moe_backend=deep_gemm_mega_moe to run the mega path with EPLB."
        )


def make_fi_moe_ep_bootstrap() -> BootstrapConfig:
    from flashinfer.moe_ep import BootstrapConfig

    ep = get_ep_group()
    return BootstrapConfig(
        world_size=ep.world_size,
        rank=ep.rank_in_group,
        process_group=ep.device_group,
        auto_bootstrap=False,
    )


def megakernel_runtime_requirements(spec: FiMoeEpBackendSpec) -> frozenset[str]:
    from flashinfer.moe_ep.core.runtime import NVSHMEM, TORCH_DIST

    if spec.needs_nvshmem:
        return frozenset({TORCH_DIST, NVSHMEM})
    return frozenset({TORCH_DIST})


def ensure_fi_moe_ep_runtime(vllm_config: VllmConfig) -> None:
    """Acquire the process-wide flashinfer moe_ep runtime once per worker."""
    global _FI_RUNTIME_HANDLE
    if _FI_RUNTIME_HANDLE is not None:
        return

    from flashinfer.moe_ep import bootstrap_moe_ep_runtime

    if not _has_fi_moe_ep_runtime():
        raise ImportError(
            "flashinfer.moe_ep runtime helpers are not available in this "
            "flashinfer build."
        )

    bootstrap = make_fi_moe_ep_bootstrap()
    spec = fi_moe_ep_backend_spec(vllm_config.kernel_config.moe_backend)
    # flashinfer's runtime/layer constructors bind the process to
    # cuda:LOCAL_RANK (falling back to bootstrap.rank). vLLM has already bound
    # this worker to its (possibly remapped) visible device, and a mismatched
    # rebind launches the weight transforms on the wrong GPU against another
    # device's pointers (observed as CUDA_ERROR_ILLEGAL_ADDRESS in the
    # deep_gemm transform_sf during load). Pin LOCAL_RANK to the device vLLM
    # chose so every internal set_device is a no-op. TODO: drop once
    # flashinfer respects the caller's current device.
    os.environ["LOCAL_RANK"] = str(torch.cuda.current_device())
    _FI_RUNTIME_HANDLE = bootstrap_moe_ep_runtime(
        bootstrap,
        megakernel_runtime_requirements(spec),
    )


def finalize_fi_moe_ep_runtime() -> None:
    """Release the process-wide flashinfer moe_ep runtime."""
    global _FI_RUNTIME_HANDLE
    if _FI_RUNTIME_HANDLE is None:
        return

    from flashinfer.moe_ep import finalize_moe_ep_runtime

    finalize_moe_ep_runtime(_FI_RUNTIME_HANDLE)
    _FI_RUNTIME_HANDLE = None


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


def _dequant_fp4_ue8m0_gran32(
    packed: torch.Tensor, sf_ue8m0: torch.Tensor
) -> torch.Tensor:
    """[rows, K//2] packed e2m1 + [rows, K//32] ue8m0-uint8 scales -> bf16 [rows, K]."""
    raw = packed.view(torch.uint8)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    vals = torch.empty(
        raw.shape[0], raw.shape[1] * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    vals[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    sf = (sf_ue8m0.to(torch.int32) << 23).view(torch.float32)
    return (vals * sf.repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def _dequant_expert_weights_to_bf16(
    weight: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """[E, N, K//2] fp4 + [E, N, K//32] ue8m0 -> [E, N, K] bf16 (expert loop)."""
    num_experts, n, k_half = weight.shape
    out = torch.empty(
        num_experts, n, k_half * 2, dtype=torch.bfloat16, device=weight.device
    )
    for e in range(num_experts):
        out[e] = _dequant_fp4_ue8m0_gran32(weight[e], scale[e])
    return out


def mega_moe_weight_pack_from_params(
    w13_weight: nn.Parameter,
    w13_weight_scale: nn.Parameter,
    w2_weight: nn.Parameter,
    w2_weight_scale: nn.Parameter,
    *,
    megakernel: str = "deep_gemm_mega",
):
    from flashinfer.moe_ep import MoEWeightPack

    if megakernel == "deep_gemm_mega":
        # Same fp4-e2m1 + ue8m0-per-32 recipe as the native path: pass verbatim,
        # flashinfer runs the identical deep_gemm transform.
        return MoEWeightPack(
            w13=w13_weight.data,
            w2=w2_weight.data,
            w13_scale=w13_weight_scale.data,
            w2_scale=w2_weight_scale.data,
        )
    # The cutedsl kernel quantizes with its own recipe (nvfp4
    # e2m1+e4m3-per-16): dequantize the checkpoint fp4 to bf16 and let the
    # backend preprocess requantize. Double quantization: outputs are close to
    # but not bit-identical with the native path.
    return MoEWeightPack(
        w13=_dequant_expert_weights_to_bf16(w13_weight.data, w13_weight_scale.data),
        w2=_dequant_expert_weights_to_bf16(w2_weight.data, w2_weight_scale.data),
    )


def build_fi_mega_config(
    *,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    megakernel: str,
    fast_math: bool = True,
):
    from flashinfer.moe_ep import (
        DeepGemmMegaMoeConfig,
        MegaConfig,
        Nvfp4CutedslMegaMoeConfig,
    )

    if megakernel == "deep_gemm_mega":
        mk = DeepGemmMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    elif megakernel == "nvfp4_cutedsl":
        mk = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
    else:
        raise ValueError(f"Unsupported fi_moe_ep megakernel {megakernel!r}")

    return MegaConfig(
        megakernel=mk,
        preprocess_weights=True,
        quantize_input=True,
    )


# All MoE layers share one symmetric workspace, like the native path's
# class-level DeepseekV4MegaMoEExperts._symm_buffer_cache. Without this the
# fi path allocates one symm buffer PER LAYER (43x memory + cold working
# sets); the workspace is stateless across forwards (kernel tail-cleans) and
# layers execute sequentially on one stream, so sharing is safe.
def build_fi_mega_layer(
    bootstrap: BootstrapConfig,
    *,
    vllm_config: VllmConfig,
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    weights,
    fast_math: bool = True,
) -> MoEEpMegaLayer:
    from flashinfer.moe_ep import FleetParams, MoEEpLayer

    megakernel = fi_moe_ep_backend_spec(
        vllm_config.kernel_config.moe_backend
    ).megakernel
    mega_config = build_fi_mega_config(
        intermediate_size=intermediate_size,
        top_k=top_k,
        activation_clamp=activation_clamp,
        megakernel=megakernel,
        fast_math=fast_math,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=hidden_size,
        ),
        weights=weights,
        backend=mega_config,
    )
    from flashinfer.moe_ep import MoEEpMegaLayer

    if not isinstance(layer, MoEEpMegaLayer):
        raise TypeError(
            f"fi_moe_ep expected MoEEpMegaLayer, got {type(layer).__name__}"
        )
    return layer


__all__ = [
    "FI_MOE_EP_BACKENDS",
    "FiMoeEpBackendSpec",
    "build_fi_mega_config",
    "build_fi_mega_layer",
    "ensure_fi_moe_ep_runtime",
    "fi_moe_ep_backend_spec",
    "finalize_fi_moe_ep_runtime",
    "is_fi_moe_ep_backend",
    "make_fi_moe_ep_bootstrap",
    "mega_moe_weight_pack_from_params",
    "megakernel_runtime_requirements",
    "validate_fi_moe_ep_config",
]

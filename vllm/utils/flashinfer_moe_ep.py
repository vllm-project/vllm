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

DeepseekV4MegaMoEExpertsFI: type | None = None


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
    # are dequantized to bf16 and requantized. See ckpt_uses_nvfp4_experts.
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


def validate_fi_moe_ep_config(vllm_config: "VllmConfig") -> None:
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


def make_fi_moe_ep_bootstrap() -> "BootstrapConfig":
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


def ensure_fi_moe_ep_runtime(vllm_config: "VllmConfig") -> None:
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
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
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


def ckpt_uses_nvfp4_experts(vllm_config: "VllmConfig") -> bool:
    """True when the loaded checkpoint quantizes experts with modelopt NVFP4
    (e2m1 + fp8-e4m3 per-16 block scales + per-tensor weight_scale_2), i.e.
    the recipe the nvfp4_cutedsl prequantized-weights path consumes verbatim.

    DeepSeek-V4 NVFP4-expert checkpoints declare ``moe_quant_algo: NVFP4`` in
    their ``quantization_config``, which DeepseekV4FP8Config surfaces as
    ``moe_quant_algo``.
    """
    return getattr(vllm_config.quant_config, "moe_quant_algo", "") == "NVFP4"


def nvfp4_prequant_pack_and_alphas(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,  # (E_local, 2) fp32: [:,0]=gate(w1), [:,1]=up(w3)
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,  # (E_local,) fp32
    *,
    intermediate_size: int,
):
    """NVFP4 checkpoint params -> (PrequantizedMoEWeights, fc1_alpha, fc2_alpha).

    Scale algebra (see mega_reference.py:_expert_reference): the kernel's fc1
    accumulator is gemm(x_fp4*x_sf, w_fp4*w_sf) * fc1_alpha. Our activation
    quant is fully dynamic (block sf carries the real magnitude,
    input_norm_const=1), so the checkpoint's static ``input_scale`` drops out
    and fc1_alpha reduces to the weight's per-tensor global: weight_scale_2.
    Same for fc2 (the internal swiglu requant is dynamic, fc1_norm_const=1):
    fc2_alpha = w2.weight_scale_2.

    fc1_alpha is ONE scalar per expert but gate (w1) and up (w3) carry their
    own weight_scale_2. When they differ, the ratio is folded into the UP
    half's e4m3 block scales — ONLY if the fold round-trips exactly
    (apples-to-apples rule: a lossy rescale silently changes the model; see
    todo_nvfp4_prequant_checkpoint.md). Power-of-two ratios (this checkpoint
    is a cast from mxfp4's power-of-two scales) fold exactly.
    """
    from flashinfer.moe_ep import MoEWeightPack

    inter = intermediate_size
    gate_s2 = w13_weight_scale_2[:, 0].float()
    up_s2 = w13_weight_scale_2[:, 1].float()
    if (gate_s2 <= 0).any() or (up_s2 <= 0).any() or (w2_weight_scale_2 <= 0).any():
        raise ValueError(
            "nvfp4 prequant: non-positive weight_scale_2 loaded — checkpoint "
            "scale tensors missing or loader routed them wrong."
        )

    w13_scale = w13_weight_scale
    if not torch.equal(gate_s2, up_s2):
        ratio = up_s2 / gate_s2  # (E_local,)
        up_sf = w13_scale[:, inter:, :].float()
        folded = up_sf * ratio[:, None, None]
        folded_e4m3 = folded.to(torch.float8_e4m3fn)
        if not torch.equal(folded_e4m3.float(), folded):
            raise ValueError(
                "nvfp4 prequant: gate/up weight_scale_2 ratio does not fold "
                "exactly into e4m3 block scales; refusing the lossy rescale "
                "(apples-to-apples rule — repackage the checkpoint with "
                "merged gate/up quantization instead)."
            )
        w13_scale = w13_scale.clone()
        w13_scale[:, inter:, :] = folded_e4m3

    fc1_alpha = gate_s2.clone().contiguous()
    fc2_alpha = w2_weight_scale_2.float().clone().contiguous()
    pack = MoEWeightPack(
        w13=w13_weight,
        w2=w2_weight,
        w13_scale=w13_scale,
        w2_scale=w2_weight_scale,
    )
    return pack, fc1_alpha, fc2_alpha


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
    bootstrap: "BootstrapConfig",
    *,
    vllm_config: "VllmConfig",
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    activation_clamp: float | None,
    weights,
    fast_math: bool = True,
) -> "MoEEpMegaLayer":
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


_MOE_SKIP_PADDING: bool | None = None


def resolve_mega_moe_is_padding(num_tokens: int) -> torch.Tensor | None:
    from vllm.forward_context import get_forward_context, is_forward_context_available

    global _MOE_SKIP_PADDING
    if _MOE_SKIP_PADDING is None:
        import vllm.envs as envs

        _MOE_SKIP_PADDING = bool(envs.VLLM_MOE_SKIP_PADDING)
    if not _MOE_SKIP_PADDING or not is_forward_context_available():
        return None
    is_padding = get_forward_context().is_padding
    if is_padding is None:
        return None
    return is_padding[:num_tokens]


def apply_mega_moe_routing_preprocess(
    topk_ids: torch.Tensor,
    *,
    is_padding: torch.Tensor | None = None,
) -> torch.Tensor:
    """Padding-only routing preprocess (EPLB hooks go here later)."""
    if is_padding is not None:
        topk_ids = torch.where(is_padding.unsqueeze(1), -1, topk_ids)
    return topk_ids


def make_fi_mega_moe_experts_cls(mega_moe_experts_cls: type[nn.Module]) -> type[nn.Module]:
    """Build ``DeepseekV4MegaMoEExpertsFI`` once the base mega experts class exists."""
    global DeepseekV4MegaMoEExpertsFI

    class _DeepseekV4MegaMoEExpertsFI(mega_moe_experts_cls):
        """Thin wrapper: same weight layout/loader as mega experts, FI compute path."""

        def __init__(
            self,
            vllm_config: "VllmConfig",
            *,
            activation_clamp: float | None = None,
            fast_math: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__(vllm_config, **kwargs)
            self._vllm_config = vllm_config
            self._activation_clamp = activation_clamp
            self._fast_math = fast_math
            self._mega_layer = None
            self._fast_ctx = None
            self._epilogue_alphas: tuple[torch.Tensor, torch.Tensor] | None = None
            self._nvfp4_prequant = ckpt_uses_nvfp4_experts(vllm_config)
            if self._nvfp4_prequant:
                megakernel = fi_moe_ep_backend_spec(
                    vllm_config.kernel_config.moe_backend
                ).megakernel
                if megakernel != "nvfp4_cutedsl":
                    raise ValueError(
                        "NVFP4-quantized expert checkpoint requires "
                        "moe_backend=flashinfer_moe_ep_mega_cutedsl, got a "
                        f"backend using megakernel {megakernel!r} "
                        "(deep_gemm consumes the MXFP4 checkpoint instead)."
                    )
                self._realloc_nvfp4_params()

        def _realloc_nvfp4_params(self) -> None:
            """Swap the mx-recipe scale params for the NVFP4 checkpoint's:
            fp8-e4m3 per-16 block scales plus the per-tensor second-level
            scales (weight_scale_2, input_scale) modelopt exports."""
            from vllm.model_executor.utils import set_weight_attrs

            n_e = self.num_local_experts
            inter = self.intermediate_size
            hidden = self.hidden_size
            attrs = {"weight_loader": self.weight_loader}

            def _param(shape: tuple, dtype: torch.dtype) -> nn.Parameter:
                p = nn.Parameter(
                    torch.zeros(*shape, dtype=dtype), requires_grad=False
                )
                set_weight_attrs(p, attrs)
                return p

            self.w13_weight_scale = _param(
                (n_e, 2 * inter, hidden // 16), torch.float8_e4m3fn
            )
            self.w13_weight_scale.quant_method = "block"
            self.w2_weight_scale = _param(
                (n_e, hidden, inter // 16), torch.float8_e4m3fn
            )
            self.w2_weight_scale.quant_method = "block"
            # (E, 2): column 0 = gate (w1), column 1 = up (w3).
            self.w13_weight_scale_2 = _param((n_e, 2), torch.float32)
            self.w2_weight_scale_2 = _param((n_e,), torch.float32)
            # Loaded for completeness; unused — activation quant is dynamic
            # (see nvfp4_prequant_pack_and_alphas).
            self.w13_input_scale = _param((n_e, 2), torch.float32)
            self.w2_input_scale = _param((n_e,), torch.float32)

        def weight_loader(
            self,
            param: nn.Parameter,
            loaded_weight: torch.Tensor,
            weight_name: str,
            shard_id: str,
            expert_id: int,
            return_success: bool = False,
        ) -> bool | None:
            # NVFP4 checkpoint second-level scalars route here; everything
            # else (packed weights, block scales) matches the base layout.
            if "weight_scale_2" in weight_name or "input_scale" in weight_name:
                local_expert_ids = self._map_global_expert_id(expert_id)
                if not local_expert_ids:
                    return False if return_success else None
                value = loaded_weight.reshape(()).to(torch.float32)
                for local_expert_id in local_expert_ids:
                    if shard_id in ("w1", "w3"):
                        if "w13_" not in weight_name:
                            return False if return_success else None
                        param.data[local_expert_id, 0 if shard_id == "w1" else 1] = value
                    elif shard_id == "w2":
                        if "w2_" not in weight_name:
                            return False if return_success else None
                        param.data[local_expert_id] = value
                    else:
                        raise ValueError(f"Unsupported expert shard id: {shard_id}")
                return True if return_success else None
            return super().weight_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success,
            )

        def finalize_weights(self) -> None:
            if self._mega_layer is not None:
                return
            if self.w13_weight is None:
                return

            self._check_runtime_supported()
            ensure_fi_moe_ep_runtime(self._vllm_config)

            if self._nvfp4_prequant:
                # NVFP4 checkpoint: hand the packed weights + both scale
                # planes straight to the backend (no dequant->requant);
                # per-expert globals become fc1/fc2 epilogue alphas staged
                # at every forward via MoEEpTensors.
                weights, fc1_alpha, fc2_alpha = nvfp4_prequant_pack_and_alphas(
                    self.w13_weight.data,
                    self.w13_weight_scale.data,
                    self.w13_weight_scale_2.data,
                    self.w2_weight.data,
                    self.w2_weight_scale.data,
                    self.w2_weight_scale_2.data,
                    intermediate_size=self.intermediate_size,
                )
                self._epilogue_alphas = (fc1_alpha, fc2_alpha)
            else:
                weights = mega_moe_weight_pack_from_params(
                    self.w13_weight,
                    self.w13_weight_scale,
                    self.w2_weight,
                    self.w2_weight_scale,
                    megakernel=fi_moe_ep_backend_spec(
                        self._vllm_config.kernel_config.moe_backend
                    ).megakernel,
                )
            self._mega_layer = build_fi_mega_layer(
                make_fi_moe_ep_bootstrap(),
                vllm_config=self._vllm_config,
                num_experts=self.num_experts,
                max_tokens_per_rank=self.max_num_tokens,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                activation_clamp=self._activation_clamp,
                weights=weights,
                fast_math=self._fast_math,
            )
            # Upstream (fi branch >= 888383f5): the layer releases the source
            # MoEWeightPack after preprocess, and workspaces are pooled across
            # same-geometry layers by core/kernel/workspace_pool.py — the old
            # `_weights = None` and `_SHARED_WORKSPACE` workarounds are gone.
            del weights
            # Allocate (or attach to) the pooled workspace before first
            # forward so warmup/capture never hits the lazy path.
            self._mega_layer._ensure_workspace()
            self.w13_weight = None
            self.w13_weight_scale = None
            self.w2_weight = None
            self.w2_weight_scale = None
            if self._nvfp4_prequant:
                self.w13_weight_scale_2 = None
                self.w2_weight_scale_2 = None
                self.w13_input_scale = None
                self.w2_input_scale = None

        # EPLB is rejected for these backends in validate_fi_moe_ep_config, so
        # nothing constructs an EplbState and none of these run. They are kept
        # as explicit errors rather than deleted because inheriting the native
        # implementations would be worse: set_eplb_state would record a map
        # this forward path never consults, silently splitting routing from
        # weights, and get_expert_weights reads transformed-weight attributes
        # the flashinfer path releases after preprocess.
        _EPLB_UNSUPPORTED = (
            "EPLB is not supported with the flashinfer moe_ep backends; "
            "validate_fi_moe_ep_config should have rejected this "
            "configuration at startup."
        )

        def set_eplb_state(
            self,
            moe_layer_idx: int,
            expert_load_view: torch.Tensor,
            logical_to_physical_map: torch.Tensor,
            logical_replica_count: torch.Tensor,
        ) -> None:
            raise NotImplementedError(self._EPLB_UNSUPPORTED)

        def get_expert_weights(self) -> list[torch.Tensor]:
            raise NotImplementedError(self._EPLB_UNSUPPORTED)

        def update_expert_map(self) -> None:
            raise NotImplementedError(self._EPLB_UNSUPPORTED)

        def forward(
            self,
            hidden_states: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            *,
            activation_clamp: float | None,
            fast_math: bool = True,
        ) -> torch.Tensor:
            if hidden_states.shape[0] > self.max_num_tokens:
                raise ValueError(
                    f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                    f"but the symmetric buffer was sized for {self.max_num_tokens}."
                )

            from flashinfer.moe_ep import MoEEpTensors

            num_tokens = hidden_states.shape[0]
            is_padding = resolve_mega_moe_is_padding(num_tokens)
            topk_ids = apply_mega_moe_routing_preprocess(
                topk_ids,
                is_padding=is_padding,
            )

            # Validated-once fast path (mirrors the microbench's cached-launch
            # loop): MoEEpMegaLayer.forward() re-runs bootstrap/dist checks and
            # input validation on every call, which costs real host time at
            # 43 MoE layers x one call per engine step. After the first
            # successful full forward the layer is immutable, so go straight
            # to the kernel backend's stage_inputs + compute.
            alphas = self._epilogue_alphas
            fc1_alpha = alphas[0] if alphas is not None else None
            fc2_alpha = alphas[1] if alphas is not None else None

            fast = self._fast_ctx
            if fast is not None:
                kernel, workspace, transformed, hidden_size, zero_copy = fast
                t = MoEEpTensors(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    fc1_alpha=fc1_alpha,
                    fc2_alpha=fc2_alpha,
                )
                kernel.stage_inputs(t, workspace, quantize_input=True)
                if zero_copy:
                    # Zero-copy (cutedsl backends): consume the workspace [:n]
                    # view directly (valid under stream ordering until the next
                    # MoE layer's launch on the shared workspace — downstream
                    # ops are enqueued first).
                    return kernel.compute(workspace, transformed, output=None)
                # deep_gemm_mega: its compute() requires a real output tensor
                # (arg0 of the pybind fp8_fp4_mega_moe) — output=None lands as
                # None in the binding and TypeErrors (found 2026-07-19; the
                # fast path had only been exercised on cutedsl backends since
                # the zero-copy change).
                out = torch.empty(
                    num_tokens,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=hidden_states.device,
                )
                return kernel.compute(workspace, transformed, output=out)

            ensure_fi_moe_ep_runtime(self._vllm_config)
            self.finalize_weights()
            assert self._mega_layer is not None

            y = self._mega_layer.forward(
                MoEEpTensors(
                    hidden_states=hidden_states,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    fc1_alpha=fc1_alpha,
                    fc2_alpha=fc2_alpha,
                )
            )
            layer = self._mega_layer
            if hidden_states.dtype == torch.bfloat16:
                self._fast_ctx = (
                    layer._kernel,
                    layer._ensure_workspace(),
                    layer._transformed,
                    layer._fleet_params.token_hidden_size,
                    # zero-copy output views are a cutedsl-backend contract
                    layer._kernel.kernel_name() != "deep_gemm_mega",
                )
            return y

    _DeepseekV4MegaMoEExpertsFI.__name__ = "DeepseekV4MegaMoEExpertsFI"
    _DeepseekV4MegaMoEExpertsFI.__qualname__ = "DeepseekV4MegaMoEExpertsFI"
    _DeepseekV4MegaMoEExpertsFI.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]

    DeepseekV4MegaMoEExpertsFI = _DeepseekV4MegaMoEExpertsFI
    return DeepseekV4MegaMoEExpertsFI


__all__ = [
    "DeepseekV4MegaMoEExpertsFI",
    "FI_MOE_EP_BACKENDS",
    "FiMoeEpBackendSpec",
    "apply_mega_moe_routing_preprocess",
    "build_fi_mega_config",
    "build_fi_mega_layer",
    "ckpt_uses_nvfp4_experts",
    "nvfp4_prequant_pack_and_alphas",
    "ensure_fi_moe_ep_runtime",
    "fi_moe_ep_backend_spec",
    "finalize_fi_moe_ep_runtime",
    "is_fi_moe_ep_backend",
    "make_fi_mega_moe_experts_cls",
    "make_fi_moe_ep_bootstrap",
    "mega_moe_weight_pack_from_params",
    "megakernel_runtime_requirements",
    "resolve_mega_moe_is_padding",
    "validate_fi_moe_ep_config",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer ``moe_ep`` helpers for DeepSeek V4 vLLM integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.config.kernel import FLASHINFER_MOE_EP_BACKENDS
from vllm.distributed import get_ep_group
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from flashinfer.moe_ep import BootstrapConfig, MoEEpMegaLayer

    from vllm.config import VllmConfig


@dataclass(frozen=True)
class FiMoeEpBackendSpec:
    """Static properties of one ``flashinfer_moe_ep_*`` backend string.

    The backend names the kernel *family*; the arch comes from the device and
    the weight handling from the checkpoint, so this only has to carry which
    megakernel to build, whether it needs NVSHMEM in the runtime set, and
    which compute capabilities the family supports. The capability allowlist
    is explicit so new archs (SM110/SM120, Rubin) fail loudly until flashinfer
    supports them.
    """

    megakernel: str
    needs_nvshmem: bool
    supported_capabilities: frozenset[tuple[int, int]]
    # Whether the model can route through DeepGEMM's fused gate kernel
    # (``bf16_mega_gate``). That kernel is SM100-only, so the Hopper
    # megakernel falls back to the eager gate + ``fused_topk_bias`` path.
    uses_fused_mega_gate: bool


_BLACKWELL_CAPABILITIES = frozenset({(10, 0), (10, 3)})
_HOPPER_CAPABILITIES = frozenset({(9, 0)})


FI_MOE_EP_BACKEND_SPECS: dict[str, FiMoeEpBackendSpec] = {
    # Consumes an MXFP4 checkpoint verbatim (e2m1 weights + E8M0 per-32
    # scales) -- the same recipe the native deep_gemm mega path uses.
    "flashinfer_moe_ep_mega_deep_gemm": FiMoeEpBackendSpec(
        megakernel="deep_gemm_mega",
        needs_nvshmem=False,
        supported_capabilities=_BLACKWELL_CAPABILITIES,
        uses_fused_mega_gate=True,
    ),
    # The checkpoint picks the weight path, not the kernel: an NVFP4
    # checkpoint is consumed prequantized (no round trip), while MXFP4 weights
    # are dequantized to bf16 and requantized. See
    # models/deepseek_v4/nvidia/fi_moe.py:ckpt_uses_nvfp4_experts.
    "flashinfer_moe_ep_mega_cutedsl": FiMoeEpBackendSpec(
        megakernel="nvfp4_cutedsl",
        needs_nvshmem=True,
        supported_capabilities=_BLACKWELL_CAPABILITIES,
        uses_fused_mega_gate=True,
    ),
    # Hopper (SM90) FP8 pull-style CuteDSL megakernel. MXFP4 weights are
    # converted directly to kernel-ready blockwise FP8; NVFP4 weights are
    # dequantized to bf16 and requantized by the backend.
    "flashinfer_moe_ep_mega_sm90_fp8": FiMoeEpBackendSpec(
        megakernel="sm90_fp8_pull",
        needs_nvshmem=True,
        supported_capabilities=_HOPPER_CAPABILITIES,
        uses_fused_mega_gate=False,
    ),
}

assert set(FI_MOE_EP_BACKEND_SPECS) == FLASHINFER_MOE_EP_BACKENDS

_FI_RUNTIME_HANDLE: Any = None


def is_fi_moe_ep_backend(moe_backend: str) -> bool:
    return moe_backend in FI_MOE_EP_BACKEND_SPECS


def fi_moe_ep_backend_spec(moe_backend: str) -> FiMoeEpBackendSpec:
    try:
        return FI_MOE_EP_BACKEND_SPECS[moe_backend]
    except KeyError:
        raise ValueError(
            f"{moe_backend!r} is not a flashinfer moe_ep backend; expected "
            f"one of {sorted(FI_MOE_EP_BACKEND_SPECS)}"
        ) from None


def validate_fi_moe_ep_config(vllm_config: VllmConfig) -> None:
    """Config-time checks for the mega-MoE backends, native and flashinfer."""
    moe_backend = vllm_config.kernel_config.moe_backend
    if not is_fi_moe_ep_backend(moe_backend):
        return

    spec = fi_moe_ep_backend_spec(moe_backend)
    # flashinfer validates the arch too, but not until the layer constructor
    # runs during weight load; check here so the error names the flag the user
    # actually typed.
    capability = current_platform.get_device_capability()
    if capability is not None:
        cc = (capability.major, capability.minor)
        if cc not in spec.supported_capabilities:
            supported = ", ".join(
                f"{m}.{n}" for m, n in sorted(spec.supported_capabilities)
            )
            raise ValueError(
                f"moe_backend={moe_backend!r} is only supported on compute "
                f"capability {supported}, but this device is {cc[0]}.{cc[1]}."
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
        # vLLM already bound this worker's (possibly remapped) device;
        # without this the runtime would rebind to cuda:LOCAL_RANK|rank and
        # launch weight transforms against another device's pointers.
        device=torch.accelerator.current_device_index(),
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

    bootstrap = make_fi_moe_ep_bootstrap()
    spec = fi_moe_ep_backend_spec(vllm_config.kernel_config.moe_backend)
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


def _dequant_nvfp4_expert_weights_to_bf16(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    *,
    gate_rows: int | None = None,
) -> torch.Tensor:
    """[E, N, K//2] e2m1 + [E, N, K//16] fp8-e4m3 + per-tensor scale -> bf16.

    modelopt NVFP4 exports a per-16 e4m3 block scale plus a per-tensor
    ``weight_scale_2``. For a fused ``w13`` tensor the two second-level
    scalars are per expert (column 0 = gate rows ``[0:gate_rows]``, column 1 =
    up rows); for ``w2`` there is one scalar per expert.
    """
    raw = weight.view(torch.uint8)
    num_experts, n, k_half = raw.shape
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    vals = torch.empty(
        num_experts, n, k_half * 2, dtype=torch.float32, device=raw.device
    )
    vals[:, :, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    vals[:, :, 1::2] = lut[(raw >> 4).to(torch.int64)]
    vals = vals * block_scale.to(torch.float32).repeat_interleave(16, dim=-1)

    s2 = weight_scale_2.to(torch.float32)
    if s2.ndim == 2:
        if gate_rows is None:
            raise ValueError(
                "gate_rows is required to dequantize a fused w13 NVFP4 tensor"
            )
        s2_rows = torch.empty(num_experts, n, dtype=torch.float32, device=raw.device)
        s2_rows[:, :gate_rows] = s2[:, 0:1]
        s2_rows[:, gate_rows:] = s2[:, 1:2]
        vals = vals * s2_rows[:, :, None]
    else:
        vals = vals * s2[:, None, None]
    return vals.to(torch.bfloat16)


def mega_moe_weight_pack_from_params(
    w13_weight: nn.Parameter,
    w13_weight_scale: nn.Parameter,
    w2_weight: nn.Parameter,
    w2_weight_scale: nn.Parameter,
    *,
    megakernel: str = "deep_gemm_mega",
    w13_weight_scale_2: nn.Parameter | None = None,
    w2_weight_scale_2: nn.Parameter | None = None,
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
    if megakernel == "sm90_fp8_pull":
        # The SM90 FP8 backend's preprocess_weights() wants canonical bf16
        # w13/w2 and requantizes to blockwise FP8 itself. Handles both the
        # MXFP4 (e2m1 + ue8m0-per-32) and NVFP4 (e2m1 + e4m3-per-16 +
        # per-tensor scale_2) checkpoint recipes.
        if w13_weight_scale_2 is not None:
            if w2_weight_scale_2 is None:
                raise ValueError(
                    "w2_weight_scale_2 is required with NVFP4 expert weights"
                )
            gate_rows = w13_weight.data.shape[1] // 2
            return MoEWeightPack(
                w13=_dequant_nvfp4_expert_weights_to_bf16(
                    w13_weight.data,
                    w13_weight_scale.data,
                    w13_weight_scale_2.data,
                    gate_rows=gate_rows,
                ),
                w2=_dequant_nvfp4_expert_weights_to_bf16(
                    w2_weight.data,
                    w2_weight_scale.data,
                    w2_weight_scale_2.data,
                ),
            )
        return MoEWeightPack(
            w13=_dequant_expert_weights_to_bf16(w13_weight.data, w13_weight_scale.data),
            w2=_dequant_expert_weights_to_bf16(w2_weight.data, w2_weight_scale.data),
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
    transformed_weights=None,
):
    from flashinfer.moe_ep import (
        DeepGemmMegaMoeConfig,
        MegaConfig,
        Nvfp4CutedslMegaMoeConfig,
        Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
    )

    # fast_math selects approximate exp/rcp in DeepGEMM's fused SwiGLU
    # epilogue; the cutedsl kernels accept it for API parity only.
    if megakernel == "deep_gemm_mega":
        mk = DeepGemmMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=True,
        )
    elif megakernel == "nvfp4_cutedsl":
        mk = Nvfp4CutedslMegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            activation_clamp=activation_clamp,
            fast_math=True,
        )
    elif megakernel == "sm90_fp8_pull":
        # Hopper FP8 pull-style CuteDSL mega kernel. Blockwise scaling is used
        # so activation quantization is fully dynamic (per-token/128-block) and
        # needs no cross-rank calibration; the SM90 tree's per-tensor mode would
        # require a static calibration scalar identical on every EP rank.
        mk = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            kind="fp8_e4m3",
            fp8_scale_mode="blockwise",
            fp8_accum_mode="1xacc",
            # DeepSeek V4's large expert dimensions favor Hopper's
            # weight-as-A specialization and its decode-oriented token tile.
            swap_ab=True,
            mma_tiler_mnk=(256, 16, 128),
            gate_up_clamp=activation_clamp,
            fast_math=True,
        )
    else:
        raise ValueError(f"Unsupported fi_moe_ep megakernel {megakernel!r}")

    # ``transformed_weights`` means the caller already emitted kernel-ready
    # weights (see build_sm90_fp8_transformed_weights), so the backend must
    # skip its own bf16 -> FP8 preprocessing.
    return MegaConfig(
        megakernel=mk,
        preprocess_weights=transformed_weights is None,
        transformed_weights=transformed_weights,
        quantize_input=True,
    )


# --- SM90 FP8 mega: fused fp4 -> fp8 weight conversion -----------------------
#
# The SM90 megakernel wants blockwise FP8 with the layout flashinfer builds in
# ``preprocess_mega_weights``. That routine only accepts canonical bf16, so the
# generic path has to materialize a full bf16 copy of every local expert first.
# For MXFP4 checkpoints (e2m1 + ue8m0-per-32) the fp4 values are all exactly
# representable, so the bf16 hop is lossless *and* redundant: the conversion can
# read the packed nibbles and emit FP8 directly.
#
# ``_fp4_ue8m0_expert_to_blockwise_fp8`` below reproduces flashinfer's
# ``quantize_fp8_weight_block_nk`` arithmetic operation-for-operation on the
# exact fp32 values (the only legal difference is the dtype of the input, which
# the upstream routine upcasts to fp32 anyway). The equivalence test in
# tests/models/test_deepseek_v4_fi_moe_ep.py pins that bit-for-bit.

# Mirrors of the upstream constants the fused conversion has to match. They are
# asserted against flashinfer in the equivalence test rather than imported, so
# this module stays importable without flashinfer.
_SM90_FP8_BLOCK = 128
_SM90_FP8_GATE_UP_INTERLEAVE = 8
_SM90_FP8_E4M3_MAX = 448.0
_SM90_FP8_BLOCK_SCALE_EPSILON = 1e-30


def _interleave_gate_up_8(
    tensor: torch.Tensor, *, intermediate_size: int
) -> torch.Tensor:
    """``(E, 2I, ...)`` gate||up halves -> 8-row gate/up interleave.

    Mirrors flashinfer's ``_interleave_gate_up_8``. It is a permutation of the
    ``N`` axis (dim 1), so it applies just as well to the packed fp4 tensor and
    to the ue8m0 scale plane -- the fp4 packing and the 32-element scale groups
    both run along the trailing axis, which the permutation leaves alone.
    """
    block = _SM90_FP8_GATE_UP_INTERLEAVE
    if intermediate_size % (2 * block) != 0:
        raise ValueError(
            "SM90 FP8 MegaMoE requires full FC1 width to be divisible by "
            f"{2 * block}, got {intermediate_size}."
        )
    if tensor.shape[1] != intermediate_size:
        raise ValueError(
            f"expected FC1 tensor with shape (experts, {intermediate_size}, ...), "
            f"got {tuple(tensor.shape)}"
        )

    half = intermediate_size // 2
    num_pairs = half // block
    # The kernel's SwiGLU epilogue folds FC1 rows as [pair, {gate, up}, 8]:
    # ``gate_rows`` holds the (num_pairs, block) gate ids of each pair and the
    # matching up ids sit ``half`` further down, so stack on a new middle axis
    # and flatten. (A flat element-wise interleave of the two halves is NOT
    # this permutation and scrambles every block.)
    offsets = torch.arange(block, device=tensor.device)
    gate_rows = torch.arange(num_pairs, device=tensor.device).unsqueeze(1) * block
    gate_rows = gate_rows + offsets
    perm = torch.stack((gate_rows, gate_rows + half), dim=1).reshape(-1)
    return tensor.index_select(1, perm)


def _fp4_ue8m0_expert_to_blockwise_fp8(
    packed: torch.Tensor,
    sf_ue8m0: torch.Tensor,
    *,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, K//2)`` e2m1 + ``(N, K//32)`` ue8m0 -> ``(N, K)`` FP8 + 128x128 scales.

    Bit-identical to running flashinfer's ``quantize_fp8_weight_block_nk`` on
    the bf16 dequantization of the same tensor, without materializing that
    bf16 pack: the nibbles are decoded straight into the fp32 operand the
    fp32 quantization already needs (upstream upcasts to fp32 itself), so the
    peak footprint is one fp32 expert instead of bf16-expert + fp32-expert.
    """
    raw = packed.view(torch.uint8)
    n, k_half = raw.shape
    k = k_half * 2
    block = _SM90_FP8_BLOCK
    if n % block != 0 or k % block != 0:
        raise ValueError(
            f"blockwise FP8 requires the expert weight shape to be a multiple "
            f"of {block}, got {(n, k)}."
        )

    # e2m1 has one mantissa bit and ue8m0 is a power of two, so both steps are
    # exact in fp32 -- the value is the one the bf16 hop would have carried.
    lut = torch.tensor(_E2M1_LUT, dtype=torch.float32, device=raw.device)
    vals = torch.empty(n, k, dtype=torch.float32, device=raw.device)
    vals[:, ::2] = lut[(raw & 0x0F).to(torch.int64)]
    vals[:, 1::2] = lut[(raw >> 4).to(torch.int64)]
    ue8m0 = sf_ue8m0.to(torch.int32)
    vals *= (ue8m0 << 23).view(torch.float32).repeat_interleave(32, dim=-1)

    # Same reduction order and same eps/no-margin scale as upstream.
    blocks = vals.view(n // block, block, k // block, block)
    absmax = blocks.abs().amax(dim=(1, 3))
    scale = (absmax / _SM90_FP8_E4M3_MAX).clamp(min=_SM90_FP8_BLOCK_SCALE_EPSILON)
    quant = (blocks / scale[:, None, :, None]).view(n, k).to(fp8_dtype)
    return quant, scale.to(torch.float32)


def build_sm90_fp8_transformed_weights(
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    *,
    intermediate_size: int,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
):
    """MXFP4 expert params -> kernel-ready SM90 FP8 weights.

    Returns the ``transformed_weights`` pair for
    ``MegaConfig(preprocess_weights=False, transformed_weights=...)``, i.e. what
    flashinfer's ``preprocess_mega_weights`` would produce from the bf16 pack:
    per leg a ``(weight, weight_sf, None, None)`` tuple in kernel layout
    ``(E, K, N)`` with K stride 1, built without the bf16 round trip.

    The FP8 payloads are allocated up front and filled expert by expert, so the
    only temporaries are one expert's fp32 operand and its decoded rows: no
    full-size bf16 pack and no gate/up interleave copy of the whole tensor.
    """
    num_experts = w13_weight.shape[0]
    hidden_size = w2_weight.shape[1]
    fc1_out = 2 * intermediate_size
    block = _SM90_FP8_BLOCK
    device = w13_weight.device

    fc1_weight = torch.empty(
        num_experts, fc1_out, hidden_size, dtype=fp8_dtype, device=device
    )
    fc1_sf = torch.empty(
        num_experts,
        fc1_out // block,
        hidden_size // block,
        dtype=torch.float32,
        device=device,
    )
    fc2_weight = torch.empty(
        num_experts, hidden_size, intermediate_size, dtype=fp8_dtype, device=device
    )
    fc2_sf = torch.empty(
        num_experts,
        hidden_size // block,
        intermediate_size // block,
        dtype=torch.float32,
        device=device,
    )

    for expert in range(num_experts):
        # FC1's N axis is gate||up-interleaved before quantization so the
        # blockwise (N, K) scale blocks line up with the kernel's SwiGLU
        # epilogue fold; the ue8m0 plane is per row, so it permutes the same way.
        # The helper works on a batch of experts; a 1-expert slice is a view.
        w13_e = _interleave_gate_up_8(
            w13_weight[expert : expert + 1], intermediate_size=fc1_out
        )[0]
        sf13_e = _interleave_gate_up_8(
            w13_weight_scale[expert : expert + 1], intermediate_size=fc1_out
        )[0]
        fc1_weight[expert], fc1_sf[expert] = _fp4_ue8m0_expert_to_blockwise_fp8(
            w13_e, sf13_e, fp8_dtype=fp8_dtype
        )
        fc2_weight[expert], fc2_sf[expert] = _fp4_ue8m0_expert_to_blockwise_fp8(
            w2_weight[expert], w2_weight_scale[expert], fp8_dtype=fp8_dtype
        )

    # (E, N, K) -> (E, K, N) with K stride 1 and no re-pack, matching the
    # upstream comment: a .contiguous() here would break the K-major invariant
    # the SM90 GEMM's TMA descriptors rely on.
    return (
        (fc1_weight.transpose(1, 2), fc1_sf, None, None),
        (fc2_weight.transpose(1, 2), fc2_sf, None, None),
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
    weights=None,
    transformed_weights=None,
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
        transformed_weights=transformed_weights,
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
    "FI_MOE_EP_BACKEND_SPECS",
    "FiMoeEpBackendSpec",
    "build_fi_mega_config",
    "build_fi_mega_layer",
    "build_sm90_fp8_transformed_weights",
    "ensure_fi_moe_ep_runtime",
    "fi_moe_ep_backend_spec",
    "finalize_fi_moe_ep_runtime",
    "is_fi_moe_ep_backend",
    "make_fi_moe_ep_bootstrap",
    "mega_moe_weight_pack_from_params",
    "megakernel_runtime_requirements",
    "validate_fi_moe_ep_config",
]

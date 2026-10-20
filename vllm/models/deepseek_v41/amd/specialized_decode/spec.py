# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's specialized decode."""

import functools
from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.models.common.specialized_decode.spec import DecodeSpec, Feature
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterSparseSWAMetadata

    from .runtime.decode import DSV41DraftKernel, DSV41Kernel, LayerLaunches

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
SWA_WIDTH = 128  # a causal decode token's window slots
MAX_ROWS = 48  # the kernels' step rows at most: 8 requests x (1 + 5 DSpark drafts)
# A gfx942 workgroup has 64 KB of LDS, not 160 KB as on gfx950. Its kernels take
# a step's tokens 6 at a time (mono.stages.gemv.TILE), and at TP 4 they take up
# to 7 requests with 5 DSpark drafts each. A larger step runs vLLM's path.
MAX_ROWS_GFX942 = 42
EXPERTS = 384  # the backbone's routed experts, top-TOPK of them a token
TOPK = 6
# FP4 groups of an ug task
UG_GROUPS = 2
# the draft's 128-expert unions are small
DRAFT_UG_GROUPS = 1
# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats, or CDNA3
# (gfx942), where the kernels compute the same layer with FNUZ FP8 MFMAs on
# their own copies of the weights (runtime/weights942.py)
CDNA_VERSIONS = (3, 4)
# the routed experts' intermediate width a rank at TP 4 (2304 / 4): vLLM pads it
# to 640 on gfx942, and the gfx942 copies keep only the real columns
INTER_TP4 = 576


def on_gfx942() -> bool:
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 3


@functools.cache
def max_rows() -> int:
    """The widest step the kernels take on this GPU."""
    return MAX_ROWS_GFX942 if on_gfx942() else MAX_ROWS


def draft_routing(vllm_config) -> tuple[int, int]:
    """The DSpark draft's routed experts and top-k (128 / 3 on V4.1-Flash)."""
    config = vllm_config.model_config.hf_config
    return (
        getattr(config, "dspark_n_routed_experts", 0) or EXPERTS,
        getattr(config, "dspark_num_experts_per_tok", 0) or TOPK,
    )


def draft_swa_width(vllm_config) -> int:
    """The DSpark draft's SWA rows: a block's non-causal window, not a causal
    decode token's."""
    from vllm.v1.attention.backends.mla.compressor_utils import (
        get_dspark_swa_index_width,
    )

    config = vllm_config.model_config.hf_config
    speculative = vllm_config.speculative_config
    assert speculative is not None
    return get_dspark_swa_index_width(
        config.sliding_window, speculative.num_speculative_tokens
    )


def _widths(vllm_config) -> tuple[int, ...]:
    """The configured capture sizes the kernels can serve, so a deployment
    that asked for fewer graphs keeps them; every size at all when it asked for
    none."""
    rows = max_rows()
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    served = tuple(s for s in sorted(sizes) if 1 <= s <= rows)
    return served or tuple(range(1, rows + 1))


def _gpu_arch() -> str:
    """The GPU's architecture without its feature flags, for example gfx942."""
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    return props.gcnArchName.split(":")[0].lower()


def _refuse_config(vllm_config) -> list[str]:
    """Every reason this config cannot run the kernels, before load."""
    reasons = []
    if vllm_config.parallel_config.tensor_parallel_size not in (2, 4):
        reasons.append("needs tensor parallel size 2 or 4")
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        reasons.append(
            "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
        )
    try:
        import vllm.models.deepseek_v41.amd.mono.runner  # noqa: F401
    except ImportError as err:
        reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({err})")
    return reasons


def _refuse_model(model) -> list[str]:
    """Every reason the loaded model and this GPU cannot run the kernels. The
    kernels read every one of the 384 experts on each rank, TP-sharded, in
    AITER's A8W4 MXFP4 layout, from CTAs that all stay resident."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.deepseek_v41.amd.mono.common.arch import (
        target_arch,
    )
    from vllm.models.deepseek_v41.amd.mono.runner import BLOCKS
    from vllm.platforms.rocm import get_cdna_version

    reasons = []
    if get_cdna_version() not in CDNA_VERSIONS:
        reasons.append(f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}")
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < BLOCKS:
        reasons.append(f"needs {BLOCKS} compute units, the GPU has {cus}")
    target, gpu = target_arch(), _gpu_arch()
    if target != gpu:
        # FlyDSL compiles the kernels for the ARCH environment variable when it
        # is set, while the choice of the gfx942 weight copies follows the GPU.
        reasons.append(
            f"needs FlyDSL to compile for the GPU's {gpu}, but the ARCH "
            f"environment variable makes it compile for {target}"
        )
    if on_gfx942():
        # The gfx942 kernels read their own copies of the experts, made from the
        # loaded tensors before vLLM's MoE backend converts them, so the backend
        # matters on gfx950 only.
        return reasons
    backends = set()
    for layer in model.layers:
        experts = getattr(getattr(layer, "ffn", None), "experts", None)
        routed = getattr(experts, "routed_experts", None)
        if routed is not None:
            backends.add(getattr(routed.quant_method, "mxfp4_backend", None))
    wrong = backends - {Mxfp4MoeBackend.AITER_MXFP4_BF16}
    if wrong:
        reasons.append(f"needs the AITER_MXFP4_BF16 MoE backend, not {wrong}")
    return reasons


def _refuse_caches(vllm_config, decode: "LayerLaunches", caches) -> list[str]:
    """Every reason the kernels cannot read these caches: they index each as
    [blocks, block, RECORD] bytes."""
    reasons = []
    for name, cache in caches.items():
        if not isinstance(cache, torch.Tensor):
            reasons.append(f"{name}: no cache tensor")
            continue
        if cache.element_size() != 1:
            reasons.append(f"{name}: {cache.dtype} records, needs bytes")
        if cache.dim() < 2:
            reasons.append(f"{name}: {cache.dim()} dims, needs [blocks, block, bytes]")
        elif cache.stride(0) < RECORD:
            reasons.append(f"{name}: block stride {cache.stride(0)} under {RECORD}")
    return reasons


def _skip_swa(decode, md: dict, width: int) -> str | None:
    """Why the SWA rows of this step are outside what the kernels read, or None
    when they take it."""
    for key in decode.swa_keys:
        swa = cast("DeepseekV41ROCMAiterSparseSWAMetadata | None", md.get(key))
        if swa is None:
            return "layer metadata missing"
        if swa.num_prefills != 0 or swa.num_decodes == 0:
            return "not a decode-only step"
        if swa.num_decode_tokens > width:
            return "more decode tokens than rows"
        if swa.decode_swa_width != decode.swa_width:
            return f"SWA width {swa.decode_swa_width}"
        if (
            swa.decode_swa_indices is None
            or swa.decode_swa_lens is None
            or swa.token_to_req_indices is None
        ):
            return "SWA decode metadata missing"
    return None


def _skip_step(vllm_config, decode: "DSV41Kernel", width: int) -> str | None:
    """Why this step keeps vLLM's path at every layer, or None when the kernels
    take it. One verdict for every kernel layer, from the batch the whole TP
    group sees alike."""
    md = get_forward_context().attn_metadata
    if not decode.plans:
        return "no layer takes a launch"
    if not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
        return "no attention metadata"
    replay = decode.model.decoder_replay_layers
    if replay is not None and replay.uses_replay_batch():
        return "replay batch step"
    if any(md.get(k) is None for k in decode.comp_keys):
        return "layer metadata missing"
    return _skip_swa(decode, md, width)


def _skip_draft_step(vllm_config, decode: "DSV41DraftKernel", width: int) -> str | None:
    """Why this draft step keeps vLLM's path at every layer. The draft reads no
    compressed keys and never runs a replay batch, so its SWA rows decide it."""
    md = get_forward_context().attn_metadata
    if not decode.plans:
        return "no draft layer takes a launch"
    if not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
        return "no attention metadata"
    return _skip_swa(decode, md, width)


def _build(vllm_config, model, widths: tuple[int, ...]) -> "DSV41Kernel":
    from .runtime.decode import DSV41Kernel

    return DSV41Kernel(vllm_config, model, widths)


def _build_draft(vllm_config, model, widths: tuple[int, ...]) -> "DSV41DraftKernel":
    from .runtime.decode import DSV41DraftKernel

    return DSV41DraftKernel(vllm_config, model, widths)


_SUPPORTS = frozenset({Feature.PIPELINE_PARALLEL, Feature.SPECULATIVE_DECODING})

DECODE_SPEC = DecodeSpec["DSV41Kernel"](
    name="dsv41-cdna4",
    supports=_SUPPORTS,
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)

DRAFT_DECODE_SPEC = DecodeSpec["DSV41DraftKernel"](
    name="dsv41-dspark-cdna4",
    supports=_SUPPORTS,
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_draft_step,
    build=_build_draft,
)

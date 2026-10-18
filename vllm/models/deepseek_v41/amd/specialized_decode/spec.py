# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1's specialized decode."""

from typing import TYPE_CHECKING, cast

import torch

import vllm.envs as envs
from vllm.forward_context import get_forward_context
from vllm.models.common.specialized_decode.spec import DecodeSpec, Feature
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterSparseSWAMetadata

    from .runtime.decode import DSV41Kernel

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
SWA_WIDTH = 128  # a causal decode token's window slots
MAX_ROWS = 48  # the kernels' step rows at most: 8 requests x (1 + 5 DSpark drafts)
# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats
CDNA_VERSIONS = (4,)


def _widths(vllm_config) -> tuple[int, ...]:
    """The configured capture sizes the kernels can serve, so a deployment
    that asked for fewer graphs keeps them; every size at all when it asked for
    none."""
    sizes = vllm_config.compilation_config.cudagraph_capture_sizes or []
    served = tuple(s for s in sorted(sizes) if 1 <= s <= MAX_ROWS)
    return served or tuple(range(1, MAX_ROWS + 1))


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
        import vllm.models.deepseek_v41.amd.specialized_decode.mono.runner  # noqa: F401
    except ImportError as err:
        reasons.append(f"needs FlyDSL and AITER's FlyDSL helpers ({err})")
    return reasons


def _refuse_model(model) -> list[str]:
    """Every reason the loaded model and this GPU cannot run the kernels. The
    kernels read every one of the 384 experts on each rank, TP-sharded, in
    AITER's A8W4 MXFP4 layout, from CTAs that all stay resident."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.models.deepseek_v41.amd.specialized_decode.mono.runner import BLOCKS
    from vllm.platforms.rocm import get_cdna_version

    reasons = []
    if get_cdna_version() not in CDNA_VERSIONS:
        reasons.append(f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}")
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < BLOCKS:
        reasons.append(f"needs {BLOCKS} compute units, the GPU has {cus}")
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


def _refuse_caches(vllm_config, decode: "DSV41Kernel", caches) -> list[str]:
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
    for key in decode.swa_keys:
        swa = cast("DeepseekV41ROCMAiterSparseSWAMetadata | None", md.get(key))
        if swa is None:
            return "layer metadata missing"
        if swa.num_prefills != 0 or swa.num_decodes == 0:
            return "not a decode-only step"
        if swa.num_decode_tokens > width:
            return "more decode tokens than rows"
        if swa.decode_swa_width != SWA_WIDTH:
            return f"SWA width {swa.decode_swa_width}"
        if (
            swa.decode_swa_indices is None
            or swa.decode_swa_lens is None
            or swa.token_to_req_indices is None
        ):
            return "SWA decode metadata missing"
    return None


def _build(vllm_config, model, widths: tuple[int, ...]) -> "DSV41Kernel":
    from .runtime.decode import DSV41Kernel

    return DSV41Kernel(vllm_config, model, widths)


DECODE_SPEC = DecodeSpec["DSV41Kernel"](
    name="dsv41-cdna4",
    supports=frozenset({Feature.PIPELINE_PARALLEL, Feature.SPECULATIVE_DECODING}),
    widths=_widths,
    refuse_config=_refuse_config,
    refuse_model=_refuse_model,
    refuse_caches=_refuse_caches,
    skip_step=_skip_step,
    build=_build,
)

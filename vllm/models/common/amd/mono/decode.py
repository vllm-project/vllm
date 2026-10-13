# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM's side of a model's mono decode layer, the parts every model shares:
the deployment the kernels need, the process's runner of a ``MonoSpec``, a
decode step's eligibility and its SWA / compressed-cache metadata, and the
caches as the kernels read them. A model's docking (``MonoDecode`` subclass)
adds its layer checks, its weights and its layer forward.

Every TP rank must take the same path at every layer (the kernels wait on each
other's pushes): the conditions here read only metadata every rank holds
alike."""

from typing import TYPE_CHECKING, Any, ClassVar

import torch

import vllm.envs as envs
from vllm.config import CUDAGraphMode
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.common.amd.mono.spec import WINDOW, MonoSpec
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.common.amd.mono.runner import MonoRunner

logger = init_logger(__name__)

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
SWA_WIDTH = WINDOW  # a causal decode token's window slots
# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats
CDNA_VERSIONS = (4,)

_runners: dict[MonoSpec, "MonoRunner"] = {}


def mono_runner(
    spec: MonoSpec, device: torch.device, max_topk: int = 0
) -> "MonoRunner":
    """The process's runner of ``spec`` (one a TP rank, shared by every layer;
    ``max_topk``: its widest key table, ``MonoRunner``). Its peer-memory
    handshake is collective over the TP group: every rank builds it at the same
    layer of the same step."""
    runner = _runners.get(spec)
    if runner is None:
        if torch.cuda.is_current_stream_capturing():
            # its peer memory is allocated and exchanged eagerly; vLLM warms
            # every graph up eagerly before capturing it
            raise RuntimeError(
                "--monokernel was first reached inside a CUDA graph "
                "capture, before an eager decode step."
            )
        from vllm.models.common.amd.mono.runner import MonoRunner

        runner = _runners[spec] = MonoRunner(
            spec,
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
            get_tp_group().cpu_group,
            device,
            max_topk=max_topk,
        )
    return runner


def record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    assert kv.stride(0) >= block * RECORD, (tuple(kv.shape), kv.stride(), block)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


def scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 scales as their bytes: a float8_e8m0fnu / uint8 tensor's own, or the
    exponents of a power-of-two fp32 one (vLLM's aiter linears upcast them)."""
    if scale.dtype == torch.float32:
        bits = scale.contiguous().view(torch.int32)
        code = (bits >> 23).to(torch.uint8)
        assert torch.equal(bits, code.to(torch.int32) << 23), "not powers of two"
        return code
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


def mx_scales(scale: torch.Tensor, block: int) -> torch.Tensor:
    """``block`` x ``block`` E8M0 weight scales as the kernels' 32 x 32 ones:
    each code repeated over its block (exact: E8M0 is a power of two)."""
    codes = scale_bytes(scale)
    if block == 32:
        return codes.contiguous()
    r = block // 32
    return codes.repeat_interleave(r, 0).repeat_interleave(r, 1).contiguous()


def _compute_units() -> int:
    return current_platform.num_compute_units(torch.accelerator.current_device_index())


def unsupported_moe(experts, vllm_config) -> str | None:
    """Why the kernels cannot run this deployment's MoE, or None. They read every
    routed expert on each rank, TP-sharded, in AITER's A8W4 MXFP4 layout."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    pc = vllm_config.parallel_config
    if pc.enable_expert_parallel or pc.enable_eplb or pc.data_parallel_size > 1:
        return "needs tensor parallelism only: no expert or data parallelism"
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        return "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
    if experts is None:
        return None  # not a routed MoE: the layer checks decline it
    backend = getattr(experts.quant_method, "mxfp4_backend", None)
    if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
        return f"needs the AITER_MXFP4_BF16 MoE backend, not {backend}"
    return None


class MonoDecode:
    """One decoder layer's mono path: the whole layer or, with ``ffn_only``, the
    FFN launch after vLLM's attention. A model subclasses it with its ``SPEC``,
    the TP sizes its kernels take and its layer's hooks."""

    SPEC: ClassVar[MonoSpec]
    TP_SIZES: ClassVar[tuple[int, ...]]
    LABEL: ClassVar[str]  # the model, in log lines

    def __init__(self, ffn_only: bool = False, max_topk: int = 0) -> None:
        self.ffn_only = ffn_only
        self.max_topk = max_topk  # the widest key table (``MonoRunner``)
        self._weights: Any = None

    @classmethod
    def deployment_error(cls, experts, vllm_config) -> str | None:
        """Why this deployment cannot run the kernels at all, or None (an
        explicit opt-in that cannot run is an error)."""
        from vllm.platforms.rocm import get_cdna_version

        if get_cdna_version() not in CDNA_VERSIONS:
            return f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}"
        tp = get_tensor_model_parallel_world_size()
        if tp not in cls.TP_SIZES:
            sizes = " or ".join(map(str, cls.TP_SIZES))
            return f"needs tensor parallel size {sizes}"
        why = unsupported_moe(experts, vllm_config)
        if why is not None:
            return why
        try:
            from vllm.models.common.amd.mono.runner import BLOCKS
        except ImportError as err:
            return f"needs FlyDSL and AITER's FlyDSL helpers ({err})"
        # every CTA of a launch stays resident: a partitioned GPU deadlocks
        cus = _compute_units()
        if cus < BLOCKS:
            return f"needs {BLOCKS} compute units, the GPU has {cus}"
        return None

    def runner(self, device: torch.device) -> "MonoRunner":
        return mono_runner(self.SPEC, device, self.max_topk)

    @classmethod
    def skip(cls, why: str) -> None:
        logger.debug_once("%s mono decode skipped a step: %s", cls.LABEL, why)
        return None

    @classmethod
    def decode_step(cls, swa_prefix: str, comp_prefix: str | None, rows: int):
        """(why this step takes the original path -- None for a decode-only step
        the kernels take -- the layer's SWA metadata, its compressed cache's)."""
        if not 1 <= rows <= cls.SPEC.max_tokens:
            return f"{rows} rows", None, None
        if not is_forward_context_available():
            return "no forward context", None, None
        fc = get_forward_context()
        md = fc.attn_metadata
        swa: Any = None
        comp: Any = None
        if isinstance(md, dict):
            swa = md.get(swa_prefix)
            if comp_prefix is not None:
                comp = md.get(comp_prefix)
        why = None
        if fc.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            # a PIECEWISE (breakable) capture is replayed for mixed batches too
            why = f"{fc.cudagraph_runtime_mode} step"
        elif not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
            why = "no attention metadata"
        elif swa is None:
            why = "layer metadata missing"
        elif swa.num_prefills != 0 or swa.num_decodes == 0:
            why = "not a decode-only step"
        elif swa.num_decode_tokens > rows:
            why = "more decode tokens than rows"
        return why, swa, comp

    @staticmethod
    def window_error(swa) -> str | None:
        """Why the kernels' attention cannot take this step's SWA metadata."""
        if swa.decode_swa_width != SWA_WIDTH:
            return f"SWA width {swa.decode_swa_width}"
        if (
            swa.decode_swa_indices is None
            or swa.decode_swa_lens is None
            or swa.token_to_req_indices is None
        ):
            return "SWA decode metadata missing"
        return None

    @staticmethod
    def contiguous(*tensors: torch.Tensor | None) -> bool:
        return all(t is not None and t.is_contiguous() for t in tensors)

    def weights(self, layer) -> Any:
        """The layer's ``MonoLayerWeights``, taken from it at the first call
        (after loading and vLLM's own post-load processing)."""
        raise NotImplementedError

    def run_ffn(
        self,
        layer,
        swa_prefix: str,
        comp_prefix: str | None,
        part: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs by the FFN launch from ``part``, wo_b's unreduced
        output, and the attention seam's outputs; None when this step takes the
        original path (the caller then reduces ``part`` itself)."""
        why, _, _ = self.decode_step(swa_prefix, comp_prefix, part.shape[0])
        tensors = (part, residual, post_mix, res_mix, pre_mix)
        if why is None and not self.contiguous(*tensors):
            why = "non-contiguous layer inputs"
        if why is not None:
            return self.skip(why)
        runner = self.runner(part.device)
        logger.info_once(
            "%s mono decode FFN launch active (decode steps of <= %d rows)",
            self.LABEL,
            self.SPEC.max_tokens,
        )
        return runner.ffn(self.weights(layer), *tensors)


__all__ = [
    "RECORD",
    "SWA_WIDTH",
    "MonoDecode",
    "mono_runner",
    "mx_scales",
    "record_view",
    "scale_bytes",
    "unsupported_moe",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 mono decode layer on ROCm CDNA4 (``VLLM_ROCM_MONO_DECODE=1``).

A decode step's backbone layer -- the attention seam, the attention, its TP
all-reduce, the FFN seam, the MoE and its all-reduce -- runs as two persistent
FlyDSL launches (``mono``, this directory) on the layer's loaded
weights, the TP reductions inside the kernels. A layer runs whole when it has
the standard seam and no compressor or indexer of its own (compress ratio 1 or
2) over ``fp8_ds_mla`` records. Any other backbone layer (the first, the Engram
layers, the compressor / indexer layers) keeps vLLM's attention seam and
attention and runs the rest as one launch, the FFN launch, from wo_b's
unreduced output. A step takes the kernels when it is decode only, at most
``MAX_ROWS`` rows (eager or a FULL CUDA graph: the model is not torch.compiled);
a whole layer also needs causal SWA windows. Anything else runs the layer as
before; the paths share every tensor at the layer boundary, so they interleave
freely.

Every TP rank must take the same path at every layer (the kernels wait on each
other's pushes): the conditions read only metadata every rank holds alike.
"""

from typing import TYPE_CHECKING, cast

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
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer
    from vllm.models.deepseek_v41.amd.mono.runner import MonoLayerWeights
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV4ROCMAiterSparseSWAMetadata
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata

logger = init_logger(__name__)

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
SWA_WIDTH = 128  # a causal decode token's window slots
MAX_ROWS = 48  # the kernels' step rows at most: 8 requests x (1 + 5 DSpark drafts)
# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats
CDNA_VERSIONS = (4,)

_runner = None


def _mono_runner(device: torch.device):
    """The process's mono runner (one a TP rank, shared by every layer). Its
    peer-memory handshake is collective over the TP group: every rank builds it
    at the same layer of the same step."""
    global _runner
    if _runner is None:
        if torch.cuda.is_current_stream_capturing():
            # its peer memory is allocated and exchanged eagerly; vLLM warms
            # every graph up eagerly before capturing it
            raise RuntimeError(
                "VLLM_ROCM_MONO_DECODE was first reached inside a CUDA graph "
                "capture, before an eager decode step."
            )
        from vllm.models.deepseek_v41.amd.mono.runner import MAX_TOKENS, DSV41MonoLayer

        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        _runner = DSV41MonoLayer(
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
            get_tp_group().cpu_group,
            device,
        )
    return _runner


def _record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    assert kv.stride(0) >= block * RECORD, (tuple(kv.shape), kv.stride(), block)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


def _compute_units() -> int:
    return current_platform.num_compute_units(torch.accelerator.current_device_index())


def _unsupported_moe(layer: "DeepseekV4DecoderLayer", vllm_config) -> str | None:
    """Why the kernels cannot run this deployment's MoE, or None. They read every
    one of the 384 experts on each rank, TP-sharded, in AITER's A8W4 MXFP4
    layout."""
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

    pc = vllm_config.parallel_config
    if pc.enable_expert_parallel or pc.enable_eplb or pc.data_parallel_size > 1:
        return "needs tensor parallelism only: no expert or data parallelism"
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        return "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
    experts = getattr(getattr(layer.ffn, "experts", None), "routed_experts", None)
    if experts is None:
        return None  # not a routed MoE: the layer checks decline it
    backend = getattr(experts.quant_method, "mxfp4_backend", None)
    if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
        return f"needs the AITER_MXFP4_BF16 MoE backend, not {backend}"
    return None


class MonoDecodeLayer:
    """One decoder layer's mono path (``create``: None when the layer is not
    eligible): the whole layer (``__call__``) or, with ``ffn_only``, the FFN
    launch after vLLM's attention (``ffn``). Its weights are taken from the
    layer at the first call, after loading."""

    def __init__(self, ffn_only: bool = False) -> None:
        self.ffn_only = ffn_only
        self._weights: MonoLayerWeights | None = None

    @staticmethod
    def create(
        layer: "DeepseekV4DecoderLayer", vllm_config
    ) -> "MonoDecodeLayer | None":
        if not envs.VLLM_ROCM_MONO_DECODE:
            return None
        from vllm.platforms.rocm import get_cdna_version

        # the deployment: an explicit opt-in that cannot run is an error
        why = None
        if get_cdna_version() not in CDNA_VERSIONS:
            why = f"needs CDNA{'/'.join(map(str, CDNA_VERSIONS))}"
        elif get_tensor_model_parallel_world_size() not in (2, 4):
            why = "needs tensor parallel size 2 or 4"
        else:
            why = _unsupported_moe(layer, vllm_config)
        if why is None:
            try:
                from vllm.models.deepseek_v41.amd.mono.runner import BLOCKS
            except ImportError as err:
                why = f"needs FlyDSL and AITER's FlyDSL helpers ({err})"
            else:
                # every CTA of a launch stays resident: a partitioned GPU deadlocks
                cus = _compute_units()
                if cus < BLOCKS:
                    why = f"needs {BLOCKS} compute units, the GPU has {cus}"
        if why is not None:
            raise ValueError(f"VLLM_ROCM_MONO_DECODE {why}.")
        # the layer: the kernels serve the backbone's seams and MoE
        config = vllm_config.model_config.hf_config
        attn, ffn = layer.attn, layer.ffn
        if layer.use_sequence_parallel:
            why = "sequence-parallel layer"
        elif not layer.fuse_seam_norm:
            why = "needs aiter's fused mHC seam"
        elif attn.layer_id >= config.num_hidden_layers:
            why = "draft layer"
        elif (
            ffn.shared_experts is None
            or ffn.gate.tid2eid is not None
            or ffn.n_routed_experts != 384
            or ffn.n_activated_experts != 6
            or ffn.routed_scaling_factor != 1.5
            or ffn.swiglu_limit != 10.0
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
        ):
            why = "MoE routing outside the kernels' (384 / 6, sqrtsoftplus, x1.5)"
        if why is not None:
            logger.info_once("DSv4.1 mono decode off for some layers: %s", why)
            return None
        # the whole layer: the kernels' attention takes the standard layers only
        if layer.engram is not None:
            why = "Engram"
        elif attn.compressor is not None or attn.indexer is not None:
            why = "a compressor / indexer"
        elif attn.compress_ratio not in (1, 2):
            why = f"compress ratio {attn.compress_ratio}"
        elif attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
            why = f"a {attn.kv_cache_dtype} KV cache"
        if why is not None:
            logger.info_once(
                "DSv4.1 mono decode: the FFN launch for layers with %s", why
            )
            return MonoDecodeLayer(ffn_only=True)
        logger.info_once("DSv4.1 mono decode: the whole standard backbone layers")
        return MonoDecodeLayer()

    def weights(self, layer: "DeepseekV4DecoderLayer"):
        if self._weights is not None:
            return self._weights
        from vllm.models.deepseek_v41.amd.mono.runner import MonoLayerWeights

        a, f = layer.attn, layer.ffn
        e, sh = f.experts.routed_experts, f.shared_experts
        assert sh is not None  # ``create`` took only layers with one
        self._weights = MonoLayerWeights(
            attn=None if self.ffn_only else self._attn_weights(a),
            hc_attn_fn=layer.hc_attn_fn,
            hc_attn_scale=layer.hc_attn_scale,
            hc_attn_base=layer.hc_attn_base,
            attn_norm=layer.attn_norm.weight,
            hc_ffn_fn=layer.hc_ffn_fn,
            hc_ffn_scale=layer.hc_ffn_scale,
            hc_ffn_base=layer.hc_ffn_base,
            ffn_norm=layer.ffn_norm.weight,
            gate_w=f.gate.weight,
            bias=f.gate.e_score_correction_bias,
            w13=e.w13_weight,
            w13_s=e.w13_weight_scale,
            w2=e.w2_weight,
            w2_s=e.w2_weight_scale,
            sgu=sh.gate_up_proj.weight,
            sgu_s=_scale_bytes(sh.gate_up_proj.weight_scale),
            sw2=sh.down_proj.weight,
            sw2_s=_scale_bytes(sh.down_proj.weight_scale),
        )
        self._weights.check(get_tensor_model_parallel_world_size())
        return self._weights

    @staticmethod
    def _attn_weights(a):
        from vllm.models.deepseek_v41.amd.mono.attention.plan import Dims
        from vllm.models.deepseek_v41.amd.mono.runner import AttnWeights

        attn = AttnWeights(
            layer_id=a.layer_id,
            wqkv=a.fused_wqa_wkv.weight,
            wqkv_scale=_scale_bytes(a.fused_wqa_wkv.weight_scale),
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wq_b=a.wq_b.weight,
            wq_b_scale=_scale_bytes(a.wq_b.weight_scale),
            wo_a=a.wo_a.weight,
            wo_a_scale=_scale_bytes(a.wo_a.weight_scale),
            wo_b=a.wo_b.weight,
            wo_b_scale=_scale_bytes(a.wo_b.weight_scale),
            attn_sink=a.attn_sink,
            cos_sin=a.rotary_emb.cos_sin_cache,
            ratio=a.compress_ratio,
        )
        attn.check(Dims(get_tensor_model_parallel_world_size()))
        return attn

    @staticmethod
    def _skip(why: str) -> None:
        logger.debug_once("DSv4.1 mono decode skipped a step: %s", why)
        return None

    @staticmethod
    def _decode_step(layer: "DeepseekV4DecoderLayer", rows: int):
        """(why this step takes the original path -- None for a decode-only step
        the kernels take -- the layer's SWA metadata, its compressed cache's)."""
        if not 1 <= rows <= MAX_ROWS:
            return f"{rows} rows", None, None
        if not is_forward_context_available():
            return "no forward context", None, None
        fc = get_forward_context()
        md = fc.attn_metadata
        attn = layer.attn
        swa, comp = None, None
        if isinstance(md, dict):
            swa = cast(
                "DeepseekV4ROCMAiterSparseSWAMetadata | None",
                md.get(attn.swa_cache_layer.prefix),
            )
            if attn.compressed_cache_prefix is not None:
                comp = cast(
                    "DeepseekV4FlashMLAMetadata | None",
                    md.get(attn.compressed_cache_prefix),
                )
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

    def __call__(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs (x, residual, post_mix, res_mix, pre_mix) by the
        mono kernels, or None when this step takes the original path (always for
        an ``ffn_only`` layer)."""
        if self.ffn_only:
            return None
        if residual is None or x.dim() != 2:
            return self._skip("first layer")
        why, swa, comp = self._decode_step(layer, x.shape[0])
        attn = layer.attn
        tensors = (x, residual, post_mix, res_mix, pre_mix)
        if why is None:
            # the kernels' attention: causal SWA windows, the compressed cache
            if comp is None:
                why = "layer metadata missing"
            elif swa.decode_swa_width != SWA_WIDTH:
                why = f"SWA width {swa.decode_swa_width}"
            elif (
                swa.decode_swa_indices is None
                or swa.decode_swa_lens is None
                or swa.token_to_req_indices is None
            ):
                why = "SWA decode metadata missing"
            elif not all(t is not None and t.is_contiguous() for t in tensors):
                why = "non-contiguous layer inputs"
        if why is not None:
            return self._skip(why)
        assert swa is not None and comp is not None
        runner = _mono_runner(x.device)
        logger.info_once(
            "DSv4.1 mono decode layer active (decode steps of <= %d rows)", MAX_ROWS
        )
        out = runner.forward(
            self.weights(layer),
            x,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            positions,
            swa.slot_mapping,
            _record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
            swa.decode_swa_indices,
            swa.decode_swa_lens,
            swa.token_to_req_indices,
            topk_indices=attn.topk_indices_buffer,
            comp_cache=_record_view(
                attn._compressed_kv_cache(), comp.block_size // attn.compress_ratio
            ),
            comp_block_table=comp.block_table,
        )
        return out

    def ffn(
        self,
        layer: "DeepseekV4DecoderLayer",
        part: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs by the FFN launch from ``part``, wo_b's unreduced
        output, and the attention seam's outputs; None when this step takes the
        original path (the caller then reduces ``part`` itself)."""
        why, _, _ = self._decode_step(layer, part.shape[0])
        tensors = (part, residual, post_mix, res_mix, pre_mix)
        if why is None and not all(t.is_contiguous() for t in tensors):
            why = "non-contiguous layer inputs"
        if why is not None:
            return self._skip(why)
        runner = _mono_runner(part.device)
        logger.info_once(
            "DSv4.1 mono decode FFN launch active (decode steps of <= %d rows)",
            MAX_ROWS,
        )
        return runner.ffn(self.weights(layer), *tensors)

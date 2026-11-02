# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Pro mono decode layer on ROCm CDNA4 (``--monokernel``).

A decode step's backbone layer -- the attention seam, the attention, its TP
all-reduce, the FFN seam, the MoE and its all-reduce -- runs as two persistent
FlyDSL launches (``mono``, this directory) on the layer's loaded weights, the
TP reductions inside the kernels. Every layer takes the whole layer: between
its two launches vLLM's compressor runs on K1's normed input (and on a CSA
layer its indexer, from K1's qr, selects the top-k), then K2 builds the key
table -- the CSA layer's top-k through the block table, the HCA layer's every
compressed row (its metadata's slots). The first three layers route by hash:
their tokens' experts are gathered here and handed to the MoE.

A step takes the kernels when it is decode only, at most ``MAX_TOKENS`` rows
(eager or a FULL CUDA graph: the model is not torch.compiled), with causal SWA
windows. Anything else runs the layer as before; the paths share every tensor
at the layer boundary, so they interleave freely. Every TP rank must take the
same path at every layer (the kernels wait on each other's pushes): the
conditions read only metadata every rank holds alike.
"""

from typing import TYPE_CHECKING, Any

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
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.amd.mono.dims import (
    EXPERTS,
    MAX_TOKENS,
    ROUTED_SCALE,
    SWIGLU_LIMIT,
    TOPK,
    WINDOW,
    Dims,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer

logger = init_logger(__name__)

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
TP_SIZE = 8
CSA, HCA = 4, 128  # the layers' compress ratios: top-k / every compressed row
BLOCK = 128  # the checkpoint's FP8 block
# the kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats
CDNA_VERSION = 4

_runner = None


def _mono_runner(device: torch.device, max_topk: int):
    """The process's mono runner (one a TP rank, shared by every layer;
    ``max_topk``: its widest key table). Its peer-memory handshake is collective
    over the TP group: every rank builds it at the same layer of the same
    step."""
    global _runner
    if _runner is None:
        if torch.cuda.is_current_stream_capturing():
            # its peer memory is allocated and exchanged eagerly; vLLM warms
            # every graph up eagerly before capturing it
            raise RuntimeError(
                "--monokernel was first reached inside a CUDA graph "
                "capture, before an eager decode step."
            )
        from vllm.models.deepseek_v4.amd.mono.runner import MonoRunner

        _runner = MonoRunner(
            get_tensor_model_parallel_world_size(),
            get_tensor_model_parallel_rank(),
            get_tp_group().cpu_group,
            device,
            max_topk=max_topk,
        )
    return _runner


def _record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    assert kv.stride(0) >= block * RECORD, (tuple(kv.shape), kv.stride(), block)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 scales as their bytes: a float8_e8m0fnu / uint8 tensor's own, or the
    exponents of a power-of-two fp32 one (vLLM's aiter linears upcast them)."""
    if scale.dtype == torch.float32:
        bits = scale.contiguous().view(torch.int32)
        code = (bits >> 23).to(torch.uint8)
        assert torch.equal(bits, code.to(torch.int32) << 23), "not powers of two"
        return code
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


def _mx_scales(scale: torch.Tensor) -> torch.Tensor:
    """The checkpoint's 128 x 128 E8M0 weight scales as the kernels' 32 x 32
    ones: each code repeated over its block (exact: E8M0 is a power of two)."""
    r = BLOCK // 32
    codes = _scale_bytes(scale)
    return codes.repeat_interleave(r, 0).repeat_interleave(r, 1).contiguous()


def _block_scale(lin) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_fp8_block_weight_scale,
    )

    scale = get_fp8_block_weight_scale(lin)
    assert scale is not None, f"{type(lin).__name__} without block scales"
    return _mx_scales(scale)


def _hca_width(vllm_config) -> int:
    """The HCA layers' key table width: vLLM's c128a table's, every compressed
    row of the longest context (``sparse_mla``)."""
    from vllm.models.deepseek_v4.sparse_mla import _C128A_TOPK_ALIGNMENT

    n = -(-vllm_config.model_config.max_model_len // HCA)
    return -(-n // _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT


def _compute_units() -> int:
    return current_platform.num_compute_units(torch.accelerator.current_device_index())


def _deployment_error(layer: "DeepseekV4DecoderLayer", vllm_config) -> str | None:
    """Why this deployment cannot run the kernels at all, or None (an explicit
    opt-in that cannot run is an error). They read every routed expert on each
    rank, TP-sharded, in AITER's A8W4 MXFP4 layout, and vLLM's preshuffled FP8
    linears with padded rows."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
    from vllm.platforms.rocm import get_cdna_version

    if get_cdna_version() != CDNA_VERSION:
        return f"needs CDNA{CDNA_VERSION}"
    if get_tensor_model_parallel_world_size() != TP_SIZE:
        return f"needs tensor parallel size {TP_SIZE}"
    pc = vllm_config.parallel_config
    if pc.enable_expert_parallel or pc.enable_eplb or pc.data_parallel_size > 1:
        return "needs tensor parallelism only: no expert or data parallelism"
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        return "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
    experts = getattr(getattr(layer.ffn, "experts", None), "routed_experts", None)
    if experts is not None:  # else not a routed MoE: the layer checks decline it
        backend = getattr(experts.quant_method, "mxfp4_backend", None)
        if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
            return f"needs the AITER_MXFP4_BF16 MoE backend, not {backend}"
    if not rocm_aiter_ops.is_linear_fp8_enabled():
        return "needs AITER's FP8 linears (VLLM_ROCM_USE_AITER_LINEAR)"
    if not envs.VLLM_ROCM_FP8_PADDING:
        return "needs vLLM's padded FP8 weight rows (VLLM_ROCM_FP8_PADDING)"
    try:
        from vllm.models.deepseek_v4.amd.mono.runner import BLOCKS
    except ImportError as err:
        return f"needs FlyDSL and AITER's FlyDSL helpers ({err})"
    # every CTA of a launch stays resident: a partitioned GPU deadlocks
    cus = _compute_units()
    if cus < BLOCKS:
        return f"needs {BLOCKS} compute units, the GPU has {cus}"
    return None


def _select(layer: "DeepseekV4DecoderLayer", positions, normed, qr) -> None:
    """The layer's compressor -- and a CSA layer's indexer, its top-k into the
    shared top-k buffer -- from K1's normed input and bf16 qr, serial."""
    attn = layer.attn
    comp = attn.compressor
    assert comp is not None
    score = torch.mm(normed, comp.fused_wkv_wgate.weight.T, out_dtype=torch.float32)
    comp(score, positions, attn.rotary_emb)
    indexer = attn.indexer
    if indexer is None:
        return
    score = torch.mm(
        normed, indexer.compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32
    )
    weights, _ = indexer.weights_proj(normed)
    q, q_scale, weights = indexer(
        normed, qr, score, weights, positions, attn.indexer_rotary_emb
    )
    if q is not None:  # else the short context's top-k is every row, filled
        assert weights is not None
        indexer.indexer_op(
            normed, (q, q_scale) if q_scale is not None else q, None, weights
        )


class MonoDecodeLayer:
    """One DeepSeek-V4-Pro decoder layer's mono path (``create``: None when the
    layer is not eligible). Its weights are taken from the layer at the first
    call, after loading and vLLM's preshuffle."""

    def __init__(self, hca_width: int) -> None:
        self.hca_width = hca_width
        self._weights: Any = None

    @classmethod
    def create(
        cls, layer: "DeepseekV4DecoderLayer", vllm_config
    ) -> "MonoDecodeLayer | None":
        if not vllm_config.kernel_config.monokernel:
            return None
        why = _deployment_error(layer, vllm_config)
        if why is not None:
            raise ValueError(f"--monokernel {why}.")
        ffn, attn = layer.ffn, layer.attn
        config = vllm_config.model_config.hf_config
        if extract_layer_index(attn.prefix) >= config.num_hidden_layers:
            why = "draft layer"
        elif not (layer.use_fused_mhc and layer.fuse_mhc_rmsnorm):
            why = "needs aiter's fused mHC seam with its norm"
        elif (
            ffn.use_mega_moe
            or ffn.shared_experts is None
            or ffn.fuse_heterogeneous_shared_expert
            or ffn.is_fused_shared_expert_enabled
            or ffn.gate.bias_vl is not None
        ):
            why = "needs the separate FP8 shared expert and the AITER MXFP4 MoE"
        elif (
            ffn.n_routed_experts != EXPERTS
            or ffn.n_activated_experts != TOPK
            or ffn.routed_scaling_factor != ROUTED_SCALE
            or ffn.swiglu_limit != SWIGLU_LIMIT
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
        ):
            why = "MoE routing outside the kernels' (384 / 6, sqrtsoftplus, x2.5)"
        elif attn.compress_ratio not in (CSA, HCA) or (
            (attn.compress_ratio == CSA) != (attn.indexer is not None)
        ):
            why = f"compress ratio {attn.compress_ratio}"
        elif attn.kv_cache_dtype != "fp8_ds_mla":
            why = f"a {attn.kv_cache_dtype} KV cache"
        if why is not None:
            logger.info_once("DSv4 mono decode off for some layers: %s", why)
            return None
        logger.info_once("DSv4 mono decode: the whole backbone layers")
        return cls(_hca_width(vllm_config))

    def weights(self, layer: "DeepseekV4DecoderLayer"):
        if self._weights is not None:
            return self._weights
        from vllm.models.deepseek_v4.amd.mono.runner import (
            AttnWeights,
            MonoLayerWeights,
        )

        a, f = layer.attn, layer.ffn
        e, sh = f.experts.routed_experts, f.shared_experts
        # the kernels read vLLM's preshuffled wqkv / wo_b / shared gate_up
        assert a._wqa_wkv_scale is not None and a._wo_b_scale is not None
        assert sh is not None and sh._gateup_scale is not None
        d = Dims(get_tensor_model_parallel_world_size())
        attn = AttnWeights(
            layer_id=extract_layer_index(a.prefix),
            wqkv=a.fused_wqa_wkv.weight,
            wqkv_scale=_mx_scales(a._wqa_wkv_scale),
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wq_b=a.wq_b.weight,
            wq_b_scale=_block_scale(a.wq_b),
            wo_a=a.wo_a.weight,
            wo_a_scale=_block_scale(a.wo_a),
            wo_b=a.wo_b.weight,
            wo_b_scale=_mx_scales(a._wo_b_scale),
            attn_sink=a.attn_sink,
            cos_sin=a.rotary_emb.cos_sin_cache,
            ratio=a.compress_ratio,
        )
        attn.check(d)
        self._weights = MonoLayerWeights(
            attn=attn,
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
            sgu_s=_mx_scales(sh._gateup_scale),
            sw2=sh.down_proj.weight,
            sw2_s=_block_scale(sh.down_proj),
        )
        self._weights.check(d)
        return self._weights

    @staticmethod
    def _skip(why: str) -> None:
        logger.debug_once("DSv4 mono decode skipped a step: %s", why)
        return None

    @staticmethod
    def _decode_step(layer: "DeepseekV4DecoderLayer", rows: int):
        """(why this step takes the original path -- None for a decode-only step
        the kernels take -- the layer's SWA metadata, its compressed cache's)."""
        if not 1 <= rows <= MAX_TOKENS:
            return f"{rows} rows", None, None
        if not is_forward_context_available():
            return "no forward context", None, None
        fc = get_forward_context()
        md = fc.attn_metadata
        attn = layer.attn
        swa: Any = None
        comp: Any = None
        if isinstance(md, dict):
            swa = md.get(attn.swa_cache_layer.prefix)
            comp = md.get(attn.prefix)
        why = None
        if fc.cudagraph_runtime_mode not in (CUDAGraphMode.NONE, CUDAGraphMode.FULL):
            # a PIECEWISE (breakable) capture is replayed for mixed batches too
            why = f"{fc.cudagraph_runtime_mode} step"
        elif not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
            why = "no attention metadata"
        elif swa is None or comp is None:
            why = "layer metadata missing"
        elif swa.num_prefills != 0 or swa.num_decodes == 0:
            why = "not a decode-only step"
        elif swa.num_decode_tokens > rows:
            why = "more decode tokens than rows"
        elif swa.decode_swa_width != WINDOW:
            why = f"SWA width {swa.decode_swa_width}"
        elif (
            swa.decode_swa_indices is None
            or swa.decode_swa_lens is None
            or swa.token_to_req_indices is None
        ):
            why = "SWA decode metadata missing"
        elif (
            attn.compress_ratio == HCA
            and getattr(comp, "c128a_global_decode_topk_indices", None) is None
        ):
            why = "C128A decode metadata missing"
        return why, swa, comp

    def __call__(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs (x, residual, post_mix, res_mix) by the mono
        kernels, or None when this step takes the original path."""
        attn, M = layer.attn, x.shape[0]
        first = residual is None
        why, swa, comp = self._decode_step(layer, M)
        tensors = (x,) if first else (x, residual, post_mix, res_mix)
        contiguous = all(t is not None and t.is_contiguous() for t in tensors)
        if why is None and not contiguous:
            why = "non-contiguous layer inputs"
        hash_route = layer.ffn.gate.tid2eid is not None
        if why is None and hash_route and input_ids is None:
            why = "a hash layer without its input ids"
        if why is not None:
            return self._skip(why)
        assert swa is not None and comp is not None
        runner = _mono_runner(x.device, self.hca_width)
        logger.info_once(
            "DSv4 mono decode layer active (decode steps of <= %d rows)", MAX_TOKENS
        )
        if first:
            # the embedding broadcast to the four streams: settled, no post owed
            # (the mixes unread without the fold)
            residual, post_mix, res_mix = x, x, x
        assert residual is not None and post_mix is not None and res_mix is not None
        route = None
        if hash_route:
            assert input_ids is not None
            route = layer.ffn.gate.tid2eid[input_ids[:M]].contiguous()
        if attn.compress_ratio == HCA:
            topk = comp.c128a_global_decode_topk_indices[:M, 0]
            kw: dict[str, Any] = dict(topk=self.hca_width, global_keys=True)
        else:
            topk = attn.topk_indices_buffer
            kw = dict(comp_block_table=comp.block_table)
        out = runner.forward(
            self.weights(layer),
            x,
            residual,
            post_mix,
            res_mix,
            positions,
            swa.slot_mapping,
            _record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
            swa.decode_swa_indices,
            swa.decode_swa_lens,
            swa.token_to_req_indices,
            topk_indices=topk,
            comp_cache=_record_view(
                attn.kv_cache, comp.block_size // attn.compress_ratio
            ),
            select=lambda normed, qr: _select(layer, positions, normed, qr),
            fold=not first,
            route_ids=route,
            **kw,
        )
        return out[:4]

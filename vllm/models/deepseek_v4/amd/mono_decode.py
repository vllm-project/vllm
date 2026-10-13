# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4-Pro mono decode layer on ROCm CDNA4 (``--monokernel``).

A decode step's backbone layer -- the attention seam, the attention, its TP
all-reduce, the FFN seam, the MoE and its all-reduce -- runs as two persistent
FlyDSL launches (``vllm.models.common.amd.mono``) on the layer's loaded
weights, the TP reductions inside the kernels. Every layer takes the whole
layer: between its two launches vLLM's compressor runs on K1's normed input
(and on a CSA layer its indexer, from K1's qr, selects the top-k), then K2
builds the key table -- the CSA layer's top-k through the block table, the HCA
layer's every compressed row (its metadata's slots). The first three layers
route by hash: their tokens' experts are gathered here and handed to the MoE.

A step takes the kernels when it is decode only, at most ``SPEC.max_tokens``
rows (eager or a FULL CUDA graph: the model is not torch.compiled), with causal
SWA windows. Anything else runs the layer as before; the paths share every
tensor at the layer boundary, so they interleave freely. Every TP rank takes the
same path at every layer: the conditions read only metadata every rank holds
alike.
"""

from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.common.amd.mono.decode import (
    MonoDecode,
    mx_scales,
    record_view,
)
from vllm.models.common.amd.mono.spec import Dims, MonoSpec

if TYPE_CHECKING:
    from vllm.models.deepseek_v4.amd.model import DeepseekV4DecoderLayer

logger = init_logger(__name__)

# DeepSeek-V4-Pro to the kernels: TP8, a step's rows at most 8 requests x
# (1 + 6 DSpark drafts) and spare; vLLM's aiter path preshuffles wqkv, wo_b and
# the shared gate_up, and norms each query head
SPEC = MonoSpec(
    name="dsv4",
    hidden=7168,
    q_rank=1536,
    n_heads=128,
    o_groups=16,
    index_topk=1024,
    experts=384,
    topk=6,
    moe_inter=3072,
    shared_inter=3072,
    routed_scale=2.5,
    swiglu_limit=10.0,
    norm_eps=1e-6,
    delayed_seam=False,
    max_tokens=64,
    preshuffled=True,
    q_head_norm=True,
    padded_rows=True,
    two_hop_rows=42,
)
CSA, HCA = 4, 128  # the layers' compress ratios: top-k / every compressed row
BLOCK = 128  # the checkpoint's FP8 block


def _hca_width(vllm_config) -> int:
    """The HCA layers' key table width: vLLM's c128a table's, every compressed
    row of the longest context (``sparse_mla``)."""
    from vllm.models.deepseek_v4.sparse_mla import _C128A_TOPK_ALIGNMENT

    n = -(-vllm_config.model_config.max_model_len // HCA)
    return -(-n // _C128A_TOPK_ALIGNMENT) * _C128A_TOPK_ALIGNMENT


def _block_scale(lin) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        get_fp8_block_weight_scale,
    )

    scale = get_fp8_block_weight_scale(lin)
    assert scale is not None, f"{type(lin).__name__} without block scales"
    return mx_scales(scale, BLOCK)


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


class MonoDecodeLayer(MonoDecode):
    """One DeepSeek-V4-Pro decoder layer's mono path (``create``: None when the
    layer is not eligible). Its weights are taken from the layer at the first
    call, after loading and vLLM's preshuffle."""

    SPEC = SPEC
    TP_SIZES = (8,)
    LABEL = "DSv4"

    def __init__(self, hca_width: int) -> None:
        super().__init__(max_topk=hca_width)
        self.hca_width = hca_width

    @classmethod
    def create(
        cls, layer: "DeepseekV4DecoderLayer", vllm_config
    ) -> "MonoDecodeLayer | None":
        if not vllm_config.kernel_config.monokernel:
            return None
        from vllm._aiter_ops import rocm_aiter_ops

        ffn, attn = layer.ffn, layer.attn
        experts = getattr(getattr(ffn, "experts", None), "routed_experts", None)
        why = cls.deployment_error(experts, vllm_config)
        if why is None and not rocm_aiter_ops.is_linear_fp8_enabled():
            why = "needs AITER's FP8 linears (VLLM_ROCM_USE_AITER_LINEAR)"
        if why is None and not envs.VLLM_ROCM_FP8_PADDING:
            why = "needs vLLM's padded FP8 weight rows (VLLM_ROCM_FP8_PADDING)"
        if why is not None:
            raise ValueError(f"--monokernel {why}.")
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
            ffn.n_routed_experts != SPEC.experts
            or ffn.n_activated_experts != SPEC.topk
            or ffn.routed_scaling_factor != SPEC.routed_scale
            or ffn.swiglu_limit != SPEC.swiglu_limit
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
        from vllm.models.common.amd.mono.runner import AttnWeights, MonoLayerWeights

        a, f = layer.attn, layer.ffn
        e, sh = f.experts.routed_experts, f.shared_experts
        # the kernels read vLLM's preshuffled wqkv / wo_b / shared gate_up
        assert a._wqa_wkv_scale is not None and a._wo_b_scale is not None
        assert sh is not None and sh._gateup_scale is not None
        d = Dims(SPEC, get_tensor_model_parallel_world_size())
        attn = AttnWeights(
            layer_id=extract_layer_index(a.prefix),
            wqkv=a.fused_wqa_wkv.weight,
            wqkv_scale=mx_scales(a._wqa_wkv_scale, BLOCK),
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wq_b=a.wq_b.weight,
            wq_b_scale=_block_scale(a.wq_b),
            wo_a=a.wo_a.weight,
            wo_a_scale=_block_scale(a.wo_a),
            wo_b=a.wo_b.weight,
            wo_b_scale=mx_scales(a._wo_b_scale, BLOCK),
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
            sgu_s=mx_scales(sh._gateup_scale, BLOCK),
            sw2=sh.down_proj.weight,
            sw2_s=_block_scale(sh.down_proj),
        )
        self._weights.check(d)
        return self._weights

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
        why, swa, comp = self.decode_step(attn.swa_cache_layer.prefix, attn.prefix, M)
        hca = attn.compress_ratio == HCA
        tensors = (x,) if first else (x, residual, post_mix, res_mix)
        if why is None:
            assert swa is not None
            if comp is None:
                why = "layer metadata missing"
            elif (
                hca and getattr(comp, "c128a_global_decode_topk_indices", None) is None
            ):
                why = "C128A decode metadata missing"
            else:
                why = self.window_error(swa)
            if why is None and not self.contiguous(*tensors):
                why = "non-contiguous layer inputs"
            if why is None and layer.ffn.gate.tid2eid is not None and input_ids is None:
                why = "a hash layer without its input ids"
        if why is not None:
            return self.skip(why)
        assert swa is not None and comp is not None
        runner = self.runner(x.device)
        logger.info_once(
            "DSv4 mono decode layer active (decode steps of <= %d rows)",
            SPEC.max_tokens,
        )
        if first:
            # the embedding broadcast to the four streams: settled, no post owed
            residual = x
            pre = runner.first_pre_mix(M)
            post_mix, res_mix = pre, pre  # unread without the fold
        route = None
        if layer.ffn.gate.tid2eid is not None:
            assert input_ids is not None
            route = layer.ffn.gate.tid2eid[input_ids[:M]].contiguous()
        if hca:
            topk = comp.c128a_global_decode_topk_indices[:M, 0]
            kw: dict[str, Any] = dict(topk=self.hca_width, global_keys=True)
        else:
            topk = attn.topk_indices_buffer
            kw = {}
        out = runner.forward(
            self.weights(layer),
            x,
            residual,
            post_mix,
            res_mix,
            runner.first_pre_mix(M),  # unread: the seam collapses with its own
            positions,
            swa.slot_mapping,
            record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
            swa.decode_swa_indices,
            swa.decode_swa_lens,
            swa.token_to_req_indices,
            topk_indices=topk,
            comp_cache=record_view(
                attn.kv_cache, comp.block_size // attn.compress_ratio
            ),
            comp_block_table=comp.block_table,
            fold=not first,
            select=lambda normed, qr: _select(layer, positions, normed, qr),
            route_ids=route,
            **kw,
        )
        return out[:4]

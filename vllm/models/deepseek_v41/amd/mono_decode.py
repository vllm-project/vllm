# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 mono decode layer on ROCm CDNA4 (``--monokernel``).

A decode step's backbone layer -- the attention seam, the attention, its TP
all-reduce, the FFN seam, the MoE and its all-reduce -- runs as two persistent
FlyDSL launches (``vllm.models.common.amd.mono``) on the layer's loaded
weights, the TP reductions inside the kernels. Every backbone layer of the
checkpoint runs whole (compress ratio 0, 1 or 2 over ``fp8_ds_mla`` records):
the first layer and the Engram layers hand K1 a settled residual, and an index
source runs vLLM's compressor and indexer between the two launches. A layer
outside that (an MXFP8 KV cache, another indexer) keeps vLLM's attention seam
and attention and runs the rest as one launch, the FFN launch, from wo_b's
unreduced output. A step takes the kernels when it is decode only, at most
``MAX_ROWS`` rows (eager or a FULL CUDA graph: the model is not torch.compiled);
a whole layer also needs causal SWA windows. Anything else runs the layer as
before; the paths share every tensor at the layer boundary, so they interleave
freely.

Every TP rank must take the same path at every layer (the kernels wait on each
other's pushes): the conditions read only metadata every rank holds alike.
"""

from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.models.common.amd.mono.decode import (
    RECORD,
    SWA_WIDTH,
    MonoDecode,
    record_view,
    scale_bytes,
)
from vllm.models.common.amd.mono.spec import Dims, MonoSpec

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV4ROCMAiterSparseSWAMetadata
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata

logger = init_logger(__name__)

# DeepSeek-V4.1-Flash to the kernels: a step's rows at most 8 requests x (1 + 5
# DSpark drafts)
SPEC = MonoSpec(
    name="dsv41",
    hidden=5120,
    q_rank=1280,
    n_heads=64,
    o_groups=8,
    index_topk=512,
    experts=384,
    topk=6,
    moe_inter=2304,
    shared_inter=2304,
    routed_scale=1.5,
    swiglu_limit=10.0,
    norm_eps=1e-20,
    delayed_seam=True,
    max_tokens=48,
)
MAX_ROWS = SPEC.max_tokens
__all__ = ["MAX_ROWS", "RECORD", "SPEC", "SWA_WIDTH", "MonoDecodeLayer"]


def _select(
    layer: "DeepseekV4DecoderLayer",
    positions: torch.Tensor,
    normed: torch.Tensor,
    qr: torch.Tensor,
) -> None:
    """An index source's compressor and top-k selection, from K1's normed input
    and bf16 qr: ``_forward_csa2_full`` / ``_forward_csa2_reindex`` and the
    indexer op, serial."""
    attn = layer.attn
    comp, indexer = attn.compressor, attn.indexer
    assert indexer is not None
    if comp is not None:
        kv_score = torch.mm(
            normed, comp.fused_wkv_wgate.weight.T, out_dtype=torch.float32
        )
        latent = comp(kv_score, positions)
        comp.insert_cache(latent, positions, attn.rotary_emb)
        indexer._produce_k(latent, positions, attn.indexer_rotary_emb)
    weights, _ = indexer.weights_proj(normed)
    q, q_scale, weights = indexer.forward_q(
        qr, None, weights, positions, attn.indexer_rotary_emb
    )
    assert weights is not None
    indexer.indexer_op(
        normed, (q, q_scale) if q_scale is not None else q, None, weights
    )


class MonoDecodeLayer(MonoDecode):
    """One DeepSeek-V4.1 decoder layer's mono path (``create``: None when the
    layer is not eligible): the whole layer (``__call__``) or, with
    ``ffn_only``, the FFN launch after vLLM's attention (``ffn``)."""

    SPEC = SPEC
    TP_SIZES = (2, 4)
    LABEL = "DSv4.1"

    @classmethod
    def create(
        cls, layer: "DeepseekV4DecoderLayer", vllm_config
    ) -> "MonoDecodeLayer | None":
        if not vllm_config.kernel_config.monokernel:
            return None
        experts = getattr(getattr(layer.ffn, "experts", None), "routed_experts", None)
        why = cls.deployment_error(experts, vllm_config)
        if why is not None:
            raise ValueError(f"--monokernel {why}.")
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
            or ffn.n_routed_experts != SPEC.experts
            or ffn.n_activated_experts != SPEC.topk
            or ffn.routed_scaling_factor != SPEC.routed_scale
            or ffn.swiglu_limit != SPEC.swiglu_limit
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
        ):
            why = "MoE routing outside the kernels' (384 / 6, sqrtsoftplus, x1.5)"
        if why is not None:
            logger.info_once("DSv4.1 mono decode off for some layers: %s", why)
            return None
        # the whole layer: the kernels' attention takes the standard layers; an
        # index source's compressor and indexer run between its two launches
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer

        if attn.compressor is not None and attn.indexer is None:
            why = "a compressor without an indexer"
        elif attn.indexer is not None and not isinstance(
            attn.indexer, DeepseekV41RocmMxfp4Indexer
        ):
            why = f"a {type(attn.indexer).__name__}"
        elif attn.compress_ratio not in (0, 1, 2):
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
        from vllm.models.common.amd.mono.runner import MonoLayerWeights

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
            sgu_s=scale_bytes(sh.gate_up_proj.weight_scale),
            sw2=sh.down_proj.weight,
            sw2_s=scale_bytes(sh.down_proj.weight_scale),
        )
        self._weights.check(Dims(SPEC, get_tensor_model_parallel_world_size()))
        return self._weights

    @staticmethod
    def _attn_weights(a):
        from vllm.models.common.amd.mono.runner import AttnWeights

        attn = AttnWeights(
            layer_id=a.layer_id,
            wqkv=a.fused_wqa_wkv.weight,
            wqkv_scale=scale_bytes(a.fused_wqa_wkv.weight_scale),
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wq_b=a.wq_b.weight,
            wq_b_scale=scale_bytes(a.wq_b.weight_scale),
            wo_a=a.wo_a.weight,
            wo_a_scale=scale_bytes(a.wo_a.weight_scale),
            wo_b=a.wo_b.weight,
            wo_b_scale=scale_bytes(a.wo_b.weight_scale),
            attn_sink=a.attn_sink,
            cos_sin=a.rotary_emb.cos_sin_cache,
            ratio=a.compress_ratio,
        )
        attn.check(Dims(SPEC, get_tensor_model_parallel_world_size()))
        return attn

    @classmethod
    def _step(cls, layer: "DeepseekV4DecoderLayer", rows: int):
        attn = layer.attn
        why, swa, comp = cls.decode_step(
            attn.swa_cache_layer.prefix, attn.compressed_cache_prefix, rows
        )
        return (
            why,
            cast("DeepseekV4ROCMAiterSparseSWAMetadata | None", swa),
            cast("DeepseekV4FlashMLAMetadata | None", comp),
        )

    def __call__(
        self,
        layer: "DeepseekV4DecoderLayer",
        x: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
        post_mix: torch.Tensor | None,
        res_mix: torch.Tensor | None,
        pre_mix: torch.Tensor | None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs (x, residual, post_mix, res_mix, pre_mix) by the
        mono kernels, or None when this step takes the original path (always for
        an ``ffn_only`` layer)."""
        if self.ffn_only:
            return None
        first = residual is None
        if first and x.dim() != 2:
            return self.skip("first layer of a pipeline stage")
        why, swa, comp = self._step(layer, x.shape[0])
        attn = layer.attn
        tensors = (x,) if first else (x, residual, post_mix, res_mix, pre_mix)
        if why is None:
            assert swa is not None
            # the kernels' attention: causal SWA windows, the compressed cache
            if comp is None and attn.compress_ratio:
                why = "layer metadata missing"
            else:
                why = self.window_error(swa)
            if why is None and not self.contiguous(*tensors):
                why = "non-contiguous layer inputs"
        if why is not None:
            return self.skip(why)
        assert swa is not None
        runner = self.runner(x.device)
        logger.info_once(
            "DSv4.1 mono decode layer active (decode steps of <= %d rows)", MAX_ROWS
        )
        # K1's seam folds the owed post into the residual, except where the
        # residual arrives settled: the embedding, or after the Engram write
        fold = True
        if first:
            residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            pre_mix = runner.first_pre_mix(x.shape[0])
            post_mix = res_mix = pre_mix  # unread without the fold
            fold = False
        elif layer.engram is not None and engram_hashes is not None:
            residual = layer.mhc_post(x, residual, post_mix, res_mix)
            hashes = engram_hashes[:, layer.engram.layer_hash_index]
            residual = layer.engram(residual, hashes, engram_mask)
            fold = False
        kw: dict[str, Any] = {}
        if attn.indexer is not None:
            kw["select"] = lambda normed, qr: _select(layer, positions, normed, qr)
        if attn.compress_ratio:
            assert comp is not None
            kw.update(
                topk_indices=attn.topk_indices_buffer,
                comp_cache=record_view(
                    attn._compressed_kv_cache(), comp.block_size // attn.compress_ratio
                ),
                comp_block_table=comp.block_table,
            )
        return runner.forward(
            self.weights(layer),
            x,
            residual,
            post_mix,
            res_mix,
            pre_mix,
            positions,
            swa.slot_mapping,
            record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
            swa.decode_swa_indices,
            swa.decode_swa_lens,
            swa.token_to_req_indices,
            fold=fold,
            **kw,
        )

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
        attn = layer.attn
        return self.run_ffn(
            layer,
            attn.swa_cache_layer.prefix,
            attn.compressed_cache_prefix,
            part,
            residual,
            post_mix,
            res_mix,
            pre_mix,
        )

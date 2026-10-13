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
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.models.common.mono import Feature, MonoOp, MonoRuntime, MonoSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer
    from vllm.models.deepseek_v41.amd.mono.runner import (
        DSV41MonoLayer,
        MonoLayerWeights,
    )
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV4ROCMAiterSparseSWAMetadata
    from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLAMetadata

logger = init_logger(__name__)

RECORD = 584  # an fp8_ds_mla KV record: 576 data bytes, 8 scale bytes
SWA_WIDTH = 128  # a causal decode token's window slots
MAX_ROWS = 48  # the kernels' step rows at most: 8 requests x (1 + 5 DSpark drafts)


def _needs_a8w4_moe(vllm_config) -> str | None:
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        return "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
    return None


DSV41_MONO = MonoSpec(
    name="DSv4.1 mono decode",
    supports=frozenset({Feature.SPECULATIVE_DECODE}),
    tp_sizes=(2, 4),
    device_archs=("gfx950",),  # the kernels' ISA: scaled MFMA and MX formats
    widths=tuple(range(1, MAX_ROWS + 1)),
    kernel_module="vllm.models.deepseek_v41.amd.mono.runner",
    resident_blocks="BLOCKS",
    constraints=(_needs_a8w4_moe,),
    opt_in=lambda c: bool(envs.VLLM_ROCM_MONO_DECODE),
    on_refusal="raise",  # the opt-in was explicit: say why it cannot be served
)


def _record_view(kv: torch.Tensor, block: int) -> torch.Tensor:
    """A layer's fp8_ds_mla cache as [blocks, block, RECORD] bytes (a block's
    data rows, then its scale words: the kernels index both from it)."""
    kv = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    assert kv.stride(0) >= block * RECORD, (tuple(kv.shape), kv.stride(), block)
    return torch.as_strided(kv, (kv.shape[0], block, RECORD), (kv.stride(0), RECORD, 1))


def _scale_bytes(scale: torch.Tensor) -> torch.Tensor:
    return scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)


class DSV41MonoDecode(MonoOp):
    """One model's mono decode path (``create``: None when no layer is eligible).

    A layer the kernels serve whole runs ``forward``; the rest run ``ffn``, the
    FFN launch after vLLM's attention. Which of the two a layer takes is the
    ``ffn_only`` entry ``build`` leaves for it, read from vLLM's own forward
    too; a layer's weights are taken at its first call, after loading.

    These kernels are built for gfx950. A port to a second architecture is a
    subclass of its own, with its own ``spec`` and layer checks, named by
    ``variant``."""

    spec = DSV41_MONO

    def __init__(self, vllm_config, layers: "Iterable[DeepseekV4DecoderLayer]") -> None:
        self.layers = [layer for layer in layers if hasattr(layer, "attn")]
        super().__init__(vllm_config)
        self.ffn_only: dict[int, bool] = {}
        self._weights: dict[int, MonoLayerWeights] = {}
        self._kernels: DSV41MonoLayer | None = None

    def refuse_model(self) -> list[str]:
        """The kernels read every one of the 384 experts on each rank, TP-sharded,
        in AITER's A8W4 MXFP4 layout: no other MoE backend is servable."""
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

        for layer in self.layers:
            experts = getattr(
                getattr(layer.ffn, "experts", None), "routed_experts", None
            )
            if experts is None:
                continue  # not a routed MoE: the layer checks skip it
            backend = getattr(experts.quant_method, "mxfp4_backend", None)
            if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
                return [f"needs the AITER_MXFP4_BF16 MoE backend, not {backend}"]
        return []

    def skip_layer(self, layer: "DeepseekV4DecoderLayer") -> str:
        """Skip the layers whose seams or MoE the kernels do not serve."""
        attn, ffn = layer.attn, layer.ffn
        hf = self.vllm_config.model_config.hf_config
        if layer.use_sequence_parallel:
            return "sequence-parallel layer"
        if not layer.fuse_seam_norm:
            return "needs aiter's fused mHC seam"
        if attn.layer_id >= hf.num_hidden_layers:
            return "draft layer"
        if (
            ffn.shared_experts is None
            or ffn.gate.tid2eid is not None
            or ffn.n_routed_experts != 384
            or ffn.n_activated_experts != 6
            or ffn.routed_scaling_factor != 1.5
            or ffn.swiglu_limit != 10.0
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
        ):
            return "MoE routing outside the kernels' (384 / 6, sqrtsoftplus, x1.5)"
        return ""

    @staticmethod
    def _ffn_only(layer: "DeepseekV4DecoderLayer") -> str:
        """Why the kernels enter this layer at the FFN seam, '' for a whole one."""
        attn = layer.attn
        if layer.engram is not None:
            return "Engram"
        if attn.compressor is not None or attn.indexer is not None:
            return "a compressor / indexer"
        if attn.compress_ratio not in (1, 2):
            return f"compress ratio {attn.compress_ratio}"
        if attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
            return f"a {attn.kv_cache_dtype} KV cache"
        return ""

    def build(self) -> None:
        """The rank's runtime, and the seam each eligible layer is entered at.

        The runtime's peer-memory handshake is collective over the TP group:
        one a rank, shared by every layer, so every rank reserves it alike."""
        from vllm.models.deepseek_v41.amd.mono.runner import (
            EPOCH_WORDS,
            MAX_TOKENS,
            peer_factory,
        )

        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        self.rt = MonoRuntime(
            DSV41_MONO,
            self.vllm_config,
            peer_factory=peer_factory,
            epoch_words=EPOCH_WORDS,
        )
        for layer in self.mono_layers(self.layers):
            why = self._ffn_only(layer)
            self.ffn_only[layer.attn.layer_id] = bool(why)
            layer.mono = self
            if why:
                logger.info_once(
                    "DSv4.1 mono decode: the FFN launch for layers with %s", why
                )
                # a decode step hands wo_b's TP partial to the FFN launch
                layer.attn.wo_b.reduce_results = False
            else:
                logger.info_once(
                    "DSv4.1 mono decode: the whole standard backbone layers"
                )

    @property
    def kernels(self) -> "DSV41MonoLayer":
        if self._kernels is None:
            from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer

            self._kernels = DSV41MonoLayer(self.rt)
        return self._kernels

    def scratch(self, width: int | None) -> torch.Tensor:
        """This width's scratch, reserved on the first step that reaches it."""
        assert width is not None, "a step that goes has a width"
        res = self.rt.reserve(width, self.kernels, width)
        assert res.scratch is not None
        return res.scratch

    def weights(self, layer: "DeepseekV4DecoderLayer"):
        from vllm.models.deepseek_v41.amd.mono.runner import MonoLayerWeights

        a, f = layer.attn, layer.ffn
        if (cached := self._weights.get(a.layer_id)) is not None:
            return cached
        e, sh = f.experts.routed_experts, f.shared_experts
        assert sh is not None  # ``skip_layer`` took only layers with one
        w = MonoLayerWeights(
            attn=None if self.ffn_only[a.layer_id] else self._attn_weights(a),
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
        w.check(get_tensor_model_parallel_world_size())
        self._weights[a.layer_id] = w
        return w

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
    def _metadata(layer: "DeepseekV4DecoderLayer"):
        """(the layer's SWA metadata, its compressed cache's), None when the
        step carries none."""
        if not is_forward_context_available():
            return None, None
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):  # a profile / dummy run, or DBO ubatches
            return None, None
        attn = layer.attn
        swa = cast(
            "DeepseekV4ROCMAiterSparseSWAMetadata | None",
            md.get(attn.swa_cache_layer.prefix),
        )
        comp = None
        if attn.compressed_cache_prefix is not None:
            comp = cast(
                "DeepseekV4FlashMLAMetadata | None",
                md.get(attn.compressed_cache_prefix),
            )
        return swa, comp

    def skip_step(self, rows: int, layer, tensors, whole_layer: bool) -> str:
        if not is_forward_context_available():
            return "no forward context"
        fc = get_forward_context()
        if not self.spec.graph_mode_ok(fc.cudagraph_runtime_mode):
            # a PIECEWISE (breakable) capture is replayed for mixed batches too
            return f"{fc.cudagraph_runtime_mode} step"
        if not isinstance(fc.attn_metadata, dict):
            return "no attention metadata"
        swa, comp = self._metadata(layer)
        if swa is None:
            return "layer metadata missing"
        if swa.num_prefills != 0 or swa.num_decodes == 0:
            return "not a decode-only step"
        if swa.num_decode_tokens > rows:
            return "more decode tokens than rows"
        if whole_layer:
            # the kernels' attention: causal SWA windows, the compressed cache
            if comp is None:
                return "layer metadata missing"
            if swa.decode_swa_width != SWA_WIDTH:
                return f"SWA width {swa.decode_swa_width}"
            if (
                swa.decode_swa_indices is None
                or swa.decode_swa_lens is None
                or swa.token_to_req_indices is None
            ):
                return "SWA decode metadata missing"
        if not all(t is not None and t.is_contiguous() for t in tensors):
            return "non-contiguous layer inputs"
        return ""

    def forward(
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
        if self.ffn_only[layer.attn.layer_id]:
            return None
        if residual is None or x.dim() != 2:
            return self._skip("first layer")
        tensors = (x, residual, post_mix, res_mix, pre_mix)
        step = self.step(x.shape[0], layer, tensors, whole_layer=True)
        if not step:
            return self._skip(step.reason)
        attn = layer.attn
        swa, comp = self._metadata(layer)
        assert swa is not None and comp is not None
        logger.info_once(
            "DSv4.1 mono decode layer active (decode steps of <= %d rows)", MAX_ROWS
        )
        return self.kernels.forward(
            self.weights(layer),
            self.scratch(step.width),
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
        tensors = (part, residual, post_mix, res_mix, pre_mix)
        step = self.step(part.shape[0], layer, tensors, whole_layer=False)
        if not step:
            return self._skip(step.reason)
        logger.info_once(
            "DSv4.1 mono decode FFN launch active (decode steps of <= %d rows)",
            MAX_ROWS,
        )
        return self.kernels.ffn(self.weights(layer), self.scratch(step.width), *tensors)

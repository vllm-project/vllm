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
from vllm.models.common.mono import Feature, MonoOp, MonoRuntime, MonoSpec, StepDecision
from vllm.platforms import current_platform

if TYPE_CHECKING:
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

_rt: MonoRuntime | None = None


def _needs_flydsl(vllm_config) -> str | None:
    try:
        from vllm.models.deepseek_v41.amd.mono.runner import BLOCKS
    except ImportError as err:
        return f"needs FlyDSL and AITER's FlyDSL helpers ({err})"
    # every CTA of a launch stays resident: a partitioned GPU deadlocks
    cus = current_platform.num_compute_units(torch.accelerator.current_device_index())
    if cus < BLOCKS:
        return f"needs {BLOCKS} compute units, the GPU has {cus}"
    return None


def _needs_a8w4_moe(vllm_config) -> str | None:
    if envs.VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4:
        return "needs AITER's A8W4 MoE (VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4 unset)"
    return None


DSV41_MONO = MonoSpec(
    name="DSv4.1 mono decode",
    supports=frozenset({Feature.SPECULATIVE_DECODE}),
    tp_sizes=(2, 4),
    cdna_versions=(4,),  # the kernels' ISA: gfx950 scaled MFMA and MX formats
    widths=tuple(range(1, MAX_ROWS + 1)),
    constraints=(_needs_flydsl, _needs_a8w4_moe),
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


def _runtime(vllm_config) -> MonoRuntime:
    """The process's mono runtime (one a TP rank, shared by every layer). Its
    peer-memory handshake is collective over the TP group: every rank builds it
    at the same layer of the same step."""
    global _rt
    if _rt is None:
        from vllm.models.deepseek_v41.amd.mono.runner import (
            EPOCH_WORDS,
            MAX_TOKENS,
            peer_factory,
        )

        assert MAX_TOKENS >= MAX_ROWS, MAX_TOKENS
        # [epoch, -, -, -, a mark per CTA, the MoE's counters]
        _rt = MonoRuntime(
            DSV41_MONO, vllm_config, peer_factory=peer_factory, epoch_words=EPOCH_WORDS
        )
    return _rt


class MonoDecodeLayer(MonoOp):
    """One decoder layer's mono path (``create``: None when the layer is not
    eligible): the whole layer (``forward``) or, with ``ffn_only``, the FFN
    launch after vLLM's attention (``ffn``). Its weights are taken from the
    layer at the first call, after loading."""

    spec = DSV41_MONO

    def __init__(self, vllm_config, layer: "DeepseekV4DecoderLayer") -> None:
        self.layer = layer
        self.ffn_only = False
        super().__init__(vllm_config)
        self._weights: MonoLayerWeights | None = None
        self._kernels: DSV41MonoLayer | None = None

    def refuse(self) -> list[str]:
        """The kernels read every one of the 384 experts on each rank, TP-sharded,
        in AITER's A8W4 MXFP4 layout: no other MoE backend is servable."""
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend

        ffn = self.layer.ffn
        experts = getattr(getattr(ffn, "experts", None), "routed_experts", None)
        if experts is None:
            return []  # not a routed MoE: the layer checks decline it
        backend = getattr(experts.quant_method, "mxfp4_backend", None)
        if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
            return [f"needs the AITER_MXFP4_BF16 MoE backend, not {backend}"]
        return []

    def decline(self) -> list[str]:
        """Decline the layers whose seams or MoE the kernels do not serve, and
        pick the seam the rest are entered at (``ffn_only``)."""
        layer = self.layer
        attn, ffn = layer.attn, layer.ffn
        hf = self.vllm_config.model_config.hf_config
        if layer.use_sequence_parallel:
            return ["sequence-parallel layer"]
        if not layer.fuse_seam_norm:
            return ["needs aiter's fused mHC seam"]
        if attn.layer_id >= hf.num_hidden_layers:
            return ["draft layer"]
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
            return ["MoE routing outside the kernels' (384 / 6, sqrtsoftplus, x1.5)"]
        # the whole layer: the kernels' attention takes the standard layers only
        why = None
        if layer.engram is not None:
            why = "Engram"
        elif attn.compressor is not None or attn.indexer is not None:
            why = "a compressor / indexer"
        elif attn.compress_ratio not in (1, 2):
            why = f"compress ratio {attn.compress_ratio}"
        elif attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
            why = f"a {attn.kv_cache_dtype} KV cache"
        self.ffn_only = why is not None
        if why is None:
            logger.info_once("DSv4.1 mono decode: the whole standard backbone layers")
        else:
            logger.info_once(
                "DSv4.1 mono decode: the FFN launch for layers with %s", why
            )
        return []

    def build(self) -> None:
        self.rt = _runtime(self.vllm_config)

    @property
    def kernels(self):
        if self._kernels is None:
            from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer

            self._kernels = DSV41MonoLayer(self.rt)
        return self._kernels

    @property
    def weights(self):
        if self._weights is not None:
            return self._weights
        from vllm.models.deepseek_v41.amd.mono.runner import MonoLayerWeights

        layer = self.layer
        a, f = layer.attn, layer.ffn
        e, sh = f.experts.routed_experts, f.shared_experts
        assert sh is not None  # ``decline`` took only layers with one
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
    def _skip(step: StepDecision) -> None:
        logger.debug_once("DSv4.1 mono decode skipped a step: %s", step.reason)
        return None

    def eligible(self, rows: int):
        """(this step's decision, the layer's SWA metadata, its compressed
        cache's)."""
        if not is_forward_context_available():
            return self.rt.step_begin(rows, "no forward context"), None, None
        fc = get_forward_context()
        md = fc.attn_metadata
        attn = self.layer.attn
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
        why = ""
        if not self.spec.graph_mode_ok(fc.cudagraph_runtime_mode):
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
        return self.rt.step_begin(rows, why), swa, comp

    def forward(
        self,
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
            return self._skip(StepDecision("first layer"))
        step, swa, comp = self.eligible(x.shape[0])
        attn = self.layer.attn
        tensors = (x, residual, post_mix, res_mix, pre_mix)
        why = step.reason
        if not why:
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
        if why:
            return self._skip(StepDecision(why))
        assert swa is not None and comp is not None
        logger.info_once(
            "DSv4.1 mono decode layer active (decode steps of <= %d rows)", MAX_ROWS
        )
        return self.kernels.forward(
            self.weights,
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
        part: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        res_mix: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs by the FFN launch from ``part``, wo_b's unreduced
        output, and the attention seam's outputs; None when this step takes the
        original path (the caller then reduces ``part`` itself)."""
        step, _, _ = self.eligible(part.shape[0])
        tensors = (part, residual, post_mix, res_mix, pre_mix)
        why = step.reason
        if not why and not all(t.is_contiguous() for t in tensors):
            why = "non-contiguous layer inputs"
        if why:
            return self._skip(StepDecision(why))
        logger.info_once(
            "DSv4.1 mono decode FFN launch active (decode steps of <= %d rows)",
            MAX_ROWS,
        )
        return self.kernels.ffn(self.weights, *tensors)

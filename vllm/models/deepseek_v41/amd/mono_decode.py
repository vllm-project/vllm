# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 mono decode layer on ROCm CDNA4 and CDNA3
(``VLLM_ROCM_MONO_DECODE=1``).

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

import functools
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
# A gfx942 workgroup has 64 KB of LDS, not 160 KB as on gfx950. Its kernels
# take a step's tokens 6 at a time (``mono.stages.gemv.TILE``), and at TP 4 they
# take up to 42 rows, 7 requests with 5 DSpark drafts each. With 5 drafts vLLM
# captures FULL CUDA graphs of 6, 12, 18, 24, 36 and 42 rows, so a step of 5
# requests runs padded to 36 rows. At 48 rows the shared expert has 8 token
# tiles of 36 tasks each, 288 units, and the MoE runs one unit a CTA on 256
# CTAs. A larger step runs vLLM's path.
MAX_ROWS_GFX942 = 42
# The kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats, or CDNA3
# (gfx942), where the kernels compute the same layer with FNUZ FP8 MFMAs on
# their own copies of the weights (mono/weights942.py).
CDNA_VERSIONS = (3, 4)
# The routed experts' intermediate width a rank at TP 4 (2304 / 4). vLLM pads
# it to 640 on gfx942, and the gfx942 copies keep only the real columns.
INTER_TP4 = 576
# The routed experts and picks a token of the target's MoE layers and of the
# DSpark draft's 3 layers (dspark_n_routed_experts, dspark_num_experts_per_tok).
TARGET_ROUTING = (384, 6)
DRAFT_ROUTING = (128, 3)
# VLLM_ROCM_MONO_SHADOW=1 (eager only): a whole mono layer runs the kernels
# and then vLLM's own layer on the same inputs, logs how far apart the
# outputs are for the first SHADOW_STEPS steps of each step width, and
# continues with vLLM's outputs. A layer that takes the FFN launch does the
# same with the part of vLLM's layer after its attention.
SHADOW_STEPS = 4

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


def _on_gfx942() -> bool:
    from vllm.platforms.rocm import get_cdna_version

    return get_cdna_version() == 3


@functools.cache
def _max_rows() -> int:
    return MAX_ROWS_GFX942 if _on_gfx942() else MAX_ROWS


def _capture_plain_experts(experts) -> None:
    """gfx942: vLLM converts the routed experts to its Triton kernels' format
    when the weights are loaded and frees the plain scales. Wrap that step so
    the mono's gfx942 copies (``weights942.moe_copies``) are made from the
    plain tensors first."""
    qm = experts.quant_method
    original = qm.process_weights_after_loading

    def process_weights_after_loading(layer):
        if layer is experts and not hasattr(layer, "mono942"):
            from vllm.models.deepseek_v41.amd.mono.weights942 import moe_copies

            layer.mono942 = moe_copies(
                layer.w13_weight.data,
                _scale_bytes(layer.w13_weight_scale.data),
                layer.w2_weight.data,
                _scale_bytes(layer.w2_weight_scale.data),
                INTER_TP4,
            )
        return original(layer)

    qm.process_weights_after_loading = process_weights_after_loading


def _shadow_report(layer_id: int, rows: int, step: int, got: tuple, ref: tuple) -> None:
    """Log, a layer and step of ``rows`` rows, each output's largest difference
    relative to the largest reference value and the relative norm of the
    difference."""
    names = ("out", "residual", "post_mix", "res_mix", "pre_mix")
    parts = []
    for name, g, r in zip(names, got, ref):
        g, r = g.float().reshape(r.shape), r.float()
        diff = (g - r).abs()
        rel_max = (diff.max() / r.abs().max().clamp_min(1e-30)).item()
        rel_norm = ((g - r).norm() / r.norm().clamp_min(1e-30)).item()
        bad = int((~torch.isfinite(g)).sum())
        parts.append(f"{name} max {rel_max:.3g} norm {rel_norm:.3g} nonfinite {bad}")
    logger.info(
        "mono shadow layer %d rows %d step %d: %s",
        layer_id,
        rows,
        step,
        "; ".join(parts),
    )


def _topk_report(
    layer_id: int, step: int, got: torch.Tensor, ref: torch.Tensor
) -> None:
    """Log, for an index layer and step, how many of each row's top-k
    entries from the mono path's inputs vLLM's own layer also picked. The
    two differ only where the inputs differ in rounding, so a few entries
    near the row's 512th largest logit can change."""
    parts = []
    for r in range(got.shape[0]):
        g, f = got[r][got[r] >= 0], ref[r][ref[r] >= 0]
        same = int(torch.isin(g, f).sum())
        parts.append(f"{same}/{f.numel()}")
    logger.info(
        "mono shadow layer %d step %d top-k shared with vLLM's: %s",
        layer_id,
        step,
        " ".join(parts),
    )


def _vllm_indexer(attn, x: torch.Tensor, qr: torch.Tensor, positions: torch.Tensor):
    """VLLM's part of an index layer after K1, from rocm.py: the compressor
    chain and the indexer's K on a CSA layer (``_forward_csa2_full``), the
    indexer's weights and queries on every index layer (both forward paths),
    then the indexer of ``_sparse_indexer_and_attn``, which writes the layer's
    top-512 into ``attn.topk_indices_buffer``. ``x`` is the attention's normed
    input row and ``qr`` the normed q latent, both bf16 as vLLM has them on
    gfx942. This runs on one stream. vLLM's forks overlap these calls with its
    q projections, which K1 replaced."""
    compressor, indexer = attn.compressor, attn.indexer
    # ReplicatedLinear returns (output, bias), and the bias is None.
    weights, _ = indexer.weights_proj(x)
    if compressor is not None:
        kv_score = torch.mm(
            x, compressor.fused_wkv_wgate.weight.T, out_dtype=torch.float32
        )
        latent = compressor(kv_score, positions)
        compressor.insert_cache(latent, positions, attn.rotary_emb)
        indexer._produce_k(latent, positions, attn.indexer_rotary_emb)
    index_q, index_q_scale, index_weights = indexer.forward_q(
        qr, None, weights, positions, attn.indexer_rotary_emb
    )
    q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
    indexer.indexer_op(x, q_quant, None, index_weights)


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
    if _on_gfx942():
        # The gfx942 kernels read their own copies of the experts. They are
        # made from the loaded tensors before vLLM's MoE backend converts
        # them (_capture_plain_experts), and moe_copies checks those tensors.
        # So the backend matters on gfx950 only.
        return None
    backend = getattr(experts.quant_method, "mxfp4_backend", None)
    if backend != Mxfp4MoeBackend.AITER_MXFP4_BF16:
        return f"needs the AITER_MXFP4_BF16 MoE backend, not {backend}"
    return None


class MonoDecodeLayer:
    """One decoder layer's mono path (``create``: None when the layer is not
    eligible): the whole layer (``__call__``) or, with ``ffn_only``, the FFN
    launch after vLLM's attention (``ffn``). Its weights are taken from the
    layer at the first call, after loading."""

    def __init__(
        self,
        ffn_only: bool = False,
        topk: int = TARGET_ROUTING[1],
        window: bool = False,
        index: bool = False,
    ) -> None:
        self.ffn_only = ffn_only
        self.topk = topk
        self.window = window
        self.index = index
        # An index layer's K1 outputs for vLLM's compressor and indexer (the
        # normed input row and q latent), one pair for each step size, so a
        # CUDA graph keeps their addresses.
        self._index_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # A window-only layer's constant K1 inputs, one set for each step size.
        self._identity_seams: dict[int, tuple[torch.Tensor, ...]] = {}
        self._weights: MonoLayerWeights | None = None
        # The shadow steps that ran so far, one count for each step width.
        self._shadow_steps: dict[int, int] = {}

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
        gfx942 = _on_gfx942()
        draft = attn.layer_id >= config.num_hidden_layers
        routing = DRAFT_ROUTING if draft else TARGET_ROUTING
        if layer.use_sequence_parallel:
            why = "sequence-parallel layer"
        elif not layer.fuse_seam_norm and not gfx942:
            # gfx942 has no fused seam kernel. vLLM rounds the collapsed
            # stream to bf16 before its RMSNorm there, which the kernels'
            # fused seam does not, so the two paths differ by that rounding.
            why = "needs aiter's fused mHC seam"
        elif gfx942 and get_tensor_model_parallel_world_size() != 4:
            why = "gfx942 builds for tensor parallel size 4 only"
        elif draft and not gfx942:
            why = "draft layer"
        elif (
            ffn.shared_experts is None
            or ffn.gate.tid2eid is not None
            or (ffn.n_routed_experts, ffn.n_activated_experts) != routing
            or ffn.routed_scaling_factor != 1.5
            or ffn.swiglu_limit != 10.0
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
        ):
            why = (
                "MoE routing outside the kernels' "
                f"({routing[0]} / {routing[1]}, sqrtsoftplus, x1.5)"
            )
        if why is not None:
            logger.info_once("DSv4.1 mono decode off for some layers: %s", why)
            return None
        if gfx942:
            _capture_plain_experts(ffn.experts.routed_experts)
        if draft:
            # gfx942 only: the DSpark draft layers take the FFN launch, built
            # for the draft's 128 / 3 routing, after vLLM's attention. The
            # kernels' attention takes the target's causal windows only, and a
            # draft block's tokens also see the block's later tokens. vLLM's
            # draft MoE multiplies bf16 activations with the FP4 experts, and
            # the FFN launch quantizes them to MXFP8 as it does for the
            # target. That changes only how many drafts the target accepts,
            # not the target's output.
            logger.info_once(
                "DSv4.1 mono decode: the FFN launch for the DSpark draft layers "
                "(%d / %d routing)",
                *routing,
            )
            return MonoDecodeLayer(ffn_only=True, topk=routing[1])
        if (
            gfx942
            and attn.indexer is not None
            and attn.compress_ratio in (1, 2)
            and not attn.kv_mxfp8
            and attn.kv_cache_dtype == "fp8_ds_mla"
        ):
            # gfx942 only: an index layer (one with the sparse indexer, 8 of
            # the 40) runs K1 instead of vLLM's attention front up to the q
            # and KV insert. K1's index build writes no keys. It writes the
            # normed input row and q latent in bf16, the inputs of vLLM's
            # compressor and indexer, which run next (``_vllm_indexer``).
            # Then ``mono.keylist`` writes the keys from the indexer's top-512
            # and K2 runs the rest of the layer. Layer 14's Engram runs before
            # K1 as layer 1's does (``_window_seam``).
            logger.info_once(
                "DSv4.1 mono decode: K1, vLLM's compressor and indexer, then K2, "
                "for the index layers"
            )
            return MonoDecodeLayer(index=True)
        if (
            gfx942
            and attn.compress_ratio == 0
            and attn.compressor is None
            and attn.indexer is None
            and not attn.kv_mxfp8
            and attn.kv_cache_dtype == "fp8_ds_mla"
        ):
            # gfx942 only: layers 0 and 1, whose attention sees only its
            # sliding window (compress ratio 0), run as whole mono layers.
            # The kernels' attention with ratio 0 reads the window keys only.
            # Layer 0 starts from the embedding and layer 1 from vLLM's
            # Engram, so K1 gets them as a zero sublayer output with an
            # identity res_mix (``_window_seam``).
            logger.info_once(
                "DSv4.1 mono decode: the whole window-only layers (compress ratio 0)"
            )
            return MonoDecodeLayer(window=True)
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
        if _on_gfx942():
            self._weights = self._weights_942(layer)
            return self._weights
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

    def _weights_942(self, layer: "DeepseekV4DecoderLayer"):
        """gfx942: the layer's tensors as the gfx942 kernels read them. The
        dense MXFP8 linears as ``weights942.linear_copy`` copies, the routed
        experts as the copies made at load (``_capture_plain_experts``)."""
        from vllm.models.deepseek_v41.amd.mono.runner import MonoLayerWeights
        from vllm.models.deepseek_v41.amd.mono.weights942 import linear_copy

        f = layer.ffn
        e, sh = f.experts.routed_experts, f.shared_experts
        assert sh is not None, "DeepSeek-V4.1 layers have a shared expert"
        w13, w13_s, w2, w2_s = e.mono942
        sgu, sgu_s = linear_copy(sh.gate_up_proj.weight, sh.gate_up_proj.weight_scale)
        sw2, sw2_s = linear_copy(sh.down_proj.weight, sh.down_proj.weight_scale)
        weights = MonoLayerWeights(
            attn=None if self.ffn_only else self._attn_weights_942(layer.attn),
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
            w13=w13,
            w13_s=w13_s,
            w2=w2,
            w2_s=w2_s,
            sgu=sgu,
            sgu_s=sgu_s,
            sw2=sw2,
            sw2_s=sw2_s,
        )
        weights.check_942(f.n_routed_experts)
        return weights

    @staticmethod
    def _attn_weights_942(a):
        from vllm.models.deepseek_v41.amd.mono.attention.plan import Dims
        from vllm.models.deepseek_v41.amd.mono.runner import AttnWeights
        from vllm.models.deepseek_v41.amd.mono.weights942 import linear_copy

        copies = {
            name: linear_copy(getattr(a, src).weight, getattr(a, src).weight_scale)
            for name, src in (
                ("wqkv", "fused_wqa_wkv"),
                ("wq_b", "wq_b"),
                ("wo_a", "wo_a"),
                ("wo_b", "wo_b"),
            )
        }
        attn = AttnWeights(
            layer_id=a.layer_id,
            wqkv=copies["wqkv"][0],
            wqkv_scale=copies["wqkv"][1],
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wq_b=copies["wq_b"][0],
            wq_b_scale=copies["wq_b"][1],
            wo_a=copies["wo_a"][0],
            wo_a_scale=copies["wo_a"][1],
            wo_b=copies["wo_b"][0],
            wo_b_scale=copies["wo_b"][1],
            attn_sink=a.attn_sink,
            cos_sin=a.rotary_emb.cos_sin_cache,
            ratio=a.compress_ratio,
        )
        attn.check(Dims(get_tensor_model_parallel_world_size()))
        return attn

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
        if not 1 <= rows <= _max_rows():
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
        input_ids: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs (x, residual, post_mix, res_mix, pre_mix) by the
        mono kernels, or None when this step takes the original path (always for
        an ``ffn_only`` layer)."""
        if self.ffn_only:
            return None
        if x.dim() != 2 or (residual is None and not self.window):
            return self._skip("first layer")
        why, swa, comp = self._decode_step(layer, x.shape[0])
        attn = layer.attn
        ratio = attn.compress_ratio
        if why is None:
            # the kernels' attention: causal SWA windows, the compressed cache
            if comp is None and ratio:
                why = "layer metadata missing"
            elif swa.decode_swa_width != SWA_WIDTH:
                why = f"SWA width {swa.decode_swa_width}"
            elif (
                swa.decode_swa_indices is None
                or swa.decode_swa_lens is None
                or swa.token_to_req_indices is None
            ):
                why = "SWA decode metadata missing"
        if why is None and layer.engram is not None and engram_hashes is None:
            why = "Engram hashes missing"
        seam = (x, residual, post_mix, res_mix, pre_mix)
        if why is None and (self.window or layer.engram is not None):
            seam = self._window_seam(layer, *seam, engram_hashes, engram_mask)
        if why is None and not all(t is not None and t.is_contiguous() for t in seam):
            why = "non-contiguous layer inputs"
        if why is not None:
            return self._skip(why)
        assert swa is not None and (comp is not None or not ratio)
        runner = _mono_runner(x.device)
        logger.info_once(
            "DSv4.1 mono decode layer active (decode steps of <= %d rows)",
            _max_rows(),
        )
        shadow = envs.VLLM_ROCM_MONO_SHADOW
        if shadow and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("VLLM_ROCM_MONO_SHADOW needs eager decode steps.")
        swa_args = (
            swa.slot_mapping,
            _record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
            swa.decode_swa_indices,
            swa.decode_swa_lens,
            swa.token_to_req_indices,
        )
        comp_cache = (
            _record_view(attn._compressed_kv_cache(), comp.block_size // ratio)
            if ratio
            else None
        )
        if self.index:
            logger.info_once(
                "DSv4.1 mono decode: K1, vLLM's compressor and indexer, then K2 "
                "active on the index layers"
            )
            x_n, qr = self._index_inputs(layer, seam[0])
            w = self.weights(layer)
            runner.index_front(w, *seam, positions, *swa_args, x_n, qr)
            _vllm_indexer(attn, x_n, qr, positions)
            topk_buffer = attn.topk_indices_buffer
            assert topk_buffer is not None, "an index layer has a top-k buffer"
            if shadow:
                topk_mine = topk_buffer[: x.shape[0]].clone()
                # A view, so that the report reads what vLLM's own indexer
                # writes into the buffer during the shadow reference below.
                topk_vllm = topk_buffer[: x.shape[0]]
            out = runner.index_back(
                w,
                positions,
                *swa_args,
                topk_buffer,
                comp_cache,
                comp.block_table,
            )
        else:
            out = runner.forward(
                self.weights(layer),
                *seam,
                positions,
                *swa_args,
                topk_indices=attn.topk_indices_buffer if ratio else None,
                comp_cache=comp_cache,
                comp_block_table=comp.block_table if ratio else None,
            )
        if shadow:
            steps = self._shadow_steps.get(x.shape[0], 0)
            ref = self._shadow(
                layer, out, x, positions, residual, post_mix, res_mix, pre_mix,
                input_ids, engram_hashes, engram_mask,
            )  # fmt: skip
            if self.index and steps < SHADOW_STEPS:
                # vLLM's layer ran its own indexer into the same buffer.
                _topk_report(
                    attn.layer_id,
                    steps,
                    topk_mine,
                    topk_vllm,
                )
            return ref
        return out

    def _index_inputs(
        self, layer: "DeepseekV4DecoderLayer", x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The buffers that an index layer's K1 writes for vLLM's compressor
        and indexer, (normed input row [M, hidden], q latent [M, q_lora_rank]),
        both bf16. They are made once for each step size, so a CUDA graph keeps
        their addresses."""
        M = x.shape[0]
        rows = self._index_rows.get(M)
        if rows is None:
            rows = (
                torch.empty_like(x),
                torch.empty(
                    M, layer.attn.q_lora_rank, dtype=torch.bfloat16, device=x.device
                ),
            )
            self._index_rows[M] = rows
        return rows

    def _window_seam(
        self, layer, x, residual, post_mix, res_mix, pre_mix, engram_hashes, engram_mask
    ) -> tuple:
        """K1's five inputs for layer 0 and the Engram layers (1 and 14). K1
        starts with the mHC post of the previous FFN, which these layers do not
        have in that form. Layer 0 has no previous FFN. Its residual is the
        embedding in each hc copy, and the pre-mix it carries in (None) selects
        copy 0. An Engram layer's Engram reads the posted residual, so vLLM
        runs that post and the Engram here, as its forward does. These layers
        then give K1 a zero sublayer output, a zero post_mix and an identity
        res_mix. K1's post only multiplies by 1 and 0 and adds zeros then, so
        the residual comes out bit for bit as it went in."""
        if residual is None:
            residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
        elif layer.engram is not None and engram_hashes is not None:
            residual = layer.mhc_post(x, residual, post_mix, res_mix)
            residual = layer.engram(
                residual, engram_hashes[:, layer.engram.layer_hash_index], engram_mask
            )
        else:
            # A layer after the first one without Engram has the usual seam.
            return x, residual, post_mix, res_mix, pre_mix
        zero_x, zero_post, identity, copy0 = self._identity_seam(x, layer.hc_mult)
        return (
            zero_x,
            residual,
            zero_post,
            identity,
            copy0 if pre_mix is None else pre_mix,
        )

    def _identity_seam(self, x: torch.Tensor, hc: int) -> tuple[torch.Tensor, ...]:
        """(zero sublayer output, zero post_mix, identity res_mix, a pre-mix
        that selects copy 0) for a step of x's rows. The tensors are made once
        for each step size and only read after that, so a CUDA graph can keep
        their addresses."""
        M = x.shape[0]
        consts = self._identity_seams.get(M)
        if consts is None:
            f32 = torch.float32
            copy0 = torch.zeros(M, hc, dtype=f32, device=x.device)
            copy0[:, 0] = 1.0
            consts = (
                torch.zeros_like(x),
                torch.zeros(M, hc, 1, dtype=f32, device=x.device),
                torch.eye(hc, dtype=f32, device=x.device)
                .expand(M, hc, hc)
                .contiguous(),
                copy0,
            )
            self._identity_seams[M] = consts
        return consts

    def _shadow(
        self, layer, out, x, positions, residual, post_mix, res_mix, pre_mix,
        input_ids, engram_hashes=None, engram_mask=None,
    ):  # fmt: skip
        """VLLM's own layer on the inputs the kernels just took, compared with
        the kernels' outputs (``_shadow_report``): its outputs carry on. Its
        attention writes this step's KV records over the kernels' ones."""
        layer.mono = None
        try:
            ref = layer.forward(
                x,
                positions,
                input_ids,
                pre_mix=pre_mix,
                post_mix=post_mix,
                res_mix=res_mix,
                residual=residual,
                engram_hashes=engram_hashes,
                engram_mask=engram_mask,
            )
        finally:
            layer.mono = self
        self._shadow_count(layer, out, ref, x.shape[0])
        return ref

    def _shadow_count(self, layer, out, ref, rows: int) -> None:
        """Report the first SHADOW_STEPS shadow steps of each step width."""
        steps = self._shadow_steps.get(rows, 0)
        if steps < SHADOW_STEPS:
            _shadow_report(layer.attn.layer_id, rows, steps, out, ref)
        self._shadow_steps[rows] = steps + 1

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
        shadow = envs.VLLM_ROCM_MONO_SHADOW
        if shadow and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("VLLM_ROCM_MONO_SHADOW needs eager decode steps.")
        runner = _mono_runner(part.device)
        logger.info_once(
            "DSv4.1 mono decode FFN launch active (decode steps of <= %d rows)",
            _max_rows(),
        )
        out = runner.ffn(self.weights(layer), *tensors, topk=self.topk)
        if shadow:
            return self._ffn_shadow(layer, out, *tensors)
        return out

    def _ffn_shadow(self, layer, out, part, residual, post_mix, res_mix, pre_mix):
        """VLLM's path after its attention, as DeepseekV4DecoderLayer.forward
        runs it when the FFN launch returns None, on the inputs the launch
        just took. It is compared with the launch's outputs
        (``_shadow_report``), and its outputs carry on."""
        from vllm.distributed import tensor_model_parallel_all_reduce

        x = tensor_model_parallel_all_reduce(part)
        residual, post_mix, res_mix, x, ffn_pre = layer.mhc_pre_delayed(
            residual,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=pre_mix,
            sublayer_out=x,
            post_layer_mix=post_mix,
            comb_res_mix=res_mix,
            norm_weight=layer.ffn_norm.weight if layer.fuse_seam_norm else None,
            norm_eps=layer.ffn_norm.variance_epsilon,
        )
        if not layer.fuse_seam_norm:
            x = layer.ffn_norm(x)
        # The MoE needs token ids for its image routing bias. Ids below the
        # image sentinels (129264 on V4.1) route as text, which is the only
        # routing the FFN launch has. The layers it takes have no hash
        # routing, so the ids have no other use here.
        ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        ref = (layer.ffn(x, ids), residual, post_mix, res_mix, ffn_pre)
        self._shadow_count(layer, out, ref, part.shape[0])
        return ref

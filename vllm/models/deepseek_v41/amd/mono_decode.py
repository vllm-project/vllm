# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 mono decode layer on ROCm CDNA4 and CDNA3 (``--monokernel``).

A decode step's backbone layer -- the attention seam, the attention, its TP
all-reduce, the FFN seam, the MoE and its all-reduce -- runs as two persistent
FlyDSL launches (``mono``, this directory) on the layer's loaded
weights, the TP reductions inside the kernels. Every backbone layer of the
checkpoint runs whole (compress ratio 0, 1 or 2 over ``fp8_ds_mla`` records):
the first layer and the Engram layers hand K1 a settled residual, and an index
source runs vLLM's compressor and indexer between the two launches. A layer
outside that (an MXFP8 KV cache, another indexer) keeps vLLM's attention seam
and attention and runs the rest as one launch, the FFN launch, from wo_b's
unreduced output. A step takes the kernels when it is decode only, at most
``MAX_ROWS`` rows (eager or a FULL CUDA graph: the model is not torch.compiled);
a whole layer also needs its SWA rows as the kernels read them (causal windows;
for the DSpark draft's layers, the block's non-causal rows). Anything else runs
the layer as before; the paths share every tensor at the layer boundary, so they
interleave freely.

On CDNA3 (gfx942) the kernels compute the same layer on their own copies of
the weights (``mono/weights942.py``), at TP 4 and on steps of up to
``MAX_ROWS_GFX942`` rows. An index source's keys are written by
``mono/keylist.py`` between vLLM's indexer and K2 there, and the DSpark draft
layers also take the FFN launch, built for the draft's routing.

Every TP rank must take the same path at every layer (the kernels wait on each
other's pushes): the conditions read only metadata every rank holds alike.
"""

import functools
from typing import TYPE_CHECKING, Any, cast

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
    from vllm.models.deepseek_v41.amd.rocm import DeepseekV41ROCMAiterSparseSWAMetadata
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
# FP4 groups of an ug task: the draft's 128-expert unions are small
DRAFT_UG_GROUPS = 1
# The kernels' ISA: CDNA4 (gfx950) scaled MFMA and MX formats, or CDNA3
# (gfx942), where the kernels compute the same layer with FNUZ FP8 MFMAs on
# their own copies of the weights (mono/weights942.py).
CDNA_VERSIONS = (3, 4)
# The routed experts' intermediate width a rank at TP 4 (2304 / 4). vLLM pads
# it to 640 on gfx942, and the gfx942 copies keep only the real columns.
INTER_TP4 = 576

_runner = None
_draft_runner = None


def _mono_runner(device: torch.device, draft: "MonoDecodeLayer | None" = None):
    """The process's mono runner (one a TP rank, shared by every layer of the
    target; the DSpark draft's layers share one of their own, as its routing and
    SWA rows differ). Its peer-memory handshake is collective over the TP group:
    every rank builds it at the same layer of the same step."""
    global _runner, _draft_runner
    if draft is not None:
        if _draft_runner is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "--monokernel's draft runner was first reached inside a CUDA "
                    "graph capture, before an eager draft step."
                )
            from vllm.models.deepseek_v41.amd.mono.runner import DSV41MonoLayer

            _draft_runner = DSV41MonoLayer(
                get_tensor_model_parallel_world_size(),
                get_tensor_model_parallel_rank(),
                get_tp_group().cpu_group,
                device,
                experts=draft.experts,
                topk=draft.topk,
                ug_groups=DRAFT_UG_GROUPS,
                swa_width=draft.swa_width,
            )
        return _draft_runner
    if _runner is None:
        if torch.cuda.is_current_stream_capturing():
            # its peer memory is allocated and exchanged eagerly; vLLM warms
            # every graph up eagerly before capturing it
            raise RuntimeError(
                "--monokernel was first reached inside a CUDA graph "
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
    the gfx942 copies (``weights942.moe_copies``) are made from the
    plain tensors first."""
    qm = experts.quant_method
    original = qm.process_weights_after_loading

    def process_weights_after_loading(layer):
        if layer is experts:
            from vllm.models.deepseek_v41.amd.mono.weights942 import moe_copies

            _keep_copies(
                layer,
                moe_copies(
                    layer.w13_weight.data,
                    _scale_bytes(layer.w13_weight_scale.data),
                    layer.w2_weight.data,
                    _scale_bytes(layer.w2_weight_scale.data),
                    INTER_TP4,
                ),
            )
        return original(layer)

    qm.process_weights_after_loading = process_weights_after_loading


def _keep_copies(layer, copies: tuple[torch.Tensor, ...]) -> None:
    """Store a layer's gfx942 copies as ``layer.mono942``. A reload of the
    weights (``reload_weights``, or a weight update in RL) runs
    ``process_weights_after_loading`` again, and the kernels and captured
    CUDA graphs read the first copies by their addresses. So the copies of a
    reload are written into the first copies in place."""
    kept = getattr(layer, "mono942", None)
    if kept is None:
        layer.mono942 = copies
        return
    for old, new in zip(kept, copies, strict=True):
        old.copy_(new)


# The attention linears that the gfx942 kernels read as ``linear_copy``
# copies, as names of the layer's attention module.
ATTN_LINEARS_942 = ("fused_wqa_wkv", "wq_b", "wo_a", "wo_b")


def _copy_linear_at_load(linear) -> None:
    """gfx942: make the dense MXFP8 linear's ``weights942.linear_copy`` right
    after vLLM has processed its weights at load. By default
    (VLLM_MXFP8_EMULATION_DEQUANT_AT_LOAD) vLLM dequantizes them to bf16
    there, and the copy re-encodes them as the gfx942 kernels read them.
    Weights that stay e4m3 are copied as they are."""
    qm = linear.quant_method
    original = qm.process_weights_after_loading

    def process_weights_after_loading(layer):
        out = original(layer)
        if layer is linear:
            from vllm.models.deepseek_v41.amd.mono.weights942 import linear_copy

            _keep_copies(layer, linear_copy(layer.weight, layer.weight_scale))
        return out

    qm.process_weights_after_loading = process_weights_after_loading


def _copy_942_at_load(layer: "DeepseekV4DecoderLayer", ffn_only: bool) -> None:
    """gfx942: the kernels read their own copies of the layer's weights. Make
    them when vLLM processes the weights at load, so that they count as model
    memory when vLLM sizes the KV cache. Copies made at the first decode step
    would count only when that step is also the CUDA graph memory profile, so
    with that profile off or with ``enforce_eager`` they would not count. A
    layer that takes only the FFN launch needs no attention copies."""
    ffn = layer.ffn
    _capture_plain_experts(ffn.experts.routed_experts)
    sh = ffn.shared_experts
    # _create refuses layers without a shared expert.
    assert sh is not None
    linears = [sh.gate_up_proj, sh.down_proj]
    if not ffn_only:
        linears += [getattr(layer.attn, name) for name in ATTN_LINEARS_942]
    for linear in linears:
        _copy_linear_at_load(linear)


def _compute_units() -> int:
    return current_platform.num_compute_units(torch.accelerator.current_device_index())


def _gpu_arch() -> str:
    """The GPU's architecture without its feature flags, for example gfx942."""
    props = torch.cuda.get_device_properties(torch.accelerator.current_device_index())
    return props.gcnArchName.split(":")[0].lower()


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


class MonoDecodeLayer:
    """One decoder layer's mono path (``create``: None when the layer is not
    eligible): the whole layer (``__call__``) or, with ``ffn_only``, the FFN
    launch after vLLM's attention (``ffn``). Its weights are taken from the
    layer at the first call, after loading."""

    def __init__(
        self,
        ffn_only: bool = False,
        draft: bool = False,
        experts: int = 384,
        topk: int = 6,
        swa_width: int = SWA_WIDTH,
    ) -> None:
        self.ffn_only = ffn_only
        # a layer of the DSpark draft: its own routing and (non-causal) SWA rows
        self.draft, self.experts, self.topk, self.swa_width = (
            draft,
            experts,
            topk,
            swa_width,
        )
        self._weights: MonoLayerWeights | None = None

    @property
    def unfolded_entry(self) -> bool:
        """gfx942: when this whole first layer runs vLLM's path, its entry seam
        projects the expanded residual with the unfolded hc_attn_fn, as K1
        does, and not the embedding with the folded copy (model.py). The
        folded form has no AITER kernel on gfx942."""
        return not self.ffn_only and _on_gfx942()

    @staticmethod
    def create(
        layer: "DeepseekV4DecoderLayer", vllm_config
    ) -> "MonoDecodeLayer | None":
        from vllm.models.deepseek_v41.amd import ops

        # The gfx942 decode helpers run in forward passes, where vLLM's
        # current config is not set, so they read the flag recorded here.
        ops.record_monokernel(vllm_config)
        mono = MonoDecodeLayer._create(layer, vllm_config)
        if mono is not None and _on_gfx942():
            _copy_942_at_load(layer, mono.ffn_only)
        return mono

    @staticmethod
    def _create(
        layer: "DeepseekV4DecoderLayer", vllm_config
    ) -> "MonoDecodeLayer | None":
        if not vllm_config.kernel_config.monokernel:
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
                from vllm.models.deepseek_v41.amd.mono.common.arch import target_arch

                # every CTA of a launch stays resident: a partitioned GPU deadlocks
                cus = _compute_units()
                target, gpu = target_arch(), _gpu_arch()
                if cus < BLOCKS:
                    why = f"needs {BLOCKS} compute units, the GPU has {cus}"
                elif target != gpu:
                    # FlyDSL compiles the kernels for the ARCH environment
                    # variable when it is set, while the choice of the gfx942
                    # weight copies follows the GPU. A stray ARCH would make
                    # the two disagree, so it stops the server here.
                    why = (
                        f"needs FlyDSL to compile for the GPU's {gpu}, but the "
                        f"ARCH environment variable makes it compile for {target}"
                    )
        if why is not None:
            raise ValueError(f"--monokernel {why}.")
        # the layer: the kernels serve the backbone's seams and MoE
        config = vllm_config.model_config.hf_config
        attn, ffn = layer.attn, layer.ffn
        gfx942 = _on_gfx942()
        if gfx942:
            # The gfx942 indexer's top-k hooks need topk512_gfx942.cu. Build it
            # when the model is created, so that a failed build shows at
            # startup and the hooks leave the top-k to vLLM from the first
            # step on. Later layers get the result of the first build.
            from vllm.models.deepseek_v41.amd.ops import topk as topk_ops

            topk_ops.build()
        draft = attn.layer_id >= config.num_hidden_layers
        # a draft layer's routing is the DSpark expert set (128 / 3 on V4.1-Flash)
        experts, topk = 384, 6
        if draft:
            experts = getattr(config, "dspark_n_routed_experts", 0) or experts
            topk = getattr(config, "dspark_num_experts_per_tok", 0) or topk
        if layer.use_sequence_parallel:
            why = "sequence-parallel layer"
        elif not layer.fuse_seam_norm and not gfx942:
            # gfx942 has no fused seam kernel. vLLM rounds the collapsed
            # stream to bf16 before its RMSNorm there, which the kernels'
            # fused seam does not, so the two paths differ by that rounding.
            why = "needs aiter's fused mHC seam"
        elif gfx942 and get_tensor_model_parallel_world_size() != 4:
            why = "gfx942 builds for tensor parallel size 4 only"
        elif (
            ffn.shared_experts is None
            or ffn.gate.tid2eid is not None
            or ffn.n_routed_experts != experts
            or ffn.n_activated_experts != topk
            or ffn.routed_scaling_factor != 1.5
            or ffn.swiglu_limit != 10.0
            or ffn.scoring_func != "sqrtsoftplus"
            or not ffn.renormalize
        ):
            why = (
                "MoE routing outside the kernels' "
                f"({experts} / {topk}, sqrtsoftplus, x1.5)"
            )
        if why is not None:
            logger.info_once("DSv4.1 mono decode off for some layers: %s", why)
            return None
        if draft and gfx942:
            # On gfx942 the DSpark draft layers take the FFN launch, built for
            # the draft's routing, after vLLM's attention. It runs on the
            # target's runner. The whole draft layer below, on a runner of its
            # own with the block's non-causal SWA rows, is checked on gfx950
            # only. vLLM's draft MoE multiplies bf16 activations with the FP4
            # experts, and the FFN launch quantizes them to MXFP8 as it does
            # for the target. That changes only how many drafts the target
            # accepts, not the target's output.
            logger.info_once(
                "DSv4.1 mono decode: the FFN launch for the DSpark draft layers "
                "(%d / %d routing)",
                experts,
                topk,
            )
            return MonoDecodeLayer(ffn_only=True, experts=experts, topk=topk)
        # the whole layer: the kernels' attention takes the standard layers; an
        # index source's compressor and indexer run between its two launches
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer
        from vllm.models.deepseek_v41.attention import DeepseekV4Indexer

        # The indexer that _select drives. The MXFP4 indexer cache needs CDNA4,
        # so gfx942 has vLLM's FP8 indexer, whose wq_b also reads K1's bf16 qr.
        indexer_cls = DeepseekV4Indexer if gfx942 else DeepseekV41RocmMxfp4Indexer
        if attn.compressor is not None and attn.indexer is None:
            why = "a compressor without an indexer"
        elif attn.indexer is not None and not isinstance(attn.indexer, indexer_cls):
            why = f"a {type(attn.indexer).__name__}"
        elif attn.compress_ratio not in (0, 1, 2):
            why = f"compress ratio {attn.compress_ratio}"
        elif attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
            why = f"a {attn.kv_cache_dtype} KV cache"
        elif draft and (attn.compress_ratio or attn.indexer is not None):
            why = "a draft layer with compressed keys"
        if why is not None and draft:
            logger.info_once("DSv4.1 mono decode off for the draft layers: %s", why)
            return None
        if why is not None:
            logger.info_once(
                "DSv4.1 mono decode: the FFN launch for layers with %s", why
            )
            return MonoDecodeLayer(ffn_only=True)
        if draft:
            from vllm.v1.attention.backends.mla.compressor_utils import (
                get_dspark_swa_index_width,
            )

            spec = vllm_config.speculative_config
            logger.info_once("DSv4.1 mono decode: the DSpark draft layers")
            return MonoDecodeLayer(
                draft=True,
                experts=experts,
                topk=topk,
                swa_width=get_dspark_swa_index_width(
                    config.sliding_window, spec.num_speculative_tokens
                ),
            )
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
        self._weights.check(get_tensor_model_parallel_world_size(), self.experts)
        return self._weights

    def _weights_942(self, layer: "DeepseekV4DecoderLayer"):
        """gfx942: the layer's tensors as the gfx942 kernels read them. The
        dense MXFP8 linears and the routed experts as the copies made at load
        (``_copy_942_at_load``)."""
        from vllm.models.deepseek_v41.amd.mono.runner import MonoLayerWeights

        f = layer.ffn
        e, sh = f.experts.routed_experts, f.shared_experts
        assert sh is not None, "DeepSeek-V4.1 layers have a shared expert"
        w13, w13_s, w2, w2_s = e.mono942
        sgu, sgu_s = sh.gate_up_proj.mono942
        sw2, sw2_s = sh.down_proj.mono942
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

        wqkv, wq_b, wo_a, wo_b = (getattr(a, name).mono942 for name in ATTN_LINEARS_942)
        attn = AttnWeights(
            layer_id=a.layer_id,
            wqkv=wqkv[0],
            wqkv_scale=wqkv[1],
            q_norm=a.q_norm.weight,
            kv_norm=a.kv_norm.weight,
            wq_b=wq_b[0],
            wq_b_scale=wq_b[1],
            wo_a=wo_a[0],
            wo_a_scale=wo_a[1],
            wo_b=wo_b[0],
            wo_b_scale=wo_b[1],
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
                "DeepseekV41ROCMAiterSparseSWAMetadata | None",
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
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...] | None:
        """The layer's outputs (x, residual, post_mix, res_mix, pre_mix) by the
        mono kernels, or None when this step takes the original path (always for
        an ``ffn_only`` layer)."""
        if self.ffn_only:
            return None
        first = residual is None
        # the draft's first layer takes the embedding already broadcast to the
        # hc copies ([rows, hc, dim]), the target's the 2-D embedding
        if first and x.dim() != (3 if self.draft else 2):
            return self._skip("first layer of a pipeline stage")
        why, swa, comp = self._decode_step(layer, x.shape[0])
        attn = layer.attn
        tensors = (x,) if first else (x, residual, post_mix, res_mix, pre_mix)
        if why is None:
            # the kernels' attention: causal SWA windows, the compressed cache
            if comp is None and attn.compress_ratio:
                why = "layer metadata missing"
            elif swa.decode_swa_width != self.swa_width:
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
        assert swa is not None
        runner = _mono_runner(x.device, self if self.draft else None)
        logger.info_once(
            "DSv4.1 mono decode %slayer active (decode steps of <= %d rows)",
            "draft " if self.draft else "",
            _max_rows(),
        )
        # K1's seam folds the owed post into the residual, except where the
        # residual arrives settled: the embedding, or after the Engram write
        fold = True
        if first:
            if self.draft:
                residual, x = x, x[:, 0].contiguous()
            else:
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
                comp_cache=_record_view(
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
            _record_view(attn.swa_cache_layer.kv_cache, swa.block_size),
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
        why, _, _ = self._decode_step(layer, part.shape[0])
        tensors = (part, residual, post_mix, res_mix, pre_mix)
        if why is None and not all(t.is_contiguous() for t in tensors):
            why = "non-contiguous layer inputs"
        if why is not None:
            return self._skip(why)
        runner = _mono_runner(part.device)
        logger.info_once(
            "DSv4.1 mono decode FFN launch active (decode steps of <= %d rows)",
            _max_rows(),
        )
        return runner.ffn(self.weights(layer), *tensors, topk=self.topk)

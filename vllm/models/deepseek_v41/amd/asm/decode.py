# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1-Flash decode steps of 17-256 rows on hand-written gfx950
kernels (MI355X, TP4), next to the mono decode layer (`--monokernel`), which
keeps the steps of up to 16 rows.

Per decoder layer:

- K1, the attention half: our seam (mHC pre-mix + norm), front (wqkv, norms,
  SWA insert, wq_b), sparse attention, back (wo_a, wo_b) with all-reduce #1 in
  the same launch; on the index layers also the indexer's q side,
  the compressor tail and the indexer top-k (logits + top-512 in one launch,
  wide rows and the candidate source / consumers up to 32 · 16384 compressed
  positions);
- K2, the FFN half: seam, router, shared expert, routed MoE (MXFP4 experts),
  combine with all-reduce #2;
- the Engram layers' wkv on the prefetch stream, their gate · value applied by
  the next attention seam (`engram.py`).

The folded all-reduces sum in bf16 as vLLM's custom all-reduce does; with a
quantized `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (the user opted into a lossy
all-reduce) they carry MXFP8 partials instead.

The mono decode layer hands this layer the decoder layer's forward for steps of
17-256 rows (`AsmDecodeLayer.__call__`, model.py's forward with our kernels
where they apply). The kernels are row-agnostic except the attention: rows of a
prefill chunk in such a step (a mixed or short prefill step) also run our seams,
front, back and FFN, while the attention, the indexer top-k and the compressor
stay vLLM's on those steps. Wider steps stay on vLLM's path.
"""

import math
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.models.deepseek_v41.amd.model import DeepseekV4DecoderLayer

logger = init_logger(__name__)

H, E = 5120, 384
MAX_M = 256
# the mono decode layer's rows (below ours)
MONO_ROWS = 16
# QuickReduce's quantized regimes: with one of them the folded all-reduces use
# the MXFP8 wire
QR_QUANTIZED = ("INT8", "INT6", "INT4", "INT3")
# --kv-cache-dtype values DeepseekV4Attention turns into fp8_ds_mla (K1's cache)
KV_DTYPES = ("auto", "fp8", "fp8_e4m3", "fp8_ds_mla")

_cfg: dict = {}


def _log(msg: str) -> None:
    logger.debug("asm decode: %s", msg)


def enabled(vllm_config) -> bool:
    """Whether this process runs the asm decode layers: `--monokernel` on a
    gfx950 GPU at TP4, DeepSeek-V4.1-Flash's shapes and an AMDGPU LLVM (clang,
    ld.lld) to build the kernels. Decided once, when the first decoder layer is
    built."""
    if "on" not in _cfg:
        why = None
        config = vllm_config.model_config.hf_config
        if not vllm_config.kernel_config.monokernel:
            why = "--monokernel is off"
        else:
            from vllm.models.deepseek_v41.amd.asm.kernels.build import (
                toolchain_missing,
            )

            props = torch.cuda.get_device_properties(
                torch.accelerator.current_device_index()
            )
            arch = props.gcnArchName.split(":")[0]
            tp = get_tensor_model_parallel_world_size()
            pc = vllm_config.parallel_config
            if arch != "gfx950" or tp != 4:
                why = f"needs gfx950 and TP4 (have {arch}, TP{tp})"
            elif pc.pipeline_parallel_size > 1 or pc.decode_context_parallel_size > 1:
                why = "no pipeline or decode context parallelism"
            elif vllm_config.lora_config is not None:
                why = "no LoRA adapters"
            elif vllm_config.kv_transfer_config is not None:
                why = "no KV transfer"
            elif config.hidden_size != H or config.n_routed_experts != E:
                why = "needs DeepSeek-V4.1-Flash's shapes"
            elif (kv := vllm_config.cache_config.cache_dtype) not in KV_DTYPES:
                why = f"needs the fp8_ds_mla KV cache (have {kv})"
            elif missing := toolchain_missing():
                why = f"needs {missing}"
        _cfg["on"] = why is None
        _cfg["layers"] = config.num_hidden_layers
        qr = envs.VLLM_ROCM_QUICK_REDUCE_QUANTIZATION
        _cfg["wire"] = "fp8" if qr in QR_QUANTIZED else "bf16"
        if why is None:
            logger.info_once(
                "DeepSeek-V4.1 asm decode: steps of %d..%d rows on the asm "
                "kernels, up to %d on the mono decode layer; all-reduces on a "
                "%s wire (VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=%s)",
                MONO_ROWS + 1,
                MAX_M,
                MONO_ROWS,
                "MXFP8" if _cfg["wire"] == "fp8" else "bf16",
                qr,
                scope="global",
            )
        elif vllm_config.kernel_config.monokernel:
            # gfx950 at TP4 expects it: say so louder there
            log = logger.warning_once if "needs gfx950" not in why else logger.info_once
            log("DeepSeek-V4.1 asm decode off: %s", why, scope="global")
    return _cfg["on"]


def mono_rows() -> int | None:
    """The mono decode layer's row limit with the asm decode layers on, else
    None (the decision is made before the mono decode layers are built)."""
    return MONO_ROWS if _cfg.get("on") else None


# ---------------------------------------------------------------- the folded
# all-reduces

_wob: dict = {"wob": None}
_comb: dict = {"car": None}


def _wob_setup() -> None:
    """K1's wo_b and all-reduce #1 in one launch. Collective: every rank calls
    it from its first layer forward (the profile run, outside graph capture)."""
    if _wob["wob"] is not None:
        return
    from vllm.models.deepseek_v41.amd.asm.ar_runner import WoBAllReduce

    tp = get_tp_group()
    _wob["wob"] = WoBAllReduce(
        tp.rank_in_group, tp.cpu_group, tp.world_size, wire=_cfg["wire"]
    )


def _comb_setup() -> None:
    """K2's MoE combine and all-reduce #2 in one launch. Collective, like
    _wob_setup."""
    if _comb["car"] is not None:
        return
    from vllm.models.deepseek_v41.amd.asm.ar_runner import CombAllReduce

    tp = get_tp_group()
    _comb["car"] = CombAllReduce(
        tp.rank_in_group, tp.cpu_group, tp.world_size, wire=_cfg["wire"]
    )


class AsmDecodeLayer:
    """One decoder layer's asm decode path: the layer's forward for steps of
    MONO_ROWS + 1..MAX_M rows. Built with the mono decode layer; its kernels,
    weights and collectives are set up on the layer's first forward (vLLM's
    eager profile run, weights loaded)."""

    @staticmethod
    def create(layer: "DeepseekV4DecoderLayer", vllm_config) -> "AsmDecodeLayer | None":
        if not enabled(vllm_config) or layer.use_sequence_parallel:
            return None
        # DSpark's draft layers (other routing, no compressed attention) stay
        # on the mono decode layer and vLLM
        if layer.attn.layer_id >= vllm_config.model_config.hf_config.num_hidden_layers:
            return None
        return AsmDecodeLayer(layer)

    def __init__(self, layer: "DeepseekV4DecoderLayer"):
        self.layer = layer
        self.installed = False
        # seam handlers, tried in order: K2's (the FFN seam), then K1's or the
        # Engram seam's (the attention seam)
        self.seams: list[Callable] = []
        self.attn_fn: Callable | None = None
        self.ffn_fn: Callable | None = None
        self.mqa_fn: Callable | None = None
        self.idx_topk_fn: Callable | None = None
        self.tk_hook: Callable | None = None
        # Engram: the layer's side chain, whether our attention seam applies
        # gate · value (else eg_out does), the step's (r', gates, value)
        self.eg_side: Any = None
        self.eg_seam = False
        self.eg_pending: tuple | None = None

    def install(self) -> None:
        if self.installed:
            return
        self.installed = True
        layer = self.layer
        lo, hi = MONO_ROWS + 1, MAX_M
        lid = layer.attn.layer_id
        k1 = _k1_install(self, layer, lo, hi)
        k2 = _k2_install(self, layer, lo, hi)
        if self.ffn_fn is not None and getattr(layer, "engram", None) is not None:
            from vllm.models.deepseek_v41.amd.asm.engram import EngramSide

            self.eg_side = EngramSide.create(layer.engram)
        from vllm.models.deepseek_v41.amd.asm.kernels.kmod import wait_builds

        # every kernel loaded before graph capture
        wait_builds()
        _summary["attention"].setdefault(k1, []).append(lid)
        _summary["FFN"].setdefault(k2, []).append(lid)
        if lid == _cfg["layers"] - 1:
            _log_summary()

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
        """model.py's DeepseekV4DecoderLayer.forward with our kernels where they
        take the step → the layer's outputs (x, residual, post_mix, res_mix,
        pre_mix), or None: not a step of ours (the mono decode layer and
        vLLM's forward take it)."""
        self.install()
        m = x.shape[0]
        # K2 on every layer we take: its FFN needs no input_ids (no hash routing)
        if self.ffn_fn is None or not MONO_ROWS < m <= MAX_M:
            return None
        if residual is None and x.dim() != 2:
            return None
        attn_args = (
            layer.hc_attn_fn,
            layer.hc_attn_scale,
            layer.hc_attn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
        )
        if residual is None:
            # the first layer: the embedding broadcast to the hc copies, projected
            # with the folded hc_attn_fn_broadcast (vLLM's seam, the separate norm)
            assert layer.hc_attn_fn_broadcast is not None
            residual = x.unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
            residual, post_mix, res_mix, x, attn_pre = layer.mhc_pre_delayed(
                residual, layer.hc_attn_fn_broadcast, *attn_args[1:], x=x
            )
            x = layer.attn_norm(x)
        else:
            norm = dict(
                pre_mix=pre_mix,
                norm_weight=layer.attn_norm.weight,
                norm_eps=layer.attn_norm.variance_epsilon,
            )
            if layer.engram is not None and engram_hashes is not None:
                # Engram between the previous post and this pre, on the full stream
                residual = layer.mhc_post(x, residual, post_mix, res_mix)
                residual = self._engram(layer, residual, engram_hashes, engram_mask)
                residual, post_mix, res_mix, x, attn_pre = self._pre(
                    layer, residual, *attn_args, **norm
                )
            else:
                residual, post_mix, res_mix, x, attn_pre = self._pre(
                    layer,
                    residual,
                    *attn_args,
                    sublayer_out=x,
                    post_layer_mix=post_mix,
                    comb_res_mix=res_mix,
                    **norm,
                )
        out = None if self.attn_fn is None else self.attn_fn(positions, x)
        if out is None:
            out = layer.attn(positions, x, None)
            if layer.mono is not None and layer.mono.ffn_only:
                # vLLM's wo_b left the reduction to the mono FFN launch
                out = tensor_model_parallel_all_reduce(out)
        x = out
        if layer.attn.layer_id == 0 and _eg["sides"] and engram_hashes is not None:
            # the Engram side chains, after layer 0's attention (the lookups
            # are done by then)
            _eg_start(m)
        residual, post_mix, res_mix, x, ffn_pre = self._pre(
            layer,
            residual,
            layer.hc_ffn_fn,
            layer.hc_ffn_scale,
            layer.hc_ffn_base,
            layer.rms_norm_eps,
            layer.hc_eps,
            layer.hc_eps,
            layer.hc_post_alpha,
            layer.hc_sinkhorn_iters,
            pre_mix=attn_pre,
            sublayer_out=x,
            post_layer_mix=post_mix,
            comb_res_mix=res_mix,
            norm_weight=layer.ffn_norm.weight,
            norm_eps=layer.ffn_norm.variance_epsilon,
        )
        out = self.ffn_fn(x)
        x = out if out is not None else layer.ffn(x, None)
        return x, residual, post_mix, res_mix, ffn_pre

    def _pre(self, layer, *args, **kw):
        """`mhc_pre_delayed` on our kernels where a seam handler takes the
        call, else vLLM's."""
        for fn in self.seams:
            outs = fn(*args, **kw)
            if outs is not None:
                return outs
        return layer.mhc_pre_delayed(*args, **kw)

    def _engram(self, layer, residual, hashes, mask) -> torch.Tensor:
        """The Engram update of r' (`residual`): through the side chain
        layer 0 started this step, as (r', gates, value) for our attention seam
        (or applied here), else vLLM's Engram."""
        engram, side = layer.engram, self.eg_side
        if side is None or not side.started:
            return engram(residual, hashes[:, engram.layer_hash_index], mask)
        side.started = False
        engram._finish_prefetch(side.done)
        gates, value = side.gate(residual, mask)
        if self.eg_seam:
            self.eg_pending = (residual, gates, value)
            return residual
        from vllm.models.deepseek_v41.amd.asm.eg_runner import eg_out

        return eg_out(residual, value, gates)

    def sparse_indexer_and_attn(
        self,
        attn,
        hidden_states: torch.Tensor,
        index_q: torch.Tensor | None,
        index_q_scale: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        q: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """VLLM's `_sparse_indexer_and_attn` with our indexer top-k and sparse
        attention where they apply."""
        if attn.indexer is not None and index_q is not None:
            assert index_weights is not None
            q_quant = (index_q, index_q_scale) if index_q_scale is not None else index_q
            args = (hidden_states, q_quant, None, index_weights)
            if self.idx_topk_fn is not None:
                self.idx_topk_fn(*args)
            else:
                attn.indexer.indexer_op(*args)
            if self.tk_hook is not None:
                self.tk_hook()
        if self.mqa_fn is not None:
            self.mqa_fn(q, q, positions, out)
        else:
            attn.forward_mqa(q, q, positions, out)


@eager_break_during_capture
def _indexer_and_attn(h: AsmDecodeLayer, attn, *args) -> None:
    """Our attention's eager break (vLLM's is the attention module's
    `_sparse_indexer_and_attn`)."""
    h.sparse_indexer_and_attn(attn, *args)


_eg: dict = {"sides": [], "seen": set()}


@eager_break_during_capture
def _eg_start(n: int) -> None:
    """Every Engram layer's side chain for this step's n rows (EngramSide.start)."""
    main = torch.cuda.current_stream()
    for side in _eg["sides"]:
        side.start(n, main)


_summary: dict = {"attention": {}, "FFN": {}, "indexer top-k": {}}


def _ranges(ids: list[int]) -> str:
    """[0, 1, 2, 5] → "0-2, 5"."""
    out: list[str] = []
    for i in sorted(ids):
        if out and int(out[-1].split("-")[-1]) == i - 1:
            out[-1] = f"{out[-1].split('-')[0]}-{i}"
        else:
            out.append(str(i))
    return ", ".join(out)


def _log_summary() -> None:
    """Which half of which layer runs on the asm kernels (steps of 17..256
    rows), once the last layer is set up."""
    for half, by in _summary.items():
        if not by:
            continue
        logger.info_once(
            "DeepSeek-V4.1 asm decode, %s: %s",
            half,
            "; ".join(f"{st}: layers {_ranges(ids)}" for st, ids in sorted(by.items())),
            scope="global",
        )


# ---------------------------------------------------------------- K2: the FFN half

_k2: dict = {"runner": None, "seen": set()}


def _k2_install(h: AsmDecodeLayer, layer, lo: int, hi: int) -> str:
    """The FFN seam call (`mhc_pre_delayed` with `hc_ffn_fn`) runs K2 for
    lo <= M <= hi and returns a token in the x slot; `ffn` then runs the
    combine + all-reduce #2 on it, as vLLM's MoE runner reduces routed +
    shared."""
    from vllm.models.deepseek_v41.amd.asm.k2_runner import K2Runner, raw_from_layer

    _comb_setup()
    if not layer.fuse_seam_norm:
        return "vLLM's (the seam doesn't fuse the norm)"
    ffn = layer.ffn
    if ffn.gate.weight.shape[0] != E:
        return f"vLLM's ({ffn.gate.weight.shape[0]} experts)"
    if (
        ffn.shared_experts is None
        or ffn.gate.tid2eid is not None
        or ffn.n_activated_experts != 6
        or ffn.routed_scaling_factor != 1.5
        or ffn.swiglu_limit != 10.0
        or ffn.scoring_func != "sqrtsoftplus"
        or not ffn.renormalize
    ):
        return "vLLM's (MoE routing)"
    assert not torch.cuda.is_current_stream_capturing(), (
        "K2 must be set up outside graph capture"
    )
    if _k2["runner"] is None:
        from vllm.models.deepseek_v41.amd.asm.kernels.build import build_lock
        from vllm.models.deepseek_v41.amd.asm.kernels.kmod import wait_builds

        # one rank builds (in parallel), the others then hit the cache
        with build_lock():
            _k2["runner"] = K2Runner()
            wait_builds()
    run = _k2["runner"]
    wt = run.weights(raw_from_layer(layer))
    state: dict = {"part": None}

    def seam_forward(*args, **kw):
        # the FFN seam call: 9 positional (residual, hc_ffn_fn, ...
        # sinkhorn_repeat), the rest by keyword
        residual: Any = args[0]
        m = residual.shape[0]
        mixes: list[Any] = [
            kw.get(k)
            for k in ("sublayer_out", "pre_mix", "post_layer_mix", "comb_res_mix")
        ]
        sub, pre_mix, post_mix, res_mix = mixes
        if (
            len(args) != 9
            or args[1] is not layer.hc_ffn_fn
            or kw.get("x") is not None
            or kw.get("norm_weight") is None
            or any(t is None for t in (sub, pre_mix, post_mix, res_mix))
            or residual.dim() != 3
            or not lo <= m <= hi
        ):
            return None
        if m not in _k2["seen"]:
            _k2["seen"].add(m)
            _log(f"K2 at M = {m}")
        part, r_new, post, comb, pre = run(
            wt,
            sub.contiguous(),
            residual.contiguous(),
            post_mix.contiguous(),
            res_mix.contiguous(),
            pre_mix.contiguous(),
        )
        # unwritten: a token for ffn_forward
        state["part"] = part
        state["comb_in"] = run.comb_in
        return r_new, post, comb, part, pre

    def ffn_forward(x):
        if state["part"] is not None and x is state["part"]:
            state["part"] = None
            return _comb["car"](*state.pop("comb_in"), x.shape[0])
        return None

    h.seams.insert(0, seam_forward)
    h.ffn_fn = ffn_forward
    return "asm"


# ---------------------------------------------------------------- K1: the attention
# half

_k1: dict = {"runner": None, "seen": set(), "logged": set()}


def _log_once(key, msg: str) -> None:
    if key not in _k1["logged"]:
        _k1["logged"].add(key)
        _log(msg)


def _k1_runner():
    """The process's one K1Runner (built on first use, under the build lock)."""
    if _k1["runner"] is None:
        from vllm.models.deepseek_v41.amd.asm.k1_runner import K1Runner
        from vllm.models.deepseek_v41.amd.asm.kernels.build import build_lock
        from vllm.models.deepseek_v41.amd.asm.kernels.kmod import wait_builds

        with build_lock():
            _k1["runner"] = K1Runner()
            wait_builds()
    return _k1["runner"]


def _idx_status(attn, status: str) -> None:
    _summary["indexer top-k"].setdefault(status, []).append(attn.layer_id)


def _idx_topk_install(h: AsmDecodeLayer, attn, run, metadata) -> None:
    """On pure decode steps the indexer op runs ours: logits + top-512 in one
    launch (decode rows only; graph-padded rows get none), writing vLLM's top-k
    buffer and the runner's fixed-stride lists. Rows up to 4 · 512 · 18
    compressed positions take one CTA per row; dense layers up to 16 · 16384
    the wide split kernels; the candidate source (ratio 1) up to 32 · 16384 the
    wide kernels with its block scores and top-2048 blocks; the candidate
    consumers the gather mode over the row's candidate blocks. Wider rows (1M
    context) keep vLLM's op, as do prefill / mixed steps."""
    from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import _layer

    op = attn.indexer.indexer_op
    width = int(op.max_model_len)
    cand: Any = getattr(op, "candidate_blocks", None)
    kind = type(op).__name__
    # the wide kernels: ≤ 32 slices of 16k columns
    wmax = 32 * 16384
    wide = cand is None and 4 * 512 * 18 < width <= 16 * 16384
    src_wide = (
        cand is not None
        and getattr(op, "candidate_write", False)
        and op.candidate_block_size == 8
        and cand.shape[1] == 2048
        and 2048 * 8 < width <= wmax
    )
    gat = (
        cand is not None
        and not getattr(op, "candidate_write", False)
        and op.candidate_block_size == 8
        and cand.shape[1] == 2048
        and width > 2048 * 8
    )
    wide = wide or src_wide
    if (
        kind not in ("RocmSparseAttnIndexer", "RocmSparseMQAIndexer")
        or op.topk_tokens != 512
        or (width > 4 * 512 * 18 and not (wide or gat))
    ):
        _idx_status(attn, f"vLLM's ({kind}, width {width})")
        return
    if (
        cand is not None
        and -(-width // op.candidate_block_size) > cand.shape[1]
        and not (src_wide or gat)
    ):
        _idx_status(attn, f"vLLM's (width {width}: candidates can filter)")
        return
    if gat:
        run.idx_gather_prepare()
    elif wide:
        run.idx_wide_prepare(width, source=src_wide)
    else:
        run.idx_logits_prepare(width)

    def fallback(why, *args):
        _log_once(
            ("idx", attn.layer_id, why),
            f"idx top-k: vLLM's op on layer {attn.layer_id} ({why})",
        )
        return op(*args)

    def forward(hidden_states, q_quant, k, weights):
        args = (hidden_states, q_quant, k, weights)
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return op(*args)
        swa_md, comp_md = metadata()
        imd: Any = md.get(op.k_cache.prefix)
        if (
            swa_md is None
            or comp_md is None
            or imd is None
            or not isinstance(q_quant, tuple)
        ):
            return fallback("no metadata", *args)
        if (
            swa_md.num_prefills
            or not swa_md.num_decode_tokens
            or imd.prefill is not None
            or imd.decode is None
        ):
            return fallback("not a decode step", *args)
        if imd.decode_row_lens is None:
            return fallback("no row lens", *args)
        # spec decode (uniform 1 + k rows per request): vLLM's dense launch
        # takes a request's rows as one sequence (decode_native); ours stays
        # per row on the row lengths and row block table
        n = swa_md.num_decode_tokens
        lens = imd.decode_row_lens
        rows = lens.shape[0]
        if rows > MAX_M or rows < n:
            return fallback(f"rows: n {n}, lens {tuple(lens.shape)}", *args)
        lay = _layer(
            op.k_cache.prefix,
            op.k_cache.kv_cache,
            q_quant[0],
            q_quant[1],
            weights,
            op.topk_indices_buffer,
            op.topk_tokens,
            op.head_dim,
            attn.compress_ratio,
            cand,
            op.candidate_block_size,
        )
        buf = op.topk_indices_buffer
        flat = lay.kv.view(lay.kv.shape[0], -1)
        nE = flat.shape[1] // 68
        assert flat.shape[1] == nE * 68 and flat.stride(1) == 1
        if src_wide and imd.decode_block_ends is None:
            return fallback("no block ends", *args)
        if gat:
            run.idx_gather(
                lay.q[:n],
                lay.q_scale[:n],
                lay.weights[:n],
                flat,
                flat.stride(0),
                nE,
                imd.decode.block_table,
                lens[:n],
                swa_md.is_valid_token[:n],
                swa_md.token_to_req_indices[:n],
                comp_md.block_table[: swa_md.num_decodes],
                comp_md.block_size // attn.compress_ratio,
                buf[:n],
                cand[:n],
            )
        else:
            (run.idx_wide if wide else run.idx_logits)(
                width,
                lay.q[:n],
                lay.q_scale[:n],
                lay.weights[:n],
                flat,
                flat.stride(0),
                nE,
                imd.decode.block_table,
                lens[:n],
                swa_md.is_valid_token[:n],
                swa_md.token_to_req_indices[:n],
                comp_md.block_table[: swa_md.num_decodes],
                comp_md.block_size // attn.compress_ratio,
                buf[:n],
                **(
                    dict(cand=cand[:rows], bends=imd.decode_block_ends)
                    if src_wide
                    else {}
                ),
            )
        _k1["tk_done"] = (attn._index_source_prefix, attn.compress_ratio, n)
        return buf

    h.idx_topk_fn = forward
    candidates = (
        "source"
        if getattr(op, "candidate_write", False)
        else "consumer"
        if cand is not None
        else "none"
    )
    _idx_status(attn, f"asm (width {width}, candidates {candidates})")


def _k1_install(h: AsmDecodeLayer, layer, lo: int, hi: int) -> str:
    """One consumer layer (compress ratio > 0, no compressor without an
    indexer) or index layer, in the three pieces of vLLM's breakable graph:

    - the attention seam (`mhc_pre_delayed` with `hc_attn_fn`) runs ours
      and returns vLLM's outputs with y' = bf16(x1 · norm.w) (the row's rstd is
      applied downstream) as the attention input;
    - `attn` on that y': our front (on index layers also the indexer's wq_b
      and q quant in the same launches, while vLLM's weights_proj and the
      compressor (GEMM + comp_tail) run on its aux streams), then vLLM's
      `_sparse_indexer_and_attn` eager break, where `forward_mqa` runs our
      sparse attention on decode-only steps (vLLM's own on mixed steps, its
      MXFP8 output converted to our layout), then our back with all-reduce #1.
    """
    _wob_setup()
    attn = layer.attn
    index = attn.indexer is not None
    if attn.compress_ratio <= 0 or (attn.compressor is not None and not index):
        if getattr(layer, "engram", None) is not None:
            # layer 1: our seam, vLLM's attention
            _eg_seam_only_install(h, layer, lo, hi)
            return "vLLM's (no compressed attention; asm Engram seam)"
        return "vLLM's (no compressed attention)"
    if attn.kv_mxfp8 or attn.kv_cache_dtype != "fp8_ds_mla":
        return f"vLLM's ({attn.kv_cache_dtype} KV cache)"
    assert not torch.cuda.is_current_stream_capturing(), (
        "K1 must be set up outside graph capture"
    )
    run = _k1_runner()

    def metadata():
        md = get_forward_context().attn_metadata
        if not isinstance(md, dict):
            return None, None
        return md.get(attn.swa_cache_layer.prefix), md.get(attn.compressed_cache_prefix)

    if index:
        # After every index source's top-k (on our kernels or not, e.g. the
        # Engram layer 14), map it into the runner's fixed-stride lists once;
        # this layer and its consumers read them instead of vLLM's ragged
        # pack. tk_src names the source, ratio and row count the lists are
        # for; anything else falls back.
        def tk_hook():
            done = _k1.pop("tk_done", None)
            _k1["tk_src"] = None
            swa_md, comp_md = metadata()
            if (
                swa_md is None
                or comp_md is None
                or swa_md.num_prefills
                or not swa_md.num_decode_tokens
            ):
                return
            n = swa_md.num_decode_tokens
            if n > MAX_M or attn.topk_indices_buffer.shape[1] != 512:
                return
            src = (attn._index_source_prefix, attn.compress_ratio, n)
            if done != src:
                # vLLM's top-k ran: our top-k writes the lists itself
                run.translate(
                    attn.topk_indices_buffer[:n],
                    swa_md.token_to_req_indices[:n],
                    swa_md.is_valid_token[:n],
                    comp_md.block_table[: swa_md.num_decodes],
                    comp_md.block_size // attn.compress_ratio,
                )
            # what vLLM's source does with fresh indices
            attn._topk_ragged_cache = {}
            _k1["tk_src"] = src

        h.tk_hook = tk_hook
        _idx_topk_install(h, attn, run, metadata)
        from vllm.models.deepseek_v41.amd.rocm import DeepseekV41RocmMxfp4Indexer

        assert isinstance(attn.indexer, DeepseekV41RocmMxfp4Indexer), type(attn.indexer)
    w = run.weights(layer)
    state: dict = {"y": None}

    h.eg_seam = getattr(layer, "engram", None) is not None

    def seam_forward(*args, **kw):
        # no attention metadata (e.g. the profile run's dummy step): vLLM's
        # attention, an Engram update pending here through the seam it reads
        swa_md, _ = metadata()
        front = swa_md is not None
        outs = _eg_seam(h, layer, run, w, args, kw, front=front, idx=index)
        if outs is not None:
            if front:
                state["y"] = outs[3]
            return outs
        residual: Any = args[0] if args else None
        mixes: list[Any] = [
            kw.get(k)
            for k in ("sublayer_out", "pre_mix", "post_layer_mix", "comb_res_mix")
        ]
        sub, pre_mix, post_mix, res_mix = mixes
        m = residual.shape[0] if residual is not None else 0
        if (
            len(args) < 2
            or args[1] is not layer.hc_attn_fn
            or kw.get("x") is not None
            or any(t is None for t in (sub, pre_mix, post_mix, res_mix))
            or residual.dim() != 3
            or not lo <= m <= hi
            or swa_md is None
            or kw.get("norm_weight") is None
        ):
            # ours needs the norm fused into the seam (its y is a placeholder)
            return None
        r_new, post, comb, y, pre = run.seam_front(
            w,
            residual.contiguous(),
            sub.contiguous(),
            post_mix.contiguous(),
            res_mix.contiguous(),
            pre_mix.contiguous(),
            idx=index,
        )
        state["y"] = y
        return r_new, post, comb, y, pre

    def mqa_forward(q, kv, positions, output):
        # The eager break re-invokes this at every graph replay with the
        # captured arguments (attn_forward doesn't run then).
        swa_md, comp_md = metadata()
        if swa_md is None:
            attn.forward_mqa(q, kv, positions, output)
            return
        m = q.shape[0]
        if swa_md.num_prefills == 0 and swa_md.decode_swa_ragged_indices is not None:
            # rows past n are graph padding (outputs dropped)
            n = swa_md.num_decode_tokens
            q = q[:n]
            positions = positions[:n]
            tk_idx = tk_ptr = None
            fixed = comp_md is not None and _k1.get("tk_src") == (
                attn._index_source_prefix,
                attn.compress_ratio,
                n,
            )
            if fixed:
                tk_idx, tk_ptr = run.s["tk"], run.s["tk_lens"]
            elif comp_md is not None:
                tk_idx, tk_ptr, _ = attn._decode_topk_ragged(
                    swa_metadata=swa_md,
                    attn_metadata=comp_md,
                    num_decodes=swa_md.num_decodes,
                    num_decode_tokens=n,
                )
                _log_once(
                    ("tk", attn.layer_id),
                    f"layer {attn.layer_id} took vLLM's ragged top-k (fixed "
                    f"lists are for {_k1.get('tk_src')}, M = {n})",
                )
            run.attend(
                w,
                q,
                positions,
                attn.swa_cache_layer.kv_cache,
                swa_md.block_size,
                swa_md.decode_swa_ragged_indices,
                swa_md.decode_swa_ragged_indptr,
                attn._compressed_kv_cache() if comp_md is not None else None,
                comp_md.block_size // attn.compress_ratio if comp_md is not None else 0,
                tk_idx,
                tk_ptr,
                fixed=fixed,
            )
            return
        # mixed step: vLLM's attention for every row (on q · rrms), then into
        # our layout
        run.q_rr(q, float(attn.eps))
        attn.forward_mqa(q, kv, positions, output)
        run.from_attn_out(output.data, output.scale, m)
        _log_once("mixed", "mixed step: vLLM's attention + layout conversion")

    # the compressor's GEMM (comp_gemm: split-K partials) + save / pool / norm,
    # compressed insert and the indexer K store in one launch (comp_tail) on
    # the aux stream, instead of vLLM's aux launches + wk and the K store after
    # the join
    ctail: dict | None = None
    if index and attn.compressor is not None:
        comp, idxr = attn.compressor, attn.indexer
        if comp.pcp_size == 1 and idxr.owns_k and idxr.use_fp4_kv:
            from vllm.models.deepseek_v41.amd.asm import comp_tail as ct
            from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
                rocm_paged_mxfp4_cache_layout,
            )

            cw = comp.fused_wkv_wgate.weight.data
            # one partial buffer for all layers (their aux chains don't
            # overlap)
            key = ("comp_part", cw.shape[0])
            if key not in _k1:
                _k1[key] = torch.empty(16, MAX_M, cw.shape[0], device=cw.device)
            ctail = dict(
                fn=ct.comp_tail,
                lat=torch.empty(
                    MAX_M, 512, dtype=torch.bfloat16, device=comp.norm.weight.device
                ),
                wk=idxr.wk.weight.data.contiguous(),
                layout=rocm_paged_mxfp4_cache_layout,
                gemm=dict(
                    fn=ct.comp_gemm, cfg=ct.gemm_cfg, w=cw.contiguous(), part=_k1[key]
                ),
            )

    norm_eps = float(layer.attn_norm.variance_epsilon)
    if index:
        # the weights' scale as vLLM's q quant gets it
        # (rocm_mxfp4_indexer_q_quant: softmax_scale · head_scale) (as the
        # kernel's f32 argument: 128^-½ · 32^-½ is 0.015625000000000003 in
        # double, 2^-6 in f32)
        w_scale = struct.unpack(
            "<f",
            struct.pack("<f", attn.indexer.softmax_scale * attn.indexer.n_head**-0.5),
        )[0]
        w_pow2 = math.frexp(w_scale)[0] == 0.5
        if "idx_wf" not in _k1:
            _k1["idx_wf"] = torch.empty(
                MAX_M, attn.indexer.n_head, device=attn.indexer.wq_b.weight.device
            )
        w_f32 = _k1["idx_wf"]
        assert (
            attn.indexer.use_fp4_kv
            and attn.indexer.indexer_weights_dtype == torch.float32
        )

    def y_of(yp):
        """Y = bf16(y' · rstd), vLLM's attention input, for vLLM's compressor
        (fallbacks); rstd[t] from the attention seam's partials (Σx1² at 25 of
        each slice's 32)."""
        rstd = torch.rsqrt(
            run.s["partial"][: yp.shape[0], :, 25].sum(-1) / H + norm_eps
        )
        return (yp.float() * rstd[:, None]).to(yp.dtype)

    def comp_tail_run(hidden_states, positions, attn_md):
        """comp_gemm + comp_tail on this layer's caches → True; False: not
        applicable here (nothing ran; vLLM's chain runs instead)."""
        comp, idxr = attn.compressor, attn.indexer
        assert ctail is not None
        if not isinstance(attn_md, dict):
            return False
        ccl = comp._static_forward_context[comp.k_cache_prefix]
        cslots = attn_md[comp.k_cache_prefix].slot_mapping
        kslots = attn_md[idxr.k_cache.prefix].slot_mapping
        ccache, kcache = ccl.kv_cache, idxr.k_cache.kv_cache
        if (
            ccache.dtype != torch.uint8
            or ccache.shape[-1] != 584
            or cslots.numel() > MAX_M
        ):
            return False
        if comp.state_cache is not None:
            smd = attn_md[comp.state_cache.prefix]
            st, ss = comp.state_cache.kv_cache, smd.slot_mapping
            qsl, req = smd.query_start_loc, smd.token_to_req_indices
        else:
            st, ss, qsl, req = None, cslots, None, None
        shuffle = ctail["layout"](idxr.n_head, idxr.head_dim, kcache.shape[1])
        g = ctail["gemm"]
        cfg = g["cfg"](hidden_states.shape[0], g["w"].shape[0])
        kv_score, split = g["fn"](hidden_states, g["w"], g["part"], cfg), cfg["split"]
        if split == 1:
            kv_score = kv_score[0, : hidden_states.shape[0]]
        # hidden_states = y' (no rstd): comp_tail applies it from the seam's
        # partials
        ctail["fn"](
            kv_score,
            positions,
            comp.compress_ratio,
            st,
            ss,
            qsl,
            req,
            comp.norm.weight,
            float(comp.rms_norm_eps),
            ctail["lat"],
            ccache,
            cslots,
            attn.rotary_emb.cos_sin_cache,
            ctail["wk"],
            idxr.k_norm.weight,
            float(idxr.k_norm.variance_epsilon),
            attn.indexer_rotary_emb.cos_sin_cache,
            kcache,
            kslots,
            shuffle,
            split,
            rs_part=run.s["partial"],
            rs_eps=norm_eps,
        )
        return True

    def index_front(positions, hidden_states, m, swa_md):
        """Our front + the indexer's wq_b, q RoPE and MXFP4 q on the default
        stream; the compressor (comp_gemm + comp_tail, or vLLM's chain) and
        weights_proj on the aux streams; then the indexer K store (unless
        comp_tail did it) → (q, index q, its scales, index weights)."""
        from vllm.platforms import current_platform
        from vllm.utils.multi_stream_utils import execute_in_parallel

        indexer, comp = attn.indexer, attn.compressor
        attn_md = get_forward_context().attn_metadata

        def default_chain():
            q = run.front(
                w,
                m,
                positions,
                swa_md.slot_mapping,
                attn.swa_cache_layer.kv_cache,
                swa_md.block_size,
                idx=True,
            )
            return q, None

        def weights_chain():
            wr = indexer.weights_proj(hidden_states)[0]
            # vLLM's q quant scales the weights (fp32) by softmax_scale ·
            # n_head^-½ (2^-6 here: exact in bf16)
            if w_pow2:
                return torch.mul(wr, w_scale, out=w_f32[:m])
            return wr.float() * w_scale

        def compressor_chain():
            if ctail is not None and comp_tail_run(hidden_states, positions, attn_md):
                # K stored too
                return None
            kv_score = torch.mm(
                y_of(hidden_states),
                comp.fused_wkv_wgate.weight.T,
                out_dtype=torch.float32,
            )
            latent = comp(kv_score, positions)
            comp.insert_cache(latent, positions, attn.rotary_emb)
            return latent

        fns: list[Callable[[], Any] | None] = [weights_chain]
        if comp is not None:
            fns.insert(0, compressor_chain)
        aux = attn.aux_stream_list
        enable = aux is not None and current_platform.enable_multi_stream_overlap(
            aux, attn_md
        )
        (q, _), res = execute_in_parallel(
            default_chain,
            fns,
            attn.ln_events[0],
            attn.ln_events[1 : 1 + len(fns)],
            aux[: len(fns)] if enable else None,
            enable=enable,
            default_first=True,
        )
        if comp is not None and res[0] is not None:
            indexer._produce_k(res[0], positions, attn.indexer_rotary_emb)
        iq, iq_scale = run.idx_q(m)
        return q, iq, iq_scale, res[-1]

    def attn_forward(positions, hidden_states):
        if state["y"] is None or hidden_states is not state["y"]:
            return None
        state["y"] = None
        m = hidden_states.shape[0]
        swa_md, _ = metadata()
        if m not in _k1["seen"]:
            _k1["seen"].add(m)
            _log(f"K1 at M = {m}")
        if index:
            q, iq, iq_scale, iw = index_front(positions, hidden_states, m, swa_md)
        else:
            q = run.front(
                w,
                m,
                positions,
                swa_md.slot_mapping,
                attn.swa_cache_layer.kv_cache,
                swa_md.block_size,
            )
            iq = iq_scale = iw = None
        attn_out = attn._alloc_attn_out(m, hidden_states)
        _indexer_and_attn(
            h, attn, hidden_states, iq, iq_scale, iw, q, positions, attn_out
        )
        return run.back_ar(w, m, positions, _wob["wob"])

    h.seams.append(seam_forward)
    h.attn_fn = attn_forward
    h.mqa_fn = mqa_forward
    return "asm (index layer)" if index else "asm"


def _eg_seam(h: AsmDecodeLayer, layer, run, w, args, kw, front=False, idx=False):
    """If the layer's Engram update left (r', gates, value) for this seam call,
    run our seam on it as a post-mix → vLLM's seam outputs with r'' as the new
    residual. None: nothing pending (an ordinary call)."""
    pend, h.eg_pending = h.eg_pending, None
    if pend is None:
        return None
    r1, gates, value = pend
    assert (
        len(args) >= 2
        and args[1] is layer.hc_attn_fn
        and args[0] is r1
        and kw.get("sublayer_out") is None
        and kw.get("x") is None
        and kw.get("pre_mix") is not None
    ), "Engram: unexpected seam call"
    if front and kw.get("norm_weight") is None:
        raise RuntimeError(
            "Engram on our seam needs the attention norm fused into the seam"
        )
    pre = kw["pre_mix"].contiguous()
    if front:
        return run.seam_eg_front(w, r1, value, gates, pre, idx=idx)
    return run.seam_eg(w, r1, value, gates, pre)


def _eg_seam_only_install(h: AsmDecodeLayer, layer, lo: int, hi: int) -> None:
    """Our seam on an Engram layer whose attention stays vLLM's (layer 1, ratio
    0): the seam call runs k1_runner.seam_eg, vLLM's attention takes its y."""
    run = _k1_runner()
    w = run.weights(layer, seam_only=True)

    def seam_forward(*args, **kw):
        return _eg_seam(h, layer, run, w, args, kw)

    h.seams.append(seam_forward)
    h.eg_seam = True
    _log(f"Engram: our seam on layer {layer.attn.layer_id} (vLLM's attention)")

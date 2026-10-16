# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer MonoKernel state and step logic of the specialized decode (decode.py).

On every pure-decode step of at most MAX_BATCH_SIZE requests, the kernel replaces vLLM's
decoder layer for the mono layers (3..77); every other step (prefill, mixed, profiling /
dummy runs, too many rows) runs vLLM's layers. The decision is taken once per step
(``begin_step``, from ``SpecializedDecode.plan`` before the layer loop), from attention
metadata identical on all TP ranks.

Boundary: the first mono layer computes ``(normed, x) = fused_allreduce_rms_norm(
h_partial, residual)``; each mono layer returns ``(zeros_L, x_out)`` and the next one
checks that its input is that zeros view; vLLM's final norm of ``(zeros, x_out)`` is
``norm(x_out)``. zeros_L is a view of a persistent per-layer zero buffer, so an output
never aliases an input; only the last layer's buffer is re-zeroed per step (the fused
final norm writes its residual into it).

Attention precision follows the checkpoint (``attention_path``): BF16 attention, or a
checkpoint's per-channel FP8 attention read with its own per-row scales (never
requantized). With FP8 attention, an FP8 indexer ``wq_b``, max_model_len <= 4096 and a
first mono layer that is not an indexer layer, the indexer layers run the fused indexer
(in-kernel index-K cache write, scores and top-k select); otherwise non-fused indexer
layers refresh vLLM's top-k buffer and index-K cache through vLLM's attention
(``indexer_mode="attn"``, output discarded, its MLA row then overwritten by the kernel)
or the indexer alone (``"indexer_only"``). vLLM's logical top-k becomes kernel CSR
slots at the first mono layer and after every non-fused indexer layer. T is padded to
the next built batch size with inactive rows (slot -1, empty sparse range, position 0,
zero hidden)."""

from __future__ import annotations

import time
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.config import (
    HIDDEN,
    KV_LORA,
    PE_DIM,
)
from vllm.models.deepseek_v32.amd.specialized_decode.runtime import DecodeConfig
from vllm.models.deepseek_v32.amd.specialized_decode.runtime.vllm_weights import (
    ATTENTION_MODULES,
)

logger = init_logger(__name__)

# GLM-5.2 decode geometry the kernel is built for (flydsl/config.py GLM5_CONFIG)
TOPK = 2048  # index_topk: sparse keys per query row
MLA_ROW = KV_LORA + PE_DIM  # 576: one fused MLA cache row
KV_LIMIT_BYTES = 1 << 32  # the kernel's 32-bit buffer offsets
ROPE_HALF = 32  # cos / sin columns per position (rotary dim 64, interleaved pairs)
# widest fused-indexer score span (its top-k selection is LDS-resident)
FUSED_INDEX_MAX_SEQ = 4096
INDEX_BLOCK = 16  # the fused indexer's KV / index-cache block (tokens)


def index_max_seq(max_model_len: int) -> int:
    """The fused indexer's score region / selection span: max_model_len rounded up to a
    multiple of 512."""
    return max(512, -(-int(max_model_len) // 512) * 512)


def attention_path(
    layers: dict, has_indexer: dict, max_model_len: int, no_fused: bool = False
) -> tuple[bool, frozenset[int], str]:
    """(FP8 attention, fused-indexer layers, why the fused indexer is off or '') from
    the loaded mono layers' weight dtypes: the checkpoint decides, nothing is
    requantized. Attention (qkv_a, q_b, kv_b, o) must be all BF16 or all FP8 on every
    mono layer; the fused indexer needs FP8 attention, the layer's FP8 ``wq_b``,
    max_model_len <= 4096 and a first mono layer without an indexer (it converts
    vLLM's top-k). Raises RuntimeError on partial / mixed FP8 attention."""
    from .vllm_weights import is_fp8

    fp8 = {
        L: tuple(
            k
            for k, m in ATTENTION_MODULES.items()
            if is_fp8(getattr(lay.self_attn, m).weight)
        )
        for L, lay in layers.items()
    }
    partial = sorted(
        L for L, f in fp8.items() if f and len(f) != len(ATTENTION_MODULES)
    )
    if partial:
        L = partial[0]
        raise RuntimeError(
            f"GLM-5.2 specialized decode: partial FP8 attention on layers {partial} "
            f"(layer {L}: FP8 {list(fp8[L])} of {list(ATTENTION_MODULES)}): the kernel "
            "reads a layer's attention all as BF16 or all as the checkpoint's FP8"
        )
    on = sorted(L for L, f in fp8.items() if f)
    if on and len(on) != len(fp8):
        off = sorted(set(fp8) - set(on))
        raise RuntimeError(
            "GLM-5.2 specialized decode: mixed attention precision: FP8 on layers "
            f"{on[0]}..{on[-1]} ({len(on)}), BF16 on {off[0]}..{off[-1]} ({len(off)})"
        )
    fp8_attention = bool(on)
    index_fp8 = {
        L: is_fp8(lay.self_attn.indexer.wq_b.weight)
        for L, lay in layers.items()
        if has_indexer[L]
    }
    first = min(layers)
    why = (
        "BF16 attention"
        if not fp8_attention
        else "debug_no_fused_indexer"
        if no_fused
        else "the indexer wq_b is not FP8"
        if not any(index_fp8.values())
        else f"max_model_len {max_model_len} > {FUSED_INDEX_MAX_SEQ} (LDS-resident "
        "top-k selection)"
        if index_max_seq(max_model_len) > FUSED_INDEX_MAX_SEQ
        else "the first mono layer is an indexer layer (it converts vLLM's top-k)"
        if first in index_fp8
        else ""
    )
    fused = frozenset() if why else frozenset(L for L, f in index_fp8.items() if f)
    return fp8_attention, fused, why


def step_reason(md, T: int, fits: bool) -> str:
    """'' if a step's attention metadata (identical on every TP rank) allows the
    kernels, else why not."""
    n_dec = int(getattr(md, "num_decode_tokens", 0))
    if md.num_prefills != 0 or n_dec != md.num_actual_tokens:
        return "prefill_or_mixed"
    if md.max_query_len != 1:
        return "query_len"
    if md.num_actual_tokens != T:
        return "padded_T"
    if not fits:
        return "too_many_rows"
    if md.topk_tokens != TOPK:
        return "topk"
    return ""


def padded_batch_size(rows: int, batch_sizes) -> int | None:
    """The smallest built batch size >= rows, or None (more rows than the largest)."""
    return min((b for b in batch_sizes if b >= rows), default=None)


def mla_cache(layer) -> torch.Tensor:
    """The layer's bound MLA KV cache (empty before vLLM binds one)."""
    kv = layer.self_attn.kv_cache
    return kv[0] if isinstance(kv, (list, tuple)) else kv


def rope_tables(layer, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel RoPE ABI: contiguous BF16 [max_pos, 32] cos / sin (interleaved pairs) from
    vLLM's ``rotary_emb.cos_sin_cache`` [max_pos, 64] = [cos | sin]."""
    c = layer.self_attn.rotary_emb.cos_sin_cache[:max_len]
    assert c.shape[1] == 2 * ROPE_HALF, c.shape
    return (
        c[:, :ROPE_HALF].to(torch.bfloat16).contiguous(),
        c[:, ROPE_HALF:].to(torch.bfloat16).contiguous(),
    )


def layer_metadata(layer_name: str):
    """(attention metadata, slot mapping) of a layer for the current step."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    md: Any = fc.attn_metadata
    if isinstance(md, dict):
        md = md.get(layer_name)
    sm = fc.slot_mapping
    return md, (sm.get(layer_name) if isinstance(sm, dict) else None)


class MonoLayers:
    """This rank's mono layers (the MoE decoder layers a MonoKernel replaces): their
    kernels of every batch size, weights, step buffers and per-step state."""

    def __init__(self, backbone, cfg: DecodeConfig):
        from vllm.distributed import get_tp_group

        self.cfg = cfg
        self.tp = get_tp_group()
        self.rank = self.tp.rank_in_group
        self.npes = self.tp.world_size
        assert self.npes == 8, "GLM MonoKernel geometry is TP8"
        self.cpu_group = self.tp.cpu_group
        self.dev = torch.device("cuda", torch.accelerator.current_device_index())
        self.batch_sizes = cfg.batch_sizes
        self.max_rows = self.batch_sizes[-1]
        self.layers = {L: backbone.layers[L] for L in cfg.layers}
        for L, layer in self.layers.items():
            assert type(layer).__name__ == "DeepseekV32DecoderLayer", type(layer)
            assert hasattr(layer.mlp, "experts"), f"layer {L} is not a MoE layer"
        self.order = sorted(self.layers)
        assert self.order == list(range(self.order[0], self.order[-1] + 1)), (
            "mono layers must be contiguous"
        )
        # mailbox epoch = step * launches_per_step + layer + 1 (kernel ABI)
        self.launches_per_step = len(self.order)
        assert 1 <= self.launches_per_step <= 128, self.launches_per_step
        self.first, self.last = self.order[0], self.order[-1]
        self.has_indexer = {
            L: (lay.self_attn.indexer is not None and not lay.self_attn.skip_topk)
            for L, lay in self.layers.items()
        }
        self.fp8_attention, self.fused_layers, no_fused = attention_path(
            self.layers, self.has_indexer, cfg.max_model_len, cfg.debug_no_fused_indexer
        )
        self.index_max_seq = index_max_seq(cfg.max_model_len)
        if self.rank == 0:
            logger.info(
                "GLM-5.2 specialized decode: %s attention (checkpoint precision); %s",
                "FP8 per-channel" if self.fp8_attention else "BF16",
                f"fused indexer on {len(self.fused_layers)} indexer layers "
                f"(max_model_len {cfg.max_model_len})"
                if self.fused_layers
                else f"indexer layers refreshed by vLLM ({cfg.indexer_mode}; fused "
                f"indexer off: {no_fused})",
            )
        # MLA cache block size of the bound KV cache (bind_caches)
        self.block_size: int | None = None
        # indexer_mode="indexer_only": the non-fused indexer layers' refresh
        self.indexer_refresh: dict[int, Any] = {}
        if cfg.indexer_mode == "indexer_only":
            from .indexer_only import IndexerRefresh

            self.indexer_refresh = {
                L: IndexerRefresh(
                    lay.self_attn,
                    self.max_rows,
                    torch.bfloat16,
                    self.dev,
                    trim=cfg.indexer_trim,
                )
                for L, lay in self.layers.items()
                if self.has_indexer[L] and L not in self.fused_layers
            }
        self.enabled = cfg.enabled
        # step_sync: the last rank-uniform vote of `enabled` (None = none yet)
        self._voted: bool | None = None
        # mono steps since install; never reset, so check steps stay rank-aligned
        self._n_steps = 0
        self.active = False
        self.ops: dict[tuple[int, int], Any] = {}
        self.packed: dict[int, dict] = {}
        self.weights: dict[int, Any] = {}
        # per step: layer -> the zeros view it returned (identity check)
        self._zret: dict[int, torch.Tensor] = {}
        self._st: dict[str, Any] = {}
        self._flat: dict[int, torch.Tensor] = {}
        self.dev_nonfinite = (
            torch.zeros(1, dtype=torch.int32, device=self.dev)
            if cfg.device_nonfinite
            else None
        )
        # one zero buffer per mono layer (separate tensors: each [:T] view has storage
        # offset 0), and T -> their [:T] views
        self._zbuf = [
            torch.zeros(self.max_rows, HIDDEN, dtype=torch.bfloat16, device=self.dev)
            for _ in self.order
        ]
        self._zviews: dict[int, list] = {}
        self._epoch_kw = {
            L: dict(layer=L - self.first, advance=False) for L in self.order
        }
        t0 = time.time()
        self._load_weights()
        self.cos, self.sin = rope_tables(self.layers[self.first], cfg.max_model_len)
        self._alloc_buffers()
        for S in self.batch_sizes:
            self._build_ops(S)
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        logger.info(
            "rank %d ready in %.1fs (%.2f GiB allocated, %.2f / %.2f GiB device used)",
            self.rank,
            time.time() - t0,
            torch.accelerator.memory_allocated() / 2**30,
            (total - free) / 2**30,
            total / 2**30,
        )

    # ------------------------------------------------------------------ setup
    def _load_weights(self):
        """Bind every mono layer's weights: views of vLLM's loaded parameters
        (vllm_weights; w_uk / w_uv and the FP8 tile packing are layout copies), plus the
        fused-indexer layers' index projections."""
        from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.config import (
            glm5_tp_config,
        )
        from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.op import (
            prepare_glm5_weights,
        )
        from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.weights import (
            LayerWeights,
        )

        from .vllm_weights import index_weights_from_model, weights_from_model

        t0 = time.time()
        # bytes of kernel-owned weight tensors (not views of vLLM's parameters)
        owned = 0
        for L in self.order:
            layer = self.layers[L]
            vllm_storages = {p.untyped_storage().data_ptr() for p in layer.parameters()}
            t, kw = weights_from_model(layer)
            if L in self.fused_layers:
                index_t = index_weights_from_model(layer.self_attn)
                assert index_t is not None, L  # attention_path: FP8 wq_b
                t.update(index_t)
            W = LayerWeights(
                8,
                t,
                config=glm5_tp_config(self.npes),
                rank=self.rank,
                npes=self.npes,
                **kw,
            )
            packed = dict(prepare_glm5_weights(W))
            self.weights[L], self.packed[L] = W, packed
            seen = set()
            for v in (*W.t.values(), *packed.values()):
                st = v.untyped_storage().data_ptr()
                if st not in seen and st not in vllm_storages:
                    seen.add(st)
                    owned += v.untyped_storage().nbytes()
        torch.accelerator.synchronize()
        logger.info(
            "bound %d layers' weights in %.1fs (rank %d, %.2f GiB kernel-owned)",
            len(self.order),
            time.time() - t0,
            self.rank,
            owned / 2**30,
        )

    def _alloc_buffers(self):
        R, dev = self.max_rows, self.dev
        self.b_x = torch.zeros(R, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.b_pos = torch.zeros(R, dtype=torch.int64, device=dev)
        self.b_slot = torch.full((R,), -1, dtype=torch.int64, device=dev)
        self.b_indptr = torch.zeros(R + 1, dtype=torch.int32, device=dev)
        self.b_indices = torch.zeros(R * TOPK, dtype=torch.int32, device=dev)
        self.b_curpos = torch.zeros(1, dtype=torch.int32, device=dev)
        # per-batch-size views of the padded step inputs
        self._sviews = {
            S: (self.b_pos[:S], self.b_slot[:S], self.b_indptr[: S + 1])
            for S in self.batch_sizes
        }
        if self.fused_layers:
            # fused-indexer block table at a stable address (param tables, graphs)
            nb = -(-int(self.cfg.max_model_len) // INDEX_BLOCK) + 1
            self.b_bt = torch.zeros(R, nb, dtype=torch.int32, device=dev)

    def runtime_ops(self, S: int) -> list:
        """The ops owning a runtime (scratch / step counter / peers) for batch size
        S: one, or two with the fused indexer."""
        return [
            op
            for (_, s), op in self.ops.items()
            if s == S and getattr(op, "_owns_runtime", False)
        ]

    def _go(self, ok: bool) -> bool:
        """Rank-uniform go / no-go and launch alignment: device sync, then a MIN
        all-reduce over the CPU group."""
        import torch.distributed as dist

        torch.accelerator.synchronize()
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.cpu_group)
        return bool(flag.item())

    def _index_caches(self) -> dict | None:
        """{layer: uint8 [blocks, 16, 132] index cache} of the fused layers, or None
        while vLLM has not bound them."""
        out = {}
        for L in self.fused_layers:
            kc = self.layers[L].self_attn.indexer.k_cache
            ic = kc.kv_cache
            if isinstance(ic, (list, tuple)):
                ic = ic[0] if len(ic) else None
            if not isinstance(ic, torch.Tensor) or ic.numel() == 0:
                return None
            # the kernel uses the SHUFFLE layout at the MLA slot (rocm.py: aiter
            # shuffles above block 1)
            if not getattr(kc, "uses_shuffled_layout", False):
                raise RuntimeError(
                    f"GLM-5.2 specialized decode: fused indexer: layer {L} index "
                    "cache not shuffled"
                )
            out[L] = ic.view(torch.uint8)
        return out

    def _set_index_tables(self) -> bool:
        """Point every fused op at its layer's index cache + b_bt (an H2D copy: never
        inside a capture); False while vLLM has not bound the index caches."""
        caches = self._index_caches()
        if caches is None:
            return False
        for L, ic in caches.items():
            for s in self.batch_sizes:
                self.ops[(L, s)].set_index_tables(ic, self.b_bt)
        return True

    def bind_caches(self):
        """Resolve every mono layer's MLA cache and point the fused-indexer ops at the
        index caches (an H2D copy: never inside a capture). Run at every KV bind
        (``SpecializedDecode.prepare_kv``): under FULL graphs vLLM's CUDA-graph memory
        profiling binds minimal caches (one block per capture row), runs mono steps on
        them, frees them and allocates the real ones; a binding kept from that run would
        address the real slots in an 8-block buffer.

        Raises unless every MLA cache is bound, < 4 GiB and its own contiguous bf16
        [..., 576] tensor (the block dim is checked against the step metadata: vLLM may
        split a KV block)."""
        flat: dict[int, torch.Tensor] = {}
        owner: dict[int, int] = {}
        for L, lay in self.layers.items():
            kv = mla_cache(lay)
            if kv is None or kv.numel() == 0:
                raise RuntimeError(
                    f"GLM-5.2 specialized decode: layer {L} has no KV cache bound"
                )
            nbytes = kv.numel() * kv.element_size()
            if nbytes >= KV_LIMIT_BYTES:
                raise RuntimeError(
                    f"GLM-5.2 specialized decode: layer {L} KV cache is "
                    f"{nbytes / 2**30:.2f} GiB >= 4 GiB; the kernel's 32-bit buffer "
                    "offsets would wrap. Bound it (e.g. --num-gpu-blocks-override) or "
                    "use decode_backend=auto."
                )
            why = []
            if kv.dtype is not torch.bfloat16:
                why.append(f"dtype {kv.dtype} (needs kv_cache_dtype=auto -> bf16)")
            if not kv.is_contiguous():
                why.append("not contiguous")
            if kv.dim() < 3 or kv.shape[-1] != MLA_ROW:
                why.append(f"shape {tuple(kv.shape)}: rows of {MLA_ROW} expected")
            if kv.data_ptr() in owner:
                why.append(f"the same tensor as layer {owner[kv.data_ptr()]}'s")
            if why:
                raise RuntimeError(
                    f"GLM-5.2 specialized decode: layer {L} MLA cache: "
                    + "; ".join(why)
                )
            owner[kv.data_ptr()] = L
            flat[L] = kv.view(-1, MLA_ROW)
        self._flat = flat
        self.block_size = int(mla_cache(self.layers[self.first]).shape[-2])
        if self.fused_layers and not self._set_index_tables():
            raise RuntimeError(
                "GLM-5.2 specialized decode: fused indexer: vLLM bound the MLA caches "
                "but not the index caches"
            )

    def _build_ops(self, S: int):
        """One Glm5MonoKernel per mono layer at batch size S, sharing the first op's
        runtime (scratch, peer buffer, step counter); the fused-indexer layers (another
        scratch geometry) share their own. Then one warm-up launch per runtime."""
        from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.op import (
            Glm5MonoKernel,
        )

        t0 = time.time()
        cfg = self.cfg
        # 64-key split tasks spill at S10 / S12; so do the idle-CTA index projections
        stage = dict(
            cache_hoist=cfg.cache_hoist, split_keys64=cfg.split_keys64 and S <= 8
        )
        fused: dict[str, Any] = dict(
            with_indexer=True,
            index_max_seq=self.index_max_seq,
            index_proj_spread=S <= 8,
        )
        runtimes: dict[bool, Any] = {}
        for L in self.order:
            is_fused = L in self.fused_layers
            op = Glm5MonoKernel(
                self.weights[L],
                S,
                rank=self.rank,
                npes=self.npes,
                group=self.cpu_group,
                topk=TOPK,
                poll_limit=cfg.poll_limit,
                prepared_weights=self.packed[L],
                runtime=runtimes.get(is_fused),
                launches_per_step=self.launches_per_step,
                poll_early_out=cfg.poll_early_out,
                **(fused if is_fused else {}),
                **{k: v for k, v in stage.items() if v},
            )
            runtimes.setdefault(is_fused, op)
            self.ops[(L, S)] = op
        if self.fused_layers:
            # inactive rows never touch it: a scratch index cache until the first step
            self._warm_index_cache = torch.zeros(
                2, INDEX_BLOCK, 132, dtype=torch.uint8, device=self.dev
            )
            for L in self.fused_layers:
                self.ops[(L, S)].set_index_tables(self._warm_index_cache, self.b_bt)
        # JIT / code-object load between two barriers, so per-rank compile skew never
        # races the bounded polls: all rows inactive (slot -1, empty sparse range)
        dev = self.dev
        wc = torch.zeros(64, MLA_ROW, dtype=torch.bfloat16, device=dev)
        z = torch.zeros(S, dtype=torch.int64, device=dev)
        for runtime in runtimes.values():
            self._go(True)
            runtime.forward(
                torch.zeros(S, HIDDEN, dtype=torch.bfloat16, device=dev),
                self.b_curpos,
                wc,
                wc,
                torch.zeros(1, dtype=torch.int32, device=dev),
                self.cos,
                self.sin,
                positions=z,
                slot_mapping=z - 1,
                sparse_kv_indptr=torch.zeros(S + 1, dtype=torch.int32, device=dev),
            )
            torch.accelerator.synchronize()
            exp = runtime.poll_error()
            if not self._go(not exp):
                raise RuntimeError(
                    f"GLM-5.2 specialized decode: warm-up launch S={S} expired polls "
                    f"{exp} on some rank"
                )
        logger.info(
            "built S=%d ops for %d layers in %.1fs",
            S,
            len(self.order),
            time.time() - t0,
        )

    # --------------------------------------------------------------- per step
    def _state_reason(self) -> str:
        """'' (go) or why not, for a step the metadata allows (rank-uniform)."""
        if not self.cfg.step_sync:
            return "" if self.enabled else "disabled"
        if self._voted is not True or self.cfg.check_every <= 0:
            self._voted = self._go(self.enabled)
        if not self._voted:
            return "disabled" if not self.enabled else "peer_no_go"
        return ""

    def begin_step(self, positions: torch.Tensor) -> int | None:
        """Once per step, before the layer loop: the rank-uniform go / no-go. On go,
        writes the padded per-step inputs and returns the padded batch size (else None).

        Metadata reasons need no vote (identical on every TP rank). Rank-local state
        (``enabled``) enters only through a step_sync vote: the one taken at the end of
        the last check step, or one taken here while no "on" vote stands. Without
        step_sync (FULL graphs) ``enabled`` is read directly."""
        self.active = False
        T = positions.shape[0]
        md, sm = layer_metadata(self.layers[self.first].self_attn.layer_name)
        S = padded_batch_size(T, self.batch_sizes)
        if md is None or sm is None or not hasattr(md, "paged_kv_indptr"):
            return None
        reason = step_reason(md, T, S is not None)
        if (reason or self._state_reason()) != "":
            return None
        assert S is not None
        if md.block_size != self.block_size:
            raise RuntimeError(
                f"GLM-5.2 specialized decode: step block size {md.block_size} != the "
                f"bound MLA cache's {self.block_size}"
            )
        self.active = True
        self._zret = {}
        # the only zero buffer written (by the fused final norm)
        self._zbuf[-1][:T].zero_()
        # padded per-step inputs (persistent buffers; graph-safe)
        self.b_pos[:T].copy_(positions[:T])
        self.b_pos[T:S].zero_()
        self.b_slot[:T].copy_(sm[:T])
        self.b_slot[T:S].fill_(-1)
        self.b_indptr[: T + 1].copy_(md.paged_kv_indptr[: T + 1])
        if S > T:
            self.b_indptr[T + 1 : S + 1].copy_(
                md.paged_kv_indptr[T : T + 1].expand(S - T)
            )
        if self.fused_layers:
            # the kernel's index-cache block, CSR slot_of and score paging use 16
            if md.block_size != INDEX_BLOCK:
                raise RuntimeError(
                    "GLM-5.2 specialized decode: the fused indexer needs a "
                    f"{INDEX_BLOCK}-token KV block, got {md.block_size}"
                )
            # vLLM's table may be wider than cdiv(max_model_len, 16): copy the prefix
            w = min(md.block_table.shape[1], self.b_bt.shape[1])
            self.b_bt[:T, :w].copy_(md.block_table[:T, :w])
        # one mailbox epoch per step; launch tag = step * lps + (L - first) + 1
        for op in self.runtime_ops(S):
            op.advance_step()
        self._st = dict(T=T, S=S, md=md)
        self._n_steps += 1
        return S

    def _convert_topk(self, layer):
        """Convert vLLM's logical per-row top-k to the kernel's CSR physical slot
        ids."""
        from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
            triton_convert_req_index_to_global_index,
        )

        T, md = self._st["T"], self._st["md"]
        triton_convert_req_index_to_global_index(
            md.req_id_per_token[:T],
            md.block_table,
            layer.self_attn.topk_indices_buffer[:T],
            self.b_indptr[: T + 1],
            self.b_indices,
            # bind_caches requires contiguous MLA caches: rows per block == block size
            BLOCK_STRIDE_ROWS=md.block_size,
            BLOCK_SIZE=md.block_size,
            NUM_TOPK_TOKENS=TOPK,
        )

    def run_layer(self, layer, positions, hidden_states, residual):
        """One mono layer of an active step -> (zeros_L, x_out[:T]), never aliasing the
        inputs; may mutate the first mono layer's hidden_states / residual in place (the
        boundary norm). ``Step.layer`` hands it to the model's layer loop."""
        L = layer.layer_idx
        if not self.active:
            raise RuntimeError(
                f"GLM-5.2 specialized decode: layer {L} called on a step without a plan"
            )
        T, S = self._st["T"], self._st["S"]
        normed = None
        if self.first == L:
            from vllm.models.common.ops.fused_allreduce_rms_norm import (
                fused_allreduce_rms_norm,
            )

            normed, x = fused_allreduce_rms_norm(
                hidden_states, residual, layer.input_layernorm
            )
        else:
            if hidden_states is not self._zret.get(L - 1):
                raise RuntimeError(
                    f"GLM-5.2 specialized decode: layer {L} runs its kernel but layer "
                    f"{L - 1} did not in this step (step state corrupted); refusing to "
                    "continue"
                )
            x = residual
        refresh = self.has_indexer[L] and L not in self.fused_layers
        if refresh:
            if normed is None:
                normed = layer.input_layernorm(x)
            if L in self.indexer_refresh:
                self.indexer_refresh[L](positions, normed)
            else:
                layer.self_attn(positions=positions, hidden_states=normed)
        if self.first == L or refresh:
            self._convert_topk(layer)
        if S == T and x.dtype is torch.bfloat16 and x.is_contiguous():
            # a step at a built batch size reads x in place (no copy node in its graph)
            h = x
        else:
            self.b_x[:T].copy_(x[:T])
            if S > T:
                self.b_x[T:S].zero_()
            h = self.b_x[:S]
        flat = self._flat[L]
        pos_v, slot_v, indptr_v = self._sviews[S]
        x_out = self.ops[(L, S)].forward(
            h,
            self.b_curpos,
            flat,
            flat,
            self.b_indices,
            self.cos,
            self.sin,
            positions=pos_v,
            slot_mapping=slot_v,
            sparse_kv_indptr=indptr_v,
            **self._epoch_kw[L],
        )
        out = x_out[:T]
        if self.last == L:
            self._end_step(out)
        zv = self._zviews.get(T)
        if zv is None:
            zv = self._zviews[T] = [b[:T] for b in self._zbuf]
        self._zret[L] = z = zv[L - self.first]
        return z, out

    def _end_step(self, out):
        """Eager health check (step_sync, every check_every kernel steps): expired
        polls fail-stop unless ``failstop`` is "warn" / "off"; then, and for non-finite
        output, the kernels are disabled from the next step through the rank-uniform
        vote. Under FULL graphs the executor's PollErrorWatch does the check."""
        if self.dev_nonfinite is not None:  # capture-safe, no host sync
            self.dev_nonfinite.add_((~torch.isfinite(out)).any().to(torch.int32))
        n = self._n_steps
        every = self.cfg.check_every
        if not self.cfg.step_sync or every <= 0 or n % every:
            return
        from vllm.models.deepseek_v32.amd.specialized_decode.runtime.guards import (
            fail_stop,
        )

        mode = self.cfg.failstop
        torch.accelerator.synchronize()
        # the poll-error words are sticky: only the fail-stop watch clears them (warn
        # mode), so leave them set unless no watch runs
        exp = tuple(
            e
            for op in self.runtime_ops(self._st["S"])
            for e in op.poll_error(clear=mode == "off")
        )
        fin = bool(torch.isfinite(out).all())
        if exp or not fin:
            why = f"GLM-5.2 specialized decode: step {n}: expired={exp} finite={fin}"
            if exp and mode == "raise":
                # this step's output and KV rows are already wrong: disabling would
                # keep serving them
                fail_stop(why, self.rank)
            self.enabled = False
            logger.error("%s -> disabling", why)
        # rank-uniform (MIN) state for the next steps; replaces their begin-step vote
        self._voted = self._go(self.enabled)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer MonoKernel state and step logic of the model-integrated dispatch
(dispatch.py).

On every pure-decode step whose rows fit a kernel width, the kernel replaces vLLM's
decoder layer for the mono layers (3..77); every other step (prefill, mixed, profiling /
dummy runs, too many rows) runs vLLM's layers. The decision is taken once per step at
the first mono layer, from attention metadata identical on all TP ranks.

Boundary: the first mono layer computes ``(normed, x) = fused_allreduce_rms_norm(
h_partial, residual)``; each mono layer returns ``(zeros_L, x_out)`` and the next one
checks that its input is that zeros view; vLLM's final norm of ``(zeros, x_out)`` is
``norm(x_out)``. zeros_L is a view of a persistent per-layer zero buffer, so an output
never aliases an input (custom-op contract); only the last layer's buffer is re-zeroed
per step (the fused final norm writes its residual into it).

Non-fused indexer layers refresh vLLM's top-k buffer and index-K cache through vLLM's
attention (``indexer_mode="attn"``, output discarded, its MLA row then overwritten by
the kernel) or the indexer alone (``"indexer_only"``); the fused indexer does it
in-kernel. vLLM's logical top-k becomes kernel CSR slots at the first mono layer and
after every non-fused indexer layer. T is padded to the next kernel width with inactive
rows (slot -1, empty sparse range, position 0, zero hidden)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.mono.spec import step_reason, width_for

logger = init_logger(__name__)

# GLM-5.2 decode geometry the kernel is built for (kernel/config.py GLM5_CONFIG)
TOPK = 2048  # index_topk: sparse keys per query row
HIDDEN = 6144
MLA_ROW = 576  # kv_lora 512 + rope 64: one fused MLA cache row
ROPE_HALF = 32  # cos / sin columns per position (rotary dim 64, interleaved pairs)


def _mla_cache(layer) -> torch.Tensor:
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
    """(attention metadata, slot mapping) of a layer for the current step; a list-form
    ``attn_metadata`` (ubatching) uses its first element."""
    from vllm.forward_context import get_forward_context

    fc = get_forward_context()
    md: Any = fc.attn_metadata
    if isinstance(md, list):
        md = md[0] if md else None
    if isinstance(md, dict):
        md = md.get(layer_name)
    sm = fc.slot_mapping
    return md, (sm.get(layer_name) if isinstance(sm, dict) else None)


@dataclass
class LiveConfig:
    ckpt: str
    layers: list[int] = field(default_factory=lambda: list(range(3, 78)))
    # kernel widths to build (subset of spec.KERNEL_WIDTHS)
    sizes: tuple[int, ...] = (1, 2, 4, 5, 6, 8)
    # MTP verify steps on the kernel; None = on iff vLLM has a speculative_config
    spec_decode: bool | None = None
    poll_limit: int = 20_000_000  # ~4 s per mailbox wait
    indexer_mode: str = "attn"  # "attn" | "indexer_only" (non-fused indexer layers)
    # eager only: rank-uniform state vote (device sync + CPU all-reduce)
    step_sync: bool = True
    check_every: int = 1  # eager health check every N mono steps (0 = never)
    max_model_len: int = 4096
    enabled: bool = True  # initial state
    poll_early_out: bool = True  # after one expired poll, skip the step's other waits
    attention_weight: str = "fp8_block128"  # or "bf16"
    indexer_trim: bool = True  # indexer_only: skip work whose results are discarded
    early_cache_checks: bool = True  # validate the MLA cache contract at install
    # in-kernel indexer on indexer layers; None = on when supported (see MonoLive)
    fused_indexer: bool | None = None
    index_q_fp8: bool = True  # FP8 index q, as vLLM's fused_q
    cache_hoist: bool = True
    fused_select_radix11: bool = True
    fused_index_proj_spread: bool = True  # widths <= 8 only
    split_keys64: bool = True  # widths <= 8 only
    # captured isfinite count of each step's output (+ fail-stop on it)
    device_nonfinite: bool = False
    failstop_nonfinite: bool = False


class MonoLive:
    def __init__(self, model, cfg: LiveConfig, vllm_config):
        from vllm.distributed import get_tp_group
        from vllm.models.deepseek_v32.amd.mono.guards import full_cudagraphs
        from vllm.models.deepseek_v32.amd.mono.kernel.config import AttentionWeight

        self.cfg = cfg
        sc = vllm_config.speculative_config
        num_spec = int(getattr(sc, "num_speculative_tokens", 0) or 0)
        self.spec_decode = num_spec > 0 if cfg.spec_decode is None else cfg.spec_decode
        if self.spec_decode and num_spec <= 0:
            raise ValueError(
                "spec_decode=True but vLLM runs without a speculative_config"
            )
        # verify steps carry 1 + k query rows per request; plain decode 1
        self.max_query_len = 1 + num_spec if self.spec_decode else 1
        if full_cudagraphs(vllm_config) and cfg.step_sync:
            raise ValueError(
                "step_sync=True under FULL cudagraphs: its device sync and CPU "
                "all-reduce cannot run inside a graph capture"
            )
        self.tp = get_tp_group()
        self.rank = self.tp.rank_in_group
        self.npes = self.tp.world_size
        assert self.npes == 8, "GLM MonoKernel geometry is TP8"
        self.cpu_group = self.tp.cpu_group
        self.dev = torch.device("cuda", torch.accelerator.current_device_index())
        self.sizes = tuple(sorted(cfg.sizes))
        self.max_rows = self.sizes[-1]
        self.layers = {L: model.model.layers[L] for L in cfg.layers}
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
        # score region / LDS selection span of the fused indexer (multiple of 512)
        self.index_max_seq = max(512, -(-int(cfg.max_model_len) // 512) * 512)
        no_fused = (
            "speculative decoding (one query row per request)"
            if self.spec_decode
            else "attention_weight != 'fp8_block128'"
            if cfg.attention_weight != "fp8_block128"
            else f"max_model_len {cfg.max_model_len} > 4096 (LDS-resident selection)"
            if self.index_max_seq > 4096
            else "the first mono layer is an indexer layer (it converts vLLM's top-k)"
            if self.has_indexer[self.first]
            else None
        )
        if cfg.fused_indexer and no_fused:
            raise ValueError(f"fused_indexer: {no_fused}")
        if cfg.fused_indexer is None and no_fused:
            logger.info("mono live: fused_indexer auto -> off (%s)", no_fused)
        self.fused_indexer = (
            no_fused is None if cfg.fused_indexer is None else cfg.fused_indexer
        )
        self.fused_layers = frozenset(
            L for L in self.layers if self.fused_indexer and self.has_indexer[L]
        )
        self._index_tables_set = False
        # {layer: (data_ptr, numel)} of the index caches the fused tables point at
        self._index_ptrs: dict[int, tuple[int, int]] | None = None
        if cfg.indexer_trim and cfg.indexer_mode == "indexer_only":
            from vllm.models.deepseek_v32.amd.mono.indexer_only import prealloc

            for L, lay in self.layers.items():
                if self.has_indexer[L]:
                    prealloc(lay.self_attn, self.max_rows, torch.bfloat16, self.dev)
        self.enabled = cfg.enabled
        # step_sync: the last rank-uniform vote of `enabled` (None = none yet)
        self._voted: bool | None = None
        # mono steps since install; never reset, so check steps stay rank-aligned
        self._n_mono = 0
        self.active = False
        self.ops: dict[tuple[int, int], Any] = {}
        self.packed: dict[int, dict] = {}
        self.weights: dict[int, Any] = {}
        self._attn_fmt = AttentionWeight(cfg.attention_weight)
        assert self._attn_fmt in (AttentionWeight.BF16, AttentionWeight.FP8_BLOCK128)
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
        # offset 0, like the op's fake output), and T -> their [:T] views
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
        for S in self.sizes:
            self._build_ops(S)
        if self.fused_layers:
            # graph mode: must happen before vLLM captures (H2D copy)
            self._set_index_tables()
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        logger.info(
            "mono live: rank %d ready in %.1fs (%.2f GiB allocated, %.2f / %.2f GiB "
            "device used)",
            self.rank,
            time.time() - t0,
            torch.accelerator.memory_allocated() / 2**30,
            (total - free) / 2**30,
            total / 2**30,
        )

    # ------------------------------------------------------------------ setup
    def _load_weights(self):
        from vllm.models.deepseek_v32.amd.mono.ckpt_weights import (
            index_weights_from_ckpt,
            load_glm5_layer,
        )
        from vllm.models.deepseek_v32.amd.mono.fp8_attention import (
            quantize_attention_fp8,
        )
        from vllm.models.deepseek_v32.amd.mono.kernel.config import (
            AttentionWeight,
            glm5_tp_config,
        )
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import prepare_glm5_weights
        from vllm.models.deepseek_v32.amd.mono.kernel.weights import LayerWeights

        t0 = time.time()
        nbytes = 0
        worst = 0.0
        fp8 = self._attn_fmt is AttentionWeight.FP8_BLOCK128
        for L in self.order:
            t = load_glm5_layer(self.cfg.ckpt, L, self.rank, self.npes, device=self.dev)
            if fp8:
                worst = max(worst, *quantize_attention_fp8(t).values())
            if L in self.fused_layers:
                t.update(index_weights_from_ckpt(self.cfg.ckpt, L, self.dev))
            W = LayerWeights(
                8, t, config=glm5_tp_config(self.npes), rank=self.rank, npes=self.npes
            )
            packed = dict(prepare_glm5_weights(W, self._attn_fmt))
            # free raw copies shadowed by a packed tensor (forward uses
            # dict(W.t, **packed)); w_uv stays: FP8 uv_scale_rows derive from its shape
            for k in packed:
                if k == "w_uv" and fp8:
                    continue
                if k in W.t and W.t[k].data_ptr() != packed[k].data_ptr():
                    W.t[k] = torch.empty(0, dtype=W.t[k].dtype, device=self.dev)
            self.weights[L], self.packed[L] = W, packed
            seen = set()
            for v in (*W.t.values(), *packed.values()):
                if v.data_ptr() not in seen:
                    seen.add(v.data_ptr())
                    nbytes += v.numel() * v.element_size()
        if fp8:
            logger.info(
                "mono live: FP8 block-128 attention, worst per-matrix rel-L2 quant "
                "error %.3e (rank %d)",
                worst,
                self.rank,
            )
        torch.accelerator.synchronize()
        logger.info(
            "mono live: loaded %d layers in %.1fs (rank %d, %.2f GiB kernel weights)",
            len(self.order),
            time.time() - t0,
            self.rank,
            nbytes / 2**30,
        )

    def _alloc_buffers(self):
        R, dev = self.max_rows, self.dev
        self.b_x = torch.zeros(R, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.b_pos = torch.zeros(R, dtype=torch.int64, device=dev)
        self.b_slot = torch.full((R,), -1, dtype=torch.int64, device=dev)
        self.b_indptr = torch.zeros(R + 1, dtype=torch.int32, device=dev)
        self.b_indices = torch.zeros(R * TOPK, dtype=torch.int32, device=dev)
        self.b_curpos = torch.zeros(1, dtype=torch.int32, device=dev)
        # per-width views of the padded step inputs
        self._sviews = {
            S: (self.b_pos[:S], self.b_slot[:S], self.b_indptr[: S + 1])
            for S in self.sizes
        }
        if self.fused_layers:
            # fused-indexer block table at a stable address (param tables, graphs)
            nb = -(-int(self.cfg.max_model_len) // 16) + 1
            self.b_bt = torch.zeros(R, nb, dtype=torch.int32, device=dev)

    def runtime_ops(self, S: int) -> list:
        """The ops owning a runtime (scratch / step counter / peers) for width S: one,
        or two with the fused indexer."""
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
                raise RuntimeError(f"fused_indexer: layer {L} index cache not shuffled")
            out[L] = ic.view(torch.uint8)
        return out

    def _ensure_caches(self, block_size: int):
        """Rebind the caches when vLLM's first-mono-layer MLA cache moved (the size is
        compared too: the allocator can reuse a freed cache's address)."""
        bound = self._flat.get(self.first)
        kv = _mla_cache(self.layers[self.first])
        if bound is None or (bound.data_ptr(), bound.numel()) != (
            kv.data_ptr(),
            kv.numel(),
        ):
            self._bind_caches(block_size)

    def _bind_caches(self, block_size: int):
        """Resolve every layer's MLA cache; drop the fused-indexer tables unless they
        point at the current index caches. Under FULL graphs vLLM's CUDA-graph memory
        profiling binds minimal caches (one block per capture row), runs mono steps on
        them, frees them and allocates the real ones: a binding kept from that run would
        address the real slots in an 8-block buffer."""
        flat = {}
        for L, lay in self.layers.items():
            kv = _mla_cache(lay)
            assert (
                kv.dtype is torch.bfloat16
                and kv.is_contiguous()
                and kv.shape[-1] == MLA_ROW
                and block_size == kv.shape[-2]
            ), (L, kv.shape, kv.dtype, block_size)
            flat[L] = kv.view(-1, MLA_ROW)
        assert len({v.data_ptr() for v in flat.values()}) == len(flat), (
            "MLA caches alias"
        )
        self._flat = flat
        if self.fused_layers and self._index_tables_set:
            caches = self._index_caches()
            cur = (
                None
                if caches is None
                else {L: (c.data_ptr(), c.numel()) for L, c in caches.items()}
            )
            if cur != self._index_ptrs:
                # re-pointed by the next eager step (_set_index_tables: an H2D copy)
                self._index_tables_set = False

    def _set_index_tables(self) -> bool:
        """Point every fused op at its layer's index cache + b_bt (an H2D copy: never
        inside a capture). Done at install if the caches are bound, else at the first
        eager mono step."""
        caches = self._index_caches()
        if caches is None:
            return False
        for L, ic in caches.items():
            for s in self.sizes:
                self.ops[(L, s)].set_index_tables(ic, self.b_bt)
        self._index_ptrs = {L: (ic.data_ptr(), ic.numel()) for L, ic in caches.items()}
        self._index_tables_set = True
        return True

    def _build_ops(self, S: int):
        """One Glm5MonoKernel per mono layer at width S, sharing the first op's runtime
        (scratch, peer buffer, step counter); the fused-indexer layers (another scratch
        geometry) share their own. Then one warm-up launch per runtime."""
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import Glm5MonoKernel

        t0 = time.time()
        cfg = self.cfg
        # 64-key split tasks spill at S10 / S12; so do the idle-CTA index projections
        stage = dict(
            cache_hoist=cfg.cache_hoist, split_keys64=cfg.split_keys64 and S <= 8
        )
        fused: dict[str, Any] = dict(
            with_indexer=True,
            index_max_seq=self.index_max_seq,
            index_q_fp8=cfg.index_q_fp8,
            select_radix11=cfg.fused_select_radix11,
            index_proj_spread=cfg.fused_index_proj_spread and S <= 8,
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
                attention_weight=self._attn_fmt,
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
                2, 16, 132, dtype=torch.uint8, device=self.dev
            )
            for L in self.fused_layers:
                self.ops[(L, S)].set_index_tables(self._warm_index_cache, self.b_bt)
        # JIT / code-object load between two barriers, so per-rank compile skew never
        # races the bounded polls: all rows inactive (slot -1, empty sparse range)
        dev = self.dev
        wc = torch.zeros(64, MLA_ROW, dtype=torch.bfloat16, device=dev)
        z = torch.zeros(S, dtype=torch.int64, device=dev)
        for op in runtimes.values():
            self._go(True)
            op.forward(
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
            exp = op.poll_error()
            if not self._go(not exp):
                raise RuntimeError(
                    f"mono live: warm-up launch S={S} expired polls {exp} on some rank"
                )
        logger.info(
            "mono live: built S=%d ops for %d layers in %.1fs",
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

    def _begin_step(self, layer, positions, hidden_states, residual):
        """Once per step, at the first mono layer: the rank-uniform go / no-go.

        Metadata reasons need no vote (identical on every TP rank). Rank-local state
        (``enabled``) enters only through a step_sync vote: the one taken at the end of
        the last check step, or one taken here while no "on" vote stands. Without
        step_sync (FULL graphs) ``enabled`` is read directly."""
        T = hidden_states.shape[0]
        md, sm = layer_metadata(layer.self_attn.layer_name)
        S = width_for(T, self.sizes)
        if md is None or sm is None or not hasattr(md, "paged_kv_indptr"):
            self.active = False
            return
        reason = step_reason(
            md, T, residual is not None, self.max_query_len, S is not None, TOPK
        )
        self.active = (reason or self._state_reason()) == ""
        if not self.active:
            return
        assert S is not None
        self._ensure_caches(md.block_size)
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
            if md.block_size != 16:
                raise RuntimeError(
                    f"fused_indexer needs a 16-token KV block, got {md.block_size}"
                )
            if not self._index_tables_set:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "mono live: fused-indexer tables not set before graph capture "
                        "(index caches were not bound at install)"
                    )
                if not self._set_index_tables():
                    raise RuntimeError(
                        "mono live: fused indexer: vLLM has not bound the index caches"
                    )
            # vLLM's table may be wider than cdiv(max_model_len, 16): copy the prefix
            w = min(md.block_table.shape[1], self.b_bt.shape[1])
            self.b_bt[:T, :w].copy_(md.block_table[:T, :w])
        # one mailbox epoch per step; launch tag = step * lps + (L - first) + 1
        for op in self.runtime_ops(S):
            op.advance_step()
        self._st = dict(T=T, S=S, md=md)
        self._n_mono += 1

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
            BLOCK_SIZE=md.block_size,
            NUM_TOPK_TOKENS=TOPK,
        )

    def mono_forward(self, layer, positions, hidden_states, residual):
        """One mono layer of an active step -> (zeros_L, x_out[:T]), never aliasing the
        inputs; may mutate the first mono layer's hidden_states / residual (the custom
        op ops/glm5_mono.py wraps exactly this)."""
        L = layer.layer_idx
        if not self.active:
            raise RuntimeError(
                f"mono live: layer {L} called on a step without a mono go decision"
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
                    f"mono live: layer {L} entered the mono path but layer {L - 1} was "
                    "not mono in this step (dispatch state corrupted); refusing to "
                    "continue"
                )
            x = residual
        refresh = self.has_indexer[L] and L not in self.fused_layers
        if refresh:
            if normed is None:
                normed = layer.input_layernorm(x)
            if self.cfg.indexer_mode == "attn":
                layer.self_attn(positions=positions, hidden_states=normed)
            else:
                from vllm.models.deepseek_v32.amd.mono.indexer_only import (
                    refresh_indexer,
                )

                refresh_indexer(
                    layer.self_attn, positions, normed, trim=self.cfg.indexer_trim
                )
        if self.first == L or refresh:
            self._convert_topk(layer)
        if S == T and x.dtype is torch.bfloat16 and x.is_contiguous():
            # graphs: capture sizes == widths, so always this branch (no copy node)
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
        """Eager health check (step_sync, every check_every mono steps): expired polls
        fail-stop unless MONO_LIVE_FAILSTOP is warn / 0; then, and for non-finite
        output, mono is disabled from the next step through the rank-uniform vote.
        Under FULL graphs the dispatch's PollErrorWatch does the check."""
        if self.dev_nonfinite is not None:  # capture-safe, no host sync
            self.dev_nonfinite.add_((~torch.isfinite(out)).any().to(torch.int32))
        n = self._n_mono
        every = self.cfg.check_every
        if not self.cfg.step_sync or every <= 0 or n % every:
            return
        from vllm.models.deepseek_v32.amd.mono.guards import fail_stop, failstop_mode

        mode = failstop_mode()
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
            why = f"mono live: step {n}: expired={exp} finite={fin}"
            if exp and mode == "raise":
                # this step's output and KV rows are already wrong: disabling would
                # keep serving them
                fail_stop(why, self.rank)
            self.enabled = False
            logger.error("%s -> disabling", why)
        # rank-uniform (MIN) state for the next steps; replaces their begin-step vote
        self._voted = self._go(self.enabled)

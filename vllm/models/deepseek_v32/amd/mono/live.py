# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer MonoKernel state and step logic of the model-integrated dispatch
(dispatch.py).

On every *pure decode* step (no prefill tokens, max_query_len == 1, T <= max_rows) the
kernel's output replaces vLLM's decoder layer for the mono layers (3..77). Every other
step (prefill, mixed, profiling / dummy runs, T too large) runs vLLM's forward for all
layers. The decision is taken once per step at the first mono layer, from attention
metadata that is identical on all TP ranks, and every later layer follows it.

Boundary:
  * first mono layer: ``(normed, x) = fused_allreduce_rms_norm(h_partial, residual)``;
    ``x`` is the bf16 pre-norm sum vLLM itself would carry;
  * each mono layer returns ``(zeros_L, x_out)``; the next mono layer checks that its
    input is the previous layer's zeros view and uses ``residual`` directly;
  * vLLM's final norm computes ``fused_allreduce_rms_norm(zeros, x_out)`` =
    ``norm(x_out)`` exactly;
  * zeros_L is a view of a persistent per-layer zero buffer, so a layer's output never
    aliases its input (custom-op contract, ops/glm5_mono.py). Only the last layer's
    buffer is re-zeroed per step: the fused final norm writes its residual into it.

Indexer layers (6, 10, ..., 74) must still refresh vLLM's top-k buffer and index-K
cache. ``indexer_mode="attn"`` runs vLLM's whole attention on the normed input and
discards the output (its MLA KV row is then overwritten by the kernel);
``indexer_mode="indexer_only"`` runs just the indexer (``_refresh_indexer``). The fused
indexer (``fused_indexer``) does all of it inside the kernel launch.

Logical top-k -> kernel CSR slots via vLLM's
``triton_convert_req_index_to_global_index``, at the first mono layer and after every
non-fused indexer layer.

Row padding: T is padded to the smallest kernel width >= T with inactive rows (slot -1,
empty sparse range, position 0, zero hidden).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.models.deepseek_v32.amd.mono.common import (
    HIDDEN,
    TOPK,
    build_width_ops,
    convert_topk,
    layer_metadata,
    rope_tables,
    tp_uniform,
    warm_up_launch,
)

logger = init_logger(__name__)


def _mla_cache(layer) -> torch.Tensor:
    kv = layer.self_attn.kv_cache
    return kv[0] if isinstance(kv, (list, tuple)) else kv


@dataclass
class LiveConfig:
    ckpt: str
    layers: list[int] = field(default_factory=lambda: list(range(3, 78)))
    # kernel widths to build (subset of 1, 2, 4, 5, 6, 8, 10, 12)
    sizes: tuple[int, ...] = (1, 2, 4, 5, 6, 8)
    # MTP verify steps on the kernel; None = on iff vLLM has a speculative_config
    spec_decode: bool | None = None
    poll_limit: int = 20_000_000  # ~4 s per mailbox wait
    indexer_mode: str = "attn"  # "attn" | "indexer_only" (non-fused indexer layers)
    # eager only: rank-uniform state vote (device sync + CPU all-reduce)
    step_sync: bool = True
    check_every: int = 1  # eager health check every N mono steps (0 = never)
    max_model_len: int = 4096
    # initial state; change via MonoLive.set_enabled (refused under FULL graphs)
    enabled: bool = True
    poll_early_out: bool = True  # after one expired poll, skip the step's other waits
    attention_weight: str = "fp8_block128"  # or "bf16"
    indexer_trim: bool = True  # indexer_only: skip work whose results are discarded
    early_cache_checks: bool = True  # validate the MLA cache contract at install
    # in-kernel indexer on indexer layers; None = on when supported (see MonoLive)
    fused_indexer: bool | None = None
    index_q_fp8: bool = True  # FP8 index q, as vLLM's fused_q
    fused_index_rowpar_cache: bool = True
    fused_index_batched_score: bool = True
    cache_hoist: bool = True
    fused_select_radix11: bool = True
    fused_index_proj_spread: bool = True  # widths <= 8 only
    split_keys64: bool = True  # widths <= 8 only
    # captured isfinite count of each step's output (+ fail-stop on it)
    device_nonfinite: bool = False
    failstop_nonfinite: bool = False
    extra: dict = field(default_factory=dict)  # debug-harness keys, unused here


class MonoLive:
    def __init__(self, model, cfg: LiveConfig, vllm_config=None):
        from vllm.distributed import get_tp_group
        from vllm.models.deepseek_v32.amd.mono.guards import (
            _vllm_config,
            full_cudagraphs,
        )

        self.cfg = cfg
        sc = getattr(_vllm_config(vllm_config), "speculative_config", None)
        self.num_spec = (
            int(getattr(sc, "num_speculative_tokens", 0) or 0) if sc is not None else 0
        )
        self.spec_decode = (
            (self.num_spec > 0) if cfg.spec_decode is None else bool(cfg.spec_decode)
        )
        if self.spec_decode and self.num_spec <= 0:
            raise ValueError(
                "spec_decode=True but vLLM runs without a speculative_config"
            )
        # verify steps carry 1 + k query rows per request; plain decode 1
        self.max_query_len = 1 + self.num_spec if self.spec_decode else 1
        if self.spec_decode and cfg.fused_indexer:
            raise ValueError(
                "spec_decode: the fused indexer takes one query row per request; "
                "leave fused_indexer at AUTO (off under spec decode)"
            )
        # FULL graphs: replays run no Python, so set_enabled is refused and a failed
        # health check fail-stops instead of disabling
        self.full_graphs = full_cudagraphs(_vllm_config(vllm_config))
        if self.full_graphs and cfg.step_sync:
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
        layers = model.model.layers
        self.layers = {}
        for L in cfg.layers:
            layer = layers[L]
            assert type(layer).__name__ == "DeepseekV32DecoderLayer", type(layer)
            assert hasattr(layer.mlp, "experts"), f"layer {L} is not a MoE layer"
            self.layers[L] = layer
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
        # AUTO (default): on when every requirement below holds
        if cfg.fused_indexer is None:
            why = (
                "speculative decoding (next_n > 1 rows per request)"
                if self.spec_decode
                else "attention_weight != 'fp8_block128'"
                if cfg.attention_weight != "fp8_block128"
                else f"max_model_len {cfg.max_model_len} > 4096"
                if self.index_max_seq > 4096
                else "the first mono layer is an indexer layer"
                if self.has_indexer[self.first]
                else None
            )
            self.fused_indexer = why is None
            if why is not None:
                logger.info("mono live: fused_indexer auto -> off (%s)", why)
        else:
            self.fused_indexer = bool(cfg.fused_indexer)
        self.fused_layers = (
            frozenset(L for L in self.layers if self.has_indexer[L])
            if self.fused_indexer
            else frozenset()
        )
        if self.fused_indexer:
            if cfg.attention_weight != "fp8_block128":
                raise ValueError(
                    "fused_indexer requires attention_weight='fp8_block128' (BF16 "
                    "attention refuses the in-kernel indexer, kernel.py)"
                )
            if self.first in self.fused_layers:
                raise ValueError(
                    "the first mono layer must not be an indexer layer (it converts "
                    "vLLM's top-k)"
                )
        if self.fused_indexer and self.index_max_seq > 4096:
            raise ValueError(
                f"fused_indexer: max_model_len {cfg.max_model_len} > 4096 "
                "(LDS-resident selection)"
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
        self.disabled_reason = ""
        # step_sync: the last rank-uniform vote of `enabled` (None = none yet)
        self._voted: bool | None = None
        # mono steps since install; never reset, so check steps stay rank-aligned
        self._n_mono = 0
        self.active = False
        self.ops: dict[tuple[int, int], Any] = {}
        self.packed: dict[int, dict] = {}
        self.weights: dict[int, object] = {}
        from vllm.models.deepseek_v32.amd.mono.kernel.config import AttentionWeight

        self._attn_fmt = AttentionWeight(cfg.attention_weight)
        assert self._attn_fmt in (AttentionWeight.BF16, AttentionWeight.FP8_BLOCK128), (
            cfg.attention_weight
        )
        self.fp8_quant_err: dict[int, dict] = {}
        # per-step state
        # layer -> the zeros view it returned this step (identity check)
        self._zret: dict[int, torch.Tensor] = {}
        self._st: dict[str, Any] = {}
        self._flat: dict[int, torch.Tensor] = {}
        # stats (host-side; eager only -- graph replays do not run Python)
        self.stats: dict[str, Any] = dict(
            steps_seen=0,
            steps_mono=0,
            steps_fallback_decode=0,
            steps_no_decode=0,
            decode_tokens=0,
            mono_tokens=0,
            mono_padded_rows=0,
            fallback_reasons={},
            mono_steps_by_S={},
            expired=[],
            nonfinite_steps=0,
        )
        # real (slot >= 0) tokens through the kernel; also counts under graph replay
        self.dev_mono_tokens = torch.zeros(1, dtype=torch.int64, device=self.dev)
        self.dev_mono_steps = torch.zeros(1, dtype=torch.int64, device=self.dev)
        self.dev_nonfinite = (
            torch.zeros(1, dtype=torch.int32, device=self.dev)
            if cfg.device_nonfinite
            else None
        )
        # one zero buffer per mono layer (module doc, boundary); separate tensors so
        # each [:T] view has storage offset 0, like the op's fake (empty_like) output
        self._zbuf = [
            torch.zeros(self.max_rows, HIDDEN, dtype=torch.bfloat16, device=self.dev)
            for _ in self.order
        ]
        # T -> the [:T] view of every layer's zero buffer (built once per T)
        self._zviews: dict[int, list] = {}
        # per-layer epoch kwargs, built once
        self._epoch_kw = {
            L: dict(layer=L - self.first, advance=False) for L in self.order
        }
        t0 = time.time()
        self._load_weights()
        self._rope_from(layers[self.first])
        self._alloc_buffers()
        for S in self.sizes:
            self._build_ops(S)
        if self.fused_layers:
            # graph mode: must happen before vLLM captures (H2D copy)
            self._set_index_tables()
        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        self.mem = dict(
            allocated_gib=torch.accelerator.memory_allocated() / 2**30,
            reserved_gib=torch.accelerator.memory_reserved() / 2**30,
            device_used_gib=(total - free) / 2**30,
            device_total_gib=total / 2**30,
            weights_gib=self._weight_bytes / 2**30,
            install_s=time.time() - t0,
        )
        logger.info("mono live: rank %d ready: %s", self.rank, self.mem)

    # ------------------------------------------------------------------ setup
    def _load_weights(self):
        from vllm.models.deepseek_v32.amd.mono.ckpt_weights import load_glm5_layer
        from vllm.models.deepseek_v32.amd.mono.kernel.config import (
            AttentionWeight,
            glm5_tp_config,
        )
        from vllm.models.deepseek_v32.amd.mono.kernel.glm.op import prepare_glm5_weights
        from vllm.models.deepseek_v32.amd.mono.kernel.weights import LayerWeights

        t0 = time.time()
        nbytes = 0
        worst = 0.0
        for L in self.order:
            t = load_glm5_layer(self.cfg.ckpt, L, self.rank, self.npes, device=self.dev)
            if self._attn_fmt is AttentionWeight.FP8_BLOCK128:
                from vllm.models.deepseek_v32.amd.mono.fp8_attention import (
                    quantize_attention_fp8,
                )

                errs = quantize_attention_fp8(t)
                self.fp8_quant_err[L] = errs
                worst = max(worst, max(errs.values()))
            if L in self.fused_layers:
                from vllm.models.deepseek_v32.amd.mono import index_weights as IW

                t = dict(t)
                t.update(IW.index_weights_from_ckpt(self.cfg.ckpt, L, self.dev))
            W = LayerWeights(
                8,
                t,
                config=glm5_tp_config(self.npes),
                rank=self.rank,
                npes=self.npes,
            )
            packed = dict(prepare_glm5_weights(W, self._attn_fmt))
            # free raw copies shadowed by a packed tensor (forward uses
            # dict(W.t, **packed)); w_uv stays: FP8 uv_scale_rows derive from its shape
            for k in packed:
                if k == "w_uv" and self._attn_fmt is AttentionWeight.FP8_BLOCK128:
                    continue
                if k in W.t and W.t[k].data_ptr() != packed[k].data_ptr():
                    W.t[k] = torch.empty(0, dtype=W.t[k].dtype, device=self.dev)
            self.weights[L], self.packed[L] = W, packed
            seen = set()
            for d in (W.t, packed):
                for v in d.values():
                    if v.data_ptr() not in seen:
                        seen.add(v.data_ptr())
                        nbytes += v.numel() * v.element_size()
        self._weight_bytes = nbytes
        if self._attn_fmt is AttentionWeight.FP8_BLOCK128:
            logger.info(
                "mono live: FP8 block-128 attention, worst per-matrix rel-L2 quant "
                "error %.3e (rank %d)",
                worst,
                self.rank,
            )
        torch.accelerator.synchronize()
        logger.info(
            "mono live: loaded %d layers in %.1fs (rank %d, %.2f GiB kernel weights, "
            "%.2f GiB alloc)",
            len(self.order),
            time.time() - t0,
            self.rank,
            nbytes / 2**30,
            torch.accelerator.memory_allocated() / 2**30,
        )

    def _rope_from(self, layer):
        self.cos, self.sin, _ = rope_tables(layer, self.cfg.max_model_len)

    def _alloc_buffers(self):
        R = self.max_rows
        self.b_x = torch.zeros(R, HIDDEN, dtype=torch.bfloat16, device=self.dev)
        self.b_pos = torch.zeros(R, dtype=torch.int64, device=self.dev)
        self.b_slot = torch.full((R,), -1, dtype=torch.int64, device=self.dev)
        self.b_indptr = torch.zeros(R + 1, dtype=torch.int32, device=self.dev)
        self.b_indices = torch.zeros(R * TOPK, dtype=torch.int32, device=self.dev)
        self.b_curpos = torch.zeros(1, dtype=torch.int32, device=self.dev)
        # per-width views of the padded step inputs
        self._sviews = {
            S: (self.b_pos[:S], self.b_slot[:S], self.b_indptr[: S + 1])
            for S in self.sizes
        }
        if self.fused_layers:
            # fused-indexer block table at a stable address (param tables, graphs)
            self.b_bt = torch.zeros(
                R,
                -(-int(self.cfg.max_model_len) // 16) + 1,
                dtype=torch.int32,
                device=self.dev,
            )

    def runtime_ops(self, S: int) -> list:
        """The ops owning a runtime (scratch / step counter / peers) for width S: one,
        or two with the fused indexer."""
        return [
            op
            for (L, s_), op in self.ops.items()
            if s_ == S and getattr(op, "_owns_runtime", False)
        ]

    def _go(self, ok: bool) -> bool:
        return tp_uniform(ok, self.cpu_group)

    def set_enabled(self, on: bool) -> bool:
        """Enable / disable the dispatch; returns False if refused.

        Eager: applied through the rank-uniform vote. FULL cudagraphs: refused, since
        captured graphs keep their decision; never raises, because a raising worker RPC
        kills the engine."""
        on = bool(on)
        if self.full_graphs and on != self.enabled:
            logger.error(
                "mono live: refusing to %s the MonoKernel under FULL cudagraphs: "
                "captured graphs keep their dispatch decision; restart (or re-capture) "
                "to change it",
                "enable" if on else "disable",
            )
            return False
        self.enabled = on
        self.disabled_reason = "" if on else "disabled by request"
        return True

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
            # the kernel writes / reads the SHUFFLE layout at the MLA slot (rocm.py:
            # aiter shuffles above block 1)
            if not getattr(kc, "uses_shuffled_layout", False):
                raise RuntimeError(
                    f"fused_indexer: layer {L} index cache is not in the shuffled "
                    "layout"
                )
            out[L] = ic.view(torch.uint8)
        return out

    def _ensure_caches(self, block_size: int):
        """(Re)bind the caches when vLLM's first-mono-layer MLA cache moved."""
        bound = self._flat.get(self.first)
        kv = _mla_cache(self.layers[self.first])
        # the allocator can reuse a freed cache's address: compare the size too
        if bound is None or (bound.data_ptr(), bound.numel()) != (
            kv.data_ptr(),
            kv.numel(),
        ):
            self._bind_caches(block_size)

    def _bind_caches(self, block_size: int):
        """Resolve every layer's MLA cache; drop the fused-indexer tables unless they
        point at the current index caches. vLLM binds KV caches more than once: under
        FULL graphs, its CUDA-graph memory profiling binds minimal caches (one block per
        capture row), runs mono steps on them, frees them and then allocates the real
        caches. A binding kept from that run addresses the real slots / block ids in an
        8-block buffer (memory access fault, or garbage output)."""
        flat = {}
        for L, lay in self.layers.items():
            kv = _mla_cache(lay)
            assert (
                kv.dtype is torch.bfloat16
                and kv.is_contiguous()
                and kv.shape[-1] == 576
                and block_size == kv.shape[-2]
            ), (L, kv.shape, kv.dtype, block_size)
            flat[L] = kv.view(-1, 576)
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
            if cur != getattr(self, "_index_ptrs", None):
                # re-pointed by the next eager step (_set_index_tables: an H2D copy)
                self._index_tables_set = False

    def _set_index_tables(self) -> bool:
        """Point every fused op at its layer's index cache + b_bt (a host->device copy:
        never inside a capture). Done at install when the caches are already bound
        (graph mode installs right before capture), else at the first eager mono
        step."""
        caches = self._index_caches()
        if caches is None:
            return False
        for L, ic in caches.items():
            for s_ in self.sizes:
                self.ops[(L, s_)].set_index_tables(ic, self.b_bt)
        self._index_ptrs = {L: (ic.data_ptr(), ic.numel()) for L, ic in caches.items()}
        self._index_tables_set = True
        return True

    def _build_ops(self, S: int):
        t0 = time.time()
        self.ops.update(
            build_width_ops(
                self.order,
                self.weights,
                S,
                rank=self.rank,
                npes=self.npes,
                group=self.cpu_group,
                poll_limit=self.cfg.poll_limit,
                attention_weight=self._attn_fmt,
                prepared_for=self.packed.__getitem__,
                launches_per_step=self.launches_per_step,
                poll_early_out=self.cfg.poll_early_out,
                stage_opts=dict(
                    cache_hoist=self.cfg.cache_hoist,
                    split_keys64=self.cfg.split_keys64,
                ),
                extra_kwargs_for=lambda L: (
                    dict(
                        with_indexer=True,
                        index_max_seq=self.index_max_seq,
                        index_q_fp8=self.cfg.index_q_fp8,
                        select_radix11=self.cfg.fused_select_radix11,
                        # the idle-CTA projections fit widths <= 8 only (MTP k=1 builds
                        # S=10)
                        index_proj_spread=self.cfg.fused_index_proj_spread and S <= 8,
                    )
                    if L in self.fused_layers
                    else None
                ),
            )
        )
        # One warm-up launch per kernel variant (JIT, all rows inactive) between two
        # barriers, so per-rank compile skew never races the bounded polls.
        warm = [self.ops[(self.first, S)]]
        if self.fused_layers:
            fop = self.ops[(min(self.fused_layers), S)]
            # inactive rows never touch it: a scratch index cache until the first step
            fop_scratch = torch.zeros(2, 16, 132, dtype=torch.uint8, device=self.dev)
            for L in self.fused_layers:
                self.ops[(L, S)].set_index_tables(fop_scratch, self.b_bt)
            self._warm_index_cache = fop_scratch
            warm.append(fop)
        for op in warm:
            self._go(True)
            warm_up_launch(op, S, self.b_curpos, self.cos, self.sin, self.dev)
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
    def _pick_S(self, T: int) -> int | None:
        for s in self.sizes:
            if s >= T:
                return s
        return None

    def _metadata(self, layer):
        return layer_metadata(layer, list_policy="first")

    def _state_reason(self) -> str:
        """'' (go) or why not, for a step the metadata allows (rank-uniform)."""
        if self.cfg.step_sync:
            if self._voted is not True or self.cfg.check_every <= 0:
                self._voted = self._go(self.enabled)
            if not self._voted:
                return "disabled" if not self.enabled else "peer_no_go"
            return ""
        return "" if self.enabled else "disabled"

    def _begin_step(self, layer, positions, hidden_states, residual):
        """Once per step, at the first mono layer: rank-uniform go / no-go.

        Metadata reasons need no vote: the metadata is identical on every TP rank.
        Rank-local state (``enabled``) enters only through a vote (step_sync): the one
        taken at the end of the last check step, or one taken here while no "on" vote
        stands. Without step_sync (FULL graphs) ``enabled`` is read directly."""
        st = self.stats
        T = hidden_states.shape[0]
        md, sm = self._metadata(layer)
        reason = ""
        if md is None or sm is None or not hasattr(md, "paged_kv_indptr"):
            reason = "no_metadata"
        else:
            st["steps_seen"] += 1
            n_dec = int(getattr(md, "num_decode_tokens", 0))
            st["decode_tokens"] += n_dec
            from vllm.models.deepseek_v32.amd.mono.spec import step_reason

            reason = step_reason(
                md,
                T,
                residual is not None,
                self.max_query_len,
                self._pick_S(T) is not None,
                TOPK,
            )
            if n_dec == 0:
                st["steps_no_decode"] += 1
        if reason == "":
            reason = self._state_reason()
        ok = reason == ""
        self.active = ok
        if not ok:
            if (
                md is not None
                and reason != "no_metadata"
                and int(getattr(md, "num_decode_tokens", 0)) > 0
            ):
                st["steps_fallback_decode"] += 1
                st["fallback_reasons"][reason] = (
                    st["fallback_reasons"].get(reason, 0) + 1
                )
            return
        S = self._pick_S(T)
        assert S is not None
        self._ensure_caches(md.block_size)
        self._zret = {}
        # the only buffer written (by the fused final norm)
        self._zbuf[-1][:T].zero_()
        # Padded per-step inputs (persistent buffers; graph-safe).
        self.b_pos[:T].copy_(positions[:T])
        self.b_pos[T:S].zero_()
        self.b_slot[:T].copy_(sm[:T])
        self.b_slot[T:S].fill_(-1)
        self.b_indptr[: T + 1].copy_(md.paged_kv_indptr[: T + 1])
        if S > T:
            self.b_indptr[T + 1 : S + 1].copy_(
                md.paged_kv_indptr[T : T + 1].expand(S - T)
            )
        self.dev_mono_tokens += (sm[:T] >= 0).sum()
        self.dev_mono_steps += 1
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
        st["steps_mono"] += 1
        st["mono_tokens"] += T
        st["mono_padded_rows"] += S - T
        st["mono_steps_by_S"][S] = st["mono_steps_by_S"].get(S, 0) + 1

    def _convert_topk(self, layer):
        st = self._st
        T, md = st["T"], st["md"]
        convert_topk(
            md,
            layer.self_attn.topk_indices_buffer[:T],
            self.b_indptr[: T + 1],
            self.b_indices,
        )

    def _refresh_indexer(self, layer, positions, normed):
        attn = layer.self_attn
        if self.cfg.indexer_mode == "attn":
            attn(positions=positions, hidden_states=normed)  # output discarded
        else:
            from vllm.models.deepseek_v32.amd.mono.indexer_only import refresh_indexer

            refresh_indexer(attn, positions, normed, trim=self.cfg.indexer_trim)

    def mono_forward(self, layer, positions, hidden_states, residual):
        """One mono layer of an active step -> (zeros_L, x_out[:T]), both fresh w.r.t.
        the inputs (no aliasing). May mutate the first mono layer's hidden_states /
        residual (fused_allreduce_rms_norm writes in place); later layers read only. The
        custom op ops/glm5_mono.py wraps exactly this (dispatch path)."""
        L = layer.layer_idx
        if not self.active:
            raise RuntimeError(
                f"mono live: layer {L} called on a step without a mono go decision"
            )
        st = self._st
        T, S = st["T"], st["S"]
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
        if self.has_indexer[L] and L not in self.fused_layers:
            if normed is None:
                normed = layer.input_layernorm(x)
            self._refresh_indexer(layer, positions, normed)
        if self.first == L or (self.has_indexer[L] and L not in self.fused_layers):
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
        z = zv[L - self.first]
        self._zret[L] = z
        return z, out

    def _end_step(self, out):
        """Health check (eager with step_sync, every check_every mono steps): expired
        polls or non-finite output, then the rank-uniform state vote.

        Expired polls fail-stop unless MONO_LIVE_FAILSTOP is warn / 0; then (and for
        non-finite output) mono is disabled from the next step. Under FULL graphs any
        failure fail-stops; there the dispatch path's PollErrorWatch does the check."""
        if self.dev_nonfinite is not None:  # capture-safe, no host sync
            self.dev_nonfinite.add_((~torch.isfinite(out)).any().to(torch.int32))
        n = self._n_mono
        if (
            not self.cfg.step_sync
            or self.cfg.check_every <= 0
            or n % self.cfg.check_every
        ):
            return
        from vllm.models.deepseek_v32.amd.mono.guards import fail_stop, failstop_mode

        mode = failstop_mode()
        torch.accelerator.synchronize()
        # The poll-error words are sticky: only the fail-stop watch clears them (and
        # only in warn mode), so leave them set unless no watch runs.
        exp = tuple(
            e
            for op in self.runtime_ops(self._st["S"])
            for e in op.poll_error(clear=mode == "off")
        )
        fin = bool(torch.isfinite(out).all())
        if exp or not fin:
            self.stats["expired"].append(dict(step=n, expired=list(exp), finite=fin))
            if not fin:
                self.stats["nonfinite_steps"] += 1
            self.disabled_reason = f"step {n}: expired={exp} finite={fin}"
            if self.full_graphs:
                fail_stop(
                    f"mono live: {self.disabled_reason} under FULL cudagraphs "
                    "(captured graphs cannot fall back)",
                    getattr(self, "rank", 0),
                )
            if exp and mode == "raise":
                # this step's output and KV rows are already wrong: disabling would
                # keep serving them
                fail_stop(
                    f"mono live: {self.disabled_reason}", getattr(self, "rank", 0)
                )
            self.enabled = False
            logger.error("mono live: disabling after %s", self.disabled_reason)
        # rank-uniform (MIN) state for the next steps; replaces their begin-step vote
        self._voted = self._go(self.enabled)

    def get_stats(self) -> dict:
        s = dict(self.stats)
        s["dev_mono_tokens"] = int(self.dev_mono_tokens.item())
        s["dev_mono_steps"] = int(self.dev_mono_steps.item())
        s["dev_nonfinite_steps"] = (
            None if self.dev_nonfinite is None else int(self.dev_nonfinite.item())
        )
        s["coverage"] = s["mono_tokens"] / max(1, s["decode_tokens"])
        s["enabled"] = self.enabled
        s["disabled_reason"] = self.disabled_reason
        s["mem"] = self.mem
        free, total = torch.accelerator.get_memory_info()
        s["mem_now"] = dict(
            allocated_gib=torch.accelerator.memory_allocated() / 2**30,
            max_allocated_gib=torch.accelerator.max_memory_allocated() / 2**30,
            device_used_gib=(total - free) / 2**30,
        )
        return s

    def reset_stats(self):
        for k, v in list(self.stats.items()):
            self.stats[k] = (
                {} if isinstance(v, dict) else ([] if isinstance(v, list) else 0)
            )
        self.dev_mono_tokens.zero_()
        self.dev_mono_steps.zero_()

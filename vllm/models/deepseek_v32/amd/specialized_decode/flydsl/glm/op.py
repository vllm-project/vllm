# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as
# vendored by ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/glm/op.py).
# The original source code was licensed under the Apache License 2.0 and included the
# following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths
#   rewritten to this package; host wrapper options for the kernel changes in
#   glm/kernel.py (poll limit / early-out, step epochs, in-kernel indexer tables and
#   options, build-time stage options), the cross-rank poll-expiry flag and per-launch
#   argument caching; torch.accelerator in place of torch.cuda device calls; BF16
#   attention weights read in place, a checkpoint's FP8 attention / index-Q weights
#   tile-packed with their per-row scales; the unused format / DCP / timeline options
#   and the debug / intermediates / close accessors removed.

"""Host wrapper for the GLM-5 sparse-MLA decode MonoKernel (optionally with the fused
indexer)."""

from __future__ import annotations

import torch

from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.config import (
    FP8_ATTENTION,
    GLM5_KERNEL_SAMPLES,
    HIDDEN,
    KV_LORA,
    N_EXPERTS,
    PE_DIM,
    glm5_tp_config,
)
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.kernel import (
    build_glm5_monokernel,
)
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.glm.layout import (
    INDEX_DIM,
    INDEX_HEADS,
    INDEX_Q_ROWS,
    POLL_STAGES,
    layout,
)
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.packing import pack_fp8
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.runtime import (
    SymmetricPeerBuffer,
)
from vllm.models.deepseek_v32.amd.specialized_decode.flydsl.weights import LayerWeights

__all__ = ["Glm5MonoKernel"]

# fused-indexer parameter table (``index_params``) slots 0..6; 7 unused, 8..11 set by
# set_index_tables (index cache, block table, its row stride, cache block stride)
_INDEX_WEIGHTS = (
    "w_index_k",
    None,  # (index-K is BF16: no scale)
    "w_index_w",
    "w_index_q",
    "s_index_q",
    "g_index_k",
    "b_index_k",
)


def fp8_attention_of(t: dict) -> tuple[str, ...]:
    """The attention matrices given as FP8 (those with an ``s_<name>`` row scale)."""
    return tuple(name for name in FP8_ATTENTION if f"s_{name}" in t)


def _fp8_rows(t: dict, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``pack_fp8`` tiles of FP8 [rows, K] ``w_<name>`` + its F32 [rows] scales."""
    w, s = t[f"w_{name}"], t[f"s_{name}"]
    _expect(w, f"w_{name}", tuple(w.shape), torch.float8_e4m3fn)
    _expect(s, f"s_{name}", (w.shape[0],), torch.float32)
    return pack_fp8(w), s


def _tsig(t):
    """What forward()'s checks depend on for one tensor argument.

    Identity, dtype, shape and strides (contiguity).
    """
    return None if t is None else (t.data_ptr(), t.dtype, t.shape, t.stride())


def _expect(
    x: torch.Tensor, name: str, shape: tuple[int, ...], dtype=None
) -> torch.Tensor:
    """The kernel's buffer loads are unbounded: a wrong layout reads garbage instead of
    faulting, so every tensor it reads in place must have exactly the expected shape /
    element size and be contiguous.
    """
    ok = (
        tuple(x.shape) == shape
        and x.is_contiguous()
        and (x.dtype is dtype if dtype else x.element_size() == 1)
    )
    if not ok:
        raise ValueError(
            f"{name}: {x.dtype} {tuple(x.shape)} strides {x.stride()}, "
            "expected contiguous "
            f"{dtype or '1-byte'} {shape}"
        )
    return x


def prepare_glm5_weights(W: LayerWeights) -> dict[str, torch.Tensor]:
    """The tensors the kernel reads, once for every graph bucket using them: vLLM's
    loaded tensors in place (BF16 attention and router row-major, routed experts in
    aiter's (16, 16) shuffle + e8m0_shuffle scales, the shared expert's own tensors);
    a checkpoint's FP8 attention matrices (``s_<name>`` given) packed into 16 x 64
    tiles (a layout copy: the values and scales are the checkpoint's).
    """
    t, cfg = W.t, W.config
    bf, hidden, inter, E = torch.bfloat16, HIDDEN, cfg.inter, N_EXPERTS
    names = (
        "w_qkv_a",
        "w_q_b",
        "w_uk",
        "w_uv",
        "w_o",
        "w_r",
        "w_ug",
        "s_ug",
        "w_dn",
        "s_dn",
        "w_ug_sh",
        "s_ug_sh",
        "w_dn_sh",
        "s_dn_sh",
    )
    missing = [name for name in names if name not in t]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    fp8 = fp8_attention_of(t)
    packed = {}
    for name in names[:5]:
        if name[2:] in fp8:
            packed[name], packed[f"s_{name[2:]}"] = _fp8_rows(t, name[2:])
        else:
            packed[name] = _expect(t[name], name, tuple(t[name].shape), bf)
    if "s_index_q" in t:  # the fused indexer's FP8 wq_b
        packed["w_index_q"], packed["s_index_q"] = _fp8_rows(t, "index_q")
    packed["w_r"] = _expect(t["w_r"], "w_r", (E, hidden), bf)
    for name, shape in (
        ("w_ug", (E, 2 * inter, hidden // 2)),
        ("s_ug", (E * 2 * inter, hidden // 32)),
        ("w_dn", (E, hidden, inter // 2)),
        ("s_dn", (E * hidden, inter // 32)),
    ):
        packed[name] = _expect(t[name], name, shape).view(torch.uint8).view(-1)
    sh = {"w_ug_sh": (2 * inter, hidden // 2), "w_dn_sh": (hidden, inter // 2)}
    if W.shared_native:  # AITER Triton FP4 linear: scales transposed
        sh.update(s_ug_sh=(hidden // 32, 2 * inter), s_dn_sh=(inter // 32, hidden))
    else:  # AITER ASM FP4 linear: the experts' shuffles
        sh.update(s_ug_sh=(2 * inter, hidden // 32), s_dn_sh=(hidden, inter // 32))
    for name, shape in sh.items():
        packed[name] = _expect(t[name], name, shape).view(torch.uint8).view(-1)
    return packed


class Glm5MonoKernel:
    """One TP rank of GLM-5's sparse-MLA decode MonoKernel.

    A single persistent launch covers the attention projections, the KV cache
    update, sparse MLA over the given top-k slots, routing, all expert compute, and
    both TP reductions. ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer uses PyTorch's caching allocator and CUDA/ROCm IPC
    storage sharing. Scratch and symmetric buffers may be shared by all layers
    because every launch uses a fresh ``tag``.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        prepared_weights: dict[str, torch.Tensor],
        rank: int = 0,
        npes: int = 1,
        group=None,
        topk: int = 2048,
        launches_per_step: int = 1,
        runtime: Glm5MonoKernel | None = None,
        poll_limit: int | None = None,
        poll_early_out: bool = False,
        cache_hoist: bool = False,
        split_keys64: bool = False,
        with_indexer: bool = False,
        index_max_seq: int = 4096,
        index_proj_spread: bool = False,
    ):
        expected_config = glm5_tp_config(npes)
        if W.config != expected_config:
            raise ValueError(
                f"Glm5MonoKernel requires {expected_config}, got {W.config}"
            )
        if samples not in GLM5_KERNEL_SAMPLES:
            raise ValueError(
                f"samples must be one of {GLM5_KERNEL_SAMPLES}, got {samples}"
            )
        if W.heads != expected_config.local_heads:
            raise ValueError(
                f"expected {expected_config.local_heads} local heads, got {W.heads}"
            )
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        if topk <= 0 or topk % 64:
            raise ValueError(f"topk must be a positive multiple of 64, got {topk}")
        if not 1 <= launches_per_step <= 128:
            raise ValueError(
                f"launches_per_step must be in [1, 128], got {launches_per_step}"
            )
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.launches_per_step = launches_per_step
        self._validated_sig = (
            None  # forward(): input signature of the last validated launch
        )
        self._wptrs = (
            None  # forward(): the fixed weight pointers (built on the first launch)
        )
        self.with_indexer = with_indexer
        self.index_max_seq = index_max_seq
        # (index cache, block table) refs set by set_index_tables
        self._index_tables: tuple | None = None
        self.packed = dict(prepared_weights)  # from prepare_glm5_weights(W)
        t = W.t
        self.fp8_attention = fp8_attention_of(t)
        if with_indexer:
            if set(self.fp8_attention) != set(FP8_ATTENTION):
                raise ValueError(
                    "with_indexer=True needs FP8 attention (all of "
                    f"{FP8_ATTENTION}), got {self.fp8_attention}"
                )
            missing = [n for n in _INDEX_WEIGHTS if n is not None and n not in t]
            if missing:
                raise ValueError(f"with_indexer=True requires: {', '.join(missing)}")
            for name, shape in (
                ("w_index_k", (INDEX_DIM, HIDDEN)),
                ("w_index_w", (INDEX_HEADS, HIDDEN)),
            ):
                _expect(t[name], name, shape, torch.bfloat16)
            for name in ("g_index_k", "b_index_k"):
                _expect(t[name], name, (INDEX_DIM,), torch.float32)
            if tuple(t["w_index_q"].shape) != (INDEX_Q_ROWS, W.config.q_lora):
                raise ValueError(f"w_index_q {tuple(t['w_index_q'].shape)}")
        self.scr_layout, self.sym_layout = layout(
            samples,
            W.heads,
            npes,
            topk,
            with_indexer,
            index_max_seq,
            inter=W.config.inter,
            split_keys=64 if split_keys64 else None,
        )
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        if with_indexer:
            it = dict(t, **self.packed)
            self.index_params = torch.tensor(
                [0 if n is None else it[n].data_ptr() for n in _INDEX_WEIGHTS]
                + [0] * 5,
                dtype=torch.int64,
                device=dev,
            )
        else:
            self.index_params = None
        self._owns_runtime = runtime is None
        if runtime is None:
            self.scratch = torch.zeros(
                self.scr_layout["_bytes"], dtype=torch.uint8, device=dev
            )
            self.peer_buffer = SymmetricPeerBuffer(
                self.sym_layout["_bytes"], rank=rank, npes=npes, group=group
            )
            self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        else:
            if (
                runtime.scr_layout != self.scr_layout
                or runtime.sym_layout != self.sym_layout
            ):
                raise ValueError("shared GLM runtime geometry mismatch")
            self.scratch = runtime.scratch
            self.peer_buffer = runtime.peer_buffer
            self.step = runtime.step
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        # per source rank: nonzero on rank 0 once that rank had an expired wait
        ox = self.sym_layout["poll_xrank"]
        self.poll_xrank = self.sym_storage[ox : ox + 4 * npes].view(torch.int32)
        if (
            self._owns_runtime
        ):  # where this rank's expired waits flag rank 0 (kernel slow path)
            oa = self.scr_layout["poll_xrank_addr"]
            self.scratch[oa : oa + 8].view(torch.int64).fill_(
                self.peer_buffer.addresses[0] + ox + 4 * rank
            )
        self.peers = self.peer_buffer.addresses
        self.launch = build_glm5_monokernel(
            samples,
            W.heads,
            npes,
            topk,
            launches_per_step=launches_per_step,
            with_indexer=with_indexer,
            index_max_seq=index_max_seq,
            fp8_attention=self.fp8_attention,
            inter=W.config.inter,
            poll_limit=poll_limit,
            poll_early_out=poll_early_out,
            cache_hoist=cache_hoist,
            split_keys64=split_keys64,
            index_proj_spread=index_proj_spread,
            shared_native=W.shared_native,
        )
        self.poll_limit = poll_limit

    def poll_error(self, clear: bool = True) -> tuple[str, ...]:
        """Stages whose bounded waits expired since the last clear (synchronizes)."""
        off = self.scr_layout["poll_err"]
        words = self.scratch[off : off + 4 * len(POLL_STAGES)].view(torch.int32)
        expired = tuple(name for name, word in zip(POLL_STAGES, words.tolist()) if word)
        if clear and expired:
            words.zero_()
        return expired

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        pe_cache,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
        positions=None,
        slot_mapping=None,
        sparse_kv_indptr=None,
    ):
        """One layer.  Mailbox epochs are ``step * launches_per_step + layer + 1``:
        layers sharing this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph.
        """
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(
                f"layer must be in [0, {self.launches_per_step}), got {layer}"
            )
        total_samples = h.shape[0]
        # Validate on the first launch and whenever an input's identity / layout
        # changes; a steady decode loop passes the same persistent buffers every step,
        # so it pays only for this signature.
        sig = (
            total_samples,
            advance,
            self._index_tables is not None,
            indices.dtype,
            _tsig(kv_cache),
            pe_cache.data_ptr(),
            _tsig(positions),
            _tsig(slot_mapping),
            _tsig(sparse_kv_indptr),
        )
        if sig != self._validated_sig:
            self._validate(
                h,
                kv_cache,
                pe_cache,
                indices,
                advance,
                positions,
                slot_mapping,
                sparse_kv_indptr,
            )
            self._validated_sig = sig
        chunks = total_samples // self.S
        wp = self._wptrs
        if wp is None:
            wp = self._wptrs = self._weight_ptrs()
        if x_out is None:
            x_out = torch.empty(
                total_samples, HIDDEN, dtype=torch.bfloat16, device=h.device
            )
        elif x_out.shape != (total_samples, HIDDEN):
            raise ValueError(
                f"x_out must have shape {(total_samples, HIDDEN)}, "
                f"got {tuple(x_out.shape)}"
            )

        def p(x):
            return x.data_ptr()

        for chunk in range(chunks):
            row = chunk * self.S
            self.launch(
                p(h) + row * HIDDEN * h.element_size(),
                p(x_out) + row * HIDDEN * x_out.element_size(),
                p(cur_pos),
                p(positions) + row * 8,
                p(slot_mapping) + row * 8,
                p(sparse_kv_indptr) + row * 4,
                p(kv_cache),
                p(pe_cache),
                p(indices),
                p(cos),
                p(sin),
                *wp,
                p(self.scratch),
                self.sym,
                p(self.peers),
                p(self.index_params) if self.with_indexer else 0,
                p(self.step),
                self.rank,
                layer,
                stream=torch.cuda.current_stream(),
            )
            if chunk + 1 < chunks or advance:
                self.advance_step()
        return x_out

    def _validate(
        self,
        h,
        kv_cache,
        pe_cache,
        indices,
        advance,
        positions,
        slot_mapping,
        sparse_kv_indptr,
    ):
        """forward()'s argument checks (raise ValueError).

        Run when forward's input signature changes.
        """
        total_samples = h.shape[0]
        if total_samples % self.S:
            raise ValueError(
                f"input rows {total_samples} must be divisible by kernel chunk {self.S}"
            )
        chunks = total_samples // self.S
        if self.with_indexer and chunks != 1:
            raise ValueError("fused indexer does not support chunked launches")
        if not advance and chunks != 1:
            raise ValueError("chunked launches must advance mailbox epochs")
        if self.with_indexer and self._index_tables is None:
            raise ValueError(
                "with_indexer: call set_index_tables(index_cache, block_table) "
                "before forward"
            )
        if self.with_indexer and indices.dtype is not torch.int32:
            # the CSR the select stage writes (min(ctx, topk) entries per row at
            # sparse_kv_indptr offsets; the caller sizes it -- warm-up launches with
            # inactive rows pass a 1-element buffer)
            raise ValueError("with_indexer: indices must be the int32 CSR buffer")
        cache_width = KV_LORA + PE_DIM
        if kv_cache.dtype is not torch.bfloat16 or not kv_cache.is_contiguous():
            raise ValueError("the MLA KV cache must be contiguous bf16")
        if kv_cache.shape[-1] != cache_width:
            raise ValueError(
                f"MLA KV cache last dimension must be {cache_width}, "
                f"got {tuple(kv_cache.shape)}"
            )
        if kv_cache.data_ptr() != pe_cache.data_ptr():
            raise ValueError(
                "kv_cache and pe_cache must be the same fused [slots, 576] tensor"
            )
        for name, value, dtype, size in (
            ("positions", positions, torch.int64, total_samples),
            ("slot_mapping", slot_mapping, torch.int64, total_samples),
            ("sparse_kv_indptr", sparse_kv_indptr, torch.int32, total_samples + 1),
        ):
            if (
                value is None
                or value.dtype is not dtype
                or value.numel() < size
                or not value.is_contiguous()
            ):
                got = None if value is None else (tuple(value.shape), value.dtype)
                raise ValueError(
                    f"{name} must be contiguous {dtype} with at least {size} values, "
                    f"got {got}"
                )

    def _weight_ptrs(self) -> tuple[int, ...]:
        """The 24 fixed weight / scale pointers of the launch ABI, in order (built once,
        on the first launch; the weight tensors are immutable after construction). The
        BF16 attention matrices have no scales: their scale slots are 0.
        """
        t = dict(self.W.t, **self.packed)

        def p(x):
            return x.data_ptr()

        return (
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]) if "s_qkv_a" in t else 0,
            p(t["w_q_b"]),
            p(t["s_q_b"]) if "s_q_b" in t else 0,
            p(t["w_uk"]),
            p(t["s_uk"]) if "s_uk" in t else 0,
            p(t["w_uv"]),
            p(t["s_uv"]) if "s_uv" in t else 0,
            p(t["w_o"]),
            p(t["s_o"]) if "s_o" in t else 0,
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
            p(t["w_ug_sh"]),
            p(t["s_ug_sh"]),
            p(t["w_dn_sh"]),
            p(t["s_dn_sh"]),
        )

    def set_index_tables(
        self, index_cache: torch.Tensor, block_table: torch.Tensor
    ) -> None:
        """Point the indexer parameter table at vLLM's uint8 [blocks, 16, 132] index
        cache and a persistent int32 [rows, W] decode block table (stable addresses:
        call once, before any capture).
        """
        if not self.with_indexer:
            raise ValueError("set_index_tables needs with_indexer=True")
        if (
            index_cache.dtype is not torch.uint8
            or index_cache.dim() != 3
            or tuple(index_cache.shape[1:]) != (16, INDEX_DIM + 4)
        ):
            raise ValueError(
                f"index cache must be uint8 [blocks, 16, {INDEX_DIM + 4}], got "
                f"{tuple(index_cache.shape)} {index_cache.dtype}"
            )
        if (
            index_cache.stride(2) != 1
            or index_cache.stride(1) != INDEX_DIM + 4
            or index_cache.stride(0) % 4
        ):
            # the kernel addresses one block as 2112 contiguous bytes (dword stores) at
            # a dword-aligned block stride
            raise ValueError(
                "index cache blocks must be contiguous with a dword block stride, "
                f"got strides {index_cache.stride()}"
            )
        if (index_cache.shape[0] - 1) * index_cache.stride(0) + 16 * (
            INDEX_DIM + 4
        ) >= 1 << 31:
            raise ValueError(
                "index cache spans >= 2 GiB: the kernel forms Int32 byte offsets"
            )
        if index_cache.data_ptr() % 4:
            raise ValueError("index cache base must be dword aligned")
        if block_table.dtype is not torch.int32 or block_table.stride(1) != 1:
            raise ValueError("block table must be int32 with unit column stride")
        vals = torch.tensor(
            [
                index_cache.data_ptr(),
                block_table.data_ptr(),
                block_table.stride(0),
                index_cache.stride(0),
            ],
            dtype=torch.int64,
        )
        self.index_params[8:12].copy_(vals)
        # keep the pointed-to storage alive
        self._index_tables = (index_cache, block_table)

    def advance_step(self):
        self.step.add_(1)

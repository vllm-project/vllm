# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2025 FlyDSL Project Contributors
# mypy: ignore-errors
#
# This file contains code copied from FlyDSL (ROCm/FlyDSL PR #1204 at 21a3d1ee), as vendored by
# ROCm/ATOM PR #2435 (head 45e4b55d, atom/model_ops/monokernel/glm/op.py). The original source code was
# licensed under the Apache License 2.0 and included the following copyright notice:
# Copyright (c) 2025 FlyDSL Project Contributors
# Modified by the vLLM project contributors (Apache-2.0 sec. 4(b)): import paths rewritten to this package;
#   host wrapper options for the kernel changes in glm/kernel.py (poll limit / early-out, step epochs,
#   in-kernel indexer tables and options, build-time stage options), the cross-rank poll-expiry flag
#   and per-launch argument caching;
#   torch.accelerator in place of torch.cuda device calls.

"""Host wrapper for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from dataclasses import replace

import torch

from vllm.models.deepseek_v32.amd.mono.kernel.config import (
    AttentionWeight,
    GLM5_CONFIG,
    GLM5_KERNEL_SAMPLES,
    HIDDEN,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    V_DIM,
    KvCacheLayout,
    MoeMode,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
    RouterWeightLayout,
    as_kv_cache_layout,
    glm5_attention_heads,
    glm5_tp_config,
    validate_shard,
)
from vllm.models.deepseek_v32.amd.mono.kernel.glm.kernel import build_glm5_monokernel
from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import (
    INDEX_DIM,
    POLL_STAGES,
    layout,
    stage_tasks,
)
from vllm.models.deepseek_v32.amd.mono.kernel.layout import TL_COLS
from vllm.models.deepseek_v32.amd.mono.kernel.packing import (
    pack_bf16,
    pack_fp8,
    pack_layer_weights,
    pack_ptpc_fp8,
)
from vllm.models.deepseek_v32.amd.mono.kernel.runtime import SymmetricPeerBuffer
from vllm.models.deepseek_v32.amd.mono.kernel.weights import LayerWeights, prepare_mxfp4_expert_storage

__all__ = ["Glm5MonoKernel"]

_FP8_DTYPES = frozenset(
    d for d in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e4m3fnuz", None)) if d is not None
)


def _tsig(t):
    """What forward()'s checks depend on for one tensor argument: identity, dtype, shape, strides (contiguity)."""
    return None if t is None else (t.data_ptr(), t.dtype, t.shape, t.stride())


def prepare_glm5_weights(
    W: LayerWeights, attention_weight: AttentionWeight | str
) -> dict[str, torch.Tensor]:
    """Pack one layer once for every graph bucket using it."""

    t = W.t
    expert_mxfp4 = t["w_ug"].dtype is torch.uint8
    moe_mode = MoeMode.A16W4 if expert_mxfp4 else MoeMode.W8A8
    profile = replace(W.config, attention_weight=AttentionWeight(attention_weight))
    if profile.attention_weight is AttentionWeight.FP8_PTPC:
        attention = {
            "w_qkv_a": t["w_qkv_a"],
            "w_q_b": t["w_q_b"],
            "w_uk": pack_ptpc_fp8(t["w_uk"]),
            "w_uv": pack_ptpc_fp8(t["w_uv"]),
            "w_o": t["w_o"],
        }
    else:
        attention = pack_layer_weights(t, moe_mode, profile, attention_only=True)
    atom_experts = (
        expert_mxfp4
        and W.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM
        and W.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM
    )
    if (
        W.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM
        or W.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM
    ) and not atom_experts:
        raise ValueError(
            "ATOM expert storage requires MXFP4 values and scales together"
        )
    if atom_experts:
        packed = attention
        packed["w_r"] = pack_bf16(t["w_r"])
        packed.update(
            dict(
                zip(
                    ("w_ug", "s_ug", "w_dn", "s_dn"),
                    prepare_mxfp4_expert_storage(W, canonical=False),
                )
            )
        )
        return packed
    if profile.attention_weight is AttentionWeight.FP8_PTPC:
        raise ValueError("PTPC attention currently requires ATOM expert storage")
    return pack_layer_weights(
        t,
        moe_mode,
        profile,
        mxfp4_weight_layout=Mxfp4WeightLayout.NATIVE,
        mxfp4_scale_layout=Mxfp4ScaleLayout.NATIVE,
        router_weight_layout=RouterWeightLayout.NATIVE,
    )


class Glm5MonoKernel:
    """One TP rank of GLM-5's indexed decode MonoKernel.

    With ``with_indexer=True``, a single persistent launch covers index K/Q/W
    projection, index K normalization/RoPE/cache update, scoring, exact sparse
    top-k selection, MLA, routing, all expert compute, and both TP reductions.
    ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer uses PyTorch's caching allocator and CUDA/ROCm IPC
    storage sharing. Scratch and symmetric buffers may be shared by all layers
    because every launch uses a fresh ``tag``.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        topk: int = 2048,
        launches_per_step: int = 1,
        with_indexer: bool = False,
        index_max_seq: int = 4096,
        attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
        kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
        kv_cache_dtype: str = "bf16",
        prepared_weights: dict[str, torch.Tensor] | None = None,
        runtime: "Glm5MonoKernel | None" = None,
        dcp_size: int = 1,
        timeline=False,
        poll_limit: int | None = None,
        poll_early_out: bool = False,
        index_paged: bool = False,
        index_q_fp8: bool = True,
        index_cache_rowpar: bool = False,
        index_score_batched: bool = False,
        cache_hoist: bool = False,
        split_keys64: bool = False,
        select_radix11: bool = False,
        index_proj_spread: bool = False,
    ):
        expected_config = glm5_tp_config(npes)
        if W.config != expected_config:
            raise ValueError(
                f"Glm5MonoKernel requires {expected_config}, got {W.config}"
            )
        output_heads = expected_config.local_heads
        attention_heads = glm5_attention_heads(npes, dcp_size)
        validate_shard(
            samples,
            W.heads,
            rank,
            npes,
            topk,
            W.config,
            supported_samples=GLM5_KERNEL_SAMPLES,
            expected_heads=attention_heads,
        )
        if not 1 <= launches_per_step <= 128:
            raise ValueError(
                f"launches_per_step must be in [1, 128], got {launches_per_step}"
            )
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.launches_per_step = launches_per_step
        self.with_indexer = with_indexer
        # fused indexer on vLLM's paged FP8 index cache (see build_glm5_monokernel)
        self.index_paged = bool(index_paged)
        self._index_tables = None  # (index cache, block table) refs set by set_index_tables (paged)
        self._validated_sig = None  # forward(): input signature of the last validated launch
        self._wptrs = None  # forward(): the fixed weight pointers (built on the first launch)
        if self.index_paged and not with_indexer:
            raise ValueError("index_paged requires with_indexer=True")
        self.index_max_seq = index_max_seq
        self.attention_weight = AttentionWeight(attention_weight)
        self.kv_cache_layout = as_kv_cache_layout(kv_cache_layout)
        if kv_cache_dtype not in ("bf16", "fp8"):
            raise ValueError(f"unsupported KV cache dtype {kv_cache_dtype!r}")
        self.kv_cache_dtype = kv_cache_dtype
        self.dcp_size = dcp_size
        self.output_heads = output_heads
        t = W.t
        self.expert_mxfp4 = t["w_ug"].dtype is torch.uint8
        self.atom_experts = (
            self.expert_mxfp4
            and W.mxfp4_weight_layout is Mxfp4WeightLayout.ATOM
            and W.mxfp4_scale_layout is Mxfp4ScaleLayout.ATOM
        )
        self.packed = dict(
            prepare_glm5_weights(W, self.attention_weight)
            if prepared_weights is None
            else prepared_weights
        )
        if with_indexer:
            required = (
                "w_index_k",
                "s_index_k",
                "w_index_w",
                "w_index_q",
                "s_index_q",
                "g_index_k",
                "b_index_k",
            )
            missing = [name for name in required if name not in t]
            if missing:
                raise ValueError(
                    f"with_indexer=True requires weights: {', '.join(missing)}"
                )
            self.packed["w_index_k"] = pack_fp8(t["w_index_k"])
            self.packed["w_index_q"] = pack_fp8(t["w_index_q"])
            self.packed["w_index_w"] = pack_bf16(t["w_index_w"])
        self.scr_layout, self.sym_layout = layout(
            samples,
            W.heads,
            npes,
            topk,
            with_indexer,
            index_max_seq,
            inter=W.config.inter,
            output_heads=output_heads,
            dcp_size=dcp_size,
            split_keys=64 if split_keys64 else None,
        )
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        self.stages = stage_tasks(
            samples,
            W.heads,
            topk,
            with_indexer,
            index_max_seq,
            self.expert_mxfp4,
            inter=W.config.inter,
            split_keys=64 if split_keys64 else None,
        )
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = (
            torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev)
            if timeline
            else None
        )
        if with_indexer:
            index_tensors = dict(t, **self.packed)
            self.index_params = torch.tensor(
                [
                    index_tensors[name].data_ptr()
                    for name in (
                        "w_index_k",
                        "s_index_k",
                        "w_index_w",
                        "w_index_q",
                        "s_index_q",
                        "g_index_k",
                        "b_index_k",
                    )
                ]
                + [0 if self.timeline is None else self.timeline.data_ptr()]
                # 8..11 (index_paged): index cache, decode block table, its row stride, cache block stride (bytes);
                # set by set_index_tables
                + [0, 0, 0, 0],
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
        if self._owns_runtime:  # where this rank's expired waits flag rank 0 (kernel slow path)
            oa = self.scr_layout["poll_xrank_addr"]
            self.scratch[oa : oa + 8].view(torch.int64).fill_(self.peer_buffer.addresses[0] + ox + 4 * rank)
        self.peers = self.peer_buffer.addresses
        self.launch = build_glm5_monokernel(
            samples,
            W.heads,
            npes,
            topk,
            launches_per_step=launches_per_step,
            with_indexer=with_indexer,
            index_max_seq=index_max_seq,
            expert_mxfp4=self.expert_mxfp4,
            atom_experts=self.atom_experts,
            attention_weight=self.attention_weight,
            kv_cache_layout=self.kv_cache_layout,
            kv_cache_dtype=self.kv_cache_dtype,
            inter=W.config.inter,
            output_heads=output_heads,
            dcp_size=dcp_size,
            uv_scale_rows=(
                128
                if self.attention_weight
                in (AttentionWeight.BF16, AttentionWeight.FP8_PTPC)
                else W.t["w_uv"].shape[0] // W.t["s_uv"].shape[0]
            ),
            timeline=timeline,
            poll_limit=poll_limit,
            poll_early_out=poll_early_out,
            index_paged=self.index_paged,
            index_q_fp8=index_q_fp8,
            index_cache_rowpar=index_cache_rowpar,
            index_score_batched=index_score_batched,
            cache_hoist=cache_hoist,
            split_keys64=split_keys64,
            select_radix11=select_radix11,
            index_proj_spread=index_proj_spread,
        )
        self.poll_limit = poll_limit

    def poll_error(self, clear: bool = True) -> tuple[str, ...]:
        """Stages whose bounded mailbox waits expired since the last clear (synchronizes)."""

        off = self.scr_layout["poll_err"]
        words = self.scratch[off : off + 4 * len(POLL_STAGES)].view(torch.int32)
        expired = tuple(
            name for name, word in zip(POLL_STAGES, words.tolist()) if word
        )
        if clear and expired:
            words.zero_()
        return expired

    def debug(
        self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False
    ) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
        ``bf2``: each pair's value word packs two bf16 elements)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = (
                self.scratch[off : off + n * 4]
                .view(torch.int32)
                .view(n // 2, 2)[:, 0]
                .contiguous()
            )
            return words.view(torch.bfloat16).float().view(shape)
        words = (
            self.scratch[off : off + n * 8]
            .view(torch.int32)
            .view(n, 2)[:, 0]
            .contiguous()
        )
        return words.view(dtype).view(shape)

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
        index_cache=None,
        positions=None,
        slot_mapping=None,
        sparse_kv_indptr=None,
    ):
        """One layer.  Mailbox epochs are ``step * launches_per_step + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(
                f"layer must be in [0, {self.launches_per_step}), got {layer}"
            )
        total_samples = h.shape[0]
        # Validate on the first launch and whenever an input's identity / layout changes; a steady
        # decode loop passes the same persistent buffers every step, so it pays only for this signature.
        sig = (total_samples, advance, self._index_tables is not None, indices.dtype, _tsig(kv_cache),
               pe_cache.data_ptr(), _tsig(positions), _tsig(slot_mapping), _tsig(sparse_kv_indptr),
               _tsig(index_cache) if (self.with_indexer and not self.index_paged) else None)
        if sig != self._validated_sig:
            self._validate(h, kv_cache, pe_cache, indices, advance, index_cache, positions, slot_mapping,
                           sparse_kv_indptr)
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
                f"x_out must have shape {(total_samples, HIDDEN)}, got {tuple(x_out.shape)}"
            )
        p = lambda x: x.data_ptr()  # noqa: E731
        for chunk in range(chunks):
            row = chunk * self.S
            self.launch(
                p(h) + row * HIDDEN * h.element_size(),
                p(x_out) + row * HIDDEN * x_out.element_size(),
                p(cur_pos),
                p(cur_pos if positions is None else positions)
                + (0 if positions is None else row * 8),
                p(cur_pos if slot_mapping is None else slot_mapping)
                + (0 if slot_mapping is None else row * 8),
                p(cur_pos if sparse_kv_indptr is None else sparse_kv_indptr)
                + (0 if sparse_kv_indptr is None else row * 4),
                p(kv_cache),
                p(pe_cache),
                p(index_cache) if (self.with_indexer and not self.index_paged) else p(indices),
                p(cos),
                p(sin),
                *wp,
                p(self.scratch),
                self.sym,
                p(self.peers),
                (
                    p(self.index_params)
                    if self.with_indexer
                    else (0 if self.timeline is None else p(self.timeline))
                ),
                p(self.step),
                self.rank,
                layer,
                stream=torch.cuda.current_stream(),
            )
            if chunk + 1 < chunks or advance:
                self.advance_step()
        return x_out

    def _validate(self, h, kv_cache, pe_cache, indices, advance, index_cache, positions, slot_mapping,
                  sparse_kv_indptr):
        """forward()'s argument checks (raise ValueError); run when forward's input signature changes."""
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
        if self.index_paged and self._index_tables is None:
            raise ValueError("index_paged: call set_index_tables(index_cache, block_table) before forward")
        if self.index_paged and indices.dtype is not torch.int32:
            # the CSR the select stage writes (min(ctx, topk) entries per row at sparse_kv_indptr offsets; the
            # caller sizes it -- warm-up launches with inactive rows pass a 1-element buffer)
            raise ValueError("index_paged: indices must be the int32 CSR buffer")
        if self.with_indexer and not self.index_paged:
            if index_cache is None:
                raise ValueError("index_cache is required when with_indexer=True")
            if (
                index_cache.shape != (self.index_max_seq, INDEX_DIM)
                or index_cache.dtype is not torch.bfloat16
            ):
                raise ValueError(
                    f"index_cache must be bf16 [{self.index_max_seq}, {INDEX_DIM}], got "
                    f"{tuple(index_cache.shape)} {index_cache.dtype}"
                )
        if self.kv_cache_layout is KvCacheLayout.ATOM:
            cache_width = GLM5_CONFIG.kv_lora + GLM5_CONFIG.pe_dim
            expected_dtype = (
                kv_cache.dtype in _FP8_DTYPES
                if self.kv_cache_dtype == "fp8"
                else kv_cache.dtype is torch.bfloat16
            )
            if not expected_dtype or not kv_cache.is_contiguous():
                raise ValueError(
                    f"ATOM KV cache must be contiguous {self.kv_cache_dtype}"
                )
            if kv_cache.shape[-1] != cache_width:
                raise ValueError(
                    f"ATOM KV cache last dimension must be {cache_width}, got {tuple(kv_cache.shape)}"
                )
            if kv_cache.data_ptr() != pe_cache.data_ptr():
                raise ValueError(
                    "ATOM KV cache layout requires the same fused tensor for kv_cache and pe_cache"
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
                        f"{name} must be contiguous {dtype} with at least {size} values, got {got}"
                    )

    def _weight_ptrs(self) -> tuple[int, ...]:
        """The 20 fixed weight / scale pointers of the launch ABI, in order (built once, on the first
        launch; the weight tensors are immutable after construction; call ``refresh_weight_ptrs`` if they change)."""
        t = dict(self.W.t, **self.packed)
        p = lambda x: x.data_ptr()  # noqa: E731
        return (
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["w_qkv_a"]),
            p(t.get("s_qkv_a", t["w_qkv_a"])),
            p(t["w_q_b"]),
            p(t.get("s_q_b", t["w_q_b"])),
            p(t["w_uk"]),
            p(t.get("s_uk", t["w_uk"])),
            p(t["w_uv"]),
            p(t.get("s_uv", t["w_uv"])),
            p(t["w_o"]),
            p(t.get("s_o", t["w_o"])),
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
        )

    def refresh_weight_ptrs(self) -> None:
        self._wptrs = None

    def set_index_tables(self, index_cache: torch.Tensor, block_table: torch.Tensor) -> None:
        """index_paged: point the indexer parameter table at vLLM's uint8 [blocks, 16, 132] index cache and a
        persistent int32 [rows, W] decode block table (stable addresses: call once, before any capture)."""
        if not self.index_paged:
            raise ValueError("set_index_tables needs index_paged=True")
        if index_cache.dtype is not torch.uint8 or index_cache.dim() != 3 or tuple(index_cache.shape[1:]) != (16, INDEX_DIM + 4):
            raise ValueError(f"index cache must be uint8 [blocks, 16, {INDEX_DIM + 4}], got "
                             f"{tuple(index_cache.shape)} {index_cache.dtype}")
        if index_cache.stride(2) != 1 or index_cache.stride(1) != INDEX_DIM + 4 or index_cache.stride(0) % 4:
            # the kernel addresses one block as 2112 contiguous bytes (dword stores) at a dword-aligned block stride
            raise ValueError(f"index cache blocks must be contiguous with a dword block stride, got strides "
                             f"{index_cache.stride()}")
        if (index_cache.shape[0] - 1) * index_cache.stride(0) + 16 * (INDEX_DIM + 4) >= 1 << 31:
            raise ValueError("index cache spans >= 2 GiB: the kernel forms Int32 byte offsets")
        if index_cache.data_ptr() % 4:
            raise ValueError("index cache base must be dword aligned")
        if block_table.dtype is not torch.int32 or block_table.stride(1) != 1:
            raise ValueError("block table must be int32 with unit column stride")
        vals = torch.tensor([index_cache.data_ptr(), block_table.data_ptr(), block_table.stride(0),
                             index_cache.stride(0)], dtype=torch.int64)
        self.index_params[8:12].copy_(vals)
        self._index_tables = (index_cache, block_table)  # keep the pointed-to storage alive

    def advance_step(self):
        self.step.add_(1)

    def close(self):
        """Release this rank's remote HIP IPC mappings."""

        if self._owns_runtime:
            self.peer_buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def timeline_report(self) -> str:
        """Per stage, in us from launch start: [first start, median hint seen, last end]
        and median per-task phases (hint wait, payload staging, compute, epilogue)."""
        # s_memrealtime ticks at 100 MHz
        tl = (
            self.timeline[:, :5].cpu().double() / 100.0
        )
        t0 = tl[:, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n].clone()
            i += n
            for c in (1, 2, 3):  # missing marks inherit the previous one
                st[:, c] = torch.where(st[:, c] > 0, st[:, c], st[:, c - 1])
            d = (st[:, 1:] - st[:, :-1]).median(0).values
            rows.append(
                f"{name:7s} x{n:4d}  [{(st[:, 0].min() - t0):6.1f} | hint {(st[:, 1].median() - t0):6.1f} | "
                f"end {(st[:, 4].max() - t0):6.1f}]  hint {d[0]:5.1f}  stage {d[1]:5.1f}  "
                f"compute {d[2]:5.1f}  epi {d[3]:5.1f}"
            )
            if name == "index_score":
                per_sample = n // self.S
                ready = [
                    (st[s * per_sample : (s + 1) * per_sample, 4].max() - t0).item()
                    for s in range(self.S)
                ]
                rows.append(
                    " " * 10
                    + "score-ready/sample "
                    + " ".join(f"{v:.1f}" for v in ready)
                )
            elif name == "index_select":
                done = [(st[s, 4] - t0).item() for s in range(self.S)]
                rows.append(
                    " " * 10
                    + "select-done/sample "
                    + " ".join(f"{v:.1f}" for v in done)
                )
        return "\n".join(rows)

    def intermediates(self):
        S, H = self.S, self.W.heads
        result = dict(
            q_a=self.debug("q_a", (S, Q_LORA)),
            kv_a=self.debug("kv_a", (S, KV_LORA + PE_DIM)),
            q_nope=self.debug("q_nope", (S, H, NOPE_DIM), bf2=True),
            q_pe=self.debug("q_pe", (S, H, PE_DIM), bf2=True),
            q_lat=self.debug("q_lat", (S, H, KV_LORA), bf2=True),
            o=self.debug("o", (S, self.output_heads * V_DIM), bf2=True),
            a=self.debug("a", (S, HIDDEN), bf2=True).to(torch.bfloat16),
            scores=self.debug("scores", (S, N_EXPERTS)),
            sel=self.debug("sel", (S, MOE_SLOTS), torch.int32),
            prob=self.debug("prob", (S, MOE_SLOTS)),
            mid=self.debug("mid", (S, MOE_SLOTS, self.W.config.inter), bf2=False),
            xq=self.debug("xqd", (S, HIDDEN), pairs=False),
        )
        if self.with_indexer:
            result["index_q"] = self.debug("index_q", (S, 32, INDEX_DIM), bf2=True)
            result["index_w"] = self.debug("index_w", (S, 32))
            result["index_k"] = self.debug("index_k", (S, INDEX_DIM))  # pre-LayerNorm projection (fp32)
            result["index_scores"] = self.debug("index_scores", (S, self.index_max_seq))
        if self.with_indexer and self.index_paged:
            result["index_k_new"] = self.debug("index_k_new", (S, INDEX_DIM), bf2=True)  # unscaled FP8 values
            result["index_k_scale"] = self.debug("index_k_scale", (S,))
            return result  # paged: the CSR is in the caller's indices buffer, not in scratch
        if self.with_indexer:
            off = self.scr_layout["indices"]
            result["indices"] = (
                self.scratch[off : off + S * self.topk * 4]
                .view(torch.int32)
                .view(S, self.topk)
            )
        return result

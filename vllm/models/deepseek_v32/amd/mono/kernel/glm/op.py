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
#   torch.accelerator in place of torch.cuda device calls; the unused format / DCP / timeline options removed.

"""Host wrapper for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

import torch

from vllm.models.deepseek_v32.amd.mono.kernel.config import (
    AttentionWeight,
    GLM5_KERNEL_SAMPLES,
    HIDDEN,
    KV_LORA,
    MOE_SLOTS,
    N_EXPERTS,
    NOPE_DIM,
    PE_DIM,
    Q_LORA,
    SCALE_BM,
    V_DIM,
    glm5_tp_config,
)
from vllm.models.deepseek_v32.amd.mono.kernel.glm.kernel import build_glm5_monokernel
from vllm.models.deepseek_v32.amd.mono.kernel.glm.layout import INDEX_DIM, POLL_STAGES, layout
from vllm.models.deepseek_v32.amd.mono.kernel.packing import pack_bf16, pack_fp8, pack_mxfp4
from vllm.models.deepseek_v32.amd.mono.kernel.runtime import SymmetricPeerBuffer
from vllm.models.deepseek_v32.amd.mono.kernel.weights import LayerWeights

__all__ = ["Glm5MonoKernel"]

# fused-indexer weights, in index_params order (slots 0..6)
_INDEX_WEIGHTS = ("w_index_k", "s_index_k", "w_index_w", "w_index_q", "s_index_q", "g_index_k", "b_index_k")


def _tsig(t):
    """What forward()'s checks depend on for one tensor argument: identity, dtype, shape, strides (contiguity)."""
    return None if t is None else (t.data_ptr(), t.dtype, t.shape, t.stride())


def prepare_glm5_weights(W: LayerWeights, attention_weight: AttentionWeight | str) -> dict[str, torch.Tensor]:
    """Pack one layer once for every graph bucket using it: attention (FP8 or BF16),
    MXFP4 experts (scales stay native) and the BF16 router."""

    t = W.t
    names = ("w_qkv_a", "w_q_b", "w_uk", "w_uv", "w_o", "w_ug", "w_dn", "w_r")
    missing = [name for name in names if name not in t]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    if t["w_ug"].dtype is not torch.uint8:
        raise ValueError("the GLM-5 MonoKernel needs MXFP4 (uint8) expert weights")
    bf16 = AttentionWeight(attention_weight) is AttentionWeight.BF16
    packed = {name: (pack_bf16 if bf16 else pack_fp8)(t[name]) for name in names[:5]}
    packed.update({name: pack_mxfp4(t[name]) for name in ("w_ug", "w_dn")})
    packed["w_r"] = pack_bf16(t["w_r"])
    return packed


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
        prepared_weights: dict[str, torch.Tensor] | None = None,
        runtime: "Glm5MonoKernel | None" = None,
        poll_limit: int | None = None,
        poll_early_out: bool = False,
        index_q_fp8: bool = True,
        cache_hoist: bool = False,
        split_keys64: bool = False,
        select_radix11: bool = False,
        index_proj_spread: bool = False,
    ):
        expected_config = glm5_tp_config(npes)
        if W.config != expected_config:
            raise ValueError(f"Glm5MonoKernel requires {expected_config}, got {W.config}")
        if samples not in GLM5_KERNEL_SAMPLES:
            raise ValueError(f"samples must be one of {GLM5_KERNEL_SAMPLES}, got {samples}")
        if W.heads != expected_config.local_heads:
            raise ValueError(f"expected {expected_config.local_heads} local heads, got {W.heads}")
        if not 0 <= rank < npes:
            raise ValueError(f"rank must be in [0, {npes}), got {rank}")
        if topk <= 0 or topk % 64:
            raise ValueError(f"topk must be a positive multiple of 64, got {topk}")
        if not 1 <= launches_per_step <= 128:
            raise ValueError(f"launches_per_step must be in [1, 128], got {launches_per_step}")
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.launches_per_step = launches_per_step
        self.with_indexer = with_indexer
        self._index_tables = None  # (index cache, block table) refs set by set_index_tables
        self._validated_sig = None  # forward(): input signature of the last validated launch
        self._wptrs = None  # forward(): the fixed weight pointers (built on the first launch)
        self.index_max_seq = index_max_seq
        self.attention_weight = AttentionWeight(attention_weight)
        t = W.t
        if (
            self.attention_weight is AttentionWeight.FP8_BLOCK128
            and t["w_uv"].shape[0] // t["s_uv"].shape[0] != SCALE_BM
        ):
            raise ValueError("w_uv FP8 scales must cover 128-row blocks")
        self.packed = dict(
            prepare_glm5_weights(W, self.attention_weight) if prepared_weights is None else prepared_weights
        )
        if with_indexer:
            missing = [name for name in _INDEX_WEIGHTS if name not in t]
            if missing:
                raise ValueError(f"with_indexer=True requires weights: {', '.join(missing)}")
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
            split_keys=64 if split_keys64 else None,
        )
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        if with_indexer:
            index_tensors = dict(t, **self.packed)
            self.index_params = torch.tensor(
                # 7: unused; 8..11 (set_index_tables): index cache, block table, its row stride, cache block stride (B)
                [index_tensors[name].data_ptr() for name in _INDEX_WEIGHTS] + [0] * 5,
                dtype=torch.int64,
                device=dev,
            )
        else:
            self.index_params = None
        self._owns_runtime = runtime is None
        if runtime is None:
            self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
            self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
            self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        else:
            if runtime.scr_layout != self.scr_layout or runtime.sym_layout != self.sym_layout:
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
            attention_weight=self.attention_weight,
            inter=W.config.inter,
            poll_limit=poll_limit,
            poll_early_out=poll_early_out,
            index_q_fp8=index_q_fp8,
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
        expired = tuple(name for name, word in zip(POLL_STAGES, words.tolist()) if word)
        if clear and expired:
            words.zero_()
        return expired

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
        ``bf2``: each pair's value word packs two bf16 elements)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = self.scratch[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
            return words.view(torch.bfloat16).float().view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
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
        positions=None,
        slot_mapping=None,
        sparse_kv_indptr=None,
    ):
        """One layer.  Mailbox epochs are ``step * launches_per_step + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(f"layer must be in [0, {self.launches_per_step}), got {layer}")
        total_samples = h.shape[0]
        # Validate on the first launch and whenever an input's identity / layout changes; a steady
        # decode loop passes the same persistent buffers every step, so it pays only for this signature.
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
            self._validate(h, kv_cache, pe_cache, indices, advance, positions, slot_mapping, sparse_kv_indptr)
            self._validated_sig = sig
        chunks = total_samples // self.S
        wp = self._wptrs
        if wp is None:
            wp = self._wptrs = self._weight_ptrs()
        if x_out is None:
            x_out = torch.empty(total_samples, HIDDEN, dtype=torch.bfloat16, device=h.device)
        elif x_out.shape != (total_samples, HIDDEN):
            raise ValueError(f"x_out must have shape {(total_samples, HIDDEN)}, got {tuple(x_out.shape)}")
        p = lambda x: x.data_ptr()  # noqa: E731
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

    def _validate(self, h, kv_cache, pe_cache, indices, advance, positions, slot_mapping, sparse_kv_indptr):
        """forward()'s argument checks (raise ValueError); run when forward's input signature changes."""
        total_samples = h.shape[0]
        if total_samples % self.S:
            raise ValueError(f"input rows {total_samples} must be divisible by kernel chunk {self.S}")
        chunks = total_samples // self.S
        if self.with_indexer and chunks != 1:
            raise ValueError("fused indexer does not support chunked launches")
        if not advance and chunks != 1:
            raise ValueError("chunked launches must advance mailbox epochs")
        if self.with_indexer and self._index_tables is None:
            raise ValueError("with_indexer: call set_index_tables(index_cache, block_table) before forward")
        if self.with_indexer and indices.dtype is not torch.int32:
            # the CSR the select stage writes (min(ctx, topk) entries per row at sparse_kv_indptr offsets; the
            # caller sizes it -- warm-up launches with inactive rows pass a 1-element buffer)
            raise ValueError("with_indexer: indices must be the int32 CSR buffer")
        cache_width = KV_LORA + PE_DIM
        if kv_cache.dtype is not torch.bfloat16 or not kv_cache.is_contiguous():
            raise ValueError("the MLA KV cache must be contiguous bf16")
        if kv_cache.shape[-1] != cache_width:
            raise ValueError(f"MLA KV cache last dimension must be {cache_width}, got {tuple(kv_cache.shape)}")
        if kv_cache.data_ptr() != pe_cache.data_ptr():
            raise ValueError("kv_cache and pe_cache must be the same fused [slots, 576] tensor")
        for name, value, dtype, size in (
            ("positions", positions, torch.int64, total_samples),
            ("slot_mapping", slot_mapping, torch.int64, total_samples),
            ("sparse_kv_indptr", sparse_kv_indptr, torch.int32, total_samples + 1),
        ):
            if value is None or value.dtype is not dtype or value.numel() < size or not value.is_contiguous():
                got = None if value is None else (tuple(value.shape), value.dtype)
                raise ValueError(f"{name} must be contiguous {dtype} with at least {size} values, got {got}")

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
        """Point the indexer parameter table at vLLM's uint8 [blocks, 16, 132] index cache and a
        persistent int32 [rows, W] decode block table (stable addresses: call once, before any capture)."""
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
        if index_cache.stride(2) != 1 or index_cache.stride(1) != INDEX_DIM + 4 or index_cache.stride(0) % 4:
            # the kernel addresses one block as 2112 contiguous bytes (dword stores) at a dword-aligned block stride
            raise ValueError(
                f"index cache blocks must be contiguous with a dword block stride, got strides {index_cache.stride()}"
            )
        if (index_cache.shape[0] - 1) * index_cache.stride(0) + 16 * (INDEX_DIM + 4) >= 1 << 31:
            raise ValueError("index cache spans >= 2 GiB: the kernel forms Int32 byte offsets")
        if index_cache.data_ptr() % 4:
            raise ValueError("index cache base must be dword aligned")
        if block_table.dtype is not torch.int32 or block_table.stride(1) != 1:
            raise ValueError("block table must be int32 with unit column stride")
        vals = torch.tensor(
            [index_cache.data_ptr(), block_table.data_ptr(), block_table.stride(0), index_cache.stride(0)],
            dtype=torch.int64,
        )
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

    def intermediates(self):
        S, H = self.S, self.W.heads
        result = dict(
            q_a=self.debug("q_a", (S, Q_LORA)),
            kv_a=self.debug("kv_a", (S, KV_LORA + PE_DIM)),
            q_nope=self.debug("q_nope", (S, H, NOPE_DIM), bf2=True),
            q_pe=self.debug("q_pe", (S, H, PE_DIM), bf2=True),
            q_lat=self.debug("q_lat", (S, H, KV_LORA), bf2=True),
            o=self.debug("o", (S, H * V_DIM), bf2=True),
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
            result["index_k_new"] = self.debug("index_k_new", (S, INDEX_DIM), bf2=True)  # unscaled FP8 values
            result["index_k_scale"] = self.debug("index_k_scale", (S,))
        return result  # the CSR is in the caller's indices buffer

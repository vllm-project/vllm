# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/aiter#6173 at b02df0db8 (Apache-2.0 License),
# Copyright (c) 2025 FlyDSL Project Contributors:
# aiter/ops/flydsl/kernels/glm5_mono/glm/op.py
# ruff: noqa: E501

"""Host wrapper for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from dataclasses import replace

import torch

from vllm.models.deepseek_v32.amd.mono.config import (
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
    AttentionWeight,
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
from vllm.models.deepseek_v32.amd.mono.glm.kernel import build_glm5_monokernel
from vllm.models.deepseek_v32.amd.mono.glm.layout import (
    INDEX_DIM,
    layout,
    stage_tasks,
)
from vllm.models.deepseek_v32.amd.mono.layout import TL_COLS
from vllm.models.deepseek_v32.amd.mono.packing import (
    pack_bf16,
    pack_fp8,
    pack_layer_weights,
    pack_ptpc_fp8,
)
from vllm.models.deepseek_v32.amd.mono.runtime import SymmetricPeerBuffer
from vllm.models.deepseek_v32.amd.mono.weights import (
    LayerWeights,
    prepare_mxfp4_expert_storage,
)

__all__ = ["Glm5MonoKernel"]


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
        runtime: Glm5MonoKernel | None = None,
        dcp_size: int = 1,
        native_fp4_mfma: bool = False,
        timeline=False,
        index_paged: bool = False,
        index_block_size: int = 64,
        index_block_bytes: int = 0,
        index_shuffled: bool = False,
        block_table_stride: int = 0,
        index_k_bf16: bool = False,
        dense_experts: int = 0,
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
        if dense_experts and W.physical_experts != dense_experts:
            raise ValueError(
                f"dense_experts={dense_experts} needs that many expert slices, "
                f"got physical_experts={W.physical_experts}"
            )
        self.dense_experts = dense_experts
        self.index_max_seq = index_max_seq
        self.index_paged = index_paged
        self.index_block_size = index_block_size
        self.index_block_bytes = index_block_bytes
        self.block_table_stride = block_table_stride
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
        if native_fp4_mfma and not self.atom_experts:
            raise ValueError("native FP4 MFMA requires ATOM MXFP4 expert storage")
        self.native_fp4_mfma = native_fp4_mfma
        self.packed = dict(
            prepare_glm5_weights(W, self.attention_weight)
            if prepared_weights is None
            else prepared_weights
        )
        if with_indexer:
            required = (
                "w_index_k",
                "w_index_w",
                "w_index_q",
                "s_index_q",
                "g_index_k",
                "b_index_k",
            ) + (() if index_k_bf16 else ("s_index_k",))
            missing = [name for name in required if name not in t]
            if missing:
                raise ValueError(
                    f"with_indexer=True requires weights: {', '.join(missing)}"
                )
            self.packed["w_index_k"] = (
                pack_bf16(t["w_index_k"]) if index_k_bf16 else pack_fp8(t["w_index_k"])
            )
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
            native_fp4_mfma=native_fp4_mfma,
        )
        dev = torch.device("cuda", torch.cuda.current_device())
        self.stages = stage_tasks(
            samples,
            W.heads,
            topk,
            with_indexer,
            index_max_seq,
            self.expert_mxfp4,
            inter=W.config.inter,
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
                        "w_index_k" if index_k_bf16 else "s_index_k",
                        "w_index_w",
                        "w_index_q",
                        "s_index_q",
                        "g_index_k",
                        "b_index_k",
                    )
                ]
                + [0 if self.timeline is None else self.timeline.data_ptr()],
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
            native_fp4_mfma=native_fp4_mfma,
            uv_scale_rows=(
                128
                if self.attention_weight
                in (AttentionWeight.BF16, AttentionWeight.FP8_PTPC)
                else W.t["w_uv"].shape[0] // W.t["s_uv"].shape[0]
            ),
            timeline=timeline,
            index_paged=index_paged,
            index_block_size=index_block_size,
            index_block_bytes=index_block_bytes,
            index_shuffled=index_shuffled,
            block_table_stride=block_table_stride,
            index_k_bf16=index_k_bf16,
            dense_experts=dense_experts,
        )

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
        block_table=None,
        req_ids=None,
        out_indices=None,
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
        if total_samples % self.S:
            raise ValueError(
                f"input rows {total_samples} must be divisible by kernel chunk {self.S}"
            )
        chunks = total_samples // self.S
        if self.with_indexer and chunks != 1:
            raise ValueError("fused indexer does not support chunked launches")
        if not advance and chunks != 1:
            raise ValueError("chunked launches must advance mailbox epochs")
        if self.index_paged:
            if index_cache is None or index_cache.element_size() != 1:
                raise ValueError("index_paged requires the 1-byte paged index cache")
            if (
                index_cache.dim() != 3
                or index_cache.shape[1:] != (self.index_block_size, INDEX_DIM + 4)
                or index_cache.stride(0) * index_cache.element_size()
                != self.index_block_bytes
                or index_cache.stride(1) != INDEX_DIM + 4
            ):
                raise ValueError(
                    f"index cache must be [blocks, {self.index_block_size}, {INDEX_DIM + 4}] "
                    f"with {self.index_block_bytes}-byte blocks, got "
                    f"{tuple(index_cache.shape)} strides {index_cache.stride()}"
                )
            for name, value in (
                ("block_table", block_table),
                ("req_ids", req_ids),
                ("out_indices", out_indices),
            ):
                if value is None or value.dtype is not torch.int32:
                    raise ValueError(f"index_paged requires int32 {name}")
            if (
                block_table.stride(0) != self.block_table_stride
                or block_table.stride(1) != 1
            ):
                raise ValueError(
                    f"block_table rows must be {self.block_table_stride} contiguous entries, "
                    f"got strides {block_table.stride()}"
                )
            if req_ids.numel() < total_samples or not req_ids.is_contiguous():
                raise ValueError(f"req_ids must hold {total_samples} contiguous ids")
        elif self.with_indexer:
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
            fp8_dtypes = {
                dtype
                for dtype in (
                    getattr(torch, "float8_e4m3fn", None),
                    getattr(torch, "float8_e4m3fnuz", None),
                )
                if dtype is not None
            }
            expected_dtype = (
                kv_cache.dtype in fp8_dtypes
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
        t = dict(self.W.t, **self.packed)
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
                p(index_cache) if self.with_indexer else p(indices),
                p(cos),
                p(sin),
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
                p(self.scratch),
                self.sym,
                p(self.peers),
                (
                    p(self.index_params)
                    if self.with_indexer
                    else (0 if self.timeline is None else p(self.timeline))
                ),
                p(self.step),
                p(block_table) if self.index_paged else 0,
                p(req_ids) + row * 4 if self.index_paged else 0,
                p(out_indices) if self.index_paged else 0,
                self.rank,
                layer,
                stream=torch.cuda.current_stream(),
            )
            if chunk + 1 < chunks or advance:
                self.advance_step()
        return x_out

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
        tl = (
            self.timeline[:, :5].cpu().double() / 100.0
        )  # s_memrealtime ticks at 100 MHz
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
            mid=self.debug("mid", (S, MOE_SLOTS, self.W.config.inter)),
            xq=self.debug("xqd", (S, HIDDEN), pairs=False),
        )
        if self.with_indexer:
            result["index_k"] = self.debug("index_k", (S, INDEX_DIM))
            result["index_q"] = self.debug("index_q", (S, 32, INDEX_DIM), bf2=True)
            result["index_w"] = self.debug("index_w", (S, 32))
            off = self.scr_layout["indices"]
            result["indices"] = (
                self.scratch[off : off + S * self.topk * 4]
                .view(torch.int32)
                .view(S, self.topk)
            )
        return result

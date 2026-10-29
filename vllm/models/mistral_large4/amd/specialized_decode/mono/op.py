# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from dataclasses import replace

import torch

from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.consensus import (
    MonoUnsupported,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.peer_memory import (
    SymmetricPeerBuffer,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.layer import (
    LayerWeights,
    prepare_mxfp4_expert_storage,
)
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.weights.packing import (
    pack_bf16,
    pack_fp8,
    pack_layer_weights,
    pack_mxfp4,
    pack_mxfp4_scale_tiles,
    pack_ptpc_fp8,
)

from .geometry import WAVES as MONO_WAVES
from .geometry_config import (
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
    glm5_tp_config,
    validate_shard,
)
from .kernel import build_glm5_monokernel
from .layout import (
    INDEX_DIM,
    INDEX_PAGE_ROW,
    INDEX_TILE,
    PARAM_TABLE_SLOTS,
    layout,
    stage_tasks,
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
    packed = pack_layer_weights(
        t,
        moe_mode,
        profile,
        mxfp4_weight_layout=Mxfp4WeightLayout.NATIVE,
        mxfp4_scale_layout=Mxfp4ScaleLayout.NATIVE,
        router_weight_layout=RouterWeightLayout.NATIVE,
        attention_only=expert_mxfp4,
    )
    if expert_mxfp4:
        packed["w_r"] = pack_bf16(t["w_r"])
        for weight, scale in (("w_ug", "s_ug"), ("w_dn", "s_dn")):
            q, scales = t[weight], t[scale]
            if weight == "w_ug":
                # Preserve the native-storage optimization: eight gate/up row
                # pairs form one contiguous MFMA tile at either TP geometry.
                def pair_rows(value):
                    experts, rows, cols = value.shape
                    return (
                        value.reshape(experts, 2, W.config.inter // 8, 8, cols)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(experts, rows, cols)
                    )

                q, scales = pair_rows(q), pair_rows(scales)
            packed[weight] = pack_mxfp4(q)
            packed[scale] = pack_mxfp4_scale_tiles(scales)
    return packed


FAULT_WORD_BYTES = 64


def fault_word_offset(sym_layout: dict[str, int]) -> int:
    """Byte offset of the fail-stop word: after the symmetric layout, 64-byte aligned."""
    return (
        (sym_layout["_bytes"] + FAULT_WORD_BYTES - 1)
        // FAULT_WORD_BYTES
        * FAULT_WORD_BYTES
    )


INDEX_PARAM_NAMES = (
    "w_index_k",
    "s_index_k",
    "w_index_w",
    "w_index_q",
    "s_index_q",
    "g_index_k",
    "b_index_k",
)
SHARED_EXPERT_NAMES = ("w_ug_sh", "s_ug_sh", "w_dn_sh", "s_dn_sh")


def param_table(
    tensors: dict[str, torch.Tensor], with_indexer: bool, separate_shared: bool
) -> list[int]:
    """The int64 pointer table a launch takes in its timeline argument (layout.PARAM_TABLE_SLOTS)."""
    table = [
        tensors[n].data_ptr() if with_indexer else 0 for n in INDEX_PARAM_NAMES
    ] + [0]
    table += [
        tensors[n].data_ptr() if separate_shared else 0 for n in SHARED_EXPERT_NAMES
    ]
    assert len(table) == PARAM_TABLE_SLOTS
    return table


class Glm5MonoKernel:
    """One TP rank of GLM-5's indexed decode MonoKernel.

    With ``with_indexer=True``, a single persistent launch covers index K/Q/W
    projection, index K normalization/RoPE/cache update, scoring, exact sparse
    top-k selection, MLA, routing, all expert compute, and both TP reductions.
    The fused path accepts flat BF16 caches for one request, or ATOM-layout
    attention KV and a physically paged index cache --
    ``index_cache_dtype="fp8"`` for ATOM's production plane, ``"bf16"`` for the
    private plane PR #1256 shipped. Paged caches share
    ``block_tables`` to resolve logical token positions; each launch groups ``index_request_width`` consecutive rows per
    request (five rows for MTP4 C1/C2).
    ``indexer_cp=True`` partitions index scoring across the TP ranks and uses
    in-kernel peer mailboxes for exact global selection. Index caches remain
    replicated; each rank must receive identical indexer weights and inputs.
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
        indexer_cp: bool = False,
        *,
        index_request_width: int | None = None,
        index_page_size: int = 16,
        index_block_table_stride: int | None = None,
        index_cache_dtype: str = "bf16",
        attention_weight: AttentionWeight | str = AttentionWeight.FP8_BLOCK128,
        kv_cache_layout: KvCacheLayout | str = KvCacheLayout.SPLIT,
        kv_cache_dtype: str = "bf16",
        prepared_weights: dict[str, torch.Tensor] | None = None,
        runtime: Glm5MonoKernel | None = None,
        dcp_size: int = 1,
        native_fp4_mfma: bool = False,
        rope_dtype: str = "fp32",
        shared_indices: bool = False,
        separate_shared: bool = False,
        fail_stop: bool = False,
    ):
        if W.config not in (GLM5_CONFIG, glm5_tp_config(4)):
            raise ValueError(f"unsupported GLM-5 shard configuration: {W.config}")
        if W.config == glm5_tp_config(4) and npes not in (
            1,
            4,
        ):  # npes=1: one TP4 rank's shard without the peer reduce (tests)
            raise ValueError("the 16-head, 512-intermediate shard requires TP4")
        output_heads = W.config.local_heads
        attention_heads = output_heads * dcp_size
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
        if dcp_size not in (1, npes):
            raise ValueError("DCP size must be 1 or the TP group size")
        if with_indexer and dcp_size != 1:
            raise ValueError("fused indexer currently requires dcp_size=1")
        if indexer_cp and not with_indexer:
            raise ValueError("indexer_cp=True requires with_indexer=True")
        if with_indexer and (index_max_seq <= 0 or index_max_seq % 64):
            raise ValueError("index_max_seq must be a positive multiple of 64")
        if not 1 <= launches_per_step <= 128:
            raise ValueError(
                f"launches_per_step must be in [1, 128], got {launches_per_step}"
            )
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.shared_indices = shared_indices
        self.separate_shared = separate_shared
        self.launches_per_step = launches_per_step
        self.with_indexer = with_indexer
        self.index_max_seq = index_max_seq
        self.index_request_width = (
            samples if index_request_width is None else index_request_width
        )
        self.index_page_size = index_page_size
        if index_cache_dtype not in ("bf16", "fp8"):
            raise ValueError(f"unsupported index cache dtype {index_cache_dtype!r}")
        self.index_cache_dtype = index_cache_dtype
        if with_indexer and (self.index_request_width <= 0 or index_page_size <= 0):
            raise ValueError(
                "fused indexer requires positive request width and page size"
            )
        if with_indexer and index_cache_dtype == "fp8" and index_page_size % INDEX_TILE:
            raise ValueError(
                f"the FP8 index plane preshuffles in {INDEX_TILE}-row tiles, got page {index_page_size}"
            )
        self.index_block_table_stride = (
            index_max_seq // index_page_size
            if index_block_table_stride is None
            else index_block_table_stride
        )
        if with_indexer and (
            samples % self.index_request_width != 0
            or index_max_seq % index_page_size != 0
            or self.index_block_table_stride < index_max_seq // index_page_size
        ):
            raise ValueError(
                "fused indexer requires whole requests and a page-aligned index_max_seq"
            )
        self.attention_weight = AttentionWeight(attention_weight)
        self.kv_cache_layout = as_kv_cache_layout(kv_cache_layout)
        if kv_cache_dtype not in ("bf16", "fp8"):
            raise ValueError(f"unsupported KV cache dtype {kv_cache_dtype!r}")
        if rope_dtype not in ("fp32", "bf16"):
            raise ValueError("rope_dtype must be fp32 or bf16")
        self.rope_dtype = rope_dtype
        self.kv_cache_dtype = kv_cache_dtype
        self.dcp_size = dcp_size
        self.output_heads = output_heads
        self.indexer_cp = indexer_cp
        # The schedule is fixed, not configurable. Each of these was measured
        # against the alternative and the loser is gone rather than switched
        # off: the shared-expert hoist (-0.94% at S=12), the head-group merge
        # (-2.47% at S=12, -0.21% at S=6) and a 6-sample qkv_a tile (-1.81% at
        # S=6, -0.87% at S=12), all bit-exact against the schedule they
        # replace. Deduplicating the routed MoE measured 7.5% / 2.7% SLOWER and
        # prefetching more than one `ug` sample measured flat, so neither is
        # here at all.
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
            indexer_cp=indexer_cp,
            inter=W.config.inter,
            output_heads=output_heads,
            dcp_size=dcp_size,
            native_fp4_mfma=native_fp4_mfma,
        )
        dev = torch.device("cuda", torch.cuda.current_device())
        self.identity_block_tables = (
            torch.arange(
                self.index_block_table_stride, dtype=torch.int32, device=dev
            ).unsqueeze(0)
            if with_indexer
            and self.kv_cache_layout is KvCacheLayout.ATOM
            and samples == self.index_request_width
            and self.index_block_table_stride == index_max_seq // index_page_size
            else None
        )
        self.stages = stage_tasks(
            samples,
            W.heads,
            topk,
            with_indexer,
            index_max_seq,
            self.expert_mxfp4,
            indexer_cp=indexer_cp,
            npes=npes,
            inter=W.config.inter,
            split_head_merge=W.heads // MONO_WAVES > 1,
        )
        if with_indexer or separate_shared:
            self.index_params = torch.tensor(
                param_table(dict(t, **self.packed), with_indexer, separate_shared),
                dtype=torch.int64,
                device=dev,
            )
        else:
            self.index_params = None
        self._owns_runtime = runtime is None
        if runtime is None:
            # `scratch` and `step` are rank-local, but `SymmetricPeerBuffer`'s
            # constructor is COLLECTIVE (it all-gathers IPC handles). Allocating
            # them straight through would let one rank fail here -- an OOM is the
            # realistic case -- and unwind into the runner's next collective while
            # its peers sit inside the peer-buffer all-gather. That is a collective
            # ORDER mismatch, which hangs, instead of the uniform fallback every
            # other refusal in this path gives.
            #
            # So agree first, in exactly the shape SymmetricPeerBuffer uses for its
            # own allocation: every rank attempts, every rank reports, and the whole
            # group raises together -- a rank that succeeded locally still raises, so
            # nobody is left holding a buffer the others do not have.
            local_error = None
            try:
                self.scratch = torch.zeros(
                    self.scr_layout["_bytes"], dtype=torch.uint8, device=dev
                )
                self.step = torch.zeros(1, dtype=torch.int32, device=dev)
            except Exception as error:
                self.scratch = None
                self.step = None
                local_error = f"{type(error).__name__}: {error}"
            if npes > 1:
                import torch.distributed as dist

                reports = [None] * npes
                dist.all_gather_object(reports, local_error, group=group)
            else:
                reports = [local_error]
            failed = [
                (peer, why) for peer, why in enumerate(reports) if why is not None
            ]
            if failed:
                self.scratch = None
                self.step = None
                # MonoUnsupported, not RuntimeError: `_build` routes on
                # MonoUnsupported, so a RuntimeError here skipped the uniform
                # fallback this whole consensus exists to give and took the
                # server down instead. Safe to fall back on because
                # the verdict is already agreed -- every rank sees the same
                # `failed` list and raises together.
                raise MonoUnsupported(
                    "mono scratch/step allocation failed: "
                    + "; ".join(f"rank {peer}: {why}" for peer, why in failed)
                )
            self.fault_offset = (
                fault_word_offset(self.sym_layout) if fail_stop else None
            )
            sym_bytes = (
                self.sym_layout["_bytes"]
                if self.fault_offset is None
                else self.fault_offset + FAULT_WORD_BYTES
            )
            self.peer_buffer = SymmetricPeerBuffer(
                sym_bytes, rank=rank, npes=npes, group=group
            )
        else:
            if (
                runtime.scr_layout != self.scr_layout
                or runtime.sym_layout != self.sym_layout
            ):
                raise ValueError("shared GLM runtime geometry mismatch")
            self.scratch = runtime.scratch
            self.fault_offset = runtime.fault_offset
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
            index_request_width=self.index_request_width,
            index_page_size=index_page_size,
            index_block_table_stride=self.index_block_table_stride,
            index_cache_dtype=self.index_cache_dtype,
            expert_mxfp4=self.expert_mxfp4,
            atom_experts=self.atom_experts,
            attention_weight=self.attention_weight,
            kv_cache_layout=self.kv_cache_layout,
            kv_cache_dtype=self.kv_cache_dtype,
            inter=W.config.inter,
            output_heads=output_heads,
            dcp_size=dcp_size,
            native_fp4_mfma=native_fp4_mfma,
            rope_dtype=rope_dtype,
            uv_scale_rows=(
                128
                if self.attention_weight
                in (AttentionWeight.BF16, AttentionWeight.FP8_PTPC)
                else W.t["w_uv"].shape[0] // W.t["s_uv"].shape[0]
            ),
            indexer_cp=indexer_cp,
            shared_indices=shared_indices,
            separate_shared=separate_shared,
            fault_offset=self.fault_offset,
        )

    @property
    def fault_word(self) -> torch.Tensor | None:
        """The sticky fail-stop word (int32[1]) of this rank, None without fail-stop."""
        if self.fault_offset is None:
            return None
        return self.peer_buffer.storage[self.fault_offset : self.fault_offset + 4].view(
            torch.int32
        )

    def debug(
        self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False
    ) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
        ``bf2``: each pair's value word packs two bf16 elements)."""
        off = self.scr_layout[name]
        storage = self.scratch
        if name == "index_q" and "index_cp_q" in self.sym_layout:
            # Debug reads the most recently published slot; production forward
            # never reads device epochs on the host.
            off = self.sym_layout["index_cp_q"]
            stride = self.sym_layout["index_cp_q_stride"]
            storage = self.sym_storage
            first_tag = storage[off + 4 : off + 8].view(torch.int32).item()
            second_tag = (
                storage[off + stride + 4 : off + stride + 8].view(torch.int32).item()
            )
            off += stride if second_tag > first_tag else 0
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return storage[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = (
                storage[off : off + n * 4]
                .view(torch.int32)
                .view(n // 2, 2)[:, 0]
                .contiguous()
            )
            return words.view(torch.bfloat16).float().view(shape)
        words = (
            storage[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
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
        block_tables=None,
        positions=None,
        slot_mapping=None,
        sparse_kv_indptr=None,
    ):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph.
        For the fused path, ``block_tables`` is contiguous int32 with one row
        per request and physical page IDs shared by both caches."""
        if not 0 <= layer < self.launches_per_step:
            raise ValueError(
                f"layer must be in [0, {self.launches_per_step}), got {layer}"
            )
        if h.ndim != 2 or h.shape[1] != HIDDEN:
            raise ValueError(f"hidden must have {HIDDEN} columns")
        if cur_pos.shape != (1,) or cur_pos.dtype is not torch.int32:
            raise ValueError("cur_pos must be int32[1]")
        expected_rope_dtype = (
            torch.bfloat16 if self.rope_dtype == "bf16" else torch.float32
        )
        if cos.dtype is not expected_rope_dtype or sin.dtype is not expected_rope_dtype:
            raise ValueError(f"RoPE tables must be {expected_rope_dtype}")
        if self.kv_cache_layout is KvCacheLayout.SPLIT:
            if (
                kv_cache.ndim != 2
                or kv_cache.shape[1] != KV_LORA
                or pe_cache.shape != (kv_cache.shape[0], PE_DIM)
            ):
                raise ValueError(
                    "split cache must be [capacity,512] KV and [capacity,64] PE"
                )
            if self.with_indexer and self.index_request_width != self.S:
                raise ValueError(
                    "split index cache supports one consecutive request per launch"
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
        if self.with_indexer:
            if index_cache is None:
                raise ValueError("index_cache is required when with_indexer=True")
            cache_rows = (
                index_cache.shape[0] * index_cache.shape[1]
                if index_cache.ndim == 3
                else index_cache.shape[0]
                if index_cache.ndim == 2
                else 0
            )
            # The FP8 plane is ATOM's: 128 key bytes and an fp32 row scale in
            # separate per-page segments, allocated 144 bytes wide. ATOM carves
            # it from a uint8 pool and binds it as fp8.
            fp8_index = self.index_cache_dtype == "fp8"
            index_width = INDEX_PAGE_ROW if fp8_index else INDEX_DIM
            index_dtypes = (
                {torch.uint8}
                | {
                    dtype
                    for dtype in (
                        getattr(torch, "float8_e4m3fn", None),
                        getattr(torch, "float8_e4m3fnuz", None),
                    )
                    if dtype is not None
                }
                if fp8_index
                else {torch.bfloat16}
            )
            if (
                index_cache.ndim not in (2, 3)
                or cache_rows < self.index_max_seq
                or index_cache.shape[-1] != index_width
                or (
                    index_cache.ndim == 3
                    and index_cache.shape[1] != self.index_page_size
                )
                or index_cache.dtype not in index_dtypes
                or not index_cache.is_contiguous()
            ):
                raise ValueError(
                    f"index_cache must be contiguous {self.index_cache_dtype} "
                    f"[physical rows >= {self.index_max_seq}, {index_width}], got "
                    f"{tuple(index_cache.shape)} {index_cache.dtype}"
                )
            if self.kv_cache_layout is KvCacheLayout.ATOM:
                block_tables = (
                    self.identity_block_tables if block_tables is None else block_tables
                )
                expected_blocks = (
                    self.S // self.index_request_width,
                    self.index_block_table_stride,
                )
                if (
                    block_tables is None
                    or block_tables.ndim != 2
                    or block_tables.shape[0] < expected_blocks[0]
                    or block_tables.shape[1] != expected_blocks[1]
                    or block_tables.dtype is not torch.int32
                    or not block_tables.is_contiguous()
                ):
                    raise ValueError(
                        f"block_tables must be contiguous int32 {expected_blocks} for fused indexer"
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
                p(block_tables)
                if (self.with_indexer or self.shared_indices)
                and self.kv_cache_layout is KvCacheLayout.ATOM
                else p(cur_pos),
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
                (p(self.index_params) if self.index_params is not None else 0),
                p(self.step),
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
            result["index_q"] = self.debug("index_q", (S, 32, INDEX_DIM), bf2=True)
            result["index_w"] = self.debug("index_w", (S, 32))
            off = self.scr_layout["indices"]
            result["indices"] = (
                self.scratch[off : off + S * self.topk * 4]
                .view(torch.int32)
                .view(S, self.topk)
            )
        return result

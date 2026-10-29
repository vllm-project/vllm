# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One decode step of 1..MAX_TOKENS tokens on the mono kernels.

Every sparse MoE layer is one launch of the fused layer kernel; it takes
``(ar, res)`` -- the pair vLLM's input norm would receive (reduced MoE output,
residual) -- and leaves ``(ar, h_mid)`` for the next layer, alternating between
two buffers by layer parity. The embedding, the leading dense-FFN layers and the
final norm run the original modules. The K/V cache is read in vLLM's interleaved
page-16 layout under the fixed unit fp8 scale; the step's tables come from the
forward context.
"""

from typing import Any, NamedTuple, cast

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.experts import rocm_aiter_moe
from vllm.models.minimax_m3.amd.mono import config as mono_config
from vllm.models.minimax_m3.amd.mono.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    N_ROUTED,
    PAGE16,
    TOP_K,
    TOPK_BLOCKS,
    TP,
    IndexHeads,
    MonoUnsupported,
)
from vllm.models.minimax_m3.amd.mono.kernels.post_attn import build_post_attn_kernel
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import K1_ARGS
from vllm.models.minimax_m3.amd.mono.kernels.pre_attn import (
    SCRATCH_BYTES as K1_SCRATCH_BYTES,
)
from vllm.models.minimax_m3.amd.mono.layout import SCRATCH_BYTES as K4_SCRATCH_BYTES
from vllm.models.minimax_m3.amd.mono.layout import sym_layout
from vllm.models.minimax_m3.amd.mono.peer_buffer import PeerBuffer
from vllm.models.minimax_m3.amd.mono.weights import SparseMoeLayer, build_sparse_layer
from vllm.platforms import current_platform


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


class StepTables(NamedTuple):
    """The per-step inputs every layer launch of a decode step shares."""

    block_table: torch.Tensor  # int32 [rows, width], one row per token
    seq_lens: torch.Tensor  # int32 [rows]
    slot_mapping: torch.Tensor  # int64, page-128 slots (K/V and index cache)
    q_len: int  # query tokens per request (speculative verify rows)


def token_rows(
    block_table: torch.Tensor, seq_lens: torch.Tensor, q: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One (block table row, seq_len) per token of the step: a request of q query
    tokens becomes q rows, token j seeing ``seq_len - (q - 1 - j)`` keys."""
    if q == 1:
        return block_table, seq_lens
    reqs = n // q
    back = torch.arange(q - 1, -1, -1, dtype=seq_lens.dtype, device=seq_lens.device)
    return (
        block_table[:reqs].repeat_interleave(q, dim=0),
        (seq_lens[:reqs, None] - back).reshape(-1),
    )


def dense_prefix(layers) -> int:
    """Leading layers without an MoE FFN; every later layer must be sparse MoE."""
    n = 0
    while n < len(layers) and not layers[n].is_moe_layer:
        n += 1
    if n == len(layers) or not all(
        lyr.is_moe_layer and hasattr(lyr.self_attn, "indexer") for lyr in layers[n:]
    ):
        raise MonoUnsupported("layer mix: dense layers must precede the MoE layers")
    return n


def _fused_shared_weight() -> float:
    """The fused shared expert's slot weight, from aiter's routing metadata."""
    meta = rocm_aiter_moe.aiter_topK_meta_data
    if meta is None:
        raise MonoUnsupported("fused shared expert routing metadata")
    total_w, total_ids = meta
    if int(total_ids[0, TOP_K].item()) != N_ROUTED:
        raise MonoUnsupported("fused shared expert id")
    return float(total_w[0, TOP_K].item())


class MonoDecodeRunner:
    """Owns the per-layer weight views, scratch, peer buffers and compiled kernels."""

    def __init__(self, model, config) -> None:
        npes = get_tensor_model_parallel_world_size()
        if npes != TP:
            raise MonoUnsupported(f"TP {npes}")
        self.model = model
        self.rank = get_tensor_model_parallel_rank()
        layers = list(model.layers)
        # Enable indexer context parallelism: each rank computes all TP index
        # heads, allowing local block scoring without cross-rank exchange.
        heads = IndexHeads(count=TP, own=self.rank)
        tp_group = get_tp_group().device_group
        self.n_dense = n_dense = dense_prefix(layers)
        self.sparse = [
            build_sparse_layer(layers[i], i, heads, rank=self.rank, tp_group=tp_group)
            for i in range(n_dense, len(layers))
        ]
        attn0 = layers[n_dense].self_attn
        moe0 = layers[n_dense].mlp
        self.shared_fp8 = self.sparse[0].shared_fp8
        if any(lw.shared_fp8 != self.shared_fp8 for lw in self.sparse):
            raise MonoUnsupported("shared expert fused in some layers only")
        # a separate MXFP8 shared expert adds unscaled: routed_scaling_factor goes
        # on the routed weights alone (route_scale), as the model's runner does
        shared_weight = 1.0 if self.shared_fp8 else _fused_shared_weight()
        self.sm_scale = attn0.scaling
        if abs(self.sm_scale - HEAD_DIM**-0.5) > 1e-12:
            raise MonoUnsupported(f"softmax scale {self.sm_scale}")
        sel = attn0.indexer.impl
        if sel.topk_blocks != TOPK_BLOCKS:
            raise MonoUnsupported(f"indexer top-{sel.topk_blocks} blocks")
        self.layer_name = attn0.layer_name
        self.index_prefix = attn0.indexer.index_cache.prefix
        # the caches this build viewed: vLLM re-binds them after its cudagraph
        # memory-profiling capture, which must rebuild the runner
        self._cache_ptrs = self.cache_ptrs(model, n_dense)
        # the K/V page views (V offset to its side of the block) and how many page
        # ids a block spans; the kernels are compiled for that geometry
        self.kv_views = [lyr.self_attn.get_aiter_sparse_pa_kv_cache() for lyr in layers]
        stride = attn0._aiter_sparse_pa_block_page_stride
        mono_config.configure(kv_page_stride=stride, static_kv_scale=True)
        self._build(
            npes, heads, float(config.rms_norm_eps), float(moe0.routed_scaling_factor),
            shared_weight,
            float(getattr(moe0.experts, "routed_experts", moe0.experts).swiglu_limit),
            sel.init_blocks, sel.local_blocks,
        )  # fmt: skip
        dev = self.h.device
        n16 = self.kv_views[0][0].shape[0]
        # the host's unit fp8 scale for every page-16 slot (K1 writes 1.0 back)
        self.kv_scale = torch.ones(n16, 1, PAGE16, dtype=torch.float32, device=dev)
        self.res0 = torch.zeros(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=dev)

    @staticmethod
    def cache_ptrs(model, n_dense: int) -> tuple[int, int]:
        """Addresses of the first MoE layer's KV cache and index cache."""
        attn = model.layers[n_dense].self_attn
        return attn.kv_cache.data_ptr(), attn.indexer.index_cache.kv_cache.data_ptr()

    def stale(self) -> bool:
        """Whether vLLM re-bound the caches this runner viewed (profiling capture)."""
        return self.cache_ptrs(self.model, self.n_dense) != self._cache_ptrs

    def step_tables(self, n: int) -> StepTables:
        """The current step's tables for ``n`` tokens, from the forward context."""
        fwd = get_forward_context()
        decode = cast(Any, fwd.attn_metadata[self.layer_name]).decode
        q = decode.decode_query_len
        block_table, seq_lens = token_rows(decode.block_table, decode.seq_lens, q, n)
        return StepTables(block_table, seq_lens, fwd.slot_mapping[self.layer_name], q)

    def start(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        aux_hidden_states: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Embedding and the dense layers on the original modules; leaves the first
        MoE layer's (ar, res) in ``ars[0][:n]`` and the returned residual (zeros
        when layer 0 is already an MoE layer: (embeddings, 0))."""
        n = positions.numel()
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else self.model.embed_input_ids(input_ids)
        )
        residual = None
        if aux_hidden_states is not None:
            self.model._maybe_add_hidden_state(aux_hidden_states, 0, hidden, residual)
        for i in range(self.n_dense):
            hidden, residual = self.model.layers[i](positions, hidden, residual)
            if aux_hidden_states is not None:
                self.model._maybe_add_hidden_state(
                    aux_hidden_states, i + 1, hidden, residual
                )
        self.ars[0][:n].copy_(hidden.view(n, HIDDEN))
        return self.res0[:n] if residual is None else residual.view(n, HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """The final hidden states of one decode step (what the model returns)."""
        aux_hidden_states: list[torch.Tensor] | None = (
            [] if self.model.aux_hidden_state_layers else None
        )
        res = self.start(input_ids, positions, inputs_embeds, aux_hidden_states)
        tables = self.step_tables(res.shape[0])
        for i, lw in enumerate(self.sparse):
            res = self.run_sparse_layer(i, lw, tables, positions, res)
            if aux_hidden_states is not None:
                self.model._maybe_add_hidden_state(
                    aux_hidden_states,
                    lw.layer_id + 1,
                    self.ars[(i + 1) % 2][: res.shape[0]],
                    res,
                )
        hidden = self.run_final_norm(res)
        if aux_hidden_states is not None:
            return hidden, aux_hidden_states
        return hidden

    def run_sparse_layer(
        self,
        i: int,
        lw: SparseMoeLayer,
        tables: StepTables,
        positions: torch.Tensor,
        res: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse layer `i`: (`ars[i % 2][:n]`, res) -> (`ars[(i + 1) % 2][:n]`, the
        returned residual)."""
        n = res.shape[0]
        block_table, seq_lens, slots, q_len = tables
        k16, v16 = self.kv_views[lw.layer_id]
        self.k4[n](
            _ptr(self.h), _ptr(self.q), _ptr(block_table), _ptr(seq_lens),
            _ptr(k16), _ptr(v16), _ptr(self.kv_scale), _ptr(self.kv_scale),
            _ptr(lw.w_o), _ptr(lw.s_o), _ptr(lw.g_post),
            _ptr(lw.gate), _ptr(lw.bias), _ptr(lw.w13), _ptr(lw.s13),
            _ptr(lw.w2), _ptr(lw.s2),
            _ptr(lw.sh_args) if lw.shared_fp8 else 0,
            _ptr(self.h_mids[(i + 1) % 2]), _ptr(self.ars[(i + 1) % 2]),
            _ptr(self.scratch4), self.peers.local, _ptr(self.peers.addresses),
            _ptr(self.step), self.rank, lw.layer_id,
            block_table.shape[1], q_len,
            0,  # timeline stamps (kernel debug build only)
            _ptr(self.k1_args[i]), _ptr(positions), _ptr(slots), _ptr(res),
            stream=torch.cuda.current_stream(),  # flydsl rejects torch.Stream
        )  # fmt: skip
        return self.h_mids[(i + 1) % 2][:n]

    def _build(
        self, npes, heads, eps, route_scale, shared_weight, swiglu_limit,
        init_blocks, local_blocks,
    ) -> None:  # fmt: skip
        """Compile the layer kernels and allocate scratch, peer buffers and the
        per-step activation buffers; ``self.sparse`` and ``self.sm_scale`` are set."""
        device_index = torch.accelerator.current_device_index()
        dev = torch.device("cuda", device_index)
        cus = current_platform.get_device_properties(device_index).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(
                f"{cus} CUs: the kernels need all {BLOCKS} CTAs co-resident"
            )
        # one layer kernel per decode batch size (every graph captures its own)
        self.k4 = {
            s: build_post_attn_kernel(
                npes,
                self.sm_scale,
                eps,
                route_scale,
                shared_weight,
                swiglu_limit,
                init_blocks,
                local_blocks,
                s,
                fuse_k1=True,
                heads=heads,
                shared_fp8=self.shared_fp8,
            )
            for s in range(1, MAX_TOKENS + 1)
        }
        self.scratch1 = torch.zeros(K1_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.scratch4 = torch.zeros(K4_SCRATCH_BYTES, dtype=torch.uint8, device=dev)
        self.peers = PeerBuffer(
            sym_layout(npes)["_bytes"], get_tp_group().cpu_group, self.rank, npes, dev
        )
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        bf16 = torch.bfloat16
        # row k = token k of the step; sparse layer i reads ars[i % 2] (the
        # previous layer's output) and writes ars[(i + 1) % 2], likewise h_mids
        self.ars = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h_mids = [
            torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev) for _ in range(2)
        ]
        self.h = torch.empty(MAX_TOKENS, HIDDEN, dtype=bf16, device=dev)
        self.q = torch.empty(
            MAX_TOKENS, LOCAL_Q_HEADS * HEAD_DIM, dtype=bf16, device=dev
        )
        self.iq = torch.empty(MAX_TOKENS, 1, HEAD_DIM, dtype=bf16, device=dev)
        # each layer's K1 pointers that do not change per step (K1_ARGS order)
        self.k1_args = []
        for i, lw in enumerate(self.sparse):
            ptrs = {
                "ar": _ptr(self.ars[i % 2]), "g_in": _ptr(lw.g_in),
                "w_qkv": _ptr(lw.w_qkv),
                "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik),
                "cos_sin": _ptr(lw.cos_sin),
                "index_cache": _ptr(lw.index_cache), "iq_out": _ptr(self.iq),
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )

    def close(self) -> None:
        self.peers.close()

    def run_final_norm(self, res: torch.Tensor) -> torch.Tensor:
        ar = self.ars[len(self.sparse) % 2]
        hidden, _ = self.model.norm(ar[: res.shape[0]], res)
        self.step.add_(1)
        return hidden

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One sparse layer of a decode step of 1..MAX_TOKENS tokens on the mono kernels
(a speculative verify's q tokens of a request count as q, see ``token_rows``).

A layer is one launch of K4 with its K1 fused in front of its stages::

    K1 part        (ar, res) -> h, q; K / V / index cache insert; indexer scores
    K4 part        top-k, attention .. FFN, both all-reduces in-kernel
                   -> (ar, h_mid)

with ``(ar, res)`` the (reduced partials, residual) pair the original path hands
its input norm, so a layer computes the same values either way. A launch reads
the previous layer's (ar, h_mid) while it writes its own, so both alternate
between two buffers by layer parity -- which is baked into each layer's argument
array at construction, so these buffers must never be reallocated.

The embedding, dense layers 0..2 and the final norm run the original modules, on
the model's own forward: ``run_layer`` takes and returns exactly the pair a
decoder layer does, so the model's loop drives either path. ``MonoDecodeLayer``
is where a layer chooses.
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.models.minimax_m3.amd.mono.config import (
    BLOCKS,
    HEAD_DIM,
    HIDDEN,
    LOCAL_Q_HEADS,
    MAX_TOKENS,
    ONE_INDEX_HEAD,
    PAGE16_SIDES,
    SHARED_EXPERT,
    SPARSE_BLOCK,
    TOP_K,
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
from vllm.models.minimax_m3.amd.mono.weights import SparseMoeLayer, sparse_selection
from vllm.models.minimax_m3.amd.ops.sparse_pa import _sides_are_packed

N_DENSE = 3
SWIGLU_ALPHA = 1.702  # aiter's swiglu_mul_batch bakes this in
FP8_CACHE = ("fp8", "fp8_e4m3")  # cache dtypes the kernels read as e4m3 bytes

# Runners a custom op can reach by name. A schema carries tensors and plain
# values, never a Python object, so an op takes the name and looks its runner up
# here -- as vLLM's attention op resolves its layer from a layer name.
_RUNNERS: dict[str, "MonoDecodeRunner"] = {}


def runner_of(name: str) -> "MonoDecodeRunner":
    """The runner registered under ``name``."""
    return _RUNNERS[name]


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def token_rows(decode, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(block table, seq_lens) with one row per token of the step.

    A request of q query tokens (a speculative verify) becomes q rows, its token
    j seeing ``seq_len - (q - 1 - j)`` keys, so the kernels treat every token
    alike and need no per-row query length.
    """
    q = decode.decode_query_len
    if q == 1:
        return decode.block_table, decode.seq_lens
    reqs = n // q
    back = torch.arange(
        q - 1, -1, -1, dtype=decode.seq_lens.dtype, device=decode.seq_lens.device
    )
    return (
        decode.block_table[:reqs].repeat_interleave(q, dim=0),
        (decode.seq_lens[:reqs, None] - back).reshape(-1),
    )


class MonoDecodeRunner:
    """Owns the per-layer weight views, scratch, peer buffers and compiled kernels."""

    def __init__(self, model, name: str = "mono") -> None:
        config = model.config
        npes = get_tensor_model_parallel_world_size()
        _need(npes == TP, f"TP {npes}, kernels are built for {TP}")
        layers = model.layers
        _need(
            model.start_layer == 0 and model.end_layer == len(layers),
            f"pipelined: layers {model.start_layer}..{model.end_layer} of "
            f"{len(layers)}",
        )
        _need(
            len(layers) > N_DENSE
            and not any(
                getattr(layers[i], "is_moe_layer", False)
                or getattr(layers[i].self_attn, "indexer", None) is not None
                for i in range(N_DENSE)
            ),
            f"layers 0..{N_DENSE - 1} must be dense full-attention layers",
        )
        self.model = model
        self.rank = get_tensor_model_parallel_rank()

        attn = layers[N_DENSE].self_attn
        # indexer context parallelism: the fused projection holds every index q
        # head, the rank's own being its TP rank. Absent on a build that does not
        # offer it, where a rank projects and scores its own head alone.
        cp = getattr(attn, "indexer_cp", False)
        heads = IndexHeads(TP, self.rank) if cp else ONE_INDEX_HEAD
        self.sparse = [
            SparseMoeLayer.from_layer(layers[i], i, heads)
            for i in range(N_DENSE, len(layers))
        ]
        self._check_dense_all_reduce(layers)
        self._check_kv_layout(attn)

        _need(
            abs(config.swiglu_alpha - SWIGLU_ALPHA) < 1e-6,
            f"swiglu alpha {config.swiglu_alpha}, kernels use {SWIGLU_ALPHA}",
        )
        self.sm_scale = attn.scaling
        _need(
            abs(self.sm_scale - HEAD_DIM**-0.5) < 1e-12,
            f"softmax scale {self.sm_scale}",
        )
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        cus = torch.cuda.get_device_properties(dev).multi_processor_count
        _need(
            cus == BLOCKS,
            f"{cus} CUs: the kernels need all {BLOCKS} CTAs co-resident",
        )

        moe = layers[N_DENSE].block_sparse_moe
        _, init_blocks, local_blocks = sparse_selection(self.sparse[0].indexer, N_DENSE)
        # one layer kernel per decode batch size (every graph captures its own)
        self.k4 = {
            s: build_post_attn_kernel(
                npes,
                self.sm_scale,
                config.rms_norm_eps,
                float(moe.routed_scaling_factor),
                self._shared_expert_weight(),
                float(config.swiglu_limit),
                init_blocks,
                local_blocks,
                s,
                fuse_k1=True,
                heads=heads,
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
                "w_qkv": _ptr(lw.w_qkv), "s_qkv": _ptr(lw.s_qkv),
                "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
                "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik),
                "cos_sin": _ptr(lw.cos_sin), "iq_out": _ptr(self.iq),
                "index_cache": _ptr(lw.indexer.index_cache.kv_cache),
                "scratch": _ptr(self.scratch1),
            }  # fmt: skip
            self.k1_args.append(
                torch.tensor([ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev)
            )
        self.name = name
        _RUNNERS[name] = self

    @staticmethod
    def _check_dense_all_reduce(layers) -> None:
        """The dense MLPs must reduce their own output.

        K1 expects ``ars[0]`` already summed over the TP group. ATOM's dense MLP
        leaves the reduce to the next layer's AR-fused input norm, but vLLM's
        input norm is plain -- its fused all-reduce sits after attention instead --
        so the reduce has to have happened by the time ``run_dense`` returns.
        """
        for i in range(N_DENSE):
            down = layers[i].mlp.down_proj
            _need(
                getattr(down, "reduce_results", False),
                f"layer {i}: dense MLP defers its all-reduce",
            )

    @staticmethod
    def _check_kv_layout(attn) -> None:
        """The caches must be paged and quantized the way the kernels read them."""
        # Both are written as e4m3 with no bf16 instantiation to fall back on, so a
        # cache of any other dtype would be read as e4m3 bytes rather than refused.
        _need(
            attn.kv_cache_dtype in FP8_CACHE,
            f"KV cache {attn.kv_cache_dtype}, kernels write e4m3",
        )
        _need(
            attn.indexer_kv_dtype in FP8_CACHE,
            f"index cache {attn.indexer_kv_dtype}, kernels write e4m3",
        )
        k_cache, v_cache = attn.get_aiter_sparse_pa_kv_cache()
        sides = 2 if _sides_are_packed(k_cache, v_cache) else 1
        _need(
            sides == PAGE16_SIDES,
            f"KV cache spans {sides} side(s) a block, built for {PAGE16_SIDES}",
        )
        blocks, _, block_size, _ = attn.kv_cache.shape
        # a build constant, since it sets the sparse selection's granularity and
        # how ``kv_page`` splits a slot; a server started with another block size
        # would address both caches wrongly rather than merely slower
        _need(
            block_size == SPARSE_BLOCK,
            f"KV cache blocks of {block_size}, kernels index {SPARSE_BLOCK}",
        )
        index_cache = attn.indexer.index_cache.kv_cache
        _need(
            index_cache.shape[:2] == (blocks, block_size),
            f"index cache {tuple(index_cache.shape)} does not share the main "
            f"cache's {blocks} blocks of {block_size}",
        )

    @staticmethod
    def _shared_expert_weight() -> float:
        """The routing weight the fused shared expert is given every step.

        Reads it off the topK metadata rather than the config, since that is what
        the router actually hands the MoE kernel. A device sync, so construction
        must not happen under graph capture.
        """
        from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
            aiter_topK_meta_data,
        )

        _need(aiter_topK_meta_data is not None, "no AITER topK metadata")
        total_w, total_ids = aiter_topK_meta_data
        _need(
            int(total_ids[0, TOP_K].item()) == SHARED_EXPERT,
            f"fused shared expert id {int(total_ids[0, TOP_K].item())}",
        )
        return float(total_w[0, TOP_K].item())

    def close(self) -> None:
        _RUNNERS.pop(self.name, None)
        self.peers.close()

    def metadata_of(self, fwd, lw: SparseMoeLayer):
        return fwd.attn_metadata[lw.attn.layer_name]

    def run_layer(
        self,
        i: int,
        fwd,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        rows: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse layer ``i`` in a decoder layer's own terms.

        ``(hidden_states, residual)`` is the kernels' ``(ar, res)``: the pair the
        original layer hands its input norm, and the pair the final norm sums. So a
        layer returns what the original returns and the model's loop drives either.

        ``ar`` has to be in ``ars[i % 2]``, which it already is when the preceding
        kernel left it there; the copy is for a step that reached this layer from
        the original modules instead.
        """
        n = hidden_states.shape[0]
        ar = self.ars[i % 2]
        if hidden_states.data_ptr() != ar.data_ptr():
            ar[:n].copy_(hidden_states.view(n, HIDDEN))
        res = self.run_sparse_layer(
            i, self.sparse[i], fwd, positions, residual.view(n, HIDDEN), rows
        )
        if i == len(self.sparse) - 1:
            # Opens the next step's epoch, which is what keeps a layer's mailbox
            # tags distinct from the same layer's tags one step ago. Enqueued after
            # the last launch of this step, so no kernel can see it early.
            self.step.add_(1)
        return self.ars[(i + 1) % 2][:n], res

    def run_sparse_layer(
        self,
        i: int,
        lw: SparseMoeLayer,
        fwd,
        positions: torch.Tensor,
        res: torch.Tensor,
        rows: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Sparse layer i: (ars[i % 2][:n], res) -> (ars[(i + 1) % 2][:n], returned
        residual). ``rows``: ``token_rows``."""
        n = res.shape[0]
        block_table, seq_lens = rows
        attn = lw.attn
        md = self.metadata_of(fwd, lw)
        # the plain slot mapping and the logical block table: kv_page reproduces
        # vLLM's host-side page-16 rebase inside the kernel
        slot_mapping = fwd.slot_mapping[attn.layer_name]
        k16, v16 = attn.get_aiter_sparse_pa_kv_cache()
        self.k4[n](
            _ptr(self.h), _ptr(self.q), _ptr(block_table), _ptr(seq_lens),
            _ptr(k16), _ptr(v16), _ptr(attn._k_scale), _ptr(attn._v_scale),
            _ptr(lw.w_o), _ptr(lw.s_o), _ptr(lw.g_post), _ptr(lw.gate),
            _ptr(lw.bias), _ptr(lw.w13), _ptr(lw.s13), _ptr(lw.w2), _ptr(lw.s2),
            _ptr(self.h_mids[(i + 1) % 2]), _ptr(self.ars[(i + 1) % 2]),
            _ptr(self.scratch4), self.peers.local, _ptr(self.peers.addresses),
            _ptr(self.step), self.rank, lw.layer_id,
            block_table.shape[1], md.decode.decode_query_len, 0,
            _ptr(self.k1_args[i]), _ptr(positions), _ptr(slot_mapping), _ptr(res),
            stream=torch.cuda.current_stream(),
        )  # fmt: skip
        return self.h_mids[(i + 1) % 2][:n]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank launch state of the Mistral Large 4 specialized decode.

The embedding and the dense layer 0 run the original modules; every MoE layer
(1..71) is one launch mapping the residual stream ``x`` ([rows, hidden]) to the
next layer's ``x`` (attention, MoE and both TP all-reduces inside the kernel),
alternating between two buffers; the final norm runs the original module. Each
row is one request's single decode token. The kernels read and write vLLM's
own caches: the bf16 MLA latent pages ([blocks, page, 576]) and the DSA
indexer's FP8 key pages ([blocks, page, 132]), through the step's block table
(one row per request) and slot mapping. A padded row (slot ``PAD_SLOT_ID``)
attends to no keys and writes nothing.

Built in two parts: the kernel weights after load (``__init__``, before vLLM's
memory profiling), the launchers (one per row count and layer kind), scratch
and peer buffer on the first decode step (``bind``: they depend on the cache
page and the block-table width).
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.mistral_large4.amd.specialized_decode.mono.common.runtime.peer_memory import (  # noqa: E501
    SymmetricPeerBuffer,
)

from .spec import DecodeConfig
from .variants import load_variant
from .weights import KINDS, LayerKind, MoeLayer, build_moe_layer, moe_kinds

logger = init_logger(__name__)

HIDDEN = 6144
FAULT_WORD_BYTES = 64

# launch-order weight arguments (see mono/op.py Glm5MonoKernel.forward)
_WEIGHT_ARGS = (
    "g_in", "g_q", "g_kv", "g_post", "w_qkv_a", "s_qkv_a", "w_q_b", "s_q_b",
    "w_uk", "s_uk", "w_uv", "s_uv", "w_o", "s_o", "w_r", "bias",
    "w_ug", "s_ug", "w_dn", "s_dn",
)  # fmt: skip


def _ptr(t: torch.Tensor) -> int:
    return t.data_ptr()


def _variant(kind: str):
    nope, kvs, _ = KINDS[kind]
    return load_variant(nope, kvs)


def _tp_agreed(error: str | None, group, npes: int) -> list[str]:
    errors: list = [None] * npes
    dist.all_gather_object(errors, error, group=group)
    return [f"rank {r}: {e}" for r, e in enumerate(errors) if e is not None]


def _kv_scale_refusal(moe) -> str | None:
    """The kernel stores and reads an fp8 latent cache unscaled: every fp8
    layer's KV scale must be 1.0."""
    scales = {
        float(getattr(lw.attn.attn, "_k_scale_float", 1.0))
        for lw in moe
        if str(lw.attn.attn.kv_cache_dtype).startswith("fp8")
    }
    if scales <= {1.0}:
        return None
    return (
        f"fp8 latent KV cache with KV scales {sorted(scales)}: the specialized "
        "decode reads and writes it unscaled (scale 1.0)"
    )


class MonoRunner:
    """The per-layer kernel arguments, scratch, peer buffer and launchers."""

    def __init__(self, model, config: DecodeConfig, max_model_len: int) -> None:
        self.npes = npes = get_tensor_model_parallel_world_size()
        self.model = model
        self.config = config
        self.widths = tuple(config.widths)
        self.max_rows = max(self.widths)
        self.max_model_len = max_model_len
        self.rank = get_tensor_model_parallel_rank()
        self.cpu_group = get_tp_group().cpu_group
        layers = list(model.layers)
        self.kinds = kinds = moe_kinds(layers)
        self.heads = layers[1].self_attn.num_local_heads
        self.layer_name = layers[1].self_attn.attn.layer_name
        self.index_name = layers[1].self_attn.indexer.k_cache.prefix
        cs = model.rotary_emb.cos_sin_cache
        if cs.shape[0] < max_model_len or cs.shape[1] != 64:
            raise ValueError(f"RoPE table {tuple(cs.shape)} for {max_model_len}")
        cs = cs.float()
        self.cos = cs[:, :32].contiguous()
        self.sin = cs[:, 32:].contiguous()
        error = None
        try:
            self.moe: list[MoeLayer] = [
                build_moe_layer(layers[i], _variant(kind), self.rank, npes)
                for i, kind in zip(range(1, len(layers)), kinds)
            ]
            torch.accelerator.synchronize()
        except Exception as why:  # e.g. OOM on one rank: every rank raises
            logger.exception("Mistral Large 4 specialized decode: weight build failed")
            error = f"{type(why).__name__}: {why}"
        failed = _tp_agreed(error, self.cpu_group, npes)
        if failed:
            raise RuntimeError(
                "Mistral Large 4 specialized decode weight build failed: "
                + "; ".join(failed)
            )
        # the KV scales are final once the weights are loaded: refuse at startup,
        # not on the first decode step
        failed = _tp_agreed(_kv_scale_refusal(self.moe), self.cpu_group, npes)
        if failed:
            raise ValueError(
                "Mistral Large 4 specialized decode unsupported: " + "; ".join(failed)
            )
        repacked = sum(
            t.numel() * t.element_size()
            for lw in self.moe
            for name, t in lw.args.items()
            if not name.startswith(("w_ug", "s_ug", "w_dn", "s_dn"))
            or name.endswith("_sh")
        )
        logger.info(
            "Mistral Large 4 specialized decode: %d layer launches (%s), rows %s, "
            "%.2f GiB of repacked weights per rank",
            len(self.moe),
            ", ".join(f"{k} x{kinds.count(k)}" for k in sorted(set(kinds))),
            self.widths,
            repacked / 2**30,
        )
        self.block_table_stride: int | None = None
        self.peers: SymmetricPeerBuffer | None = None
        dev = layers[0].input_layernorm.weight.device
        # the fail-stop word reads 0 until the peer buffer exists
        self.fault_word = torch.zeros(1, dtype=torch.int32, device=dev)

    def _kv_cache_dtype(self) -> str:
        """The kernel's latent cache dtype, from the bound caches: ``bf16`` or
        ``fp8`` (E4M3, ATOM layout)."""
        # vLLM allocates an fp8 cache as uint8 and views it as fp8 per call
        fp8 = {torch.float8_e4m3fn, torch.uint8}
        dtypes = {lw.attn.attn.kv_cache.dtype for lw in self.moe}
        if dtypes == {torch.bfloat16}:
            return "bf16"
        if len(dtypes) != 1 or not dtypes <= fp8:
            raise ValueError(f"unsupported latent KV cache dtypes {dtypes}")
        return "fp8"

    @property
    def bound(self) -> bool:
        return self.block_table_stride is not None

    def bind(self, page: int, block_table_stride: int) -> None:
        """Launchers, scratch and peer buffer for the bound caches (collective:
        every TP rank calls it on the same step, outside graph capture)."""
        assert not self.bound
        npes, heads, cfg = self.npes, self.heads, self.config
        kv_cache_dtype = self._kv_cache_dtype()
        unit = math.lcm(64, page)
        index_max_seq = block_table_stride * page // unit * unit
        if index_max_seq < self.max_model_len:
            raise ValueError(
                f"block table covers {index_max_seq} < max_model_len "
                f"{self.max_model_len}"
            )
        build_kwargs = dict(
            heads=heads, npes=npes, topk=cfg.topk,
            launches_per_step=len(self.moe), index_max_seq=index_max_seq,
            index_request_width=cfg.q, index_page_size=page,
            index_block_table_stride=block_table_stride, index_cache_dtype="fp8",
            expert_mxfp4=True, atom_experts=True, attention_weight="fp8_block128",
            kv_cache_layout="atom", kv_cache_dtype=kv_cache_dtype, native_fp4_mfma=True,
            rope_dtype="fp32", uv_scale_rows=128, separate_shared=True,
        )  # fmt: skip
        scratch_bytes: dict[tuple[bool, bool], int] = {}
        sym_bytes = 0
        self.indices_offset: dict[int, int] = {}
        builds = []
        for rows in self.widths:
            sym: dict[str, int] = {}
            for kind in sorted(set(self.kinds)):
                nope, kvs, with_indexer = KINDS[kind]
                v = _variant(kind)
                inter = v.geometry_config.glm5_tp_config(npes).inter
                scr, sym_l = v.layout.layout(
                    rows, heads, npes, cfg.topk, with_indexer, index_max_seq,
                    inter=inter, native_fp4_mfma=True,
                )  # fmt: skip
                builds.append((rows, kind, v, inter, with_indexer))
                key = (nope, kvs)
                scratch_bytes[key] = max(scratch_bytes.get(key, 0), scr["_bytes"])
                if kind == LayerKind.ROPE:
                    self.indices_offset[rows] = scr["indices"]
                for name, off in sym_l.items():
                    if name != "_bytes" and sym.setdefault(name, off) != off:
                        raise ValueError(
                            f"peer layouts disagree on {name} at {rows} rows"
                        )
                sym_bytes = max(sym_bytes, sym_l["_bytes"])
        fault_offset = -(-sym_bytes // FAULT_WORD_BYTES) * FAULT_WORD_BYTES
        self.launch = {
            (rows, kind): v.op.build_glm5_monokernel(
                **build_kwargs,
                S=rows,
                inter=inter,
                output_heads=heads,
                with_indexer=with_indexer,
                shared_indices=kind == LayerKind.ROPE_SHARED,
                fault_offset=fault_offset,
            )  # fmt: skip
            for rows, kind, v, inter, with_indexer in builds
        }
        dev = self.cos.device
        # one rank-local scratch per kernel package (sized for the widest
        # step): mailboxes of different geometries never share bytes; the RoPE
        # package's holds the top-k the index-sharing layers read
        self.scratch = {
            key: torch.zeros(n, dtype=torch.uint8, device=dev)
            for key, n in scratch_bytes.items()
        }
        self.peers = SymmetricPeerBuffer(
            fault_offset + FAULT_WORD_BYTES,
            rank=self.rank,
            npes=npes,
            group=self.cpu_group,
        )
        self.fault_word = self.peers.storage[fault_offset : fault_offset + 4].view(
            torch.int32
        )
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.x = [
            torch.empty(self.max_rows, HIDDEN, dtype=torch.bfloat16, device=dev)
            for _ in range(2)
        ]
        self.indptr = torch.zeros(self.max_rows + 1, dtype=torch.int32, device=dev)
        self.block_table_stride = block_table_stride
        logger.info(
            "Mistral Large 4 specialized decode bound: page %d, index capacity %d, "
            "%d launchers",
            page,
            index_max_seq,
            len(self.launch),
        )

    def close(self) -> None:
        if self.peers is not None:
            self.peers.close()
            self.peers = None

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        assert self.peers is not None
        rows = positions.shape[0]
        assert rows in self.widths
        model = self.model
        hidden = (
            inputs_embeds
            if inputs_embeds is not None
            else model.embed_input_ids(input_ids)
        )
        log_positions = torch.log(positions.to(torch.float32) + 1)
        layer0 = model.layers[0]
        hidden, residual = layer0.input_norm(hidden, None)
        # aux hidden-state taps (DSpark): the residual stream entering layer idx,
        # as _stock_forward takes it after that layer's input norm
        taps = model.aux_hidden_state_layers
        aux_hidden_states: list[torch.Tensor] = []
        if 0 in taps:
            aux_hidden_states.append(residual.clone())
        hidden, residual = layer0(
            positions, hidden, residual, log_positions, model.rotary_emb
        )
        if layer0.defer_allreduce:
            # the dense MLP left a TP-partial sum (its reduction is fused into
            # the next layer's input norm, which the kernels replace)
            hidden = tensor_model_parallel_all_reduce(hidden)
        torch.add(hidden, residual, out=self.x[0][:rows])
        fwd = get_forward_context()
        block_table = fwd.attn_metadata[self.layer_name].block_table
        slots = fwd.slot_mapping[self.layer_name]
        # keys per row (cumulative): the whole context up to top-k, none for a
        # padded row
        nkeys = torch.where(
            slots[:rows] >= 0,
            torch.clamp(positions[:rows] + 1, max=self.config.topk),
            0,
        ).to(torch.int32)
        self.indptr[1 : rows + 1].copy_(torch.cumsum(nkeys, 0))
        rope_scratch = self.scratch[(False, False)]
        stream = torch.cuda.current_stream()
        for i, lw in enumerate(self.moe):
            a = lw.args
            kv = lw.attn.attn.kv_cache
            if lw.kind == LayerKind.ROPE_SHARED:
                index_arg = _ptr(rope_scratch) + self.indices_offset[rows]
            else:
                index_arg = _ptr(lw.attn.indexer.k_cache.kv_cache)
            nope, kvs, _ = KINDS[lw.kind]
            if i + 1 in taps:
                # launch i is layer i + 1 and reads x[i % 2]; launch i + 1
                # overwrites it
                aux_hidden_states.append(self.x[i % 2][:rows].clone())
            self.launch[(rows, lw.kind)](
                _ptr(self.x[i % 2]), _ptr(self.x[(i + 1) % 2]), _ptr(block_table),
                _ptr(positions), _ptr(slots), _ptr(self.indptr),
                _ptr(kv), _ptr(kv), index_arg, _ptr(self.cos), _ptr(self.sin),
                *(_ptr(a[name]) for name in _WEIGHT_ARGS),
                _ptr(self.scratch[(nope, kvs)]), self.peers.local_address,
                _ptr(self.peers.addresses), _ptr(lw.params), _ptr(self.step),
                self.rank, i,
                stream=stream,
            )  # fmt: skip
        self.step.add_(1)
        x = self.x[len(self.moe) % 2][:rows]
        if len(model.layers) in taps:
            aux_hidden_states.append(x.clone())
        hidden = model.norm(x)
        if aux_hidden_states:
            return hidden, aux_hidden_states
        return hidden

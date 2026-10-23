# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3's MTP drafter owned by the model: the MTP layer and its draft head as CuTe
DSL kernels.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from vllm.models.glm53.nvidia import arena as glm53_arena
from vllm.models.glm53.nvidia import decode as glm53_decode
from vllm.models.glm53.nvidia import draft_providers as dp
from vllm.models.glm53.nvidia import weights as w
from vllm.models.glm53.nvidia.ops import launch as glm53_launch
from vllm.models.glm53.nvidia.ops.cute_dsl.fused_moe import LAYOUT
from vllm.models.glm53.nvidia.ops.launch import (
    BLOCK_TABLE,
    Glm53Geometry,
    Region,
    sizes_of,
)

logger = logging.getLogger(__name__)

#: Hidden columns each program of the row-copy kernels moves.
COPY_BLOCK = 1024


@triton.jit
def _catchup_words_kernel(positions, accept, position_out, accept_out, context_out):
    position = tl.load(positions).to(tl.int32)
    accepted = tl.load(accept).to(tl.int32)
    tl.store(position_out, position)
    tl.store(accept_out, accepted)
    tl.store(context_out, position + accepted + 1)
    tl.store(context_out + 1, tl.load(context_out + 1) + 1)


@triton.jit(do_not_specialize=["num_rows"])
def _sync_inputs_kernel(
    last_token_indices,
    positions,
    token_ids,
    rows,
    position_out,
    accept_out,
    context_out,
    num_rows,
    T: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Move an eager draft prefill's last rows (up to T, ending at the sampled row) to
    rows 0...
    """
    last = tl.minimum(tl.load(last_token_indices).to(tl.int32), num_rows - 1)
    accepted = tl.minimum(last, T - 1)
    first = last - accepted
    block = tl.program_id(0)
    columns = block * BLOCK + tl.arange(0, BLOCK)
    for i in range(T):
        if i <= accepted:
            tl.store(
                rows + i * WIDTH + columns,
                tl.load(rows + (first + i) * WIDTH + columns),
            )
    if block == 0:
        for i in range(T):
            if i <= accepted:
                tl.store(token_ids + i, tl.load(token_ids + first + i))
        position = tl.load(positions + first).to(tl.int32)
        tl.store(position_out, position)
        tl.store(accept_out, accepted)
        tl.store(context_out, position + accepted + 1)
        tl.store(context_out + 1, tl.load(context_out + 1) + 1)


@triton.jit
def _proposal_inputs_kernel(
    positions, input_ids, hidden, position_out, token_out, row_out, BLOCK: tl.constexpr
):
    block = tl.program_id(0)
    if block == 0:
        tl.store(position_out, tl.load(positions).to(tl.int32))
        tl.store(token_out, tl.load(input_ids).to(tl.int32))
    columns = block * BLOCK + tl.arange(0, BLOCK)
    tl.store(row_out + columns, tl.load(hidden + columns))


def _scratch_like(tensor: torch.Tensor, numel: int) -> torch.Tensor:
    """Return a zeroed scratch tensor of ``tensor``'s dtype and 16-byte phase (Triton's
    key).
    """
    size = tensor.element_size()
    base = torch.zeros(numel + 16 // size, dtype=tensor.dtype, device=tensor.device)
    shift = (tensor.data_ptr() - base.data_ptr()) % 16 // size
    return base[shift : shift + numel]


class Glm53DraftSegment:
    """One rank's drafter segment: its parts' providers bound to the drafter's peer
    arena.
    """

    def __init__(
        self,
        geometry: Glm53Geometry,
        rank: int,
        tensors: Mapping[str, torch.Tensor],
        group: torch.distributed.ProcessGroup,
        embedding: glm53_arena.PeerArena,
    ) -> None:
        device = tensors["position"].device
        self.placement: glm53_arena.Glm53Placement = glm53_arena.Glm53Placement(device)
        self.parts: list[list[dp.DraftProvider]] = dp.program(
            geometry, rank, tensors, self.placement
        )
        providers = [provider for part in self.parts for provider in part]
        self.arena: glm53_arena.PeerArena = glm53_arena.PeerArena(
            dp.draft_slices(geometry), group
        )
        arenas = (self.arena, embedding)

        def peers(name: str) -> list[int]:
            """Return slice ``name``'s address on every rank (drafter arena or target
            embedding).
            """
            arena = next(arena for arena in arenas if name in arena.offsets)
            return [base + arena.offsets[name] for base in arena.peer_addresses]

        for provider in providers:
            provider.bind(peers)
        self.launches: list[list[glm53_launch.Launch]] = [
            [provider.launch for provider in part if provider.launch is not None]
            for part in self.parts
        ]
        self.bound: list[glm53_launch.Bound] = [
            glm53_launch.bind(part, device.index) for part in self.launches
        ]
        self.state: dict[str, torch.Tensor] = {
            name: tensor
            for provider in providers
            for name, tensor in provider.state.items()
        }
        self.results: dict[str, torch.Tensor] = {
            f"%{provider.site}.{name}": tensor
            for provider in providers
            for name, tensor in provider.results.items()
        }

    @property
    def regions(self) -> dict[str, Region]:
        """Every result and arena slice by name (the multi-rank tests compare them)."""
        results = {
            name: glm53_decode.tensor_region(t) for name, t in self.results.items()
        }
        return {**results, **self.arena.regions()}

    def launch(self, part: int, stream: int) -> None:
        """Launch the catch-up (``part`` -1) or proposal ``part`` on ``stream``
        (capturable).
        """
        glm53_launch.run(self.bound[part + 1], stream)

    def reset_seeds(self) -> None:
        """Reset the catch-up selectors' seeds (stream-ordered)."""
        for provider in self.parts[0]:
            if isinstance(provider, dp.CatchupSelector):
                provider.reset_seeds()

    def close(self) -> None:
        """Release the peer arena (after every graph that launched the segment is
        destroyed) and give the placed MoE flags back to their pools.
        """
        self.arena.close()
        self.placement.release()


class Glm53Draft:
    """The drafter model's hook (module docstring)."""

    normalized: bool = True

    def __init__(
        self,
        drafter: torch.nn.Module,
        geometry: Glm53Geometry,
        forms: Mapping[str, torch.Tensor],
        drafts: int,
    ) -> None:
        self.geometry: Glm53Geometry = geometry
        self.layer: int = dp.mtp_layer(geometry)
        self.rows: int = drafts + 1
        self.drafts: int = drafts
        self._drafter: torch.nn.Module = drafter
        self._forms: dict[str, torch.Tensor] = dict(forms)
        self._segment: Glm53DraftSegment | None = None
        self._tensors: dict[str, torch.Tensor] = {}
        # The target block-table row and layer-78 cache the segment was built over.
        self._table: torch.Tensor | None = None
        self._cache: int = 0
        self._last_token_indices: torch.Tensor | None = None
        self.captures: dict[str, int] = {"prefill": 0, "decode": 0}
        self._tokens: torch.Tensor | None = None
        self._decoding: bool = False
        self._sequence_captures: int = 0
        self._warmed: bool = False
        self._positions: torch.Tensor | None = None

    @property
    def built(self) -> bool:
        """Whether the drafter segment is built."""
        return self._segment is not None

    def observe_slots(self, slot_ids: torch.Tensor) -> None:
        """Keep the speculator's sampled-row indices (handed to
        ``compact_topk_indices``).
        """
        if self._last_token_indices is None:
            self._last_token_indices = slot_ids
        self._warm_up()

    def decode_phase(self, decoding: bool) -> None:
        """Track the speculator's draft-decode phase (``set_skip_topk``)."""
        if not decoding and 0 < self._sequence_captures != self.drafts - 1:
            raise RuntimeError(
                "GLM-5.3 drafter: the draft-decode graph holds one proposal; it needs "
                "vLLM's "
                "fused draft decode"
            )
        self._decoding = decoding
        self._sequence_captures = 0

    def claims_proposals(self) -> bool:
        """Return whether the proposals are this drafter's (always, once captured)."""
        return (
            self._segment is not None
            and self.captures["decode"] > 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def claims_capture(self) -> bool:
        """Return whether a draft forward captured now is this segment's (it is
        built).
        """
        return self._segment is not None and torch.cuda.is_current_stream_capturing()

    def claimed_tokens(self) -> torch.Tensor | None:
        """Return (once) the greedy drafts of the forward just claimed, else None."""
        tokens, self._tokens = self._tokens, None
        return tokens

    def after_stock_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Run the catch-up after an eager (stock) draft prefill: the proposals stay
        valid.
        """
        if (
            self._segment is None
            or self.captures["decode"] == 0
            or self._decoding
            or self._last_token_indices is None
            or torch.cuda.is_current_stream_capturing()
            or self._stale()
        ):
            return
        tensors = self._tensors
        if (
            input_ids.data_ptr() != tensors["token_ids"].data_ptr()
            or hidden_states.data_ptr() != tensors["draft.rows"].data_ptr()
        ):
            raise RuntimeError(
                "GLM-5.3 drafter: an eager draft prefill on other buffers than the "
                "drafter binds"
            )
        self._sync_inputs(input_ids, positions, hidden_states, self._last_token_indices)
        target = glm53_decode.current()
        assert target is not None
        target.sync_block_table()
        self._segment.launch(-1, torch.cuda.current_stream().cuda_stream)

    def _sync_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        words: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Launch :func:`_sync_inputs_kernel` (``words``: the words it writes; default
        bound).
        """
        words = self._tensors if words is None else words
        hidden = self.geometry.hidden
        _sync_inputs_kernel[(hidden // COPY_BLOCK,)](
            last_token_indices,
            positions,
            input_ids,
            hidden_states,
            words["position"],
            words["mtp.accept"],
            words["draft.context"],
            int(input_ids.shape[0]),
            T=self.rows,
            WIDTH=hidden,
            BLOCK=COPY_BLOCK,
        )

    def _warm_up(self) -> None:
        """Compile the eager catch-up's input kernel at startup (scratch tensors, equal
        keys).
        """
        indices, positions = self._last_token_indices, self._positions
        if (
            self._warmed
            or self._segment is None
            or indices is None
            or positions is None
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        tensors = self._tensors
        rows, hidden = self.rows, self.geometry.hidden
        words = {
            name: _scratch_like(tensors[name], 1)
            for name in ("position", "mtp.accept", "draft.context")
        }
        self._sync_inputs(
            _scratch_like(tensors["token_ids"], rows),
            _scratch_like(positions, rows),
            _scratch_like(tensors["draft.rows"], rows * hidden).view(rows, hidden),
            _scratch_like(indices, 1),
            words,
        )
        torch.cuda.current_stream().synchronize()
        self._warmed = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Claim a captured draft prefill or decode; return its row, else None."""
        self._tokens = None
        if input_ids is None or not input_ids.is_cuda:
            return None
        rows = int(input_ids.shape[0])
        if not torch.cuda.is_current_stream_capturing():
            if self._segment is not None and self._stale():
                self.close()
            if self._segment is None and rows == self.rows:
                self._try_build(input_ids, positions, hidden_states)
            elif self._segment is not None:
                self._segment.reset_seeds()
            return None
        if self._segment is None:
            if rows in (1, self.rows):
                logger.warning(
                    "GLM-5.3 drafter: a %d-row capture before the drafter segment was "
                    "built "
                    "(stock)",
                    rows,
                )
            return None
        if self._stale():
            raise RuntimeError(
                "GLM-5.3 drafter: vLLM captures over another KV cache or block table "
                "than the "
                "drafter segment was built for"
            )
        tensors = self._tensors
        stream = torch.cuda.current_stream().cuda_stream
        if rows == self.rows:
            if input_ids.data_ptr() != tensors["token_ids"].data_ptr():
                raise RuntimeError(
                    "GLM-5.3 drafter: vLLM captures other input_ids than the drafter "
                    "binds"
                )
            if self._last_token_indices is None:
                raise RuntimeError(
                    "GLM-5.3 drafter: a draft-prefill capture before an eager propose"
                )
            _catchup_words_kernel[(1,)](
                positions,
                self._last_token_indices,
                tensors["position"],
                tensors["mtp.accept"],
                tensors["draft.context"],
            )
            self._segment.launch(-1, stream)
            self.captures["prefill"] += 1
            self._tokens = tensors["draft.tokens"]
            return tensors["draft.hidden"]
        if rows != 1:
            return None
        proposal = self._sequence_captures
        if proposal >= self.drafts - 1:
            raise RuntimeError(
                f"GLM-5.3 drafter: draft-decode capture {proposal} past the "
                f"{self.drafts - 1} proposals"
            )
        self._sequence_captures += 1
        if hidden_states.is_contiguous():
            _proposal_inputs_kernel[(self.geometry.hidden // COPY_BLOCK,)](
                positions,
                input_ids,
                hidden_states,
                tensors["draft.propose.position"],
                tensors["draft.propose.token"],
                tensors["draft.propose.row"],
                BLOCK=COPY_BLOCK,
            )
        else:
            tensors["draft.propose.position"].copy_(positions[:1])
            tensors["draft.propose.token"].copy_(input_ids[:1])
            tensors["draft.propose.row"].copy_(hidden_states[:1])
        self._segment.launch(proposal, stream)
        self.captures["decode"] += 1
        self._tokens = tensors["draft.propose.tokens"]
        return tensors["draft.propose.hidden"]

    def _layer_caches(
        self,
    ) -> tuple[
        dict[str, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]
    ]:
        """Return vLLM's MTP-layer caches in the forward context (empty before
        allocation).
        """
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return {}, {}, {}
        layers = get_forward_context().no_compile_layers
        caches = glm53_decode.vllm_caches(layers, wanted=(self.layer,))
        mla, index = glm53_decode._layer_caches(caches)
        return caches, mla, index

    def _stale(self) -> bool:
        """Return whether the target segment or vLLM's MTP-layer cache changed since the
        build.
        """
        target = glm53_decode.current()
        if target is None or not target.built:
            return True
        if target.shared_tensors()[BLOCK_TABLE] is not self._table:
            return True
        _, mla, _ = self._layer_caches()
        return self.layer in mla and mla[self.layer].data_ptr() != self._cache

    def _try_build(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Build at the first eager draft prefill once segment 1 is built."""
        from vllm.distributed import get_tp_group
        from vllm.forward_context import get_forward_context

        target = glm53_decode.current()
        if target is None or not target.built:
            return
        caches, mla, index = self._layer_caches()
        layer = self.layer
        if layer not in mla or layer not in index:
            return
        geometry = self.geometry
        group = get_tp_group()
        rank = group.rank_in_group
        device = mla[layer].device
        shared = target.shared_tensors()
        embedding = target.embedding_arena
        if (
            self._drafter.model.embed_tokens.weight.data_ptr()
            != embedding.view(glm53_launch.EMBEDDING_SLICE).data_ptr()
        ):
            raise RuntimeError(
                "GLM-5.3 drafter: its embedding is not the target's shard the "
                "prologues read"
            )
        head = pad_draft_head(self._drafter.model.layers[str(layer)].shared_head.head)
        tensors: dict[str, torch.Tensor] = {
            **self._forms,
            "lm_head": head,
            **shared,
            f"layers.{layer}.attention.kv_cache": mla[layer],
            f"layers.{layer}.attention.index_key_cache": index[layer],
            "token_ids": input_ids[: self.rows],
            "draft.rows": hidden_states[: self.rows],
            **{
                name: torch.zeros(
                    dp.CONTEXT_WORDS if name == "draft.context" else 1,
                    dtype=torch.int32,
                    device=device,
                )
                for name in dp.WORDS
            },
            **{
                name: torch.zeros(
                    (1, geometry.hidden), dtype=torch.bfloat16, device=device
                )
                for name in dp.ROWS
            },
        }
        tensors["draft.context"].fill_(1)
        layers = get_forward_context().no_compile_layers
        glm53_decode.check_mla_caches(
            layers, caches, geometry, int(mla[layer].shape[-2])
        )
        segment = Glm53DraftSegment(
            geometry, rank, tensors, group.device_group, embedding
        )
        target.add_layer(layer, caches, reach=self.drafts)
        self._tensors, self._segment = tensors, segment
        self._table, self._cache = shared[BLOCK_TABLE], mla[layer].data_ptr()
        self._positions = positions
        # Warm every part on the null block (the table zeroed), then restore vLLM's row.
        table = shared[BLOCK_TABLE]
        kept = table.clone()
        table.zero_()
        stream = torch.cuda.current_stream().cuda_stream
        for part in range(-1, self.drafts - 1):
            segment.launch(part, stream)
        table.copy_(kept)
        torch.cuda.current_stream().synchronize()
        self._warm_up()
        logger.info(
            "GLM-5.3 drafter segment built on rank %d (%d proposals)",
            rank,
            self.drafts - 1,
        )

    def close(self) -> None:
        """Drop the segment, its tensors and every capture; the next eager draft prefill
        rebuilds.
        """
        if self._segment is not None:
            torch.cuda.synchronize()
            self._segment.close()
        self._segment = None
        self._tensors = {}
        self._table, self._cache, self._positions = None, 0, None
        self._tokens = None
        self._decoding = False
        self._sequence_captures = 0
        self.captures = {"prefill": 0, "decode": 0}
        self._warmed = False


_CURRENT: Glm53Draft | None = None


def current() -> Glm53Draft | None:
    """Return this process's installed drafter hook, if any."""
    return _CURRENT


def claims_draft_prefill() -> bool:
    """Return whether the drafter replays the T-row draft prefill."""
    draft = _CURRENT
    return draft is not None and draft.built and draft.captures["prefill"] > 0


def install(
    drafter: torch.nn.Module,
    geometry: Glm53Geometry,
    forms: Mapping[str, torch.Tensor] | None,
    drafts: int,
) -> Glm53Draft | None:
    """Install the drafter hook (with the model-owned segment 1)."""
    if forms is None or not glm53_decode.segment_enabled():
        return None
    supported = dp.proposals_of(geometry) + 1
    if drafts != supported:
        raise ValueError(
            f"the GLM-5.3 drafter segment drafts {supported} tokens, not {drafts}"
        )
    global _CURRENT
    draft = Glm53Draft(drafter, geometry, forms, drafts)
    drafter.model.glm53_segment = draft
    _CURRENT = draft
    logger.info(
        "GLM-5.3 drafter segment installed (T = %d, %d proposals)",
        drafts + 1,
        drafts - 1,
    )
    return draft


def draft_linears(geometry: Glm53Geometry) -> tuple[w.VllmLinear, ...]:
    """Return the draft worker's linears read in place."""
    g = geometry
    query = w.VllmLinear(
        "q_b_proj", "query_weight", g.qk_head * g.heads, g.q_lora, g.block
    )
    return (*w.worker_linears(geometry).values(), query)


def pad_draft_head(module: torch.nn.Module) -> torch.Tensor:
    """Pad vLLM's drafter head shard to the head kernel's padded rows (zero rows) in
    place.
    """
    padded = getattr(module, "_glm53_padded", None)
    if padded is not None:
        return padded
    sizes = sizes_of(dp.HEAD_KERNEL)
    rows, hidden = sizes["kVocabShard"], sizes["kHidden"]
    weight = module.weight
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (rows, hidden):
        raise ValueError(
            f"vLLM's drafter head is not a [{rows}, {hidden}] BF16 shard: "
            f"{tuple(weight.shape)}"
        )
    padded = torch.zeros(
        (sizes["kPaddedRows"], hidden), dtype=weight.dtype, device=weight.device
    )
    padded[:rows].copy_(weight.data)
    weight.data = padded[:rows]
    module._glm53_padded = padded
    return padded


def adopt_once(
    linear: w.VllmLinear, module: torch.nn.Module, layer: int
) -> torch.Tensor:
    """Return vLLM's ``module`` of MTP layer ``layer`` moved into one buffer by
    ``linear``.
    """
    buffer = getattr(module, "_glm53_adopted", None)
    if buffer is None:
        buffer = linear.adopt(module, layer)
        module._glm53_adopted = buffer
    return buffer


def selector_forms(
    attention: torch.nn.Module, geometry: Glm53Geometry, gamma_x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the catch-up selector's MTP-layer forms."""
    layer = dp.mtp_layer(geometry)
    indexer = attention.indexer
    return {
        "gamma_x": gamma_x,
        "input_weight": adopt_once(
            w.worker_linears(geometry)["input"], attention.fused_qkv_a_proj, layer
        ),
        "wis_weight": w.wk_weights_proj(indexer, geometry, layer),
        "key_gamma": w._norm(indexer.k_norm).float().contiguous(),
        "key_bias": indexer.k_norm.bias.data.float().contiguous(),
        "gamma_q": w._norm(attention.q_a_layernorm).float().contiguous(),
        "wqb_weight": adopt_once(w.selector_linear(geometry), indexer.wq_b, layer),
    }


def draft_weights(
    drafter: torch.nn.Module, geometry: Glm53Geometry, rank: int
) -> dict[str, torch.Tensor]:
    """Bind this rank's MTP-layer forms to vLLM's drafter tensors in place (one
    copy).
    """
    start = time.perf_counter()
    index = dp.mtp_layer(geometry)
    layer = drafter.model.layers[str(index)]
    block = layer.mtp_block
    attention = block.self_attn
    prefix, catchup = f"layers.{index}.attention.", "mtp.catchup.attention."
    ffn = f"layers.{index}.ffn."
    forms: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for linear in draft_linears(geometry):
            forms[prefix + linear.operand] = adopt_once(
                linear, getattr(attention, linear.key), index
            )
        gamma_x = w._norm(block.input_layernorm).float().contiguous()
        gamma_q = w._norm(attention.q_a_layernorm).float().contiguous()
        forms.update(
            {
                prefix + "gamma_x": gamma_x,
                prefix + "gamma_q": gamma_q,
                prefix + "gamma_kv": w._norm(attention.kv_a_layernorm)
                .float()
                .contiguous(),
            }
        )
        if rank < geometry.selector_ranks:
            forms.update(
                {
                    catchup + n: t
                    for n, t in selector_forms(attention, geometry, gamma_x).items()
                }
            )
        mlp = block.mlp
        experts = w._routed_experts(mlp, index)
        buffers = w.repoint_routed_experts(experts, index, LAYOUT.slots)
        w.share_shared_expert(mlp, buffers, index)
        forms.update({ffn + name: t for name, t in buffers.kernel_inputs().items()})
        forms[ffn + "gamma"] = w._norm(block.post_attention_layernorm)
        forms[ffn + "router_weight"] = mlp.gate.weight.data
        forms[ffn + "correction_bias"] = mlp.gate.e_score_correction_bias.data
        eh = layer.eh_proj.weight.data
        shape = (geometry.hidden, 2 * geometry.hidden)
        if (
            tuple(eh.shape) != shape
            or eh.dtype != torch.bfloat16
            or not eh.is_contiguous()
        ):
            raise ValueError(
                f"layer {index}: vLLM's eh_proj is not a contiguous BF16 {list(shape)} "
                "tensor"
            )
        forms["mtp.preprocess.eh_weight"] = eh
        norm = layer.enorm if rank < geometry.world // 2 else layer.hnorm
        forms["mtp.preprocess.gamma"] = w._norm(norm).float().contiguous()
        forms["mtp.shared_head_norm"] = layer.shared_head.norm.weight.data
        torch.cuda.empty_cache()
    logger.info(
        "GLM-5.3 drafter segment weights, rank %d: %d tensors in place, %.2f s",
        rank,
        len(forms),
        time.perf_counter() - start,
    )
    return forms

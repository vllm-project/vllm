# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3's MTP drafter owned by the model: the MTP layer and its draft head as CuTe
DSL kernels.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from vllm.models.glm53.nvidia import arena as glm53_arena
from vllm.models.glm53.nvidia import decode as glm53_decode
from vllm.models.glm53.nvidia import draft_providers as dp
from vllm.models.glm53.nvidia import switch
from vllm.models.glm53.nvidia import weights as w
from vllm.models.glm53.nvidia.ops import launch as glm53_launch
from vllm.models.glm53.nvidia.ops.launch import (
    BLOCK_TABLE,
    Glm53Geometry,
    Region,
    sizes_of,
)
from vllm.triton_utils import tl, triton

logger = logging.getLogger(__name__)

#: Hidden columns each program of the row-copy kernels moves.
COPY_BLOCK = 1024


@triton.jit
def _catchup_words_kernel(
    positions,
    last_token_indices,
    table,
    position_out,
    accept_out,
    context_out,
    T: tl.constexpr,
):
    request = tl.program_id(0)
    if request < tl.load(table + 1):
        first = request * T
        position = tl.load(positions + first).to(tl.int32)
        last = tl.load(last_token_indices + request).to(tl.int32)
        accepted = tl.minimum(tl.maximum(last - first, 0), T - 1)
        tl.store(position_out + 2 * first, position)
        tl.store(accept_out + request, accepted)
        tl.store(context_out + 2 * request, position + accepted + 1)
        generation = context_out + 2 * request + 1
        tl.store(generation, tl.load(generation) + 1)


@triton.jit(do_not_specialize=["num_rows"])
def _gather_inputs_kernel(
    last_token_indices,
    query_start_loc,
    positions,
    token_ids,
    rows,
    token_scratch,
    rows_scratch,
    position_out,
    accept_out,
    context_out,
    table,
    num_rows,
    T: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    request = tl.program_id(1)
    if request < tl.load(table + 1):
        last = tl.minimum(
            tl.load(last_token_indices + request).to(tl.int32), num_rows - 1
        )
        start = tl.load(query_start_loc + request).to(tl.int32)
        accepted = tl.minimum(tl.maximum(last - start, 0), T - 1)
        first = last - accepted
        columns = block * BLOCK + tl.arange(0, BLOCK)
        for i in range(T):
            if i <= accepted:
                tl.store(
                    rows_scratch + (request * T + i) * WIDTH + columns,
                    tl.load(rows + (first + i) * WIDTH + columns),
                )
        if block == 0:
            for i in range(T):
                if i <= accepted:
                    tl.store(
                        token_scratch + request * T + i,
                        tl.load(token_ids + first + i),
                    )
            position = tl.load(positions + first).to(tl.int32)
            tl.store(position_out + 2 * request * T, position)
            tl.store(accept_out + request, accepted)
            tl.store(context_out + 2 * request, position + accepted + 1)
            generation = context_out + 2 * request + 1
            tl.store(generation, tl.load(generation) + 1)


@triton.jit
def _proposal_inputs_kernel(
    positions,
    input_ids,
    hidden,
    table,
    position_out,
    token_out,
    row_out,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    request = tl.program_id(1)
    if request < tl.load(table + 1):
        if block == 0:
            tl.store(
                position_out + 2 * request, tl.load(positions + request).to(tl.int32)
            )
            tl.store(token_out + request, tl.load(input_ids + request).to(tl.int32))
        columns = block * BLOCK + tl.arange(0, BLOCK)
        tl.store(
            row_out + request * WIDTH + columns,
            tl.load(hidden + request * WIDTH + columns),
        )


@triton.jit
def _window_inputs_kernel(
    position_t,
    accept_t,
    src_tokens,
    src_rows,
    table,
    blocks,
    block_stride,
    block_size,
    tokens_out,
    rows_out,
    position_out,
    accept_out,
    context_out,
    T: tl.constexpr,
    PASS: tl.constexpr,
    RESERVED: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    request = tl.program_id(1)
    if request < tl.load(table + 1):
        position = tl.load(position_t + 2 * T * request)
        accepted = tl.load(accept_t + request)
        low = accepted < 4
        if PASS == 0:
            start = tl.where(low, T - 4, 0)
            row = start * 0
        else:
            start = tl.where(low, 0, T - 4)
            row = accepted - start
        columns = block * BLOCK + tl.arange(0, BLOCK)
        for i in range(4):
            # Rows past a short step's T (dead rows) repeat its last row.
            source = request * T + tl.minimum(start + i, T - 1)
            tl.store(
                rows_out + (request * 4 + i) * WIDTH + columns,
                tl.load(src_rows + source * WIDTH + columns),
            )
        if block == 0:
            for i in range(4):
                source = request * T + tl.minimum(start + i, T - 1)
                tl.store(
                    tokens_out + request * 4 + i,
                    tl.load(src_tokens + source).to(tl.int32),
                )
            tl.store(position_out + 8 * request, position + start)
            tl.store(accept_out + request, row)
            tl.store(context_out + 2 * request, position + start + row + 1)
            generation = context_out + 2 * request + 1
            tl.store(generation, tl.load(generation) + 1)
            # vLLM reserves RESERVED slots from the position (the step's rows and K
            # lookahead): a dead row past them gets the null block, not a stale entry.
            if RESERVED < 4:
                for k in tl.static_range(RESERVED, 4):
                    dead = (position + k) // block_size
                    if dead > (position + RESERVED - 1) // block_size:
                        tl.store(blocks + request * block_stride + dead, 0)


def stage_window(
    tensors: Mapping[str, torch.Tensor],
    tokens: torch.Tensor,
    rows: torch.Tensor,
    table: torch.Tensor,
    requests: int,
    drafts: int,
    window: int,
    block_size: int,
) -> None:
    """Copy each live request's four catch-up rows of ``window`` (0: the half without
    the accepted row, 1: the half with it) out of vLLM's ``1 + drafts`` rows a request
    into the kernels' four-row buffers, with their step words (``table``: the row
    table, whose request count gates the requests).
    """
    blocks = tensors[BLOCK_TABLE]
    hidden = rows.shape[-1]
    _window_inputs_kernel[(hidden // COPY_BLOCK, requests)](
        tensors["draft.t.position"],
        tensors["draft.t.accept"],
        tokens,
        rows,
        table,
        blocks,
        blocks.stride(0) if blocks.dim() == 2 else 0,
        block_size,
        tensors["token_ids"],
        tensors["draft.rows"],
        tensors["position"],
        tensors["mtp.accept"],
        tensors["draft.context"],
        T=drafts + 1,
        PASS=window,
        RESERVED=2 * drafts + 1,
        WIDTH=hidden,
        BLOCK=COPY_BLOCK,
    )


def _scratch_like(tensor: torch.Tensor, numel: int) -> torch.Tensor:
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
        requests: int = 1,
        proposals: int | None = None,
    ) -> None:
        device = tensors["position"].device
        self.requests: int = requests
        self.placement: glm53_arena.Glm53Placement = glm53_arena.Glm53Placement(device)
        self.parts: list[dp.DraftPart] = dp.program(
            geometry, rank, tensors, self.placement, requests, proposals
        )
        providers = [provider for part in self.parts for provider in part]
        self.arena: glm53_arena.PeerArena = glm53_arena.PeerArena(
            dp.draft_slices(geometry, requests, proposals), group
        )
        arenas = (self.arena, embedding)

        def peers(name: str) -> list[int]:
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
        writes = [
            provider.launch
            for provider in self.parts[0]
            if provider.launch is not None
            and not isinstance(provider, (dp.DraftMoe, dp.DraftHead))
        ]
        self.writes: glm53_launch.Bound = glm53_launch.bind(writes, device.index)
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
        results = {
            name: glm53_decode.tensor_region(t) for name, t in self.results.items()
        }
        return {**results, **self.arena.regions()}

    def launch(self, part: int, stream: int) -> None:
        """Launch the catch-up (``part`` -1) or proposal ``part`` on ``stream``
        (capturable).
        """
        glm53_launch.run(self.bound[part + 1], stream)

    def launch_writes(self, stream: int) -> None:
        """Launch the catch-up without its MoE and head on ``stream`` (capturable)."""
        glm53_launch.run(self.writes, stream)

    def step(self, live: int) -> None:
        """Write the row tables of a step of ``live`` requests (stream-ordered, before
        the replay it belongs to); the one-request tier's never change.
        """
        if self.requests > 1:
            for part in self.parts:
                part.table.set_uniform_step(live, part.rows_per_request)

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


@dataclass
class DraftTier:
    """One request tier's drafter segment and the buffers it was built over."""

    requests: int
    segment: Glm53DraftSegment
    tensors: dict[str, torch.Tensor]
    table: torch.Tensor
    cache: int
    rows_scratch: torch.Tensor
    tokens_scratch: torch.Tensor
    hidden: torch.Tensor | None
    captures: dict[str, int] = field(
        default_factory=lambda: {"prefill": 0, "decode": 0}
    )


class Glm53Draft:
    """The drafter model's hook (module docstring)."""

    normalized: bool = True

    def __init__(
        self,
        drafter: torch.nn.Module,
        geometry: Glm53Geometry,
        forms: Mapping[str, torch.Tensor],
        drafts: int,
        tiers: tuple[int, ...] = (1,),
    ) -> None:
        self.geometry: Glm53Geometry = geometry
        self.layer: int = dp.mtp_layer(geometry)
        self.rows: int = drafts + 1
        self.drafts: int = drafts
        #: The catch-up's windows of four rows: two at K = 4..7, one padded at K = 1, 2
        #: (vLLM's rows staged into the kernels' buffers), none at K = 3.
        self.windows: tuple[int, ...] = ()
        if self.rows != geometry.rows:
            self.windows = (0, 1) if self.rows > geometry.rows else (1,)
        self._block_size: int = 0
        self.sizes: tuple[int, ...] = tuple(tiers)
        self._drafter: torch.nn.Module = drafter
        self._forms: dict[str, torch.Tensor] = dict(forms)
        self._tiers: dict[int, DraftTier] = {}
        self._last_token_indices: torch.Tensor | None = None
        #: The step's requests and its ``query_start_loc`` (:meth:`step`).
        self._live: int = 1
        self._batch: tuple[str, ...] = ()
        self._query_start_loc: torch.Tensor | None = None
        self._tokens: torch.Tensor | None = None
        self._decoding: bool = False
        self._sequence_captures: int = 0
        self._warmed: bool = False
        self._positions: torch.Tensor | None = None
        self._eager_rows: int = 0
        self._lean_rows: int | None = None
        #: K = 1: each request's sampled row, from the target's sampled-token counts
        #: (:meth:`observe_sampled`; vLLM compacts no top-k indices to hand over).
        self._sampled: torch.Tensor | None = None
        device = next((t.device for t in forms.values()), None)
        if drafts == 1 and device is not None and device.type == "cuda":
            first_rows = torch.arange(max(tiers), dtype=torch.int64, device=device)
            self._first_rows: torch.Tensor = first_rows * self.rows - 1
            self._sampled = torch.zeros_like(first_rows)
            self._last_token_indices = self._sampled

    @property
    def built(self) -> bool:
        return bool(self._tiers)

    def target_rows(self, requests: int) -> switch.Build:
        """Return the target build that runs ``requests`` requests of this drafter's
        ``T`` rows (its block table and RoPE table are the build's).
        """
        return switch.shape_build(requests, self.rows)

    def _foreign_target(self, requests: int) -> bool:
        bound = self.target_rows(requests)
        served = (b for b in range(1, requests + 1) if self.tier_of(b) == requests)
        return any(switch.shape_build(b, self.rows) != bound for b in served)

    def _vllm_name(self, name: str) -> str:
        return f"vllm.{name}" if self.windows else name

    def _catch_up(
        self,
        tier: DraftTier,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        stream: int,
    ) -> None:
        if not self.windows:
            tier.segment.launch(-1, stream)
            return
        for window in self.windows:
            stage_window(
                tier.tensors,
                tokens,
                rows,
                tier.segment.parts[0].table.tensor,
                tier.requests,
                self.drafts,
                window,
                self._block_size,
            )
            if window == 0:
                tier.segment.launch_writes(stream)
            else:
                tier.segment.launch(-1, stream)

    def tier_of(self, requests: int) -> int | None:
        """Return the smallest tier that holds ``requests`` requests (None: none)."""
        return next((size for size in self.sizes if size >= requests), None)

    def _tier(self, requests: int) -> DraftTier | None:
        size = self.tier_of(requests)
        return None if size is None else self._tiers.get(size)

    def observe_slots(self, slot_ids: torch.Tensor) -> None:
        """Keep the speculator's sampled-row indices (handed to
        ``compact_topk_indices``).
        """
        if self._last_token_indices is None:
            self._last_token_indices = slot_ids
        self._warm_up()

    def observe_sampled(self, num_sampled: torch.Tensor | int) -> None:
        """K = 1: write each request's sampled row of a uniform step, its first row
        plus its sampled tokens minus one (stream-ordered after the target's sampling,
        before the draft prefill; captured with vLLM's post-forward graph).
        """
        if self._sampled is None or not isinstance(num_sampled, torch.Tensor):
            return
        n = min(num_sampled.shape[0], self._sampled.numel())
        torch.add(self._first_rows[:n], num_sampled[:n], out=self._sampled[:n])

    def decode_phase(self, decoding: bool) -> None:
        """Track the speculator's draft-decode phase (``set_skip_topk``)."""
        if (
            not decoding
            and self.drafts > 1
            and self._sequence_captures % (self.drafts - 1)
        ):
            raise RuntimeError(
                "GLM-5.3 drafter: the draft-decode graph holds one proposal; it needs "
                "vLLM's "
                "fused draft decode"
            )
        self._decoding = decoding
        self._sequence_captures = 0
        self._lean_rows = None

    def step(
        self,
        request_ids: Sequence[str],
        query_start_loc: torch.Tensor,
    ) -> None:
        """Account one forward before it runs: its requests and the row tables."""
        self._live = len(request_ids)
        self._query_start_loc = query_start_loc
        tier = self._tier(self._live)
        if tier is None:
            self._batch = ()
            return
        tier.segment.step(self._live)
        if tuple(request_ids) != self._batch:
            tier.segment.reset_seeds()
        self._batch = tuple(request_ids)

    def claims_proposals(self) -> bool:
        """Return whether the proposals are this drafter's (once captured)."""
        tier = self._tier(self._live)
        return (
            tier is not None
            and tier.captures["decode"] > 0
            and not torch.cuda.is_current_stream_capturing()
        )

    def claims_draft_prefill(self, requests: int) -> bool:
        """Return whether the tier of ``requests`` requests replays a captured draft
        prefill.
        """
        tier = self._tier(requests)
        return tier is not None and tier.captures["prefill"] > 0

    def claims_capture(self) -> bool:
        """Return whether the draft decode captured now is this drafter's: every tier is
        built and one tier holds exactly the requests of the capture (a size without
        a tier, such as three requests when the largest tier is two, captures vLLM's
        drafter, which needs its slot-mapping and metadata updates).
        """
        self._lean_rows = None
        if not (
            all(size in self._tiers for size in self.sizes)
            and torch.cuda.is_current_stream_capturing()
        ):
            return False
        tier = self._tier(self._eager_rows)
        if tier is None or tier.requests != self._eager_rows:
            return False
        self._lean_rows = self._eager_rows
        return True

    def claimed_tokens(self, rows: int | None = None) -> torch.Tensor | None:
        """Return (once) the greedy drafts of the forward just claimed (its first
        ``rows``), else None.
        """
        tokens, self._tokens = self._tokens, None
        return tokens if tokens is None or rows is None else tokens[:rows]

    def after_stock_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Run the catch-up after an eager (stock) draft prefill: the proposals stay
        valid.
        """
        tier = self._tier(self._live)
        if (
            tier is None
            or tier.captures["decode"] == 0
            or self._decoding
            or self._last_token_indices is None
            or self._query_start_loc is None
            or torch.cuda.is_current_stream_capturing()
            or self._stale()
        ):
            return
        tensors = tier.tensors
        tokens = tensors[self._vllm_name("token_ids")]
        rows = tensors[self._vllm_name("draft.rows")]
        if (
            input_ids.data_ptr() != tokens.data_ptr()
            or hidden_states.data_ptr() != rows.data_ptr()
        ):
            raise RuntimeError(
                "GLM-5.3 drafter: an eager draft prefill on other buffers than the "
                "drafter binds"
            )
        self._sync_inputs(tier, input_ids, positions, hidden_states)
        target = glm53_decode.current()
        assert target is not None
        # The padded block table is refreshed only inside the target's captured graph.
        target.sync_block_table(self.target_rows(tier.requests))
        self._catch_up(
            tier,
            tier.tokens_scratch,
            tier.rows_scratch,
            torch.cuda.current_stream().cuda_stream,
        )

    def _sync_inputs(
        self,
        tier: DraftTier,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        words: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        assert self._last_token_indices is not None
        assert self._query_start_loc is not None
        if words is None and self.windows:
            words = {
                name: tier.tensors[f"draft.t.{name.split('.')[-1]}"]
                for name in ("position", "mtp.accept", "draft.context")
            }
        words = tier.tensors if words is None else words
        hidden = self.geometry.hidden
        _gather_inputs_kernel[(hidden // COPY_BLOCK, tier.requests)](
            self._last_token_indices,
            self._query_start_loc,
            positions,
            input_ids,
            hidden_states,
            tier.tokens_scratch,
            tier.rows_scratch,
            words["position"],
            words["mtp.accept"],
            words["draft.context"],
            tier.segment.parts[0].table.tensor,
            int(input_ids.shape[0]),
            T=self.rows,
            WIDTH=hidden,
            BLOCK=COPY_BLOCK,
        )
        if words is tier.tensors:
            tier.tensors["token_ids"].copy_(tier.tokens_scratch)
            tier.tensors["draft.rows"].copy_(tier.rows_scratch)

    def _warm_up(self) -> None:
        indices, positions = self._last_token_indices, self._positions
        if (
            self._warmed
            or not self._tiers
            or indices is None
            or positions is None
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        tier = next(iter(self._tiers.values()))
        tensors = tier.tensors
        rows, hidden = self.rows, self.geometry.hidden
        names = ("position", "mtp.accept", "draft.context")
        sources = (
            ("draft.t.position", "draft.t.accept", "draft.t.context")
            if self.windows
            else names
        )
        words = {
            name: _scratch_like(tensors[source], tensors[source].numel())
            for name, source in zip(names, sources)
        }
        saved = self._last_token_indices, self._query_start_loc
        self._last_token_indices = _scratch_like(indices, tier.requests)
        self._query_start_loc = _scratch_like(tensors["mtp.accept"], tier.requests + 1)
        self._sync_inputs(
            tier,
            _scratch_like(tensors[self._vllm_name("token_ids")], rows),
            _scratch_like(positions, rows),
            _scratch_like(tensors[self._vllm_name("draft.rows")], rows * hidden).view(
                rows, hidden
            ),
            words,
        )
        if self.windows:
            # The windows' kernel over no live request (a zeroed table): compiled only.
            table = _scratch_like(tier.segment.parts[0].table.tensor, 2)
            for window in self.windows:
                stage_window(
                    tensors,
                    tier.tokens_scratch,
                    tier.rows_scratch,
                    table,
                    1,
                    self.drafts,
                    window,
                    self._block_size,
                )
        self._last_token_indices, self._query_start_loc = saved
        torch.cuda.current_stream().synchronize()
        self._warmed = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Claim a captured draft prefill or decode; return its rows, else None."""
        self._tokens = None
        if input_ids is None or not input_ids.is_cuda:
            return None
        rows = int(input_ids.shape[0])
        if not torch.cuda.is_current_stream_capturing():
            self._eager_rows = rows
            if self._tiers and self._stale():
                self.close()
            size = rows // self.rows
            built = False
            if (
                rows % self.rows == 0
                and size in self.sizes
                and size not in self._tiers
                and not self._decoding
            ):
                self._try_build(size, input_ids, positions, hidden_states)
                built = size in self._tiers
            if self._tiers and not built:
                for tier in self._tiers.values():
                    tier.segment.reset_seeds()
            return None
        decoding = self._decoding or rows < self.rows
        requests = rows if decoding else rows // self.rows
        if decoding and self._lean_rows is not None and self._lean_rows != rows:
            raise RuntimeError(
                f"GLM-5.3 drafter: the draft decode of {self._lean_rows} requests was "
                f"claimed lean but {rows} rows are captured"
            )
        if not self._tiers or (not decoding and rows % self.rows):
            if not self._tiers and rows in (1, self.rows):
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
        tier = self._tier(requests)  # type: ignore[assignment]
        # K = 1 has no proposals: vLLM captures no draft decode to claim.
        if tier is None or (decoding and (rows != tier.requests or self.drafts == 1)):
            return None
        tensors = tier.tensors
        stream = torch.cuda.current_stream().cuda_stream
        if not decoding:
            if input_ids.data_ptr() != tensors[self._vllm_name("token_ids")].data_ptr():
                raise RuntimeError(
                    "GLM-5.3 drafter: vLLM captures other input_ids than the drafter "
                    "binds"
                )
            if self._last_token_indices is None:
                raise RuntimeError(
                    "GLM-5.3 drafter: a draft-prefill capture before an eager propose"
                )
            names = ("position", "mtp.accept", "draft.context")
            if self.windows:
                names = ("draft.t.position", "draft.t.accept", "draft.t.context")
            _catchup_words_kernel[(tier.requests,)](
                positions,
                self._last_token_indices,
                tier.segment.parts[0].table.tensor,
                *(tensors[name] for name in names),
                T=self.rows,
            )
            if self._foreign_target(tier.requests):
                target = glm53_decode.current()
                assert target is not None
                target.sync_block_table(self.target_rows(tier.requests))
            self._catch_up(tier, input_ids, hidden_states, stream)
            tier.captures["prefill"] += 1
            if tier.captures["prefill"] == 1:
                logger.info(
                    "GLM-5.3 drafter: tier of %d requests claims the draft prefill "
                    "of %d rows",
                    tier.requests,
                    rows,
                )
            self._tokens = tensors["draft.tokens"]
            if tier.hidden is None:
                return tensors["draft.hidden"]
            # vLLM reads request b's row at its sampled index in b's group of rows.
            tier.hidden.view(tier.requests, self.rows, -1).copy_(
                tensors["draft.hidden"][: tier.requests]
                .unsqueeze(1)
                .expand(-1, self.rows, -1)
            )
            return tier.hidden[:rows]
        # vLLM captures each size's draft decodes one after another within one phase.
        proposal = self._sequence_captures % (self.drafts - 1)
        self._sequence_captures += 1
        if hidden_states.is_contiguous():
            hidden = self.geometry.hidden
            _proposal_inputs_kernel[(hidden // COPY_BLOCK, tier.requests)](
                positions,
                input_ids,
                hidden_states,
                tier.segment.parts[proposal + 1].table.tensor,
                tensors["draft.propose.position"],
                tensors["draft.propose.token"],
                tensors["draft.propose.row"],
                WIDTH=hidden,
                BLOCK=COPY_BLOCK,
            )
        else:
            n = tier.requests
            tensors["draft.propose.position"].view(-1, 2)[:n, 0].copy_(positions[:n])
            tensors["draft.propose.token"][:n].copy_(input_ids[:n])
            tensors["draft.propose.row"][:n].copy_(hidden_states[:n])
        tier.segment.launch(proposal, stream)
        tier.captures["decode"] += 1
        if tier.captures["decode"] == 1:
            logger.info(
                "GLM-5.3 drafter: tier of %d requests claims the draft decode of %d "
                "rows",
                tier.requests,
                rows,
            )
        self._tokens = tensors["draft.propose.tokens"]
        return tensors["draft.propose.hidden"][:rows]

    def _layer_caches(
        self,
    ) -> tuple[
        dict[str, torch.Tensor], dict[int, torch.Tensor], dict[int, torch.Tensor]
    ]:
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
        target = glm53_decode.current()
        if target is None or not target.built:
            return True
        _, mla, _ = self._layer_caches()
        for tier in self._tiers.values():
            shared = target.shared_tensors(self.target_rows(tier.requests))
            if shared.get(BLOCK_TABLE) is not tier.table:
                return True
            if self.layer in mla and mla[self.layer].data_ptr() != tier.cache:
                return True
        return False

    def _try_build(
        self,
        requests: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        from vllm.distributed import get_tp_group
        from vllm.forward_context import get_forward_context

        target = glm53_decode.current()
        capacity = self.rows * requests
        if target is None or not target.has_tier(self.target_rows(requests)):
            return
        caches, mla, index = self._layer_caches()
        layer = self.layer
        if layer not in mla or layer not in index:
            return
        geometry = self.geometry
        group = get_tp_group()
        rank = group.rank_in_group
        device = mla[layer].device
        shared = target.shared_tensors(self.target_rows(requests))
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
        words, rows = dp.step_words(requests), dp.buffer_rows(requests)
        tensors: dict[str, torch.Tensor] = {
            **self._forms,
            "lm_head": head,
            **shared,
            f"layers.{layer}.attention.kv_cache": mla[layer],
            f"layers.{layer}.attention.index_key_cache": index[layer],
            self._vllm_name("token_ids"): input_ids[:capacity],
            self._vllm_name("draft.rows"): hidden_states[:capacity],
            **{
                name: torch.zeros(words[name], dtype=torch.int32, device=device)
                for name in dp.WORDS
            },
            **{
                name: torch.zeros(
                    (rows, geometry.hidden), dtype=torch.bfloat16, device=device
                )
                for name in dp.ROWS
            },
        }
        if self.windows:
            window = geometry.rows * requests
            tensors["token_ids"] = torch.zeros(window, dtype=torch.int32, device=device)
            tensors["draft.rows"] = torch.zeros(
                (window, geometry.hidden), dtype=torch.bfloat16, device=device
            )
            for name, size in (
                ("position", 2 * self.rows * requests),
                ("accept", requests),
                ("context", dp.CONTEXT_WORDS * requests),
            ):
                tensors[f"draft.t.{name}"] = torch.zeros(
                    size, dtype=torch.int32, device=device
                )
        tensors["draft.context"].fill_(1)
        layers = get_forward_context().no_compile_layers
        self._block_size = int(mla[layer].shape[-2])
        glm53_decode.check_mla_caches(layers, caches, geometry, self._block_size)
        segment = Glm53DraftSegment(
            geometry,
            rank,
            tensors,
            group.device_group,
            embedding,
            requests,
            self.drafts - 1,
        )
        target.add_layer(layer, caches, reach=self.drafts)
        self._tiers[requests] = DraftTier(
            requests,
            segment,
            tensors,
            shared[BLOCK_TABLE],
            mla[layer].data_ptr(),
            torch.zeros(
                (capacity, geometry.hidden), dtype=torch.bfloat16, device=device
            ),
            torch.zeros(capacity, dtype=torch.int32, device=device),
            None
            if requests == 1
            else torch.zeros(
                (capacity, geometry.hidden), dtype=torch.bfloat16, device=device
            ),
        )
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
            "GLM-5.3 drafter segment built on rank %d (%d requests, %d proposals)",
            rank,
            requests,
            self.drafts - 1,
        )

    def close(self) -> None:
        """Drop the segments, their tensors and every capture; the next eager draft
        prefill of a size rebuilds its tier.
        """
        if self._tiers:
            torch.accelerator.synchronize()
            for tier in self._tiers.values():
                tier.segment.close()
        self._tiers = {}
        self._positions = None
        self._tokens = None
        self._decoding = False
        self._sequence_captures = 0
        self._batch = ()
        self._warmed = False


_CURRENT: Glm53Draft | None = None


def current() -> Glm53Draft | None:
    """Return this process's installed drafter hook, if any."""
    return _CURRENT


def claims_draft_prefill(requests: int = 1) -> bool:
    """Return whether the drafter replays the draft prefill of ``requests`` requests."""
    draft = _CURRENT
    return draft is not None and draft.claims_draft_prefill(requests)


def install(
    drafter: torch.nn.Module,
    geometry: Glm53Geometry,
    forms: Mapping[str, torch.Tensor] | None,
    drafts: int,
    tiers: tuple[int, ...] = (1,),
) -> Glm53Draft | None:
    """Install the drafter hook (with the model-owned segment 1) for the request
    ``tiers`` of ``switch.draft_tiers``.
    """
    if forms is None or not glm53_decode.segment_enabled():
        return None
    if drafts not in switch.DRAFT_KS:
        raise ValueError(
            f"the GLM-5.3 drafter segment drafts {switch.DRAFT_KS} tokens, not {drafts}"
        )
    global _CURRENT
    draft = Glm53Draft(drafter, geometry, forms, drafts, tiers)
    drafter.model.glm53_segment = draft
    _CURRENT = draft
    logger.info(
        "GLM-5.3 drafter segment installed (T = %d, %d proposals, request tiers %s)",
        drafts + 1,
        drafts - 1,
        draft.sizes,
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
) -> dict[str, torch.Tensor]:
    """Return the forms of vLLM's ``module`` of MTP layer ``layer`` read in place by
    ``linear``: its weight and its decoded scales (decoded once per module).
    """
    adopted = getattr(module, "_glm53_adopted", None)
    if adopted is None:
        adopted = linear.adopt(module, layer)
        module._glm53_adopted = adopted
    weight, scales = adopted
    return {linear.operand: weight, linear.scales_operand: scales}


def selector_forms(
    attention: torch.nn.Module, geometry: Glm53Geometry, gamma_x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Return the catch-up selector's MTP-layer forms."""
    layer = dp.mtp_layer(geometry)
    indexer = attention.indexer
    return {
        "gamma_x": gamma_x,
        **adopt_once(
            w.worker_linears(geometry)["input"], attention.fused_qkv_a_proj, layer
        ),
        "wis_weight": w.wk_weights_proj(indexer, geometry, layer),
        "key_gamma": w._norm(indexer.k_norm).float().contiguous(),
        "key_bias": indexer.k_norm.bias.data.float().contiguous(),
        "gamma_q": w._norm(attention.q_a_layernorm).float().contiguous(),
        **adopt_once(w.selector_linear(geometry), indexer.wq_b, layer),
    }


def draft_weights(
    drafter: torch.nn.Module, geometry: Glm53Geometry, rank: int
) -> dict[str, torch.Tensor]:
    """Bind this rank's MTP-layer forms to vLLM's drafter tensors in place
    (one copy).
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
            adopted = adopt_once(linear, getattr(attention, linear.key), index)
            forms.update({prefix + name: t for name, t in adopted.items()})
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
        experts = w.layer_experts(mlp, index).kernel_inputs()
        forms.update({ffn + name: t for name, t in experts.items()})
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
        torch.accelerator.empty_cache()
    logger.info(
        "GLM-5.3 drafter segment weights, rank %d: %d tensors in place, %.2f s",
        rank,
        len(forms),
        time.perf_counter() - start,
    )
    return forms

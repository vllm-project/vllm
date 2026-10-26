# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch providers of GLM-5.3's specialized MTP drafter (the MTP layer,
``draft.py``).
"""

from __future__ import annotations

from abc import abstractmethod
from collections import ChainMap
from collections.abc import Mapping

import torch

from vllm.models.glm53.nvidia import providers
from vllm.models.glm53.nvidia.arena import Glm53Placement, Slice
from vllm.models.glm53.nvidia.ops.cute_dsl import registry, row_table
from vllm.models.glm53.nvidia.ops.cute_dsl.lm_head import CTAS as HEAD_CTAS
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_draft import CONTEXT_WORDS
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_selector import StateWords
from vllm.models.glm53.nvidia.ops.launch import (
    BLOCK_TABLE,
    EMBEDDING_SLICE,
    Argument,
    Glm53Geometry,
    Launch,
    sizes_of,
)
from vllm.models.glm53.nvidia.providers import (
    BLOCK_STRIDE_ARGUMENT,
    ROW_TABLE_ARGUMENT,
    ROW_TABLE_INPUT,
    Peers,
)
from vllm.models.glm53.nvidia.rows import RowTable

DRAFT_SLICE = "glm53.mtp.{part}.{node}.rows"
DRAFT_SELECTION = "glm53.mtp.catchup.attention.selection"
DRAFT_TOP1 = "glm53.head.mtp.{part}.top1"
HEAD_KERNEL = "lm_head_top1_q1"
HEAD_KERNEL_Q4 = "lm_head_top1_q4"
#: Rows of the preprocess, MoE and head kernels of a request tier (the four-row builds).
LAUNCH_ROWS = 4
WORDS = (
    "position",
    "mtp.accept",
    "draft.context",
    "draft.propose.token",
    "draft.propose.position",
    "draft.tokens",
    "draft.propose.tokens",
)
ROWS = ("draft.propose.row", "draft.hidden", "draft.propose.hidden")


def mtp_layer(geometry: Glm53Geometry) -> int:
    """Return the MTP layer's index, after the target layers (vLLM's
    ``mtp_start_layer_idx``).
    """
    return geometry.layers


def proposals_of(geometry: Glm53Geometry, drafts: int | None = None) -> int:
    """Return the proposals after the catch-up: ``drafts - 1`` (default: the drafts of
    the four-row step, ``geometry.rows - 1``).
    """
    return (geometry.rows - 1 if drafts is None else drafts) - 1


def row_launches(requests: int) -> int:
    """Return the four-row launches of the MoE and the head of a part (a request's four
    rows of the one-request build; four requests' rows in a tier).
    """
    return -(-requests // LAUNCH_ROWS)


def preprocess_launches(proposal: int | None, requests: int) -> int:
    """Return the preprocess launches of a part: one per request in the catch-up (its
    four rows), one per four requests in a proposal tier.
    """
    return requests if proposal is None else row_launches(requests)


def buffer_rows(requests: int) -> int:
    """Return the rows of a per-request row tensor: the one request's single row, else
    the requests rounded up to the four-row launches that read them.
    """
    return 1 if requests == 1 else row_launches(requests) * LAUNCH_ROWS


def step_words(requests: int) -> dict[str, int]:
    """Return the element count of each step word (:data:`WORDS`) of a tier of
    ``requests`` requests.
    """
    rows = buffer_rows(requests)
    return {
        "position": 2 * LAUNCH_ROWS * requests,
        "mtp.accept": requests,
        "draft.context": CONTEXT_WORDS * requests,
        "draft.propose.token": requests,
        "draft.propose.position": 2 * requests,
        "draft.tokens": rows,
        "draft.propose.tokens": rows,
    }


def _node(name: str, index: int) -> str:
    return name if index == 0 else f"{name}{index}"


def _suffix(proposal: int | None) -> str:
    return "" if proposal is None else f"#{proposal + 1}"


def _part(proposal: int | None) -> str:
    return "catchup" if proposal is None else f"propose{proposal}"


def _peers(prefix: str, addresses: list[int]) -> dict[str, Argument]:
    return {f"{prefix}{peer}": address for peer, address in enumerate(addresses)}


def only_row_kernel(geometry: Glm53Geometry, rank: int, requests: int = 1) -> str:
    """Return the catch-up's only-row selector entry on ``rank``, for the index-key
    cache's scale format (as ``Glm53Geometry.selector_kernel``) and the request tier.
    """
    tag = "q4" if requests == 1 else f"r{geometry.rows * requests}"
    shard = f"shard{geometry.selector_ranks}r{rank}"
    return f"mla_selector_{tag}_{shard}_only_row_paged"


def worker_kernel(geometry: Glm53Geometry, proposal: int | None, requests: int) -> str:
    """Return the draft worker's entry of a part: the one-request build's, else the
    request tier's.
    """
    if requests == 1:
        return f"mla_worker_draft_q{geometry.rows if proposal is None else 1}_paged"
    if proposal is None:
        return f"mla_worker_draft_r{geometry.rows * requests}_paged"
    return f"mla_worker_draft_p{requests}_paged"


def prologue_kernel(
    geometry: Glm53Geometry, proposal: int | None, requests: int
) -> str:
    """Return the prologue's entry of a part (the row tier's for the catch-up)."""
    if requests == 1:
        return "prologue_vllm_q4" if proposal is None else "prologue_vllm"
    if proposal is None:
        return f"prologue_vllm_r{geometry.rows * requests}"
    return f"prologue_vllm_p{requests}"


def attention_site(
    geometry: Glm53Geometry, proposal: int | None, selector: bool = False
) -> str:
    """Return the site of a part's draft worker (``selector``: the
    catch-up's selector).
    """
    layer = mtp_layer(geometry)
    if proposal is None:
        if selector:
            return (
                f"layer{layer}.mla_selector_q4_shard{geometry.selector_ranks}_only_row"
            )
        return f"layer{layer}.mla_worker_draft_q4"
    # Proposal 0 is the first site of its own operator, so it has no suffix.
    suffix = "" if proposal == 0 else f"#{proposal}"
    return f"layer{layer}.mla_worker_draft_q1{suffix}"


class DraftProvider(providers.Provider):
    """A drafter launch whose host-wrapper arguments are given by name
    (:meth:`values`).
    """

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        requests: int = 1,
        index: int = 0,
    ) -> None:
        super().__init__(geometry, inputs["position"].device, inputs)
        self.proposal: int | None = proposal
        self.rank: int = rank
        self.requests: int = requests
        self.index: int = index
        self.group_rows: int = geometry.rows if proposal is None else 1
        self.part_rows: int = self.group_rows * requests
        self.part: str = _part(proposal)
        self.suffix: str = _suffix(proposal)
        self.layer_key: str = f"layers.{mtp_layer(geometry)}"

    def part_hidden(self, name: str, rows: int | None = None) -> torch.Tensor:
        """Allocate zeroed BF16 result ``name`` of ``rows`` (default: the part's) hidden
        rows (the rows past the live requests are read by four-row launches).
        """
        rows = self.part_rows if rows is None else rows
        self.result(name, (rows, self.geometry.hidden), torch.bfloat16).zero_()
        return self.results[name]

    def node(self, name: str) -> str:
        """Return ``name`` of this launch among the part's replicated launches."""
        return _node(name, self.index)

    def word_at(self, name: str, first: int) -> int:
        """Return the address of element ``first`` of input ``name``."""
        tensor = self.inputs[name]
        return tensor.data_ptr() + first * tensor.element_size()

    @abstractmethod
    def values(self, peers: Peers) -> dict[str, Argument]:
        """Return the host-wrapper arguments by name (a missing pointer is null)."""
        raise NotImplementedError

    def bind(self, peers: Peers) -> None:
        """Fix the launch's arguments (:meth:`arguments` names every one, the row
        table's and the block stride's included, so none is inserted).
        """
        self.launch = Launch(self.kernel, self.site, tuple(self.arguments(peers)))

    def arguments(self, peers: Peers) -> list[Argument]:
        """The named values in the entry's argument order."""
        values = self.values(peers)
        names = registry.arguments(self.kernel)
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise ValueError(f"{self.kernel}: unknown arguments {unknown}")
        if ROW_TABLE_ARGUMENT in names:
            values[ROW_TABLE_ARGUMENT] = self.address(ROW_TABLE_INPUT)
        if BLOCK_STRIDE_ARGUMENT in names:
            values[BLOCK_STRIDE_ARGUMENT] = self.inputs[BLOCK_TABLE].stride(0)
        return [values.get(name) for name in names]

    def word(self, name: str) -> str:
        """Return the step word ``name`` of the catch-up, or its proposal
        counterpart.
        """
        return name if self.proposal is None else f"draft.propose.{name}"


class DraftPrologue(DraftProvider):
    """Embed the part's rows from the target's embedding shards (``prologue_vllm``)."""

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        requests: int = 1,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs, requests)
        self.kernel = prologue_kernel(geometry, proposal, requests)
        self.site = f"prologue_vllm{self.suffix}"
        # The proposals' hidden rows are read four at a time by the preprocess.
        rows = self.part_rows if proposal is None else buffer_rows(requests)
        self.hidden: torch.Tensor = self.part_hidden("hidden", rows)
        self.freq: torch.Tensor = self.result(
            "freq", (self.part_rows, geometry.rope_dim), torch.float32
        )

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Token IDs, position, RoPE table, rows, RoPE rows, the table's capacity,
        shards.
        """
        rope = self.inputs["rope_table"]
        token = "token_ids" if self.proposal is None else "draft.propose.token"
        return {
            "TokenIds": self.address(token),
            "Position": self.address(self.word("position")),
            "RopeTable": rope.data_ptr(),
            "Hidden": self.hidden.data_ptr(),
            "Freq": self.freq.data_ptr(),
            "rope_capacity": rope.shape[0],
            **_peers("EmbeddingShard", peers(EMBEDDING_SLICE)),
        }


class DraftPreprocess(DraftProvider):
    """The MTP preprocess: enorm or hnorm, then ``eh_proj`` (input columns split over
    ranks).
    """

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        embedded: torch.Tensor,
        requests: int = 1,
        index: int = 0,
        first: DraftPreprocess | None = None,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs, requests, index)
        rows = LAUNCH_ROWS if proposal is None or requests > 1 else 1
        self.kernel = f"mtp_preprocess_q{rows}"
        self.site = f"mtp_preprocess{self.suffix}" + ("" if index == 0 else f".{index}")
        #: This launch's first row among the part's rows, and its row count.
        self.first_row: int = index * rows
        self.launch_rows: int = rows
        offset = self.first_row * geometry.hidden * embedded.element_size()
        hidden = "draft.rows" if proposal is None else "draft.propose.row"
        self.rows_in: int = offset + (
            embedded.data_ptr() if rank < geometry.world // 2 else self.address(hidden)
        )
        self.result("local_o", (rows, geometry.hidden), torch.bfloat16)
        # The part's rows of the worker's input, one tensor over the launches.
        self.final: torch.Tensor = (
            self.part_hidden(
                "final", self.part_rows if proposal is None else buffer_rows(requests)
            )
            if first is None
            else first.final
        )
        self.owned(f"mtp.{self.part}.{self.node('preprocess')}.full_epoch", 1)

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Rows, weights, position, results, launch ticket, peers."""
        final = self.final[self.first_row : self.first_row + self.launch_rows]
        return {
            "Rows": self.rows_in,
            "Gamma": self.address("mtp.preprocess.gamma"),
            "Weight": self.address("mtp.preprocess.eh_weight"),
            "Position": self.word_at(self.word("position"), 2 * self.first_row),
            "LocalO": self.results["local_o"].data_ptr(),
            "Final": final.data_ptr(),
            "FullEpoch": self.state[
                f"mtp.{self.part}.{self.node('preprocess')}.full_epoch"
            ].data_ptr(),
            "peer_rank": self.rank,
            **_peers(
                "peer_base",
                peers(DRAFT_SLICE.format(part=self.part, node=self.node("preprocess"))),
            ),
        }


class CatchupSelector(DraftProvider):
    """The catch-up's only-row selector shard (ranks 0-2) over the MTP layer's index
    keys.
    """

    WEIGHTS = {
        "GammaX": "gamma_x",
        "QakiWeight": "input_weight",
        "QakiScales": "input_scales",
        "WisWeight": "wis_weight",
        "KeyGamma": "key_gamma",
        "KeyBias": "key_bias",
        "GammaQ": "gamma_q",
        "WqbWeight": "wqb_weight",
        "WqbScales": "wqb_scales",
    }
    RESULTS = {
        "QA": "qa",
        "RawKey": "raw_key",
        "HeadWeights": "head_weights",
        "RawQuery": "raw_query",
        "IndexQuery": "index_query",
        "Scores": "scores",
        "SelectedIds": "selected_ids",
    }

    def __init__(
        self,
        geometry: Glm53Geometry,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        freq: torch.Tensor,
        requests: int = 1,
    ) -> None:
        super().__init__(geometry, None, rank, inputs, requests)
        self.x: torch.Tensor = x
        self.freq: torch.Tensor = freq
        self.kernel = only_row_kernel(geometry, rank, requests)
        self.site = attention_site(geometry, None, selector=True)
        sizes = sizes_of(self.kernel)
        self.capacity: int = inputs["rope_table"].shape[0]
        if self.capacity % sizes["kKeyTile"] or self.capacity > sizes["kMaxContext"]:
            raise ValueError(f"selector key capacity {self.capacity} does not fit")
        rows, index = self.part_rows, (geometry.index_heads, geometry.index_dim)
        shapes = (
            (rows, geometry.q_lora),
            (rows, geometry.index_dim),
            (rows, geometry.index_heads),
            (rows, *index),
            (rows, *index),
        )
        for name, shape in zip(
            ("qa", "raw_key", "head_weights", "raw_query", "index_query"), shapes
        ):
            self.result(name, shape, torch.bfloat16)
        self.result("scores", (rows, self.capacity), torch.float32)
        self.result("selected_ids", (rows, geometry.index_topk), torch.int32)
        self.prefix: str = "mtp.catchup.attention."
        # One launch ticket per row group, 16 bytes apart.
        self.owned(self.prefix + "full_epoch", 4 * requests)
        self.owned(self.prefix + "flags", sizes["kFlagWords"])
        self.select_state: torch.Tensor = self.owned(
            self.prefix + "select_state", sizes["kStateWords"]
        )
        # The seeds are constant: keep them on the device so a reset is a device copy.
        self.seeds: torch.Tensor = torch.tensor(
            providers.Selector.nominal_seed_words("mtp.catchup", self.part_rows),
            dtype=torch.int32,
            device=self.select_state.device,
        )

    def reset_seeds(self) -> None:
        """Reset the selector's seed words (stream-ordered): each row group's state
        block holds the four seed words of its request's rows.
        """
        groups = self.requests
        stride = self.select_state.numel() // groups
        width = self.seeds.numel() // groups
        for group in range(groups):
            first = group * stride + StateWords.SEEDS
            self.select_state[first : first + width].copy_(
                self.seeds[group * width : (group + 1) * width]
            )

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Rows, weights, RoPE, position, keys, results, state, capacity, peers, blocks,
        row A.
        """
        prefix = self.prefix
        return {
            "X": self.x.data_ptr(),
            **{
                name: self.address(prefix + form) for name, form in self.WEIGHTS.items()
            },
            "Freq": self.freq.data_ptr(),
            "CurPos": self.address("position"),
            "IndexKeyCache": self.address(
                f"{self.layer_key}.attention.index_key_cache"
            ),
            **{name: self.results[r].data_ptr() for name, r in self.RESULTS.items()},
            "FullEpoch": self.state[prefix + "full_epoch"].data_ptr(),
            "Flags": self.state[prefix + "flags"].data_ptr(),
            "SelectState": self.select_state.data_ptr(),
            "score_stride": self.capacity,
            **_peers(
                "peer_base", peers(DRAFT_SLICE.format(part="catchup", node="attention"))
            ),
            **{
                f"selection_base{peer}": address
                for peer, address in enumerate(peers(DRAFT_SELECTION))
                if peer > 0
            },
            "BlockTable": self.address(BLOCK_TABLE),
            "OnlyRow": self.address("mtp.accept"),
        }


class DraftWorker(DraftProvider):
    """The draft worker of the catch-up (T rows) or a proposal (one row) over the rank's
    heads.
    """

    WEIGHTS = {
        "GammaX": "gamma_x",
        "InputWeight": "input_weight",
        "InputScales": "input_scales",
        "GammaQ": "gamma_q",
        "QueryWeight": "query_weight",
        "QueryScales": "query_scales",
        "GammaKV": "gamma_kv",
        "KVbWeight": "kvb_weight",
        "KVbScales": "kvb_scales",
        "OWeight": "o_weight",
        "OScales": "o_scales",
    }

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        freq: torch.Tensor,
        requests: int = 1,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs, requests)
        self.x: torch.Tensor = x
        self.freq: torch.Tensor = freq
        self.kernel = worker_kernel(geometry, proposal, requests)
        self.site = attention_site(geometry, proposal)
        sizes = sizes_of(self.kernel)
        f32 = torch.float32.itemsize
        self.result("partial_o", (sizes["kPartialOBytes"] // f32,), torch.float32)
        self.result("partial_lse", (sizes["kPartialLseBytes"] // f32,), torch.float32)
        self.result(
            "final",
            (max(geometry.buffer_rows, requests + LAUNCH_ROWS - 1), geometry.hidden),
            torch.bfloat16,
        ).zero_()
        self.final: torch.Tensor = self.results["final"]
        self.prefix: str = (
            "mtp.catchup.worker." if proposal is None else f"mtp.{self.part}.attention."
        )
        self.owned(self.prefix + "full_epoch", 4 * requests)
        self.owned(self.prefix + "flags", sizes["kFlagWords"])
        self.owned(self.prefix + "handoff", sizes["kHandoffWords"])

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Head and tail records (the tail's Extra: row A and its key count), then pool
        bytes.
        """
        pool = self.inputs[f"{self.layer_key}.attention.kv_cache"]
        weights = f"{self.layer_key}.attention."
        state = {
            name: self.state[self.prefix + name].data_ptr()
            for name in ("handoff", "full_epoch", "flags")
        }
        return {
            "X": self.x.data_ptr(),
            **{
                name: self.address(weights + form)
                for name, form in self.WEIGHTS.items()
            },
            "Freq": self.freq.data_ptr(),
            "CurPos": self.address(self.word("position")),
            "KVCachePtr": pool.data_ptr(),
            "PECache": pool.data_ptr(),
            "SelectionMailbox": peers(DRAFT_SELECTION)[self.rank],
            "Handoff": state["handoff"],
            "PartialO": self.results["partial_o"].data_ptr(),
            "PartialLSE": self.results["partial_lse"].data_ptr(),
            "Hidden": self.x.data_ptr(),
            "Final": self.final.data_ptr(),
            "FullEpoch": state["full_epoch"],
            "Flags": state["flags"],
            "peer_rank": self.rank,
            **_peers(
                "peer_base", peers(DRAFT_SLICE.format(part=self.part, node="attention"))
            ),
            "BlockTable": self.address(BLOCK_TABLE),
            "SelectRow": self.address("mtp.accept"),
            "SelectContext": self.address("draft.context"),
            "KvBytes": pool.numel() * pool.element_size(),
        }


class DraftMoe(DraftProvider):
    """The target's T-row MoE (``fused_moe_q4_ckpt_shared``) over the worker's repeated
    row.
    """

    kernel = "fused_moe_q4_ckpt_shared"
    WEIGHTS = {
        "Gamma": "gamma",
        "RouterWeight": "router_weight",
        "CorrectionBias": "correction_bias",
        "RoutedUp": "routed_up",
        "RoutedDown": "routed_down",
        "SharedUp": "shared_up",
        "SharedDown": "shared_down",
        "RoutedUpScales": "routed_up_scales",
        "RoutedDownScales": "routed_down_scales",
        "SharedScales": "shared_scales",
    }
    RESULTS = {
        "RouterLogits": "router_logits",
        "Activation": "activation",
        "RouteIds": "route_ids",
        "RouteWeights": "route_weights",
        "LocalO": "local_o",
        "Final": "final",
    }

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        placement: Glm53Placement,
        requests: int = 1,
        index: int = 0,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs, requests, index)
        self.x: torch.Tensor = x
        self.site = f"layer{mtp_layer(geometry)}.fused_moe{self.suffix}" + (
            "" if index == 0 else f".{index}"
        )
        rows, top_k = geometry.rows, geometry.top_k
        self.result("router_logits", (rows, geometry.experts), torch.float32)
        self.result(
            "activation",
            ((1 + rows * top_k) * rows, geometry.expert_shard),
            torch.bfloat16,
        )
        self.result("route_ids", (rows, top_k), torch.int32)
        self.result("route_weights", (rows, top_k), torch.float32)
        self.rows("local_o")
        self.final: torch.Tensor = self.rows("final")
        self.prefix: str = f"mtp.{self.part}.{self.node('ffn')}."
        self.owned(self.prefix + "full_epoch", 1)
        words = sizes_of(self.kernel)["kFlagWords"]
        flags, mirror = placement.moe_flags()
        self.owned(self.prefix + "flags", words, placed=flags)
        self.owned(self.prefix + "flags_mirror", words, placed=mirror)

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Rows, weights, results, state, peers."""
        ffn = f"{self.layer_key}.ffn."
        return {
            "X": self.x.data_ptr(),
            **{name: self.address(ffn + form) for name, form in self.WEIGHTS.items()},
            **{name: self.results[r].data_ptr() for name, r in self.RESULTS.items()},
            "FullEpoch": self.state[self.prefix + "full_epoch"].data_ptr(),
            "Flags": self.state[self.prefix + "flags"].data_ptr(),
            "peer_rank": self.rank,
            **_peers(
                "peer_base",
                peers(DRAFT_SLICE.format(part=self.part, node=self.node("moe"))),
            ),
        }


class DraftHead(DraftProvider):
    """The BF16 draft head: shared-head RMSNorm, the rank's vocabulary
    shard, TP top-1.
    """

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        requests: int = 1,
        index: int = 0,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs, requests, index)
        self.kernel = HEAD_KERNEL if requests == 1 else HEAD_KERNEL_Q4
        self.x: torch.Tensor = x
        self.site = f"lm_head_top1{self.suffix}" + ("" if index == 0 else f".{index}")
        sizes = sizes_of(self.kernel)
        rows = sizes["kQTokens"]
        self.result("logits", (rows, sizes["kVocabShard"]), torch.float32)
        # Per CTA: its best logit and that logit's index.
        self.result("partials", (rows, HEAD_CTAS, 2), torch.float32)
        self.result("rank_record", (rows, sizes["kRecordWords"]), torch.int32)
        # The TP winner: its logit's bits and its index.
        self.result("winner", (rows, 2), torch.int32)
        self.done: torch.Tensor = self.owned(
            f"mtp.{self.part}.{self.node('head')}.done", sizes["kDoneWords"]
        )

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Row, weights, results, the caller's normalized row and token, peers."""
        out = "draft" if self.proposal is None else "draft.propose"
        results = self.results
        return {
            "X": self.x.data_ptr(),
            "NormWeight": self.address("mtp.shared_head_norm"),
            "LmHead": self.address("lm_head"),
            "Logits": results["logits"].data_ptr(),
            "Partials": results["partials"].data_ptr(),
            "RankRecord": results["rank_record"].data_ptr(),
            "Winner": results["winner"].data_ptr(),
            "Done": self.done.data_ptr(),
            # This launch's rows of the part's normalized hidden rows and tokens.
            "NormRows": self.word_at(
                f"{out}.hidden", self.index * LAUNCH_ROWS * self.geometry.hidden
            ),
            "Tokens": self.word_at(f"{out}.tokens", self.index * LAUNCH_ROWS),
            "peer_rank": self.rank,
            **_peers(
                "peer_base",
                peers(DRAFT_TOP1.format(part=self.node(self.part))),
            ),
        }


class DraftPart(list[DraftProvider]):
    """One part's providers in launch order, and the row table its kernels read (the
    host rewrites it before each replay of a tier above one request).
    """

    def __init__(
        self, launches: list[DraftProvider], table: RowTable, rows_per_request: int
    ) -> None:
        super().__init__(launches)
        self.table: RowTable = table
        self.rows_per_request: int = rows_per_request


def part(
    geometry: Glm53Geometry,
    proposal: int | None,
    rank: int,
    inputs: Mapping[str, torch.Tensor],
    placement: Glm53Placement,
    requests: int = 1,
) -> DraftPart:
    """Return one part's providers on ``rank`` in launch order (module docstring)."""
    group_rows = geometry.rows if proposal is None else 1
    table = RowTable(
        row_table.RowTableLayout(group_rows * requests, requests),
        inputs["position"].device,
    )
    table.set_uniform_step(requests, group_rows)
    inputs = ChainMap({ROW_TABLE_INPUT: table.tensor}, inputs)  # type: ignore[arg-type]
    prologue = DraftPrologue(geometry, proposal, rank, inputs, requests)
    preprocess: list[DraftPreprocess] = []
    for index in range(preprocess_launches(proposal, requests)):
        preprocess.append(
            DraftPreprocess(
                geometry,
                proposal,
                rank,
                inputs,
                prologue.hidden,
                requests,
                index,
                first=preprocess[0] if preprocess else None,
            )
        )
    x = preprocess[0].final
    launches: list[DraftProvider] = [prologue, *preprocess]
    if proposal is None and rank < geometry.selector_ranks:
        launches.append(
            CatchupSelector(geometry, rank, inputs, x, prologue.freq, requests)
        )
    worker = DraftWorker(geometry, proposal, rank, inputs, x, prologue.freq, requests)
    launches.append(worker)
    for index in range(row_launches(requests)):
        first = index * LAUNCH_ROWS
        moe = DraftMoe(
            geometry,
            proposal,
            rank,
            inputs,
            worker.final[first : first + LAUNCH_ROWS],
            placement,
            requests,
            index,
        )
        launches += [
            moe,
            DraftHead(geometry, proposal, rank, inputs, moe.final, requests, index),
        ]
    return DraftPart(launches, table, group_rows)


def program(
    geometry: Glm53Geometry,
    rank: int,
    inputs: Mapping[str, torch.Tensor],
    placement: Glm53Placement,
    requests: int = 1,
    proposals: int | None = None,
) -> list[DraftPart]:
    """Return rank ``rank``'s parts: the catch-up's providers, then each proposal's."""
    if not 0 <= rank < geometry.world:
        raise ValueError(f"rank {rank} is not one of {geometry.world} TP ranks")
    if requests != 1 and requests not in registry.DRAFT_REQUESTS:
        raise ValueError(f"no drafter tier of {requests} requests")
    count = proposals_of(geometry) if proposals is None else proposals
    return [
        part(geometry, proposal, rank, inputs, placement, requests)
        for proposal in (None, *range(count))
    ]


def draft_slices(
    geometry: Glm53Geometry, requests: int = 1, proposals: int | None = None
) -> list[Slice]:
    """Return the slices of the drafter's peer arena (every rank reserves the same)."""
    preprocess = sizes_of("mtp_preprocess_q4")["kMailboxBytes"]
    moe = sizes_of(DraftMoe.kernel)["kMailboxBytes"]
    top1 = sizes_of(HEAD_KERNEL if requests == 1 else HEAD_KERNEL_Q4)["kMailboxBytes"]
    selector = only_row_kernel(geometry, 0, requests)
    slices: list[Slice] = []
    count = proposals_of(geometry) if proposals is None else proposals
    for proposal in (None, *range(count)):
        name = _part(proposal)
        attention = sizes_of(worker_kernel(geometry, proposal, requests))[
            "kMailboxBytes"
        ]
        for i in range(preprocess_launches(proposal, requests)):
            node = _node("preprocess", i)
            slices.append(Slice(DRAFT_SLICE.format(part=name, node=node), preprocess))
        slices.append(Slice(DRAFT_SLICE.format(part=name, node="attention"), attention))
        for i in range(row_launches(requests)):
            slices += [
                Slice(DRAFT_SLICE.format(part=name, node=_node("moe", i)), moe),
                Slice(DRAFT_TOP1.format(part=_node(name, i)), top1),
            ]
    slices.append(Slice(DRAFT_SELECTION, sizes_of(selector)["kShardSelectionBytes"]))
    return slices

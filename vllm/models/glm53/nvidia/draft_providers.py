# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch providers of GLM-5.3's specialized MTP drafter (the MTP layer,
``draft.py``).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping

import torch

from vllm.models.glm53.nvidia import providers
from vllm.models.glm53.nvidia.arena import Glm53Placement, Slice
from vllm.models.glm53.nvidia.ops.cute_dsl import registry
from vllm.models.glm53.nvidia.ops.cute_dsl.lm_head import CTAS as HEAD_CTAS
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_selector import StateWords
from vllm.models.glm53.nvidia.ops.launch import (
    BLOCK_TABLE,
    EMBEDDING_SLICE,
    Argument,
    Glm53Geometry,
    sizes_of,
)
from vllm.models.glm53.nvidia.providers import Peers

DRAFT_SLICE = "glm53.mtp.{part}.{node}.rows"
DRAFT_SELECTION = "glm53.mtp.catchup.attention.selection"
DRAFT_TOP1 = "glm53.head.mtp.{part}.top1"
HEAD_KERNEL = "lm_head_top1_q1"
CONTEXT_WORDS = 2
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


def proposals_of(geometry: Glm53Geometry) -> int:
    """Return the proposals after the catch-up."""
    return geometry.rows - 2


def _suffix(proposal: int | None) -> str:
    """Return the site suffix of the catch-up (``""``) or proposal ``proposal`` (``#<k +
    1>``).
    """
    return "" if proposal is None else f"#{proposal + 1}"


def _part(proposal: int | None) -> str:
    """Return the state and slice prefix part: ``catchup`` or ``propose<k>``."""
    return "catchup" if proposal is None else f"propose{proposal}"


def _peers(prefix: str, addresses: list[int]) -> dict[str, Argument]:
    """Return ``{<prefix><p>: address}`` for every rank."""
    return {f"{prefix}{peer}": address for peer, address in enumerate(addresses)}


def only_row_kernel(geometry: Glm53Geometry, rank: int) -> str:
    """Return the catch-up's only-row selector entry on ``rank``, for the index-key
    cache's scale format (as ``Glm53Geometry.selector_kernel``).
    """
    suffix = "_mx" if geometry.index_scale_packed else ""
    return (
        f"mla_selector_q4_shard{geometry.selector_ranks}r{rank}_only_row_paged{suffix}"
    )


def attention_site(
    geometry: Glm53Geometry, proposal: int | None, selector: bool = False
) -> str:
    """Return the site of a part's draft worker (``selector``: the catch-up's
    selector).
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
    ) -> None:
        super().__init__(geometry, inputs["position"].device, inputs)
        self.proposal: int | None = proposal
        self.rank: int = rank
        #: Rows of this part: the step's rows for the catch-up, one for a proposal.
        self.part_rows: int = geometry.rows if proposal is None else 1
        self.part: str = _part(proposal)
        self.suffix: str = _suffix(proposal)
        #: The MTP layer's input names (``layers.<L>``).
        self.layer_key: str = f"layers.{mtp_layer(geometry)}"

    def part_hidden(self, name: str) -> torch.Tensor:
        """Allocate BF16 result ``name`` of the part's hidden rows."""
        return self.result(name, (self.part_rows, self.geometry.hidden), torch.bfloat16)

    @abstractmethod
    def values(self, peers: Peers) -> dict[str, Argument]:
        """Return the host-wrapper arguments by name (a missing pointer is null)."""
        raise NotImplementedError

    def arguments(self, peers: Peers) -> list[Argument]:
        """The named values in the entry's argument order."""
        values = self.values(peers)
        names = registry.arguments(self.kernel)
        unknown = sorted(set(values) - set(names))
        if unknown:
            raise ValueError(f"{self.kernel}: unknown arguments {unknown}")
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
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs)
        self.kernel = "prologue_vllm_q4" if proposal is None else "prologue_vllm"
        self.site = f"prologue_vllm{self.suffix}"
        self.hidden: torch.Tensor = self.part_hidden("hidden")
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
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs)
        self.kernel = "mtp_preprocess_q4" if proposal is None else "mtp_preprocess_q1"
        self.site = f"mtp_preprocess{self.suffix}"
        hidden = "draft.rows" if proposal is None else "draft.propose.row"
        self.rows_in: int = (
            embedded.data_ptr() if rank < geometry.world // 2 else self.address(hidden)
        )
        self.part_hidden("local_o")
        self.final: torch.Tensor = self.part_hidden("final")
        self.owned(f"mtp.{self.part}.preprocess.full_epoch", 1)

    def values(self, peers: Peers) -> dict[str, Argument]:
        """Rows, weights, position, results, launch ticket, peers."""
        return {
            "Rows": self.rows_in,
            "Gamma": self.address("mtp.preprocess.gamma"),
            "Weight": self.address("mtp.preprocess.eh_weight"),
            "Position": self.address(self.word("position")),
            "LocalO": self.results["local_o"].data_ptr(),
            "Final": self.final.data_ptr(),
            "FullEpoch": self.state[
                f"mtp.{self.part}.preprocess.full_epoch"
            ].data_ptr(),
            "peer_rank": self.rank,
            **_peers(
                "peer_base",
                peers(DRAFT_SLICE.format(part=self.part, node="preprocess")),
            ),
        }


class CatchupSelector(DraftProvider):
    """The catch-up's only-row selector shard (ranks 0-2) over the MTP layer's index
    keys.
    """

    WEIGHTS = {
        "GammaX": "gamma_x",
        "QakiWeight": "input_weight",
        "WisWeight": "wis_weight",
        "KeyGamma": "key_gamma",
        "KeyBias": "key_bias",
        "GammaQ": "gamma_q",
        "WqbWeight": "wqb_weight",
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
    ) -> None:
        super().__init__(geometry, None, rank, inputs)
        self.x: torch.Tensor = x
        self.freq: torch.Tensor = freq
        self.kernel = only_row_kernel(geometry, rank)
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
        self.owned(self.prefix + "full_epoch", 1)
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
        """Reset the selector's seed words (stream-ordered)."""
        seeds = StateWords.SEEDS
        self.select_state[seeds : seeds + self.seeds.numel()].copy_(self.seeds)

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
        "GammaQ": "gamma_q",
        "QueryWeight": "query_weight",
        "GammaKV": "gamma_kv",
        "KVbWeight": "kvb_weight",
        "OWeight": "o_weight",
    }

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
        freq: torch.Tensor,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs)
        self.x: torch.Tensor = x
        self.freq: torch.Tensor = freq
        self.kernel = f"mla_worker_draft_q{self.part_rows}_paged"
        self.site = attention_site(geometry, proposal)
        sizes = sizes_of(self.kernel)
        f32 = torch.float32.itemsize
        self.result("partial_o", (sizes["kPartialOBytes"] // f32,), torch.float32)
        self.result("partial_lse", (sizes["kPartialLseBytes"] // f32,), torch.float32)
        # The worker repeats its attended row over the MoE's T rows.
        self.final: torch.Tensor = self.rows("final")
        self.prefix: str = (
            "mtp.catchup.worker." if proposal is None else f"mtp.{self.part}.attention."
        )
        self.owned(self.prefix + "full_epoch", 1)
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
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs)
        self.x: torch.Tensor = x
        self.site = f"layer{mtp_layer(geometry)}.fused_moe{self.suffix}"
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
        self.prefix: str = f"mtp.{self.part}.ffn."
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
                "peer_base", peers(DRAFT_SLICE.format(part=self.part, node="moe"))
            ),
        }


class DraftHead(DraftProvider):
    """The BF16 draft head: shared-head RMSNorm, the rank's vocabulary shard, TP
    top-1.
    """

    kernel = HEAD_KERNEL

    def __init__(
        self,
        geometry: Glm53Geometry,
        proposal: int | None,
        rank: int,
        inputs: Mapping[str, torch.Tensor],
        x: torch.Tensor,
    ) -> None:
        super().__init__(geometry, proposal, rank, inputs)
        self.x: torch.Tensor = x
        self.site = f"lm_head_top1{self.suffix}"
        sizes = sizes_of(self.kernel)
        rows = sizes["kQTokens"]
        self.result("logits", (rows, sizes["kVocabShard"]), torch.float32)
        # Per CTA: its best logit and that logit's index.
        self.result("partials", (rows, HEAD_CTAS, 2), torch.float32)
        self.result("rank_record", (rows, sizes["kRecordWords"]), torch.int32)
        # The TP winner: its logit's bits and its index.
        self.result("winner", (rows, 2), torch.int32)
        self.done: torch.Tensor = self.owned(
            f"mtp.{self.part}.head.done", sizes["kDoneWords"]
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
            "NormRows": self.address(f"{out}.hidden"),
            "Tokens": self.address(f"{out}.tokens"),
            "peer_rank": self.rank,
            **_peers("peer_base", peers(DRAFT_TOP1.format(part=self.part))),
        }


def part(
    geometry: Glm53Geometry,
    proposal: int | None,
    rank: int,
    inputs: Mapping[str, torch.Tensor],
    placement: Glm53Placement,
) -> list[DraftProvider]:
    """Return one part's providers on ``rank`` in launch order (module docstring)."""
    prologue = DraftPrologue(geometry, proposal, rank, inputs)
    preprocess = DraftPreprocess(geometry, proposal, rank, inputs, prologue.hidden)
    launches: list[DraftProvider] = [prologue, preprocess]
    if proposal is None and rank < geometry.selector_ranks:
        launches.append(
            CatchupSelector(geometry, rank, inputs, preprocess.final, prologue.freq)
        )
    worker = DraftWorker(
        geometry, proposal, rank, inputs, preprocess.final, prologue.freq
    )
    moe = DraftMoe(geometry, proposal, rank, inputs, worker.final, placement)
    return [
        *launches,
        worker,
        moe,
        DraftHead(geometry, proposal, rank, inputs, moe.final),
    ]


def program(
    geometry: Glm53Geometry,
    rank: int,
    inputs: Mapping[str, torch.Tensor],
    placement: Glm53Placement,
) -> list[list[DraftProvider]]:
    """Return rank ``rank``'s parts: the catch-up's providers, then each proposal's."""
    if not 0 <= rank < geometry.world:
        raise ValueError(f"rank {rank} is not one of {geometry.world} TP ranks")
    proposals = (None, *range(proposals_of(geometry)))
    return [part(geometry, proposal, rank, inputs, placement) for proposal in proposals]


def draft_slices(geometry: Glm53Geometry) -> list[Slice]:
    """Return the slices of the drafter's peer arena (every rank reserves the same)."""
    preprocess = sizes_of("mtp_preprocess_q4")["kMailboxBytes"]
    rows = sizes_of("mla_worker_draft_q4_paged")["kMailboxBytes"]
    moe = sizes_of(DraftMoe.kernel)["kMailboxBytes"]
    top1 = sizes_of(HEAD_KERNEL)["kMailboxBytes"]
    selector = only_row_kernel(geometry, 0)
    slices: list[Slice] = []
    for name in ["catchup", *(f"propose{k}" for k in range(proposals_of(geometry)))]:
        slices += [
            Slice(DRAFT_SLICE.format(part=name, node="preprocess"), preprocess),
            Slice(DRAFT_SLICE.format(part=name, node="attention"), rows),
            Slice(DRAFT_SLICE.format(part=name, node="moe"), moe),
            Slice(DRAFT_TOP1.format(part=name), top1),
        ]
    slices.append(Slice(DRAFT_SELECTION, sizes_of(selector)["kShardSelectionBytes"]))
    return slices

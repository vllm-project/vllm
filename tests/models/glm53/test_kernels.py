# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 kernel registry and launch program, on the CPU."""

import struct
from collections.abc import Sequence
from types import SimpleNamespace

import torch

from vllm.models.glm53.nvidia import draft_providers, providers, switch
from vllm.models.glm53.nvidia.ops import launch as glm53_launch
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_selector import IndexScale, StateWords
from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import Handoff, MlaWorker


class _AnyTensors(dict):
    """Every input resolves to a small CPU tensor (the target providers); no replicated
    embedding.
    """

    def __missing__(self, name: str) -> torch.Tensor:
        if name == glm53_launch.BLOCK_TABLE:
            tensor = torch.zeros(16384 // 64, dtype=torch.int32)
        elif name == "rope_table":
            tensor = torch.zeros(16384, 64)
        else:
            tensor = torch.zeros(8, dtype=torch.int32)
        self[name] = tensor
        return tensor

    def __contains__(self, name: object) -> bool:
        return name != "embedding"


class _ReplicatedEmbedding(_AnyTensors):
    """A replicated ``embedding`` table instead of vLLM's shards."""

    def __contains__(self, name: object) -> bool:
        return True


class _Placement:
    """CPU stand-in for ``Glm53Placement``: plain handoff and flags (``issued``: the
    ``moe_flags`` pairs handed out, in order).
    """

    def __init__(self) -> None:
        self.issued: list[tuple[torch.Tensor, torch.Tensor]] = []

    def worker_handoff(self, by_sm: bool, groups: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            handoff=torch.zeros(groups * Handoff.WORDS, dtype=torch.int32),
            mirror=torch.zeros(groups * Handoff.WORDS, dtype=torch.int32),
        )

    def moe_flags(self, rows: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
        words = glm53_launch.sizes_of(f"fused_moe_q{rows}_ckpt_shared")["kFlagWords"]
        pair = (
            torch.zeros(words, dtype=torch.int32),
            torch.zeros(words, dtype=torch.int32),
        )
        self.issued.append(pair)
        return pair


def _bound(program: Sequence[providers.Provider]) -> list[glm53_launch.Launch]:
    launches = []
    for provider in program:
        provider.bind(lambda name: [peer << 40 for peer in range(8)])
        assert provider.launch is not None
        launches.append(provider.launch)
    return launches


def _deployment_geometries(
    shape: glm53_launch.Glm53Geometry,
) -> list[glm53_launch.Glm53Geometry]:
    geometries: list[glm53_launch.Glm53Geometry] = []
    for per_request in range(1, 9):
        for requests in range(1, switch.MAX_REQUESTS + 1):
            if not switch.serves_shape(requests, per_request):
                continue
            tier = switch.shape_tier(requests, per_request)
            packed = switch.packed_tier(tier, per_request)
            rows = per_request if packed else switch.lone_rows(per_request)
            w8a16 = switch.shape_w8a16_rows(requests, per_request)
            live = 0 if packed else per_request
            geometry = shape.for_tier(tier, rows - 1, packed, w8a16, live)
            if geometry in geometries:
                continue
            geometries.append(geometry)
    return geometries


def test_shipped_kernels_are_the_served_kernels(geometry) -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    launched: list[glm53_launch.Launch] = []
    for tier in glm53_launch.TIERS:
        tier_geometry = geometry.for_tier(tier, geometry.drafts)
        for rank in range(geometry.world):
            for inputs in (_AnyTensors(), _ReplicatedEmbedding()):
                launched += _bound(
                    providers.program(tier_geometry, rank, inputs, _Placement())
                )
    # The drafter runs beside the one-request build, and has its own request tiers.
    for requests in (1, *registry.DRAFT_REQUESTS):
        for rank in range(geometry.world):
            for part in draft_providers.program(
                geometry, rank, _AnyTensors(), _Placement(), requests
            ):
                launched += _bound(part)
    # The shapes of every MTP K and selection split.
    for tier_geometry in _deployment_geometries(geometry):
        for rank in range(geometry.world):
            for inputs in (_AnyTensors(), _ReplicatedEmbedding()):
                launched += _bound(
                    providers.program(tier_geometry, rank, inputs, _Placement())
                )
    for launch in launched:
        assert len(launch.arguments) == len(registry.arguments(launch.kernel)), (
            launch.site
        )
    kernels = {launch.kernel for launch in launched}
    assert set(registry.ENTRIES) == kernels, sorted(set(registry.ENTRIES) ^ kernels)


def test_b2_q8_launches_use_eight_row_buffers_and_mirrored_flags(geometry) -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    shaped = geometry.for_tier(8, drafts=3)
    placement = _Placement()
    parts = providers.program(shaped, 0, _AnyTensors(), placement, layers=8)
    ffns = [p for p in parts if isinstance(p, providers.Ffn) and p.layer >= 3]
    assert {p.kernel for p in ffns} == {
        "fused_moe_q8_ckpt_shared",
        "fused_moe_q8_seam_ckpt_shared",
    }
    assert shaped.moe_rows == shaped.buffer_rows == 8
    for part in ffns:
        assert part.work is None
        assert part.results["router_logits"].shape == (8, 256)
        assert part.results["activation"].shape == (65 * 8, 256)
        assert part.results["route_ids"].shape == (8, 8)
        assert (
            part.slices[0].size == registry.ENTRIES[part.kernel].host["kMailboxBytes"]
        )
        key = f"layers.{part.layer}.ffn.flags"
        assert (
            part.state[key].numel() == registry.ENTRIES[part.kernel].host["kFlagWords"]
        )
        assert key + "_mirror" in part.state
        launch = _bound([part])[0]
        names = registry.arguments(part.kernel)
        assert "Work" not in names
        assert len(launch.arguments) == len(names)
        assert (
            launch.arguments[names.index("Activation")]
            == part.results["activation"].data_ptr()
        )


def test_draft_tier_blocks_are_the_workers_group_strides() -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry
    from vllm.models.glm53.nvidia.ops.cute_dsl.lm_head import DoneWords, Mailbox
    from vllm.models.glm53.nvidia.ops.cute_dsl.mla_worker import PUSH_MAILBOX_BYTES

    for requests in (1, *registry.DRAFT_REQUESTS):
        tier = requests > 1
        for name, group_rows in (
            (
                f"mla_worker_draft_r{4 * requests}_paged"
                if tier
                else "mla_worker_draft_q4_paged",
                4,
            ),
            (
                f"mla_worker_draft_p{requests}_paged"
                if tier
                else "mla_worker_draft_q1_paged",
                1,
            ),
        ):
            worker = registry.kernel(name)
            sizes = registry.ENTRIES[name].host
            assert group_rows == worker.GROUP_ROWS and requests == worker.GROUPS
            strides = worker.GROUP_STRIDES
            assert strides["Handoff"] * requests == sizes["kHandoffWords"]
            assert strides["PartialO"] * 4 * requests == sizes["kPartialOBytes"]
            assert strides["PartialLSE"] * 4 * requests == sizes["kPartialLseBytes"]
            assert strides["Flags"] * requests == sizes["kFlagWords"]
            if tier:
                assert sizes["kMailboxBytes"] == PUSH_MAILBOX_BYTES * requests
    # The four-row head: one ticket, the CTAs' records and the two tail counters.
    head = registry.ENTRIES[draft_providers.HEAD_KERNEL_Q4].host
    assert head["kDoneWords"] == DoneWords.counter(4) + 2
    assert head["kMailboxBytes"] == Mailbox.packet(1, 7, 3) + 16


def test_draft_step_words_hold_every_request() -> None:
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    for requests in (1, *registry.DRAFT_REQUESTS):
        words = draft_providers.step_words(requests)
        assert set(words) <= set(draft_providers.WORDS)
        # Two words a row, four rows a request: the layout of vLLM's positions.
        assert words["position"] == 8 * requests
        assert words["draft.propose.position"] == 2 * requests
        assert words["draft.tokens"] == draft_providers.buffer_rows(requests)
        assert words["draft.tokens"] >= requests


def test_drafter_moe_flags_are_placed_like_the_targets(geometry) -> None:
    placement = _Placement()
    parts = draft_providers.program(geometry, 0, _AnyTensors(), placement)
    assert len(placement.issued) == len(parts)
    for part, (flags, mirror) in zip(parts, placement.issued):
        moe = next(p for p in part if isinstance(p, draft_providers.DraftMoe))
        assert moe.state[moe.prefix + "flags"] is flags
        assert moe.state[moe.prefix + "flags_mirror"] is mirror
    target = providers.program(geometry, 0, _AnyTensors(), _Placement())
    ffn = next(p for p in target if isinstance(p, providers.Ffn) and p.layer > 3)
    assert {n.rsplit(".", 1)[-1] for n in ffn.state} >= {"flags", "flags_mirror"}


def test_layer_layout(geometry) -> None:
    layers = range(geometry.layers)
    shared = [layer for layer in layers if geometry.is_shared(layer)]
    assert shared[:6] == [3, 4, 5, 7, 8, 9] and len(shared) == 78 - 3 - 18
    kinds = [geometry.ffn_kind(layer) for layer in layers]
    assert kinds[:7] == ["dense"] * 3 + ["moe_seam", "moe_seam", "moe", "moe_seam"]
    assert kinds[-1] == "moe" and kinds.count("moe") == 19
    assert kinds.count("moe_seam") == 56


def test_placed_handoff_layout() -> None:
    assert Handoff.ROLE_TABLE == Handoff.VO + 4 * Handoff.VO_ROW
    assert Handoff.ROLE_TABLE + MlaWorker.GRID <= Handoff.SPLIT_DONE
    assert Handoff.SPLIT_DONE + 4 * 32 <= Handoff.WORDS
    assert Handoff.WORDS % 32 == 0 and Handoff.WORDS < Handoff.MIRROR
    # The registry's tier blocks are whole placed Handoffs.
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    assert registry._HANDOFF_BLOCK == Handoff.WORDS


def test_seed_words(geometry) -> None:
    Selector = providers.Selector
    assert Selector.nominal_seed_words("layers.2", 4) == [0] * 4
    key = providers.ordered_key(100.0) | 1
    assert Selector.nominal_seed_words("layers.1", 4) == [key - (1 << 32)] * 4
    inputs = _AnyTensors()
    x = torch.zeros(geometry.rows, geometry.hidden, dtype=torch.bfloat16)
    selector = Selector(geometry, 1, 0, inputs, x)
    selector.reset_seeds()
    state = selector.select_state
    seeds = state[StateWords.SEEDS : StateWords.SEEDS + 4]
    assert seeds.tolist() == [key - (1 << 32)] * 4
    assert int(state.count_nonzero()) == 4


def test_packed_seed_words_leave_absent_rows_unknown(geometry) -> None:
    nominal = (providers.ordered_key(100.0) | 1) - (1 << 32)
    for drafts in (0, 1, 2, 4, 5, 6):
        per_request = drafts + 1
        shaped = geometry.for_tier(16, drafts, True, False)
        x = torch.zeros(shaped.rows, shaped.hidden, dtype=torch.bfloat16)
        selector = providers.Selector(shaped, 1, 0, _AnyTensors(), x)
        words = selector.seeds.view(-1, 4).tolist()
        assert len(words) == shaped.selector_slots
        lives = set()
        for slot, row in enumerate(words):
            live = providers.Selector.packed_slot_rows(per_request, slot)
            lives.add(live)
            assert row == [nominal] * live + [0] * (4 - live), (drafts, slot)
        # One slot per request up to four rows; two (four rows, the rest) above.
        assert lives == ({per_request} if per_request <= 4 else {4, per_request - 4})


def test_index_scale_words() -> None:
    from cutlass import Uint32

    for exponent in range(1, 255):
        scale = struct.unpack("<f", struct.pack("<I", exponent << 23))[0]
        assert scale == 2.0 ** (exponent - 127)
        packed = ((exponent << 23) >> 23 & 0xFF) * 0x01010101
        assert int(IndexScale.word(Uint32(exponent))) == packed
        decoded = struct.unpack("<f", struct.pack("<I", (packed & 255) << 23))[0]
        assert decoded == scale

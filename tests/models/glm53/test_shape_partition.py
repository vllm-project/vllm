# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Every decode shape (MTP depth K, request count B) is claimed by exactly one path."""

import pytest
import regex as re

from vllm.models.glm53.nvidia import switch
from vllm.models.glm53.nvidia.switch import ShapePath

MAX_B = 17  # one past the request capacity: always vLLM's stock forward

TABLE = {
    0: "Q4 P8q4 P8q4 P8q4 P8q8 P8q8 P8q8 P8q8",
    1: "Q4 P8q4 P8q8 P8q8 P16 P16 P16 P16",
    2: "Q4 P8q8 P16 P16 P16 S S S",
    3: "Q4 A8 A16 A16 S S S S",
    4: "L8 P16 P16 S S S S S",
    5: "L8 P16 S S S S S S",
    6: "L8 P16 S S S S S S",
    7: "L8 A16 S S S S S S",
}
CODES = {
    ShapePath.Q4: "Q4",
    ShapePath.LONE8: "L8",
    ShapePath.ALIGNED: "A",
    ShapePath.PACK: "P",
    ShapePath.STOCK: "S",
}


def cell(requests: int, drafts: int) -> str:
    """The table cell of a shape: the path's code, then its tier when it runs on one."""
    per_request = 1 + drafts
    path = switch.shape_path(requests, per_request)
    code = CODES[path]
    if path in (ShapePath.ALIGNED, ShapePath.PACK):
        w8a16 = switch.shape_w8a16_rows(requests, per_request)
        moe = f"q{w8a16}" if w8a16 else ""
        return f"{code}{switch.shape_tier(requests, per_request)}{moe}"
    return code


def shapes():
    """Every (K, B) of K = 0..7 and B = 1..17."""
    return [(k, b) for k in range(8) for b in range(1, MAX_B + 1)]


def test_the_claim_function_gives_the_table():
    found = {k: " ".join(cell(b, k) for b in range(1, 9)) for k in range(8)}
    assert found == TABLE


@pytest.mark.parametrize(("drafts", "requests"), shapes())
def test_a_shape_has_one_path_and_the_arithmetic_agrees(drafts, requests):
    per_request = 1 + drafts
    rows = requests * per_request
    path = switch.shape_path(requests, per_request)
    assert isinstance(path, ShapePath)
    stock = requests > 16 or rows > 16
    assert (path is ShapePath.STOCK) == stock
    assert switch.serves_shape(requests, per_request) == (not stock)
    assert switch.shape_w8a16_rows(requests, per_request) == (
        (4 if rows <= 4 else 8) if path is ShapePath.PACK and rows <= 8 else 0
    )
    assert (path is ShapePath.Q4) == (requests == 1 and per_request <= 4)
    assert (path is ShapePath.LONE8) == (requests == 1 and per_request > 4)
    assert (path is ShapePath.ALIGNED) == (
        requests >= 2 and per_request % 4 == 0 and not stock
    )
    assert (path is ShapePath.PACK) == (
        requests >= 2 and per_request % 4 != 0 and not stock
    )
    # --max-num-seqs = B: the graphs claim the served counts, stock steps none.
    served = switch.served_requests(requests, per_request)
    assert (served < requests) == stock
    found = switch.graphs(requests, per_request)
    claimed = {b for graph in found for b in graph.requests}
    assert claimed == set(range(1, served + 1))
    assert switch.derived_tiers(requests, per_request)


@pytest.mark.parametrize(("drafts", "requests"), shapes())
def test_the_tier_and_the_entry_names_follow_the_claim(drafts, requests, geometry):
    per_request = 1 + drafts
    path = switch.shape_path(requests, per_request)
    if path is ShapePath.STOCK:
        return
    tier = switch.shape_tier(requests, per_request)
    rows = requests * per_request
    packed = switch.packed_tier(tier, per_request)
    assert packed == (path is ShapePath.PACK)
    assert tier >= rows
    kernel_rows = per_request if packed else switch.lone_rows(per_request)
    w8a16 = switch.shape_w8a16_rows(requests, per_request)
    shaped = geometry.for_tier(
        tier, kernel_rows - 1, packed, w8a16, 0 if packed else per_request
    )
    if packed:  # the q4 build (one request) has its own MoE whatever the flag says
        assert ("fused_moe_q4_" in shaped.moe_kernel(3)) == (w8a16 == 4)
    assert shaped.max_requests >= requests
    for layer in (3, 4):
        uses_q8 = shaped.moe_kernel(layer).startswith("fused_moe_q8")
        # The tier-8 builds: K = 3 at B = 2, and a lone request of five to eight rows.
        lone8 = requests == 1 and drafts >= 4
        assert uses_q8 == (w8a16 == 8 or (requests, drafts) == (2, 3) or lone8)
    names = [
        shaped.selector_kernel(0),
        shaped.worker_kernel(0, 0),
        shaped.worker_kernel(1, 0),
        shaped.prologue_kernel(False),
    ]
    for name in names:
        assert bool(re.search(r"_r\d+p(_|$)", name)) == packed, name
    for name in (shaped.moe_kernel(3), shaped.dense_kernel):
        assert not re.search(r"r\d+p", name), name
    dead_rows = path is ShapePath.LONE8 and per_request < kernel_rows
    assert bool(re.search(r"d\d+x\d+", shaped.moe_kernel(3))) == dead_rows
    assert not re.search(r"d\d+x\d+", shaped.dense_kernel), shaped.dense_kernel


@pytest.mark.parametrize(
    ("drafts", "tag"),
    [(4, "q8d8x5"), (5, "q8d8x6"), (6, "q8d8x7"), (7, "q8")],
)
def test_a_lone_request_with_dead_rows_runs_the_masked_moe(geometry, drafts, tag):
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    per_request = drafts + 1
    assert switch.shape_path(1, per_request) is ShapePath.LONE8
    kernel_rows = switch.lone_rows(per_request)
    assert kernel_rows == 8
    shaped = geometry.for_tier(8, kernel_rows - 1, False, False, per_request)
    assert shaped.moe_tag == tag
    for layer in (3, 4):
        knobs = dict(registry.ENTRIES[shaped.moe_kernel(layer)].constants)
        dead = (knobs.get("DEAD_STRIDE", 0), knobs.get("DEAD_LIVE", 0))
        assert dead == ((8, per_request) if per_request < 8 else (0, 0))

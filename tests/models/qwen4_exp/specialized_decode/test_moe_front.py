# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TP 1 routed MoE path of Qwen4Exp's specialized decode (`RoutedPath`):
the CuTe DSL front (`moe/front.py`: top-10 routing, routing metadata and
NVFP4 quantization in one launch) and FlashInfer's expert body on its record
(`runtime/moe.py`). The path replaces the stock quantization and routing
of calls of up to 32 rows and must leave every output bit unchanged.

- Bits (GB300/B200). Against FlashInfer's FP4 MoE operation routing from the
  logits, with the same tactic in both arms (FlashInfer's fallback, which
  the served 20-32-row calls run, and a tuned 8-row tile): the top-10 ids
  and BF16 weights, the quantized input, the routing record's padded rows,
  tile count and per-tile expert and row limit, the finalized output and,
  for the folded finalize, every GEMM2 row a token reads with its weight.
  At 1, 4, 5, 8, 16, 17, 20, 24, 31 and 32 rows (FlashInfer's routing kernel
  changes at 5, 17 and 32 rows), on random logits and on concentrated
  routing: every row the same logits, groups of four rows sharing logits
  (the verify rows of one request), and one expert taken by every row.
- Weight rounding. The front's weights must repeat FlashInfer's arithmetic,
  not only its function: a softmax over all experts renormalized over the
  ten rounds one weight in about 10^5 to the other BF16 neighbour. Two
  million random weights at every row count from 1 to 32 must all match.
- PDL. A front that sleeps after its wait keeps the output bits, also when
  it triggers at entry so that GEMM1 waits through the sleep (a GEMM reading
  its record before its own wait would read another input's); so do CUDA
  graph replays.
- Order. `runtime/moe.py` states the ordering argument; a simulation of two
  decode steps applies CUDA's PDL rules to the front's global accesses (read
  from its kernel source) and FlashInfer's GEMMs, and catches injected
  defects.
- Dispatch (CPU). `Boundaries.moe_output` runs the routed path for a call
  with a tactic, with the runner's gate and shared expert in the runner's
  order, and the stock runner otherwise.
"""

import ast
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch

from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock

Access = tuple[str, str]  # (wait | trigger | load | store, buffer)
WAIT: Access = ("wait", "")
TRIGGER: Access = ("trigger", "")
GLOBAL_LOADS = ("load_bf16", "load_bf16x8", "load_f32")
GLOBAL_STORES = ("store_u32", "store_u8", "store_u32x2", "store_bf16")
# The front's record buffers (its outputs).
RECORD = (
    "x_fp4",
    "x_scale",
    "topk_ids",
    "topk_weights",
    "padded_rows",
    "permuted_of",
    "token_of",
    "tile_expert",
    "tile_limit",
    "tiles",
)
LAYERS = 3
REPLAYS = 2
FRONT_SOURCE = "models/qwen4_exp/nvidia/specialized_decode/moe/front.py"


@dataclass(frozen=True)
class Node:
    """A kernel: whether it is launched with the PDL attribute, and its
    global accesses and PDL controls in program order."""

    name: str
    pdl: bool
    accesses: tuple[Access, ...]


def _root(expression: ast.expr, aliases: dict[str, str]) -> str | None:
    """The pointer parameter an address expression starts from."""
    while isinstance(expression, ast.BinOp):
        expression = expression.left
    if isinstance(expression, ast.Name):
        return aliases.get(expression.id, expression.id)
    return None


def front_accesses(early_trigger: bool = True) -> list[Access]:
    """The front kernel's global loads and stores (named by pointer
    parameter), waits and triggers in program order: `RoutedFront.kernel`
    with the methods it calls expanded in place."""
    # Located without importing the package: the check needs no GPU stack.
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.origin is not None
    tree = ast.parse((Path(spec.origin).parent / FRONT_SOURCE).read_text())
    front = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RoutedFront"
    )
    methods = {
        node.name: node for node in front.body if isinstance(node, ast.FunctionDef)
    }

    def walk(function: ast.FunctionDef) -> list[Access]:
        """The accesses of `function`, calls to other methods expanded."""
        aliases: dict[str, str] = {}
        assignments = sorted(
            (
                (node.lineno, node)
                for node in ast.walk(function)
                if isinstance(node, ast.Assign) and len(node.targets) == 1
            ),
            key=lambda item: item[0],
        )
        for _, node in assignments:
            target = node.targets[0]
            root = _root(node.value, aliases)
            if isinstance(target, ast.Name) and root is not None:
                aliases[target.id] = root
        events: list[tuple[int, int, list[Access]]] = []
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            accesses: list[Access] = []
            if name.endswith("griddepcontrol_wait"):
                accesses = [WAIT]
            elif name.endswith("griddepcontrol_launch_dependents"):
                accesses = [TRIGGER] if early_trigger else []
            elif name in GLOBAL_LOADS or name in GLOBAL_STORES:
                root = _root(node.args[0], aliases)
                assert root is not None, ast.unparse(node)
                kind = "load" if name in GLOBAL_LOADS else "store"
                accesses = [(kind, root)]
            elif name.startswith("self.") and name[5:] in methods:
                accesses = walk(methods[name[5:]])
            if accesses:
                events.append((node.lineno, node.col_offset, accesses))
        return [access for *_, accesses in sorted(events) for access in accesses]

    return walk(methods["kernel"])


def completed_before(nodes: list[Node]) -> list[list[frozenset[str]]]:
    """For each access of each node, the nodes that have completed, their
    stores visible, before the access runs: a node launched without PDL
    starts after its predecessor completed; a PDL node starts once its
    predecessor triggers (its first trigger, else its exit) and only its own
    wait waits for the predecessor to complete."""
    seen: list[list[frozenset[str]]] = []
    done: frozenset[str] = frozenset()  # complete once the previous node is
    released: frozenset[str] = frozenset()  # complete once it triggers
    for node in nodes:
        now = released if node.pdl else done
        trigger: frozenset[str] | None = None
        seen.append([])
        for access in node.accesses:
            if access == WAIT and node.pdl:
                now = now | done
            if access == TRIGGER and trigger is None:
                trigger = now
            seen[-1].append(now)
        released = now if trigger is None else trigger
        done = now | {node.name}
    return seen


def violations(nodes: list[Node]) -> list[str]:
    """Each access to a buffer that an earlier node stores, or a store to a
    buffer an earlier node accesses, without the earlier node complete."""
    found: list[str] = []
    for index, (node, seen) in enumerate(zip(nodes, completed_before(nodes))):
        for (kind, buffer), done in zip(node.accesses, seen):
            if kind not in ("load", "store"):
                continue
            for earlier in nodes[:index]:
                conflict = any(
                    other == buffer and "store" in (kind, other_kind)
                    for other_kind, other in earlier.accesses
                )
                if conflict and earlier.name not in done:
                    found.append(
                        f"{node.name} {kind}s {buffer} while {earlier.name} "
                        "may still run"
                    )
    return found


def gemm(name: str, loads: tuple[str, ...], store: str) -> Node:
    """A FlashInfer body kernel: waits, then loads and stores."""
    return Node(
        name, True, (WAIT, *(("load", buffer) for buffer in loads), ("store", store))
    )


def decode_steps(front: list[Access], gemm1: Node | None = None) -> list[Node]:
    """`REPLAYS` replays of `LAYERS` served layers, `NODES` kernels a
    layer: the router GEMM (no PDL), the front, GEMM1, GEMM2, the finalize,
    the boundary kernel and the next read (PDL, waiting before their
    accesses). Each layer owns its record; `gemm1` replaces GEMM1 (its
    record buffers named without a layer)."""
    nodes: list[Node] = []
    for replay in range(REPLAYS):
        for layer in range(LAYERS):
            tag = f"{replay}.{layer}"

            def buf(name: str, layer: int = layer) -> str:
                """The layer's buffer for a record buffer name."""
                return f"{name}[{layer}]" if name in RECORD else name

            x, logits, out = f"x[{tag}]", f"logits[{tag}]", f"routed[{tag}]"
            inputs = {"x": x, "logits": logits}
            first = gemm1 or gemm(
                "gemm1",
                ("token_of", "tile_expert", "tile_limit", "tiles", "padded_rows")
                + ("x_fp4", "x_scale"),
                "gemm1_out",
            )
            nodes += [
                Node(f"router {tag}", False, (WAIT, ("load", x), ("store", logits))),
                Node(
                    f"front {tag}",
                    True,
                    tuple((kind, inputs.get(name, buf(name))) for kind, name in front),
                ),
                replace(
                    first,
                    name=f"gemm1 {tag}",
                    accesses=tuple((kind, buf(name)) for kind, name in first.accesses),
                ),
                gemm(
                    f"gemm2 {tag}",
                    ("gemm1_out", buf("tile_expert"), buf("tile_limit"), buf("tiles")),
                    "gemm2_out",
                ),
                gemm(
                    f"finalize {tag}",
                    ("gemm2_out", buf("permuted_of"), buf("topk_weights")),
                    out,
                ),
                gemm(f"boundary {tag}", (out,), f"streams[{tag}]"),
                gemm(f"read {tag}", (f"streams[{tag}]",), f"x[{replay}.{layer + 1}]"),
            ]
    return nodes


NODES = 7  # kernels a layer in decode_steps
ROUTED = ("front", "gemm1", "gemm2", "finalize")


def launched_early(nodes: list[Node]) -> list[Node]:
    """`nodes` with every node outside the routed path launched with PDL
    and triggering at entry, its waits kept: the earliest that a node keeping
    the PDL contract lets its successor start. A replay's first node keeps
    its launch (a CUDA graph launch follows the previous one's completion)."""
    early = []
    for index, node in enumerate(nodes):
        first_of_replay = index % (NODES * LAYERS) == 0
        if node.name.split()[0] not in ROUTED and not first_of_replay:
            node = replace(node, pdl=True, accesses=(TRIGGER, *node.accesses))
        early.append(node)
    return early


def check(nodes: list[Node]) -> list[str]:
    """The order violations of `nodes` as modeled and with every other node
    launched early."""
    return [f"as modeled: {found}" for found in violations(nodes)] + [
        f"with the other nodes launched early: {found}"
        for found in violations(launched_early(nodes))
    ]


def test_front_accesses_come_from_the_kernel_source():
    accesses = front_accesses()
    assert accesses[0] == TRIGGER and accesses[1] == WAIT
    loads = {buffer for kind, buffer in accesses if kind == "load"}
    stores = {buffer for kind, buffer in accesses if kind == "store"}
    assert loads == {"logits", "x", "global_scale"}
    assert stores == set(RECORD)
    assert front_accesses(early_trigger=False)[0] == WAIT


def test_the_routed_path_orders_its_record():
    assert check(decode_steps(front_accesses(early_trigger=False))) == []


def test_no_trigger_position_breaks_the_order():
    accesses = front_accesses(early_trigger=False)
    for position in range(len(accesses) + 1):
        moved = accesses[:position] + [TRIGGER] + accesses[position:]
        assert check(decode_steps(moved)) == [], position


def _gemm1_reads_before_its_wait() -> Node:
    """GEMM1 gathering the activations before its wait."""
    return Node(
        "gemm1",
        True,
        (("load", "x_fp4"), ("load", "x_scale"), WAIT, ("store", "gemm1_out")),
    )


DEFECTS = {
    "the front loads before its wait": lambda: decode_steps(
        [access for access in front_accesses() if access != WAIT]
    ),
    "GEMM1 reads the record before its wait": lambda: decode_steps(
        front_accesses(), gemm1=_gemm1_reads_before_its_wait()
    ),
}


@pytest.mark.parametrize("defect", sorted(DEFECTS))
def test_the_order_check_catches_a_broken_order(defect):
    assert check(DEFECTS[defect]()) != []


def test_the_fallback_tactic_resolves_to_the_fronts_tile():
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.moe import (
        MAX_ROWS,
        fallback_tile,
    )

    # Up to 32 rows, the average rows per expert stay below the 8-row tile.
    assert {fallback_tile(rows, 10, 512) for rows in range(1, MAX_ROWS + 1)} == {8}
    assert fallback_tile(1024, 10, 512) == 16
    assert fallback_tile(8192, 10, 512) == 128


# On the GPU: the front and the body against FlashInfer's FP4 MoE operation
# routing from the logits, with the same tactic, on random NVFP4 experts of
# the model's shapes (every expert's bytes differ, so a wrong expert index
# changes the output).
requires_sm100 = pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(100),
    reason="the routed path requires SM100 (FlashInfer's TRT-LLM MoE)",
)
EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 512, 10, 2560, 640
FP8 = torch.float8_e4m3fn
FALLBACK = [-1, -1]
ROWS = (1, 4, 5, 8, 16, 17, 20, 24, 31, 32)
MODES = ("random", "same", "groups of four", "one expert in every row")


@pytest.fixture(scope="module")
def experts():
    """Random NVFP4 experts: packed FP4 bytes, E4M3 block scales in
    [0.25, 1) and per-expert output scales (as test_boundary_fold's)."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.moe import (
        RENORMALIZE_NAIVE,
        ExpertParams,
    )

    generator = torch.Generator(device="cuda").manual_seed(0)

    def randint(low: int, high: int, *shape: int) -> torch.Tensor:
        return torch.randint(
            low, high, shape, dtype=torch.uint8, generator=generator, device="cuda"
        )

    e, h, i = EXPERTS, HIDDEN, INTERMEDIATE
    scale = torch.full((e,), 1 / 64, dtype=torch.float32, device="cuda")
    return ExpertParams(
        gemm1_weights=randint(0, 256, e, 2 * i, h // 2),
        gemm1_weights_scale=randint(0x28, 0x38, e, 2 * i, h // 16).view(FP8),
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=randint(0, 256, e, h, i // 2),
        gemm2_weights_scale=randint(0x28, 0x38, e, h, i // 16).view(FP8),
        output1_scale_scalar=scale,
        output1_scale_gate_scalar=scale.clone(),
        output2_scale_scalar=scale.clone(),
        input_scale=torch.tensor([448.0], device="cuda"),
        num_experts=EXPERTS,
        top_k=TOP_K,
        intermediate_size=INTERMEDIATE,
        activation_type=3,  # FlashInfer's ActivationType.Swiglu
        routing_method_type=RENORMALIZE_NAIVE,
        tune_max_num_tokens=256,
    )


def _inputs(rows: int, seed: int, mode: str = "random"):
    """Router logits of `mode` (`MODES`) and an input of `rows` rows."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(rows, EXPERTS, device="cuda", generator=generator) * 1.5
    x = torch.randn(rows, HIDDEN, device="cuda", generator=generator)
    if mode == "same":
        logits = logits[:1].expand(rows, -1)
    elif mode == "groups of four":
        noise = torch.randn(rows, EXPERTS, device="cuda", generator=generator)
        logits = logits[torch.arange(rows, device="cuda") // 4 * 4] + noise * 0.02
    elif mode == "one expert in every row":
        logits = logits.clone()
        logits[:, 7] += 30.0
    else:
        assert mode == "random"
    return logits.bfloat16().contiguous(), x.bfloat16()


def _stock(params, logits, x, tactic, finalize):
    """VLLM's quantization, then FlashInfer's FP4 MoE operation routing from
    `logits` with `tactic`. Returns the quantized input, the top-10 ids,
    and the finalized output or, without `finalize`, the GEMM2 rows, the
    routing weights and slots (given the finalized-shape output, as the
    stock deferred call)."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.quantization.nvfp4_quantization_utils import (
        NVFP4_4OVER6_CODE_FROM_ENV,
    )

    from vllm import _custom_ops as ops

    rows = x.shape[0]
    xq, xs = ops.scaled_fp4_quant(x, params.input_scale, is_sf_swizzled_layout=False)
    output = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.empty(rows, TOP_K, dtype=torch.int32, device="cuda")
    weights = torch.empty(rows, TOP_K, dtype=torch.bfloat16, device="cuda")
    replay = torch.full((rows, TOP_K), -1, dtype=torch.int16, device="cuda")
    result = get_trtllm_moe_sm100_module().moe_op.trtllm_fp4_block_scale_moe(
        0,  # routing from the logits
        logits,
        ids,
        weights,
        None,  # routing bias
        xq,
        xs.view(FP8).reshape(rows, -1),
        params.gemm1_weights,
        params.gemm1_weights_scale,
        None,  # gemm1 bias
        None,  # gemm1 LoRA delta
        None,
        None,
        None,
        params.gemm2_weights,
        params.gemm2_weights_scale,
        None,  # gemm2 bias
        params.output1_scale_scalar,
        params.output1_scale_gate_scalar,
        params.output2_scale_scalar,
        None,  # per-token scale
        EXPERTS,
        TOP_K,
        0,  # fused shared experts
        0,  # expert groups
        0,  # top-k groups
        INTERMEDIATE,
        0,  # local expert offset
        EXPERTS,
        None,  # routed scaling factor
        params.routing_method_type,
        finalize,
        True,  # programmatic dependent launch
        params.activation_type,
        output,
        tactic,
        True,  # renormalize the top-k
        replay,
        [],  # no prepared routing
        [],
        False,
        None,
        None,
        None,  # workspace buffer
        False,  # not a workspace size query
        NVFP4_4OVER6_CODE_FROM_ENV,
    )
    if finalize:
        return xq, xs, replay, output
    rows_out, expert_weights, slots = result
    if expert_weights is None or expert_weights.numel() == 0:
        # Routing from the logits, the operation fills the caller's top-k
        # weights buffer and returns none (FlashInfer's own wrapper,
        # ``unpack_trtllm_moe_output``, returns that buffer).
        expert_weights = weights
    rows_out, slots = torch.from_dlpack(rows_out), torch.from_dlpack(slots)
    return xq, xs, replay, (rows_out, expert_weights, slots[: rows * TOP_K])


def _tile8_tactic(rows: int) -> list[int]:
    """A valid tactic of FlashInfer's FP4 MoE with 8-row tiles for `rows`
    rows (the autotuner's choice for the served 8-16-row calls)."""
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.tllm_enums import DtypeTrtllmGen, Fp8QuantizationType, WeightLayout

    tactics = get_trtllm_moe_sm100_module().moe_op.trtllm_get_valid_moe_configs(
        DtypeTrtllmGen.E2m1,
        DtypeTrtllmGen.E2m1,
        Fp8QuantizationType.NoneFp8,
        TOP_K,
        HIDDEN,
        HIDDEN,
        INTERMEDIATE,
        EXPERTS,  # local experts
        EXPERTS,
        3,
        True,
        WeightLayout.MajorK,
        False,
        rows,
        False,
    )
    return [int(v) for v in next(t for t in tactics if t[0] == 8)]


def _record():
    """A routing record for calls of up to 32 rows."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.moe import (
        RoutingRecord,
    )

    return RoutingRecord.allocate(torch.device("cuda"))


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The tensor's bits as integers (equal NaNs compare equal)."""
    return tensor.contiguous().view(
        torch.int16 if tensor.element_size() == 2 else torch.uint8
    )


def _assert_record_matches_stock_routing(record, logits, rows):
    """The record's fields that the GEMMs read independently of the row
    order inside an expert (FlashInfer orders an expert's rows by arrival,
    the front by row) against FlashInfer's routing kernels."""
    from flashinfer.fused_moe import trtllm_gen_routing
    from flashinfer.tllm_enums import RoutingMethodType

    stock = trtllm_gen_routing(
        logits.clone(),
        None,
        RoutingMethodType.RenormalizeNaive,
        TOP_K,
        local_num_experts=EXPERTS,
        tile_tokens_dim=8,
    )
    metadata = record.metadata
    padded, tiles = int(metadata.padded_rows[0]), int(metadata.tiles[0])
    assert padded == int(stock.total_num_padded_tokens[0])
    assert tiles == int(stock.num_non_exiting_ctas[0])
    assert torch.equal(
        metadata.tile_expert[:tiles], stock.cta_idx_xy_to_batch_idx[:tiles]
    )
    assert torch.equal(
        metadata.tile_limit[:tiles], stock.cta_idx_xy_to_mn_limit[:tiles]
    )
    # Each (row, k) slot points at a padded row of that row's token.
    slots = metadata.permuted_of[: rows * TOP_K].long()
    rows_of_slots = torch.arange(rows, device="cuda").repeat_interleave(TOP_K)
    assert torch.equal(metadata.token_of[slots].long(), rows_of_slots)


@requires_sm100
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", ROWS)
def test_the_front_and_body_match_the_stock_operation(experts, rows, mode):
    logits, x = _inputs(rows, rows, mode)
    xq, xs, ids, expected = _stock(experts, logits, x, FALLBACK, True)
    *_, (stock_rows, stock_weights, stock_slots) = _stock(
        experts, logits, x, FALLBACK, False
    )
    record = _record()
    record.front(logits, x, experts.input_scale)
    torch.accelerator.synchronize()
    assert torch.equal(record.metadata.ids[:rows], ids.int())
    assert torch.equal(
        _bits(record.metadata.weights[:rows]), _bits(stock_weights.view(rows, TOP_K))
    )
    assert torch.equal(record.x_fp4[:rows], xq.view(torch.uint8))
    assert torch.equal(record.x_scale[:rows], xs.view(torch.uint8).reshape(rows, -1))
    _assert_record_matches_stock_routing(record, logits, rows)

    output = record.body(experts, rows, FALLBACK, True)
    assert torch.equal(_bits(output), _bits(expected))
    unfinalized = record.body(experts, rows, FALLBACK, False)
    torch.accelerator.synchronize()
    # The folded finalize reads, per (row, k), the GEMM2 row of the slot and
    # the weight: equal bits whatever the order of an expert's rows.
    assert unfinalized.expert_weights is not None
    assert torch.equal(
        _bits(unfinalized.expert_weights.view(rows, TOP_K)),
        _bits(stock_weights.view(rows, TOP_K)),
    )
    ours = unfinalized.gemm2_permuted[unfinalized.expanded_idx_to_permuted_idx.long()]
    theirs = stock_rows[stock_slots.long()]
    assert torch.equal(_bits(ours), _bits(theirs))


@requires_sm100
@pytest.mark.parametrize("rows", [8, 16])
def test_a_tuned_tactic_matches_the_stock_operation(experts, rows):
    logits, x = _inputs(rows, 50 + rows)
    tactic = _tile8_tactic(rows)
    *_, expected = _stock(experts, logits, x, tactic, True)
    record = _record()
    record.front(logits, x, experts.input_scale)
    output = record.body(experts, rows, tactic, True)
    torch.accelerator.synchronize()
    assert torch.equal(_bits(output), _bits(expected))


@requires_sm100
def test_every_weight_rounds_as_the_stock_routing():
    """Two million weights of random logits (several scales), every row count
    from 1 to 32: a softmax over all experts renormalized over the ten
    differs in about 20 of them; the front's arithmetic in none."""
    from flashinfer.fused_moe import trtllm_gen_routing
    from flashinfer.tllm_enums import RoutingMethodType

    generator = torch.Generator(device="cuda").manual_seed(5)
    record = _record()
    x = torch.zeros(32, HIDDEN, dtype=torch.bfloat16, device="cuda")
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    ours, theirs = [], []
    weights = 0
    call = 0
    while weights < 2_000_000:
        rows = call % 32 + 1
        sigma = (0.5, 1.0, 1.5, 2.5)[call % 4]
        logits = torch.randn(rows, EXPERTS, device="cuda", generator=generator) * sigma
        logits = logits.bfloat16()
        record.front(logits, x[:rows], one)
        stock = trtllm_gen_routing(
            logits,
            None,
            RoutingMethodType.RenormalizeNaive,
            TOP_K,
            local_num_experts=EXPERTS,
            tile_tokens_dim=8,
        )
        ours.append(
            torch.cat(
                [
                    record.metadata.weights[:rows].view(torch.int16),
                    record.metadata.ids[:rows].to(torch.int16),
                ],
                dim=1,
            )
        )
        theirs.append(
            torch.cat(
                [
                    stock.topk_weights.view(rows, TOP_K).view(torch.int16),
                    stock.topk_ids.view(rows, TOP_K).to(torch.int16),
                ],
                dim=1,
            )
        )
        weights += rows * TOP_K
        call += 1
    differing = int((torch.cat(ours) != torch.cat(theirs)).sum())
    assert differing == 0, f"{differing} of {weights} weights or ids differ"


@requires_sm100
@pytest.mark.parametrize("early_trigger", [True, False])
def test_a_stalled_front_keeps_the_body_bitwise(experts, early_trigger):
    """Each front sleeps after its wait, its record first holding another
    input's routing and activations. With `early_trigger` the front lets
    GEMM1 launch at its entry, so GEMM1 spends the sleep in its own wait and
    would compute on the stale record if it read the routing or the
    activations before that wait (FlashInfer's contract that the front relies
    on, `runtime/moe.py`); without, the production launch."""
    rows = 32
    calls = [_inputs(rows, 200 + seed) for seed in range(3)]
    expected = [_stock(experts, logits, x, FALLBACK, True)[-1] for logits, x in calls]
    records = [_record() for _ in calls]
    for record, (logits, x) in zip(records, calls[1:] + calls[:1]):
        record.front(logits, x, experts.input_scale)
    torch.accelerator.synchronize()
    outputs = []
    for record, (logits, x) in zip(records, calls):
        record.front(
            logits, x, experts.input_scale, early_trigger=early_trigger, stall_us=200
        )
        outputs.append(record.body(experts, rows, FALLBACK, True))
    torch.accelerator.synchronize()
    for output, reference in zip(outputs, expected):
        assert torch.equal(_bits(output), _bits(reference))


@requires_sm100
def test_graph_replays_match_the_stock_operation(experts):
    rows = 20
    calls = [_inputs(rows, 300 + seed) for seed in range(2)]
    expected = [_stock(experts, logits, x, FALLBACK, True)[-1] for logits, x in calls]
    records = [_record() for _ in calls]
    for record, (logits, x) in zip(records, calls):  # compile outside capture
        record.front(logits, x, experts.input_scale)
        record.body(experts, rows, FALLBACK, True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = []
        for record, (logits, x) in zip(records, calls):
            record.front(logits, x, experts.input_scale)
            outputs.append(record.body(experts, rows, FALLBACK, True))
    for _ in range(2):
        graph.replay()
        torch.accelerator.synchronize()
        for output, reference in zip(outputs, expected):
            assert torch.equal(_bits(output), _bits(reference))


# The dispatch on the CPU: Boundaries.moe_output runs the routed path for a
# call with a tactic, and RoutedPath.forward runs the runner's shared expert,
# gate and routed experts in the runner's order.
class _SharedExperts:
    """The runner's shared-experts wrapper, on a side stream or not."""

    def __init__(self, layer, log: list[str], overlap: bool) -> None:
        self._layer = layer
        self.log = log
        self.overlap = overlap
        self._output = None

    def _determine_shared_experts_order(self, x):
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
            SharedExpertsOrder,
        )

        if self.overlap:
            return SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
        return SharedExpertsOrder.NO_OVERLAP

    def maybe_forward_async(self, x) -> bool:
        if self.overlap:
            self.log.append("shared on the side stream")
            self._output = self._layer(x)
        return self.overlap

    def __call__(self, x, order) -> None:
        self.log.append("shared")
        self._output = self._layer(x)

    def wait(self) -> None:
        self.log.append("join")

    @property
    def output(self):
        output, self._output = self._output, None
        return output


def _routed_path(log: list[str], tactics: dict[int, list[int]]):
    """A `RoutedPath` whose front and body are a logged stand-in."""
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.moe import (
        RoutedPath,
    )

    path = object.__new__(RoutedPath)

    def tactic(position: int, rows: int) -> list[int] | None:
        return tactics.get(position)

    def routed(position, logits, x, tactic, deferred=None):
        log.append(f"routed {position} {tactic}")
        return torch.tanh(logits)

    # Instance attributes over the methods (setattr: assigning a method).
    setattr(path, "tactic", tactic)  # noqa: B010
    setattr(path, "routed", routed)  # noqa: B010
    return path


@pytest.mark.parametrize("overlap", [True, False])
def test_the_routed_path_keeps_the_runners_order(overlap):
    from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.boundary import (
        Boundaries,
    )

    from .test_executor import HIDDEN as STAND_IN_HIDDEN
    from .test_executor import StandInBoundaryKernel, StandInMoe

    log: list[str] = []
    mlp = StandInMoe("mlp", log, torch.Generator().manual_seed(2))
    runner = mlp.experts

    def gate(x):
        log.append("gate")
        return x * 2, None

    # The runner members the routed path uses, beyond the stand-in's.
    runner.shared_experts = _SharedExperts(mlp.shared_expert, log, overlap)
    setattr(  # noqa: B010
        runner,
        "routed_experts",
        SimpleNamespace(_ensure_moe_quant_config_init=lambda: None),
    )
    setattr(runner, "gate", gate)  # noqa: B010
    x = torch.randn(5, STAND_IN_HIDDEN, generator=torch.Generator().manual_seed(3))
    layer = cast("Qwen4ExpSparseMoeBlock", mlp)
    boundaries = Boundaries(
        StandInBoundaryKernel(),
        False,
        [layer, layer],
        routed=_routed_path(log, {0: [8, 3]}),
    )

    pending = boundaries.moe_output(0, x, True)

    if overlap:
        assert log == ["shared on the side stream", "gate", "routed 0 [8, 3]", "join"]
    else:
        assert log == ["gate", "shared", "routed 0 [8, 3]"]
    assert pending is not None
    assert torch.equal(pending.partial, torch.tanh(x * 2))
    gate_logit, _ = mlp.shared_expert_gate(x)
    assert torch.equal(pending.gate_logit, gate_logit)
    # The shared expert's output before its gate.
    assert torch.equal(pending.shared, torch.cos(x * mlp.shared_expert.weight))
    # The layer is restored; a layer without a tactic runs the stock runner.
    assert mlp.shared_expert.expert_gate is mlp.shared_expert_gate
    log.clear()
    assert boundaries.moe_output(1, x, True) is not None
    assert log == ["mlp"]

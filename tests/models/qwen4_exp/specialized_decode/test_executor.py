# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp's specialized decode executor on CPU: its layer loop makes the
stock layers' submodule calls in the stock order and computes the stock
result (`Qwen4ExpModel.forward` after the embedding), on deterministic
stand-ins for the submodules, and its steps of up to 64 rows take the read
kernels for every hyper-connection read; its GDN layers take the fused
core for the speculative decodes it covers, in a stream order that a
happens-before checker proves over three layers. With sublayer boundaries, the
blocks whose outputs the boundary kernel reduces return their partial outputs
and the next reads run the (stand-in) kernel, with the same result. With the row
split, four ranks in threads read their quarters of the rows from the last PLE
layer's MLP read on and gather the block inputs and the final streams, with the
same result on every rank."""

import contextlib
import sys
import threading
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    MAX_FUSED_GDN_MTP_TOKENS,
)
from vllm.models.qwen4_exp.nvidia import model as stock_model
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer, Qwen4ExpModel
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime import (
    decode as decode_module,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.boundary import (
    Boundaries,
    GateLogits,
    agree_across_ranks,
    ungated_shared_expert,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode import (
    DecodeStep,
    SpecializedDecode,
    stock_project,
    stock_read,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.decode_metadata import (
    DecodeMetadataGraphs,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gdn import (
    MAX_CORE_TOKENS,
    GdnLayers,
    fused_core_unsupported,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.gemm import (
    PlannedGemms,
)
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.ple import PleGathers
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.qsa import QsaRows
from vllm.models.qwen4_exp.nvidia.specialized_decode.runtime.split import (
    SPLIT_MIN_ROWS,
    RowSplit,
    split_max_rows,
    split_start,
    split_unsupported,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

if TYPE_CHECKING:
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpSparseMoeBlock
    from vllm.models.qwen4_exp.nvidia.specialized_decode.gdn.core import Schedule
    from vllm.v1.attention.backend import AttentionMetadata

STREAMS = 4
HIDDEN = 8
Q = 4  # rows per request: three MTP draft tokens plus one
# One linear-attention layer with PLE (nothing pending), two QSA layers, one
# linear-attention layer with PLE after a pending combine.
LAYERS = (
    ("linear_attention", True),
    ("qwen_sparse_attention", False),
    ("qwen_sparse_attention", False),
    ("linear_attention", True),
)


class StandIn:
    """A submodule stand-in: fixed random weights and a shared call log."""

    def __init__(self, name: str, log: list[str], generator: torch.Generator):
        self.name = name
        self.log = log
        self.weight = torch.randn(STREAMS * HIDDEN, generator=generator)


class StandInHyperConnection(StandIn):
    """`GatedResidual`: a weighted stream sum as the block input, injection
    logits from it, and a sigmoid-weighted combine."""

    def __init__(
        self,
        name: str,
        log: list[str],
        generator: torch.Generator,
        use_combine: bool = True,
    ):
        super().__init__(name, log, generator)
        self.use_combine = use_combine
        self.hc_norm = SimpleNamespace(weight=self.weight)
        self.config = SimpleNamespace(rms_norm_eps=1e-6)

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        self.log.append(f"{self.name}.combine")
        streams = hidden_states.unflatten(-1, (STREAMS, HIDDEN))
        added = block_output.unsqueeze(-2)
        added = added * (
            1.0 if injection is None else torch.sigmoid(injection).unsqueeze(-1)
        )
        return (streams + added).flatten(-2)

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self.log.append(f"{self.name}.mix")
        xn = hidden_states * self.weight
        lowrank, injection = self._down_and_inject(xn)
        return hidden_states, self._up_gate_mix(lowrank, xn), injection

    def _down_and_inject(
        self, xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        block_input = xn.unflatten(-1, (STREAMS, HIDDEN)).sum(-2)
        injection = block_input[:, :STREAMS].tanh() if self.use_combine else None
        return block_input, injection

    def _up_gate_mix(self, lowrank: torch.Tensor, xn: torch.Tensor) -> torch.Tensor:
        return lowrank

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.mix(self.combine(hidden_states, block_output, injection))


class StandInBlock(StandIn):
    """An attention block: `tanh` of a scaled input; records whether its
    output projection reduced its output."""

    # The layer type's names for the block and its output projection.
    prefix: str
    out_proj: SimpleNamespace
    o_proj: SimpleNamespace

    def __init__(self, name: str, log: list[str], generator: torch.Generator):
        super().__init__(name, log, generator)
        self.projection = SimpleNamespace(reduce_results=True)
        self.reduced: list[bool] = []

    def __call__(
        self,
        block_input: torch.Tensor | None = None,
        *,
        hidden_states: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.log.append(self.name)
        self.reduced.append(self.projection.reduce_results)
        x = block_input if block_input is not None else hidden_states
        assert x is not None
        if positions is not None:
            x = x + positions.unsqueeze(-1)
        return torch.tanh(x * self.weight[:HIDDEN])


class StandInLinear(nn.Module):
    """A replicated linear layer: returns its output and no bias."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x @ self.weight, None


class StandInSharedExpert(nn.Module):
    """`Qwen2MoeMLP` with its sigmoid gate (`expert_gate`)."""

    def __init__(self, weight: torch.Tensor, expert_gate: StandInLinear):
        super().__init__()
        self.weight = weight
        self.expert_gate = expert_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cos(x * self.weight)
        if self.expert_gate is not None:
            out = torch.sigmoid(self.expert_gate(x)[0]) * out
        return out


class StandInRunner:
    """The runner of a MoE layer: its body (`_forward_impl`) returns the
    shared expert's output (through the `shared_experts` wrapper) and the
    routed output, logged under the layer's name."""

    def __init__(self, name: str, log: list[str], shared: nn.Module, weight):
        self.name = name
        self.log = log
        self.weight = weight
        self.shared_experts = nn.Module()
        self.shared_experts._layer = shared

    def _forward_impl(self, hidden_states, router_logits, shared_input, input_ids):
        self.log.append(self.name)
        shared = self.shared_experts._layer(shared_input)
        return shared, torch.tanh(hidden_states * self.weight)


class StandInMoe(StandIn):
    """`Qwen4ExpSparseMoeBlock`: the routed output plus the gated shared
    expert's output."""

    def __init__(self, name: str, log: list[str], generator: torch.Generator):
        super().__init__(name, log, generator)
        self.shared_expert_gate = StandInLinear(
            torch.randn(HIDDEN, 1, generator=generator)
        )
        self.shared_expert = StandInSharedExpert(
            torch.randn(HIDDEN, generator=generator), self.shared_expert_gate
        )
        self.experts = StandInRunner(
            name, log, self.shared_expert, self.weight[:HIDDEN]
        )

    def __call__(self, block_input: torch.Tensor) -> torch.Tensor:
        shared, routed = self.experts._forward_impl(
            block_input, block_input, block_input, None
        )
        return shared + routed


class StandInBoundaryKernel:
    """`SublayerBoundary` of one rank: the stand-in combine (as
    `StandInHyperConnection.combine`), then the stand-in norm (the product
    with the norm weight); records the norm weight of every call. A split call
    combines rank `rank`'s quarter of the rows (the partial, the same on
    every rank here, stands for the reduced block output)."""

    def __init__(self, rank: int = 0):
        self.rank = rank
        self.split = True
        self.calls: list[tuple[torch.Tensor, str]] = []

    def _combine_norm(self, block, streams, injection, weight, split):
        if split:
            owned = streams.shape[0]
            block = block[self.rank * owned : (self.rank + 1) * owned]
        scale = torch.sigmoid(injection)
        combined = (
            streams.unflatten(-1, (STREAMS, HIDDEN))
            + block.unsqueeze(-2) * scale.unsqueeze(-1)
        ).flatten(-2)
        return combined, combined * weight

    def __call__(self, partial, streams, injection, weight, eps, split=False):
        self.calls.append((weight, "attention split" if split else "attention"))
        return self._combine_norm(partial, streams, injection, weight, split)

    def moe(
        self, routed, shared, gate_logit, streams, injection, weight, eps, split=False
    ):
        self.calls.append((weight, "moe split" if split else "moe"))
        block = torch.sigmoid(gate_logit) * shared + routed
        return self._combine_norm(block, streams, injection, weight, split)


class StandInPle(StandIn):
    """The PLE layer: adds a function of the token ids to the streams."""

    def start_prefetch(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> None:
        self.log.append(f"{self.name}.prefetch")

    def __call__(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        self.log.append(self.name)
        return hidden_states + input_ids.unsqueeze(-1) * self.weight


def _layer(
    index: int, layer_type: str, ple: bool, log: list[str], generator: torch.Generator
) -> Qwen4ExpDecoderLayer:
    """A decoder layer whose submodules are stand-ins."""
    layer = object.__new__(Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = layer_type
    layer.use_hc_sequence_parallel = False
    layer.use_sequence_parallel = False
    layer.ple = StandInPle(f"{index}.ple", log, generator) if ple else None
    layer.attn_hyper_connection = StandInHyperConnection(
        f"{index}.attn_hc", log, generator
    )
    attention = StandInBlock(f"{index}.attention", log, generator)
    if layer_type == "linear_attention":
        attention.prefix = attention.name
        attention.out_proj = attention.projection
        layer.linear_attn = attention
    else:
        attention.o_proj = attention.projection
        layer.self_attn = attention
    layer.mlp_hyper_connection = StandInHyperConnection(
        f"{index}.mlp_hc", log, generator
    )
    layer.mlp = StandInMoe(f"{index}.mlp", log, generator)
    return layer


def _backbone(log: list[str]) -> Qwen4ExpModel:
    """A backbone of the `LAYERS` stand-in layers and the MTP buffer."""
    generator = torch.Generator().manual_seed(0)
    backbone = object.__new__(Qwen4ExpModel)
    nn.Module.__init__(backbone)
    backbone.config = SimpleNamespace(hc_count=STREAMS, hidden_size=HIDDEN)
    backbone.use_hc_sequence_parallel = False
    backbone.layers = [
        _layer(index, layer_type, ple, log, generator)
        for index, (layer_type, ple) in enumerate(LAYERS)
    ]
    backbone.start_layer = 0
    backbone.end_layer = len(LAYERS)
    backbone.hyper_connection_mixer = StandInHyperConnection(
        "final_hc", log, generator, use_combine=False
    )
    backbone._mtp_hidden_buffer = torch.zeros(256, STREAMS * HIDDEN)
    return backbone


# The boundary kernel's sites: none, TP > 1 (attention and MoE blocks) or TP 1
# (MoE blocks only).
SITES = ("none", "tp", "local")


def _boundaries(backbone: Qwen4ExpModel, sites: str, rank: int = 0) -> Boundaries:
    """`Boundaries` with the stand-in kernel (of `rank`) for `sites`; the
    MoE before a PLE layer keeps its own reduction, as `Boundaries.build`
    decides."""
    layers = backbone.layers
    moe_layers = [
        None
        if sites == "none" or (index + 1 < len(layers) and layers[index + 1].ple)
        else layer.mlp
        for index, layer in enumerate(layers)
    ]
    return Boundaries(StandInBoundaryKernel(rank), sites == "tp", moe_layers)


def _without_reads(log: list[str], names: set[str]) -> list[str]:
    """`log` without the combine and mix of the reads of `names`."""
    kept: list[str] = []
    index = 0
    while index < len(log):
        name, _, call = log[index].rpartition(".")
        if (
            call == "combine"
            and name in names
            and log[index + 1 : index + 2] == [f"{name}.mix"]
        ):
            index += 2
            continue
        kept.append(log[index])
        index += 1
    return kept


@pytest.mark.parametrize("rows", [4, 64, 65])
@pytest.mark.parametrize("deepstack", [False, True])
@pytest.mark.parametrize("sites", SITES)
def test_the_loop_runs_the_stock_layers(rows, deepstack, sites, monkeypatch):
    log: list[str] = []
    backbone = _backbone(log)
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(rows, STREAMS * HIDDEN, generator=generator)
    positions = torch.arange(rows)
    input_ids = torch.randint(0, 100, (rows,), generator=generator)
    query_start_loc = torch.tensor([0, rows])
    ngram_context = torch.zeros(rows, 2, dtype=torch.int64)
    # Deepstack embeddings after the first two layers.
    deepstack_input_embeds = (
        IntermediateTensors(
            {
                f"deepstack_input_embeds_{index}": torch.randn(
                    rows, HIDDEN, generator=generator
                )
                for index in range(2)
            }
        )
        if deepstack
        else None
    )
    inputs = (
        positions,
        input_ids,
        query_start_loc,
        ngram_context,
        deepstack_input_embeds,
    )

    # The stock forward of the last pipeline stage after the embedding: a
    # stage that is not the first takes the multi-stream state as it is.
    monkeypatch.setattr(
        stock_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    expected = backbone(
        input_ids,
        positions,
        IntermediateTensors({"hidden_states": hidden_states}),
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
        deepstack_input_embeds=deepstack_input_embeds,
    )
    expected_mtp_hidden = backbone._mtp_hidden_buffer[:rows].clone()
    expected_log = list(log)

    log.clear()
    backbone._mtp_hidden_buffer.zero_()
    kernel_reads: list[str] = []
    kernel_projects: list[str] = []

    def kernel_read(hc, *args):
        kernel_reads.append(hc.name)
        return stock_read(hc, *args)

    def kernel_project(hc, xn):
        kernel_projects.append(hc.name)
        lowrank, injection = hc._down_and_inject(xn)
        return hc._up_gate_mix(lowrank, xn), injection

    executor = object.__new__(SpecializedDecode)
    executor.backbone = backbone
    executor.layers = list(backbone.layers)
    executor.kernel_read = kernel_read
    executor.kernel_project = kernel_project
    executor.kernel_read_rows = 64
    executor.boundaries = _boundaries(backbone, sites)
    executor.split = None
    # Every GDN layer gets its own metadata from the step, and runs stock here.
    gdn_calls: list[tuple[str, object]] = []

    class RecordingGdnLayers(GdnLayers):
        def __call__(self, attn, hidden_states, metadata):
            gdn_calls.append((attn.name, metadata))
            return super().__call__(attn, hidden_states, metadata)

    executor.gdn = RecordingGdnLayers([], Q, 16, None, None, None)
    # The layers' GEMM plans are active over the whole loop.
    plans_open: list[bool] = []

    class RecordingGemms(PlannedGemms):
        @contextlib.contextmanager
        def active(self) -> Iterator[None]:
            plans_open.append(True)
            try:
                yield
            finally:
                plans_open.append(False)

    executor.gemms = RecordingGemms([])
    # The QSA layers' row attention is chosen by the step's rows, over the
    # whole loop.
    qsa_open: list[int | None] = []

    class RecordingQsa(QsaRows):
        @contextlib.contextmanager
        def active(self, rows: int) -> Iterator[None]:
            qsa_open.append(rows)
            try:
                yield
            finally:
                qsa_open.append(None)

    executor.qsa = RecordingQsa([])
    # The PLE layers' gathers are active over the whole loop (none here: the
    # stand-in PLE layer runs its own forward).
    ple_open: list[bool] = []

    class RecordingPle(PleGathers):
        @contextlib.contextmanager
        def active(self) -> Iterator[None]:
            ple_open.append(True)
            try:
                yield
            finally:
                ple_open.append(False)

    executor.ple = RecordingPle([])
    # No overlapped metadata replay (test_the_loop_waits_for_the_metadata_...).
    executor.metadata = None
    attn_metadata = {f"{index}.attention": f"metadata {index}" for index in (0, 3)}
    actual = executor.run(
        DecodeStep(rows, executor, attn_metadata), hidden_states, *inputs
    )

    # The reads that took the boundary kernel, in order, by the norm weight.
    hcs = {
        id(hc.weight): hc.name
        for layer in backbone.layers
        for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection)
    }
    mixer = backbone.hyper_connection_mixer
    assert mixer is not None
    hcs[id(mixer.weight)] = "final_hc"
    kernel = executor.boundaries.kernel
    assert isinstance(kernel, StandInBoundaryKernel)
    boundary_reads = [(hcs[id(weight)], kind) for weight, kind in kernel.calls]
    order = [f"{index}.{hc}" for index in range(4) for hc in ("attn_hc", "mlp_hc")]
    order.append("final_hc")
    expected_boundary_reads = []
    if sites == "tp":
        expected_boundary_reads += [(f"{i}.mlp_hc", "attention") for i in range(4)]
    if sites != "none" and not deepstack:
        # The MoE of layer 2 precedes a PLE layer.
        expected_boundary_reads += [
            ("1.attn_hc", "moe"),
            ("2.attn_hc", "moe"),
            ("final_hc", "moe"),
        ]
    expected_boundary_reads.sort(key=lambda read: order.index(read[0]))

    assert torch.equal(actual, expected)
    assert torch.equal(backbone._mtp_hidden_buffer[:rows], expected_mtp_hidden)
    assert boundary_reads == expected_boundary_reads
    # The boundary kernel replaces those reads' combine-norm and mix.
    boundary_names = {name for name, _ in boundary_reads}
    assert log == _without_reads(expected_log, boundary_names)
    assert kernel_reads == (
        [name for name in order if name not in boundary_names] if rows <= 64 else []
    )
    assert kernel_projects == (
        [name for name, _ in boundary_reads] if rows <= 64 else []
    )
    for layer in backbone.layers:
        attention = getattr(layer, "linear_attn", None) or layer.self_attn
        # The stock run's call, then the executor's.
        assert attention.reduced == [True, sites != "tp"]
        assert attention.projection.reduce_results
    assert gdn_calls == [
        ("0.attention", "metadata 0"),
        ("3.attention", "metadata 3"),
    ]
    assert plans_open == [True, False]
    assert qsa_open == [rows, None]
    assert ple_open == [True, False]


TP = 4


class StandInExchange:
    """The TP group's all-gather between threads, one a rank."""

    def __init__(self) -> None:
        self.barrier = threading.Barrier(TP, timeout=60)
        self.parts: list[torch.Tensor] = [torch.empty(0)] * TP
        self.calls = 0

    def all_gather(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        """Every rank's `x` concatenated in rank order (every rank calls)."""
        self.parts[rank] = x.clone()
        self.barrier.wait()
        out = torch.cat(self.parts)
        if rank == 0:
            self.calls += 1
        self.barrier.wait()
        return out


def _row_split(rank: int, exchange: StandInExchange, start: int) -> RowSplit:
    """`RowSplit` of `rank` with the stand-in read's projections and the
    exchange's all-gather in place of the kernels."""
    split = object.__new__(RowSplit)
    split.world = TP
    split.rank = rank
    split.max_rows = 256
    split.start = start
    split.gather = None  # the stand-in projections gather

    def project(hc, xn, gather):
        lowrank, injection = hc._down_and_inject(xn)
        block_input = hc._up_gate_mix(lowrank, xn)
        return exchange.all_gather(rank, block_input), injection

    def gather_streams(rows: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return out.copy_(exchange.all_gather(rank, rows))

    split._project = project
    split.streams = gather_streams
    return split


@pytest.mark.parametrize("rows", [16, SPLIT_MIN_ROWS, 64, 128, 256])
def test_split_steps_compute_the_stock_result_on_every_rank(rows, monkeypatch):
    """Four ranks (threads) of a TP4 executor with the row split: from the MLP
    read of the last PLE layer (layer 3) on, each reads its quarter of the
    rows through the boundary kernel's split calls and gathers the block
    inputs, and the final mixer's streams are gathered into the MTP buffer;
    every rank returns the stock result and holds the stock MTP buffer. Steps
    below `SPLIT_MIN_ROWS` read replicated; steps above the read kernels'
    64 rows read replicated with the stock reads before the split."""
    generator = torch.Generator().manual_seed(1)
    hidden_states = torch.randn(rows, STREAMS * HIDDEN, generator=generator)
    positions = torch.arange(rows)
    input_ids = torch.randint(0, 100, (rows,), generator=generator)
    query_start_loc = torch.tensor([0, rows])
    ngram_context = torch.zeros(rows, 2, dtype=torch.int64)
    inputs = (positions, input_ids, query_start_loc, ngram_context, None)
    monkeypatch.setattr(
        stock_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    stock = _backbone([])
    expected = stock(
        input_ids,
        positions,
        IntermediateTensors({"hidden_states": hidden_states}),
        query_start_loc=query_start_loc,
        ngram_context=ngram_context,
    )
    expected_mtp_hidden = stock._mtp_hidden_buffer[:rows].clone()

    exchange = StandInExchange()
    results: list[tuple[torch.Tensor, torch.Tensor, list[tuple[str, str]]] | None]
    results = [None] * TP
    errors: list[BaseException] = []

    def rank_main(rank: int) -> None:
        try:
            log: list[str] = []
            backbone = _backbone(log)
            executor = _waiting_executor(backbone, log)
            executor.metadata = None
            executor.boundaries = _boundaries(backbone, "tp", rank)
            start = split_start(executor.layers)
            assert split_unsupported(executor.boundaries, start) == ""
            executor.split = _row_split(rank, exchange, start)
            actual = executor.run(DecodeStep(rows, executor), hidden_states, *inputs)
            hcs = {
                id(hc.weight): hc.name
                for layer in backbone.layers
                for hc in (layer.attn_hyper_connection, layer.mlp_hyper_connection)
            }
            mixer = backbone.hyper_connection_mixer
            assert mixer is not None
            hcs[id(mixer.weight)] = "final_hc"
            kernel = executor.boundaries.kernel
            assert isinstance(kernel, StandInBoundaryKernel)
            reads = [(hcs[id(weight)], kind) for weight, kind in kernel.calls]
            results[rank] = (actual, backbone._mtp_hidden_buffer[:rows].clone(), reads)
        except BaseException as error:  # reported by the main thread
            errors.append(error)
            exchange.barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(rank,)) for rank in range(TP)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    split = rows >= SPLIT_MIN_ROWS
    for result in results:
        assert result is not None
        actual, mtp_hidden, reads = result
        assert torch.equal(actual, expected)
        assert torch.equal(mtp_hidden, expected_mtp_hidden)
        # Layer 3 has the last PLE: its MLP read and the final mixer's split.
        assert reads == [
            ("0.mlp_hc", "attention"),
            ("1.attn_hc", "moe"),
            ("1.mlp_hc", "attention"),
            ("2.attn_hc", "moe"),
            ("2.mlp_hc", "attention"),
            ("3.mlp_hc", "attention split" if split else "attention"),
            ("final_hc", "moe split" if split else "moe"),
        ]
    # The two split reads' block inputs and the final streams.
    assert exchange.calls == (3 if split else 0)


@pytest.mark.parametrize("max_rows, rows", [(16, 16), (64, 64), (256, 256), (512, 256)])
def test_a_split_rank_reads_at_most_64_rows(max_rows, rows):
    """The largest step that splits is the executor's largest, up to 64 rows
    a rank of four (what the read kernels take)."""
    assert split_max_rows(max_rows) == rows


def test_the_split_needs_every_later_moe_block_served():
    log: list[str] = []
    backbone = _backbone(log)
    boundaries = _boundaries(backbone, "tp")
    assert split_start(backbone.layers) == 3
    assert split_unsupported(boundaries, 3) == ""
    # The MoE of layer 2 precedes the PLE layer 3 and keeps its own reduction.
    assert "positions [2]" in split_unsupported(boundaries, 2)
    assert "attention" in split_unsupported(_boundaries(backbone, "local"), 3)
    kernel = boundaries.kernel
    assert isinstance(kernel, StandInBoundaryKernel)
    kernel.split = False
    assert "split calls" in split_unsupported(boundaries, 3)


class StandInMetadata:
    """`DecodeMetadataGraphs` as the executor sees it: logs the waits it
    places and the captures it notes."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def join(self, gdn_only: bool) -> None:
        self.log.append("metadata wait gdn" if gdn_only else "metadata wait all")

    def joined(self, rows: int) -> None:
        self.log.append(f"metadata joined {rows}")

    def captured_stock(self, rows: int) -> None:
        self.log.append(f"metadata stock {rows}")


def _waiting_executor(backbone: Qwen4ExpModel, log: list[str]) -> SpecializedDecode:
    """An executor over `backbone` with stand-ins, the stock reads and
    `StandInMetadata` as its replayed metadata."""
    executor = object.__new__(SpecializedDecode)
    executor.backbone = backbone
    executor.layers = list(backbone.layers)
    executor.kernel_read = stock_read
    executor.kernel_project = stock_project
    executor.kernel_read_rows = 64
    executor.boundaries = _boundaries(backbone, "none")
    executor.split = None
    executor.gdn = GdnLayers([], Q, 16, None, None, None)
    executor.gemms = PlannedGemms([])
    executor.qsa = QsaRows([])
    executor.ple = PleGathers([])
    executor.metadata = cast("DecodeMetadataGraphs", StandInMetadata(log))
    return executor


# LAYERS without the first layer's PLE: its GDN layer reads its metadata
# before any PLE or QSA layer, so the wait for the GDN metadata comes first.
GDN_FIRST = (("linear_attention", False), *LAYERS[1:])


@pytest.mark.parametrize("layers", [LAYERS, GDN_FIRST], ids=["ple-first", "gdn-first"])
def test_the_loop_waits_for_the_metadata_before_reading_it(layers, monkeypatch):
    """The capture waits for the GDN groups' metadata (or all of it) before
    the first GDN layer and for all of it before the first PLE or QSA layer,
    once each, after the first read, then notes the captured rows
    (`CaptureWaits`)."""
    monkeypatch.setattr(sys.modules[__name__], "LAYERS", layers)
    log: list[str] = []
    executor = _waiting_executor(_backbone(log), log)
    rows = 8
    generator = torch.Generator().manual_seed(3)
    executor.run(
        DecodeStep(
            rows,
            executor,
            cast(
                "dict[str, AttentionMetadata]",
                {f"{i}.attention": f"metadata {i}" for i in (0, 3)},
            ),
        ),
        torch.randn(rows, STREAMS * HIDDEN, generator=generator),
        torch.arange(rows),
        torch.randint(0, 100, (rows,), generator=generator),
        torch.tensor([0, rows]),
        torch.zeros(rows, 2, dtype=torch.int64),
        None,
    )
    gdn_layers = {
        f"{index}.attention"
        for index, (kind, _) in enumerate(layers)
        if kind == "linear_attention"
    }
    readers_of_all = {
        f"{index}.attention"
        for index, (kind, _) in enumerate(layers)
        if kind != "linear_attention"
    } | {f"{index}.ple" for index, (_, ple) in enumerate(layers) if ple}
    waits = [entry for entry in log if entry.startswith("metadata")]
    assert waits[-1] == f"metadata joined {rows}"
    assert waits.count("metadata wait all") == 1
    assert waits.count("metadata wait gdn") == (0 if layers[0][1] else 1)
    for index, entry in enumerate(log):
        before = log[:index]
        if entry in gdn_layers:
            assert "metadata wait gdn" in before or "metadata wait all" in before
        if entry in readers_of_all:
            assert "metadata wait all" in before, entry
    first_wait = log.index(waits[0])
    first_reader = min(
        index for index, entry in enumerate(log) if entry in gdn_layers | readers_of_all
    )
    assert first_wait == first_reader - 1
    if not layers[0][1]:
        # After the first hyper-connection read, which reads no metadata.
        assert first_wait > log.index("0.attn_hc.mix")


def test_a_stock_full_capture_is_noted_for_the_metadata(monkeypatch):
    """A FULL capture that runs the stock layers tells the replayed metadata
    (its recordings of those rows must not overlap); a PIECEWISE one does
    not."""
    log: list[str] = []
    executor = _waiting_executor(_backbone(log), log)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(decode_module, "is_forward_context_available", lambda: True)
    for mode, noted in [(CUDAGraphMode.FULL, True), (CUDAGraphMode.PIECEWISE, False)]:
        log.clear()
        monkeypatch.setattr(
            decode_module,
            "get_forward_context",
            lambda mode=mode: SimpleNamespace(cudagraph_runtime_mode=mode),
        )
        executor._note_stock_capture(12)
        assert log == (["metadata stock 12"] if noted else [])


def test_moe_output_leaves_the_gate_to_the_kernel():
    log: list[str] = []
    mlp = StandInMoe("mlp", log, torch.Generator().manual_seed(2))
    x = torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(3))
    boundaries = Boundaries(
        StandInBoundaryKernel(), False, [cast("Qwen4ExpSparseMoeBlock", mlp), None]
    )

    pending = boundaries.moe_output(0, x, True)

    assert pending is not None and log == ["mlp"]
    gate_logit, _ = mlp.shared_expert_gate(x)
    assert torch.equal(pending.gate_logit, gate_logit)
    assert torch.equal(
        torch.sigmoid(pending.gate_logit) * pending.shared + pending.partial, mlp(x)
    )
    # The layer is restored.
    assert mlp.shared_expert.expert_gate is mlp.shared_expert_gate
    assert mlp.experts.shared_experts._layer is mlp.shared_expert
    # Layers and steps that keep the stock MoE.
    assert boundaries.moe_output(1, x, True) is None
    assert boundaries.moe_output(0, x, False) is None


def test_ungated_shared_expert_restores_the_layer_on_errors():
    mlp = StandInMoe("mlp", [], torch.Generator().manual_seed(2))
    with pytest.raises(RuntimeError), ungated_shared_expert(mlp, GateLogits(mlp)):
        raise RuntimeError
    assert mlp.shared_expert.expert_gate is mlp.shared_expert_gate
    assert mlp.experts.shared_experts._layer is mlp.shared_expert


def test_ranks_that_disagree_on_the_moe_sites_raise(monkeypatch):
    # Another rank serves MoE layers 0 and 2.
    other = torch.tensor([1, 0, 1], dtype=torch.int32)

    def all_reduce(tensor, op, group):
        both = torch.minimum if op == torch.distributed.ReduceOp.MIN else torch.maximum
        tensor.copy_(both(tensor, other))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    agree_across_ranks([True, False, True], None)
    with pytest.raises(RuntimeError, match=r"positions \[1\]"):
        agree_across_ranks([True, True, True], None)


class StandInGdn:
    """The parts of `QwenGatedDeltaNetAttention` that `GdnLayers` uses:
    projections that log their calls, the caches and the parameters of a GDN
    layer with 4 key heads, and the stock forward."""

    def __init__(self, log: list[str], state_dtype: torch.dtype = torch.float32):
        self.log = log
        self.prefix = "model.layers.0.linear_attn"
        self.num_k_heads, self.num_v_heads, self.tp_size = 4, 12, 1
        self.head_k_dim = self.head_v_dim = 128
        self.value_dim = self.num_v_heads * 128
        conv_dim = (2 * self.num_k_heads + self.num_v_heads) * 128
        self.conv_kernel_size = 4
        self.activation = "silu"
        self.state_dtype = state_dtype
        self.enable_fused_gdn_spec_decode = True
        self.gqa_interleaved_layout = False
        self.disable_tp_for_ba_proj = False
        self._fused_gdn_gated_rms_norm = None
        self.conv1d = SimpleNamespace(weight=torch.zeros(conv_dim, 1, 4), bias=None)
        self.A_log = torch.zeros(self.num_v_heads)
        self.dt_bias = torch.zeros(self.num_v_heads, dtype=torch.bfloat16)
        self.norm = SimpleNamespace(
            weight=torch.ones(128, dtype=torch.bfloat16),
            activation="sigmoid",
            norm_before_gate=True,
            group_size=None,
        )
        self.layer_norm_epsilon = 1e-6
        # vLLM's default (SD) convolution layout: [slots, q + 2, conv_dim].
        self.kv_cache = [torch.zeros(3, Q + 2, conv_dim), torch.zeros(3, 12, 8, 8)]

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return torch.bfloat16, self.state_dtype

    def in_proj_qkvz(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.log.append("qkvz")
        return x + 1, None

    def in_proj_ba(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.log.append("ba")
        return x + 2, None

    def out_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.log.append("out_proj")
        return x[:, :2] * 10, None

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.log.append("stock")
        return hidden_states * 2


def _as_gdn(*attns: "StandInGdn") -> list["QwenGatedDeltaNetAttention"]:
    """Stand-ins in place of the GDN layers `GdnLayers` takes."""
    return cast("list[QwenGatedDeltaNetAttention]", list(attns))


def _spec_metadata(requests: int, **fields) -> GDNAttentionMetadata:
    """The GDN metadata of a speculative decode of `requests` requests."""
    values = dict(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=requests,
        num_spec_decode_tokens=Q * requests,
        num_actual_tokens=Q * requests,
        spec_query_start_loc=torch.arange(0, Q * requests + 1, Q, dtype=torch.int32),
        spec_state_indices_tensor=torch.ones(requests, Q, dtype=torch.int32),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
    )
    values.update(fields)
    return GDNAttentionMetadata(**values)


def _gdn_layers(attn: StandInGdn, log: list[str]) -> tuple[GdnLayers, list[tuple]]:
    """`GdnLayers` over `attn` with a core that logs its arguments."""
    calls: list[tuple] = []

    def core(*args):
        log.append("core")
        calls.append(args)
        args[12].fill_(3.0)
        return args[12]

    def schedule(key_heads: int, requests: int) -> str:
        return f"schedule {key_heads}x{requests}"

    return GdnLayers(_as_gdn(attn), Q, 16, core, schedule, None), calls


@pytest.mark.parametrize("requests", [1, 5, 16])
def test_gdn_layers_take_the_fused_core_for_speculative_decodes(requests):
    log: list[str] = []
    attn = StandInGdn(log)
    gdn, calls = _gdn_layers(attn, log)
    metadata = _spec_metadata(requests)
    hidden_states = torch.zeros(Q * requests, 6)

    output = gdn(attn, hidden_states, metadata)

    assert log == ["qkvz", "ba", "core", "out_proj"]
    assert torch.equal(output, torch.full((Q * requests, 2), 30.0))
    (args,) = calls
    qkvz, ba, conv_weight, conv_state, state = args[:5]
    assert torch.equal(qkvz, hidden_states + 1) and torch.equal(ba, hidden_states + 2)
    assert conv_weight.shape == (attn.conv1d.weight.size(0), 4)
    assert torch.equal(conv_state, attn.kv_cache[0].transpose(-1, -2))
    assert state is attn.kv_cache[1]
    assert args[8] is metadata.spec_state_indices_tensor
    assert args[9] is metadata.num_accepted_tokens
    assert args[10] is metadata.spec_query_start_loc
    assert args[11] == requests
    assert args[13] == attn.layer_norm_epsilon
    assert args[14] == f"schedule 4x{requests}"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "not GDN metadata",
        _spec_metadata(2, num_prefills=1),
        _spec_metadata(2, num_decodes=1),
        _spec_metadata(0),
        _spec_metadata(17),
        _spec_metadata(2, spec_state_indices_tensor=torch.ones(2, Q + 1)),
        _spec_metadata(2, spec_query_start_loc=None),
        _spec_metadata(2, num_accepted_tokens=None),
    ],
)
def test_gdn_layers_run_other_steps_stock(metadata):
    log: list[str] = []
    attn = StandInGdn(log)
    gdn, calls = _gdn_layers(attn, log)
    hidden_states = torch.ones(8, 6)

    output = gdn(attn, hidden_states, metadata)

    assert log == ["stock"] and not calls
    assert torch.equal(output, hidden_states * 2)


def test_gdn_layers_without_the_fused_core_run_stock():
    log: list[str] = []
    supported = StandInGdn(log)
    assert fused_core_unsupported(_as_gdn(supported), Q) == ""
    bf16_state = StandInGdn(log, state_dtype=torch.bfloat16)
    silu_gate = StandInGdn(log)
    silu_gate.norm.activation = "silu"
    missing = fused_core_unsupported(_as_gdn(supported, bf16_state, silu_gate), Q)
    assert "an FP32 recurrent state" in missing and "sigmoid output gate" in missing
    assert fused_core_unsupported([], Q) == "GDN layers"

    gdn = GdnLayers.build(_as_gdn(bf16_state), Q, 16)
    assert gdn.core is None and gdn.warm_up() == []
    output = gdn(bf16_state, torch.ones(Q, 6), _spec_metadata(1))
    assert log == ["stock"] and torch.equal(output, torch.full((Q, 6), 2.0))


@pytest.mark.parametrize("tokens", [2, Q, MAX_CORE_TOKENS])
def test_the_fused_core_takes_the_validated_depths(tokens):
    assert MAX_CORE_TOKENS <= MAX_FUSED_GDN_MTP_TOKENS
    assert fused_core_unsupported(_as_gdn(StandInGdn([])), tokens) == ""


def test_the_fused_core_refuses_deeper_drafts():
    # The core is validated up to MAX_CORE_TOKENS rows per request, and above
    # MAX_FUSED_GDN_MTP_TOKENS stock runs the Triton core, which it does not
    # reproduce.
    log: list[str] = []
    attn = StandInGdn(log)
    tokens = MAX_CORE_TOKENS + 1
    missing = fused_core_unsupported(_as_gdn(attn), tokens)
    assert missing.startswith(f"at most {MAX_CORE_TOKENS} rows per request")
    assert f"not {tokens}" in missing

    gdn = GdnLayers.build(_as_gdn(attn), tokens, 16)
    assert gdn.core is None and gdn.warm_up() == []
    metadata = _spec_metadata(
        1, spec_state_indices_tensor=torch.ones(1, tokens, dtype=torch.int32)
    )
    gdn(attn, torch.ones(tokens, 6), metadata)
    assert log == ["stock"]


# The fork and join of the GDN layers' side stream, checked by simulation:
# what GdnLayers enqueues on stand-in streams is recorded as (operation,
# stream, name) and replayed under CUDA's ordering rules (stream order; an
# event wait orders the stream after everything before the event's latest
# record).
Op = tuple[str, str, str]


def happens_before(ops: list[Op]) -> dict[str, set[str]]:
    """For each launch in `ops`, the launches that complete before it
    starts."""
    pending: dict[str, set[str]] = {}  # what precedes a stream's next op
    recorded: dict[str, set[str]] = {}
    before: dict[str, set[str]] = {}
    for op, stream, name in ops:
        frontier = pending.setdefault(stream, set())
        if op == "launch":
            before[name] = set(frontier)
            frontier.add(name)
        elif op == "record":
            recorded[name] = set(frontier)
        else:  # wait; waiting for a never-recorded event is a no-op
            frontier |= recorded.get(name, set())
    return before


def check_stream_order(ops: list[Op], layers: list[str]) -> None:
    """Assert the fused GDN layers' stream invariants (`runtime/gdn.py`)
    over `ops`, in which each layer is followed by a `residual` launch on
    the main stream that produces the next layer's input."""
    before = happens_before(ops)
    stream_of = {name: stream for op, stream, name in ops if op == "launch"}
    for i, layer in enumerate(layers):
        core, ba = before[f"core {layer}"], before[f"ba {layer}"]
        assert stream_of[f"core {layer}"] == "main", f"{layer}: core off main"
        assert f"qkvz {layer}" in core, f"{layer}: core before its qkvz"
        assert f"ba {layer}" in core, f"{layer}: core before its ba (no join)"
        assert f"qkvz {layer}" not in ba, f"{layer}: ba forked after qkvz"
        if i:
            assert f"residual {layers[i - 1]}" in ba, (
                f"{layer}: ba before the input it reads (no fork wait)"
            )
    main = [name for op, stream, name in ops if op == "launch" and stream == "main"]
    side = {name for name, stream in stream_of.items() if stream != "main"}
    assert side <= before[main[-1]], "the side stream does not rejoin main"


class Recorder:
    """Stand-in streams and events, and the current stream, recording what
    `GdnLayers` enqueues."""

    def __init__(self) -> None:
        self.ops: list[Op] = []
        self.current = "main"
        recorder = self

        class Stream:
            def __init__(self, name: str) -> None:
                self.name = name

            def wait_event(self, event: "Event") -> None:
                recorder.ops.append(("wait", self.name, event.name))

        class Event:
            def __init__(self, name: str) -> None:
                self.name = name

            def record(self, stream: Stream) -> None:
                recorder.ops.append(("record", stream.name, self.name))

        self.streams = {name: Stream(name) for name in ("main", "side")}
        self.events = (Event("start"), Event("done"))

    def launch(self, name: str) -> None:
        """Record a kernel launch on the current stream."""
        self.ops.append(("launch", self.current, name))

    @contextlib.contextmanager
    def stream(self, stream) -> Iterator[None]:
        """`torch.cuda.stream`: make `stream` current."""
        previous, self.current = self.current, stream.name
        try:
            yield
        finally:
            self.current = previous


class TracedGdn(StandInGdn):
    """A GDN layer whose projections record their launches."""

    def __init__(self, name: str, recorder: Recorder) -> None:
        super().__init__([])
        self.name, self.recorder = name, recorder

    def in_proj_qkvz(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.recorder.launch(f"qkvz {self.name}")
        return x + 1, None

    def in_proj_ba(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.recorder.launch(f"ba {self.name}")
        return x + 2, None

    def out_proj(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.recorder.launch(f"out_proj {self.name}")
        return x[:, :6], None


def _record_fused_layers(monkeypatch, layers: list[str]) -> list[Op]:
    """What GdnLayers enqueues for `layers` in a speculative decode step of
    two requests, with a residual launch after each layer."""
    recorder = Recorder()
    monkeypatch.setattr(torch.cuda, "stream", recorder.stream)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: recorder.streams[recorder.current]
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attns = [TracedGdn(name, recorder) for name in layers]
    names = {id(attn.kv_cache[1]): attn.name for attn in attns}

    def core(*args):
        recorder.launch(f"core {names[id(args[4])]}")
        return args[12]

    gdn = GdnLayers(
        attns,
        Q,
        16,
        core,
        lambda key_heads, requests: cast("Schedule", "schedule"),
        recorder.streams["side"],
        recorder.events,
    )
    hidden_states = torch.zeros(2 * Q, 6)
    for attn in attns:
        hidden_states = gdn(attn, hidden_states, _spec_metadata(2))
        recorder.launch(f"residual {attn.name}")
    return recorder.ops


LAYER_NAMES = ["layer 0", "layer 1", "layer 2"]


def test_gdn_layers_fork_and_join_the_side_stream(monkeypatch):
    ops = _record_fused_layers(monkeypatch, LAYER_NAMES)

    check_stream_order(ops, LAYER_NAMES)
    assert ops[:9] == [
        ("record", "main", "start"),
        ("wait", "side", "start"),
        ("launch", "side", "ba layer 0"),
        ("record", "side", "done"),
        ("launch", "main", "qkvz layer 0"),
        ("wait", "main", "done"),
        ("launch", "main", "core layer 0"),
        ("launch", "main", "out_proj layer 0"),
        ("launch", "main", "residual layer 0"),
    ]


def _move_after(ops: list[Op], op: Op, anchor: Op) -> list[Op]:
    """`ops` with the last `op` before `anchor` moved right after it."""
    at = ops.index(anchor)
    index = at - 1 - ops[:at][::-1].index(op)
    moved = ops[:index] + ops[index + 1 :]
    at = moved.index(anchor)
    return moved[: at + 1] + [op] + moved[at + 1 :]


def _drop_before(ops: list[Op], op: Op, anchor: Op) -> list[Op]:
    """`ops` without the last `op` before `anchor`."""
    at = ops.index(anchor)
    index = at - 1 - ops[:at][::-1].index(op)
    return ops[:index] + ops[index + 1 :]


# Defects injected into layer 1 of a recorded trace; each must fail the check.
JOIN, CORE_1 = ("wait", "main", "done"), ("launch", "main", "core layer 1")
FORK, QKVZ_1 = ("record", "main", "start"), ("launch", "main", "qkvz layer 1")
FORK_WAIT, BA_1 = ("wait", "side", "start"), ("launch", "side", "ba layer 1")
DEFECTS: dict[str, Callable[[list[Op]], list[Op]]] = {
    "join after the core": lambda ops: _move_after(ops, JOIN, CORE_1),
    "no join": lambda ops: _drop_before(ops, JOIN, CORE_1),
    "no fork wait on the side stream": lambda ops: _drop_before(ops, FORK_WAIT, BA_1),
    "fork after the qkvz projection": lambda ops: _move_after(ops, FORK, QKVZ_1),
}


@pytest.mark.parametrize("defect", DEFECTS)
def test_the_stream_order_check_catches_a_broken_fork_or_join(monkeypatch, defect):
    ops = _record_fused_layers(monkeypatch, LAYER_NAMES)
    broken = DEFECTS[defect](ops)
    assert broken != ops

    with pytest.raises(AssertionError):
        check_stream_order(broken, LAYER_NAMES)

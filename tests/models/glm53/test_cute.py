# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 CuTe DSL kernel families
(vllm/models/glm53/nvidia/ops/cute_dsl), on a GPU.
"""

import importlib.util

import pytest
import torch

from vllm.models.glm53.nvidia.ops.cute_dsl import registry

HIDDEN, VOCAB, SHARD_ROWS = 6144, 154880, 19360
ROPE_CAPACITY = 16384


def _sm100() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    return importlib.util.find_spec("cutlass") is not None


sm100 = pytest.mark.skipif(not _sm100(), reason="needs an SM100 GPU and CuTe DSL")


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    device = torch.device("cuda")
    embedding = torch.randn(VOCAB, HIDDEN, device=device).to(torch.bfloat16)
    specials = torch.tensor(
        [0x7FC1, -127, 0x7F80, -128, 0x0001, -32767, 0, -32768], dtype=torch.int16
    )
    embedding.view(torch.int16)[VOCAB - 1, :8] = specials.to(device)
    tokens = torch.tensor([VOCAB - 1, 0, SHARD_ROWS - 1, 77777], dtype=torch.int32)
    return {
        "embedding": embedding,
        "token_ids": tokens.to(device),
        "position": torch.tensor([12345], dtype=torch.int32, device=device),
        "rope": torch.randn(ROPE_CAPACITY, 64, device=device),
        # The q4 row table: R = 4 live rows of B = 1 request starting at row 0.
        "row_table": torch.tensor(
            [4, 1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.int32, device=device
        ),
        # The one-row build's table (the drafter's proposals): R = 1 live row.
        "row_table_q1": torch.tensor(
            [1, 1, 0, 0, 0, 0], dtype=torch.int32, device=device
        ),
    }


def _outputs() -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    out = {
        "hidden_q4": torch.empty(4, HIDDEN, dtype=torch.bfloat16, device=device),
        "freq_q4": torch.empty(4, 64, device=device),
        "hidden_v1": torch.empty(4, HIDDEN, dtype=torch.bfloat16, device=device),
        "freq_v1": torch.empty(4, 64, device=device),
        "hidden_v4": torch.empty(4, HIDDEN, dtype=torch.bfloat16, device=device),
        "freq_v4": torch.empty(4, 64, device=device),
    }
    for tensor in out.values():
        tensor.view(torch.uint8).fill_(0xA5)
    return out


def _launches(x, out) -> None:
    from cuda.bindings.driver import CUstream

    device = torch.accelerator.current_device_index()
    stream = CUstream(torch.cuda.current_stream().cuda_stream)

    def launch(name: str, values: list) -> None:
        registry.executor(name, device)(*registry.bind(name, values), stream)

    shards = [x["embedding"].data_ptr() + p * SHARD_ROWS * HIDDEN * 2 for p in range(8)]
    common = [x["token_ids"].data_ptr(), x["position"].data_ptr()]
    launch(
        "prologue_q4",
        [
            *common,
            x["embedding"].data_ptr(),
            x["rope"].data_ptr(),
            out["hidden_q4"].data_ptr(),
            out["freq_q4"].data_ptr(),
            ROPE_CAPACITY,
            x["row_table"].data_ptr(),
        ],
    )
    for name, tag, table in (
        ("prologue_vllm", "v1", "row_table_q1"),
        ("prologue_vllm_q4", "v4", "row_table"),
    ):
        launch(
            name,
            [
                *common,
                x["rope"].data_ptr(),
                out[f"hidden_{tag}"].data_ptr(),
                out[f"freq_{tag}"].data_ptr(),
                ROPE_CAPACITY,
                x[table].data_ptr(),
                *shards,
            ],
        )


def _programmatic_edges(graph: torch.cuda.CUDAGraph) -> int:
    from cuda.bindings import driver

    handle = driver.CUgraph(graph.raw_cuda_graph())
    error, _, _, _, count = driver.cuGraphGetEdges(handle, 0)
    assert int(error) == 0, error
    error, _, _, data, count = driver.cuGraphGetEdges(handle, count)
    assert int(error) == 0, error
    programmatic = driver.CUgraphDependencyType.CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
    return sum(1 for edge in data[:count] if edge.type == programmatic)


@sm100
def test_cute_entries_build_and_bind():
    device = torch.accelerator.current_device_index()
    for name in registry.ENTRIES:
        assert registry.executor(name, device) is registry.executor(name, device), name


CHAINED = ("prologue_q4", "prologue_vllm", "prologue_vllm_q4")


@sm100
def test_cute_entries_replay_bitwise_in_a_pdl_graph():
    x = _inputs()
    out = _outputs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _launches(x, out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.accelerator.synchronize()
    eager = {k: v.clone() for k, v in out.items()}
    for tensor in out.values():
        tensor.view(torch.uint8).fill_(0xA5)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        _launches(x, out)
    assert _programmatic_edges(graph) == len(CHAINED) - 1
    graph.instantiate()
    graph.replay()
    torch.accelerator.synchronize()
    for key in out:
        assert torch.equal(out[key].view(torch.uint8), eager[key].view(torch.uint8)), (
            key
        )

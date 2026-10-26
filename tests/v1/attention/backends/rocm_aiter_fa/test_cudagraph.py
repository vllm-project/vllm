# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-graph capture and replay of ROCM_AITER_FA's decode path.

The builder declares ``UNIFORM_BATCH``: FULL_DECODE_ONLY and
FULL_AND_PIECEWISE capture uniform decode batches (q_len 1, or 1 + spec
tokens) as full graphs. This mirrors the runner: metadata tensors live in
persistent buffers, capture uses ``build_for_cudagraph_capture`` with
``max_seq_len = max_model_len``, and every replay rebuilds metadata into the
same buffers with different sequence lengths and block tables. Mode-level
behavior (NONE / PIECEWISE / FULL_DECODE_ONLY / FULL_AND_PIECEWISE through the
runner) is in tests/v1/e2e/attention/test_rocm_aiter_fa_runtime.py.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_aiter_fa.fa_harness import (
    DEVICE,
    FaCase,
    set_shuffle,
)
from tests.v1.attention.model_shapes import (
    MUSE_SPARK_SLIDING,
    QWEN3_8_27B,
    QWEN3_235B,
)

MAX_MODEL_LEN = 4096
NUM_REQS = 8


def _capture_and_replay(case: FaCase, q_len: int, replays: list[list[int]]):
    max_blocks = MAX_MODEL_LEN // case.block_size
    capture = case.batch([q_len] * NUM_REQS, [MAX_MODEL_LEN] * NUM_REQS)
    common = capture.common
    common.max_seq_len = MAX_MODEL_LEN
    assert common.block_table_tensor.shape[1] == max_blocks
    num_tokens = q_len * NUM_REQS
    query = case.query(num_tokens)
    key = torch.zeros(
        num_tokens,
        case.shape.num_kv_heads,
        case.shape.head_size,
        dtype=case.dtype,
        device=DEVICE,
    )
    value = torch.zeros_like(key)
    output = torch.empty_like(query)
    layer = case.layer
    metadata = case.builder.build_for_cudagraph_capture(common)

    def step():
        layer.impl.forward(layer, query, key, value, case.kv_cache, metadata, output)

    step()  # warm up JIT kernels outside capture
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    for seq_lens in replays:
        replay = case.batch([q_len] * NUM_REQS, seq_lens)
        width = replay.common.block_table_tensor.shape[1]
        common.block_table_tensor.zero_()
        common.block_table_tensor[:, :width].copy_(replay.common.block_table_tensor)
        common.seq_lens.copy_(replay.common.seq_lens)
        common.slot_mapping.copy_(replay.common.slot_mapping)
        replay_common = common.replace(
            seq_lens_cpu_upper_bound=replay.common.seq_lens_cpu_upper_bound,
            max_seq_len=max(seq_lens),
        )
        # The runner rebuilds metadata each step; persistent builder buffers
        # (sliding-window tables) are rewritten in place.
        case.builder.build(0, replay_common)
        query.copy_(case.query(num_tokens))
        output.fill_(float("nan"))
        graph.replay()
        torch.accelerator.synchronize()
        case.check(replay, query, output)


REPLAYS = [
    [1 + 37 * i for i in range(NUM_REQS)],
    [MAX_MODEL_LEN - i for i in range(NUM_REQS)],
    [300] * NUM_REQS,
]


@pytest.mark.parametrize("shape", [QWEN3_235B, QWEN3_8_27B], ids=lambda s: s.id)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_decode_graph_replays_new_lengths(shape, kv_cache_dtype, shuffle):
    case = FaCase(
        shape, kv_cache_dtype=kv_cache_dtype, max_model_len=MAX_MODEL_LEN
    ).setup(NUM_REQS * MAX_MODEL_LEN // 16 + 8)
    _capture_and_replay(case, 1, REPLAYS)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_sliding_window_decode_graph_replays(kv_cache_dtype, monkeypatch):
    set_shuffle(monkeypatch, False)
    case = FaCase(
        MUSE_SPARK_SLIDING,
        kv_cache_dtype=kv_cache_dtype,
        sliding_window=256,
        max_model_len=MAX_MODEL_LEN,
    ).setup(NUM_REQS * MAX_MODEL_LEN // 16 + 8)
    _capture_and_replay(case, 1, REPLAYS)


@pytest.mark.parametrize("q_len", [2, 4])
@pytest.mark.parametrize("shape", [QWEN3_235B, MUSE_SPARK_SLIDING], ids=lambda s: s.id)
def test_spec_decode_graph_replays(shape, q_len, monkeypatch):
    """Uniform multi-token decode (q_len = 1 + spec tokens) under full graphs."""
    set_shuffle(monkeypatch, False)
    case = FaCase(
        shape, sliding_window=shape.sliding_window, max_model_len=MAX_MODEL_LEN
    ).setup(NUM_REQS * MAX_MODEL_LEN // 16 + 8)
    case.builder.reorder_batch_threshold = q_len
    _capture_and_replay(
        case,
        q_len,
        [[s + q_len for s in r] for r in REPLAYS[:1]]
        + [[MAX_MODEL_LEN - i for i in range(NUM_REQS)]],
    )

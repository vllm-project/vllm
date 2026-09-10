# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DecoderReplayLayers runs the replay layers on the forward's replay batch."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.models.deepseek_v41.decoder_replay_layers import (
    DecoderReplayLayers,
    ReplayBatch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
NUM_TOKENS = 401
GRAPH_SIZE = 512
# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
WINDOW = 128
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


def _replay_layers(run_layers, topk_buffer=None, candidate_buffer=None, graphs=False):
    cfg = MagicMock()
    cfg.parallel_config.data_parallel_size = 1
    cfg.compilation_config.cudagraph_capture_sizes = [GRAPH_SIZE] if graphs else None
    source_attn = SimpleNamespace(
        topk_indices_buffer=topk_buffer, candidate_block_buffer=candidate_buffer
    )
    return DecoderReplayLayers(cfg, WINDOW, source_attn, run_layers)


def _context(metadata, cudagraph_mode=CUDAGraphMode.NONE):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(NUM_TOKENS, dtype=torch.bool, device=DEVICE),
        cudagraph_runtime_mode=cudagraph_mode,
    )


def _replay_batch(rows, metadata, graph_size=None, trims=True):
    rows = torch.tensor(rows, device=DEVICE)
    return ReplayBatch(
        rows=rows,
        trims=trims,
        graph_size=graph_size,
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(len(rows), dtype=torch.bool, device=DEVICE),
        dp_metadata=None,
    )


def test_run_gathers_states_and_realigns_shared_indexer_buffers():
    topk = torch.arange(NUM_TOKENS * 4, device=DEVICE).view(NUM_TOKENS, 4).int()
    candidates = torch.arange(NUM_TOKENS * 3, device=DEVICE).view(NUM_TOKENS, 3).int()
    topk_before, candidates_before = topk.clone(), candidates.clone()
    seen = {}

    def run_layers(hidden_states, *rest):
        seen["hidden_states"] = hidden_states.clone()
        replay_context = get_forward_context()
        seen["attn_metadata"] = replay_context.attn_metadata
        seen["batch_descriptor"] = replay_context.batch_descriptor
        return (hidden_states, rest[2])  # pre_mix

    layers = _replay_layers(run_layers, topk, candidates)
    hidden = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.float32)[:, None]
    states = (hidden, hidden.long(), None, hidden, hidden, hidden, hidden)
    full, replay = object(), object()
    layers.replay_batch = _replay_batch(REPLAY_ROWS, replay)
    context = _context(full)
    with override_forward_context(context):
        outputs = layers(*states)
        assert get_forward_context() is context

    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(seen["hidden_states"], hidden[rows])
    assert seen["attn_metadata"] is replay
    assert seen["batch_descriptor"].num_tokens == len(REPLAY_ROWS)
    assert torch.equal(topk[: len(REPLAY_ROWS)], topk_before[rows])
    assert torch.equal(candidates[: len(REPLAY_ROWS)], candidates_before[rows])
    assert outputs[0].shape[0] == NUM_TOKENS
    assert torch.equal(outputs[0][rows], hidden[rows])
    assert outputs[0][1:173].abs().sum() == 0


def test_no_replay_batch_runs_the_whole_batch():
    seen = {}

    def run_layers(hidden_states, *rest):
        seen["attn_metadata"] = get_forward_context().attn_metadata
        return (hidden_states, rest[2])

    layers = _replay_layers(run_layers)
    hidden = torch.zeros(NUM_TOKENS, 1, device=DEVICE)
    states = (hidden, hidden.long(), None, hidden, hidden, hidden, hidden)
    full = object()
    with override_forward_context(_context(full)):
        outputs = layers(*states)
    assert outputs[0] is hidden and seen["attn_metadata"] is full
    # A graph capture's eager warmup carries the capture's replay batch.
    layers.replay_batch = _replay_batch(REPLAY_ROWS, object(), graph_size=GRAPH_SIZE)
    with override_forward_context(_context(full)):
        outputs = layers(*states)
    assert outputs[0] is hidden and seen["attn_metadata"] is full


class _Metadata:
    """Persistent buffers the fake kernels read, refilled per step."""

    def __init__(self):
        self.slot_mapping = torch.zeros(GRAPH_SIZE, dtype=torch.int64, device=DEVICE)
        self.token_to_req_indices = torch.zeros(
            GRAPH_SIZE, dtype=torch.int32, device=DEVICE
        )
        self.num_tokens = 0

    def fill(self, rows: list[int]) -> "_Metadata":
        rows_t = torch.tensor(rows, device=DEVICE)
        self.slot_mapping[: len(rows)] = rows_t * 7
        self.token_to_req_indices[: len(rows)] = (rows_t // 100).int()
        self.num_tokens = len(rows)
        return self


def _fake_attention(x: torch.Tensor, out: torch.Tensor) -> None:
    """Reads the current metadata through an eager graph break, like the real
    attention kernels."""

    def run() -> None:
        md = get_forward_context().attn_metadata
        n = md.num_tokens
        out[:n] = x[:n] + md.token_to_req_indices[:n].to(x.dtype)[:, None]

    capture = BreakableCUDAGraphCapture.current()
    if capture is not None and capture.capturing:
        capture.add_eager(run)
    else:
        run()


def _fake_replay_layers(hidden, positions, _, pre_mix, post_mix, res_mix, residual):
    # Like the window KV insert, this captured op takes the metadata's slot
    # mapping by address.
    slots = get_forward_context().attn_metadata.slot_mapping
    x = hidden * 2 + slots[: hidden.shape[0], None].to(hidden.dtype)
    out = torch.empty_like(x)
    _fake_attention(x, out)
    return (out + positions[:, None].to(out.dtype), pre_mix + residual)


def _states(seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    hidden = torch.randn(NUM_TOKENS, 3, device=DEVICE, generator=g)
    return (
        hidden,
        torch.arange(NUM_TOKENS, device=DEVICE),
        None,
        hidden + 1,
        hidden + 2,
        hidden + 3,
        hidden + 4,
    )


def test_replay_graph_matches_eager(monkeypatch):
    """The replay graph captured with the model graph matches the eager path on
    a batch it never saw, from the model graph's replay and from an eager
    model forward."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import _current_stream_tls

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    eager = _replay_layers(_fake_replay_layers)
    graphed = _replay_layers(_fake_replay_layers, graphs=True)
    metadata = _Metadata()
    states = _states(0)
    all_rows = list(range(NUM_TOKENS))

    prev_stream = getattr(_current_stream_tls, "value", None)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            with override_forward_context(_context(metadata.fill(all_rows))):
                graphed(*states)  # the profile run sizes the fixed buffers
            graphed.replay_batch = _replay_batch(
                all_rows, metadata, GRAPH_SIZE, trims=False
            )
            with override_forward_context(_context(None, CUDAGraphMode.PIECEWISE)):
                outer = BreakableCUDAGraphCapture(
                    current_platform.get_global_graph_pool()
                )
                with outer:
                    hidden_out, pre_mix_out = graphed(*states)
            assert set(graphed.graphs.wrapper.entries) == {
                BatchDescriptor(num_tokens=GRAPH_SIZE)
            }
            for dst, src in zip(states, _states(1)):
                if dst is not None:
                    dst.copy_(src)
            graphed.replay_batch = _replay_batch(
                REPLAY_ROWS, metadata.fill(REPLAY_ROWS), GRAPH_SIZE
            )
            with override_forward_context(_context(None, CUDAGraphMode.PIECEWISE)):
                outer.replay()
                torch.accelerator.synchronize()
                outputs = graphed(*states)
            eager.replay_batch = _replay_batch(REPLAY_ROWS, metadata.fill(REPLAY_ROWS))
            with override_forward_context(_context(None)):
                expected = eager(*states)
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        _current_stream_tls.value = prev_stream
    assert torch.equal(hidden_out, expected[0])
    assert torch.equal(pre_mix_out, expected[1])
    assert torch.equal(outputs[0], expected[0])
    assert torch.equal(outputs[1], expected[1])

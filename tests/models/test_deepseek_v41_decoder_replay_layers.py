# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DecoderReplayLayers runs the replay layers on the forward's replay batch."""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.config.kernel import FLASHINFER_MOE_EP_BACKENDS, NATIVE_MEGA_MOE_BACKENDS
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
from vllm.models.deepseek_v41.nvidia.decoder_replay_cudagraph import (
    DecoderReplayCudaGraphManager,
)
from vllm.v1.worker.gpu import cudagraph_utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
NUM_TOKENS = 401
HC, HIDDEN = 2, 3
# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
WINDOW = 128
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


def _context(metadata):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(NUM_TOKENS, dtype=torch.bool, device=DEVICE),
    )


def _set_replay_batch(layers, rows, metadata):
    layers.replay_batch = ReplayBatch(
        torch.tensor(rows, device=DEVICE),
        ForwardContext(no_compile_layers={}, attn_metadata=metadata, slot_mapping={}),
    )


def _states(num_tokens: int):
    """(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual),
    shaped as the model's: hidden states [T, H], mixes [T, HC], residual [T, HC, H]."""
    hidden = torch.arange(num_tokens, device=DEVICE, dtype=torch.float32)
    hidden = hidden[:, None].expand(num_tokens, HIDDEN).contiguous()
    mix = hidden[:, :HC].contiguous()
    residual = hidden[:, None, :].expand(num_tokens, HC, HIDDEN).contiguous()
    return (
        hidden,
        torch.arange(num_tokens, device=DEVICE),
        None,
        mix + 1,
        mix + 2,
        mix + 3,
        residual,
    )


def test_run_gathers_states():
    """Every row's KV is written first; the layers then run on the replay rows."""
    seen = {}

    def run_layers(hidden_states, positions, input_ids, pre_mix, post, res, residual):
        seen["hidden_states"] = hidden_states.clone()
        replay_context = get_forward_context()
        seen["attn_metadata"] = replay_context.attn_metadata
        seen["is_padding"] = replay_context.is_padding
        return residual, pre_mix

    write_batch_kv = MagicMock()
    layers = DecoderReplayLayers(WINDOW, run_layers, write_batch_kv, set(), "swa")
    states = _states(NUM_TOKENS)
    hidden, residual = states[0], states[-1]
    full, sub = object(), object()
    _set_replay_batch(layers, REPLAY_ROWS, sub)
    context = _context(full)
    with override_forward_context(context):
        outputs = layers(*states)
        assert get_forward_context() is context

    write_batch_kv.assert_called_once_with(*states)
    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(seen["hidden_states"], hidden[rows])
    assert seen["attn_metadata"] is sub and seen["is_padding"] is None
    assert len(outputs) == 2 and outputs[0].shape == residual.shape
    assert torch.equal(outputs[0][rows], residual[rows])
    assert outputs[0][1:173].abs().sum() == 0


def test_no_replay_batch_runs_the_whole_batch():
    seen = {}

    def run_layers(hidden_states, *rest):
        seen["attn_metadata"] = get_forward_context().attn_metadata
        return (hidden_states,)

    write_batch_kv = MagicMock()
    layers = DecoderReplayLayers(WINDOW, run_layers, write_batch_kv, set(), "swa")
    states = _states(NUM_TOKENS)
    full = object()
    with override_forward_context(_context(full)):
        (output,) = layers(*states)
    assert output is states[0] and seen["attn_metadata"] is full
    write_batch_kv.assert_not_called()


def _graph_context(num_tokens):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
    )


def test_piecewise_graphs_break_out_by_size():
    """A PIECEWISE graph's KV write and break are fixed at capture, when no replay
    batch is set, so both follow the graph's size rather than the replay batch."""
    write_batch_kv = MagicMock()
    layers = DecoderReplayLayers(
        WINDOW, lambda *states: ("whole",), write_batch_kv, set(), "swa"
    )
    layers._run_in_graph_break = lambda *states: ("break",)  # type: ignore[method-assign]
    states = _states(NUM_TOKENS)
    with override_forward_context(_graph_context(1024)):
        assert layers(*states) == ("whole",)  # no threshold: no break
        layers.trim_threshold = 768
        assert layers(*states) == ("break",)
    _set_replay_batch(layers, REPLAY_ROWS, object())
    with override_forward_context(_graph_context(576)):
        assert not layers.uses_replay_batch()  # a replay batch alone does not break out
    write_batch_kv.assert_called_once_with(*states)


@pytest.mark.parametrize(
    ("moe_backend", "ids_dtype"),
    [
        ("auto", torch.int32),
        *((b, torch.int64) for b in sorted(NATIVE_MEGA_MOE_BACKENDS)),
        *((b, torch.int32) for b in sorted(FLASHINFER_MOE_EP_BACKENDS)),
    ],
    ids=str,
)
def test_replay_graph_matches_eager(monkeypatch, moe_backend, ids_dtype):
    """The replay graph of the next captured size pads and runs the rows. Input ids
    arrive as the model passes them: int32, cast to int64 for native MegaMoE only."""
    monkeypatch.setattr(cudagraph_utils, "get_pp_group", MagicMock)
    capture = torch.cuda.stream(torch.cuda.Stream())
    monkeypatch.setattr(cudagraph_utils, "graph_capture", lambda device: capture)
    compilation = CompilationConfig(decoder_replay_cudagraph_capture_sizes=[4])
    cfg = MagicMock(compilation_config=compilation, speculative_config=None)
    # Mocking it true would capture into the cuMem pool, which breaks on ROCm.
    cfg.use_cumem_cudagraph_pool = False
    cfg.scheduler_config = MagicMock(max_num_seqs=2, max_num_batched_tokens=64)
    cfg.cache_config.use_kda_recoverssm = False
    cfg.model_config.hf_config = MagicMock(hc_mult=HC, hidden_size=HIDDEN)
    cfg.model_config.dtype = torch.float32
    cfg.kernel_config.moe_backend = moe_backend

    def run_layers(hidden, positions, ids, pre, post, mix, residual):
        return residual + hidden[:, None] + positions[:, None, None], pre + ids[:, None]

    layers = DecoderReplayLayers(4, run_layers, MagicMock(), set(), "swa")
    manager = DecoderReplayCudaGraphManager(cfg, DEVICE, layers)

    def prepare(desc):
        rows = torch.arange(desc.num_tokens)
        layers.replay_batch = ReplayBatch(rows, _graph_context(desc.num_tokens))

    manager.capture_replay_graphs(prepare)
    states = [
        torch.randn(12, *b.shape[1:], device=DEVICE).to(b.dtype)
        for b in manager.input_buffers
    ]
    states[2] = torch.arange(12, device=DEVICE, dtype=ids_dtype)  # input_ids
    rows = torch.tensor([11, 2, 7], device=DEVICE)
    desc = manager.dispatch(2, 3, None, 0)
    with override_forward_context(_graph_context(desc.num_tokens)):
        actual = manager.run(rows, states)
    for a, e in zip(actual, run_layers(*(t[rows] for t in states))):
        torch.testing.assert_close(a, e, rtol=0, atol=0)

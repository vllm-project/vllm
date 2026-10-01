# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DecoderReplayLayers runs the replay layers on the forward's replay batch."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
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
from vllm.v1.worker.gpu import cudagraph_utils as gpu_cudagraph_utils

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


def test_run_gathers_states_and_realigns_shared_indexer_buffers():
    topk = torch.arange(NUM_TOKENS * 4, device=DEVICE).view(NUM_TOKENS, 4).int()
    candidates = torch.arange(NUM_TOKENS * 3, device=DEVICE).view(NUM_TOKENS, 3).int()
    topk_before, candidates_before = topk.clone(), candidates.clone()
    seen = {}

    def run_layers(hidden_states, positions, input_ids, pre_mix, post, res, residual):
        seen["hidden_states"] = hidden_states.clone()
        replay_context = get_forward_context()
        seen["attn_metadata"] = replay_context.attn_metadata
        seen["is_padding"] = replay_context.is_padding
        return residual, pre_mix

    layers = DecoderReplayLayers(WINDOW, run_layers, [topk, candidates])
    states = _states(NUM_TOKENS)
    hidden, residual = states[0], states[-1]
    full, sub = object(), object()
    _set_replay_batch(layers, REPLAY_ROWS, sub)
    context = _context(full)
    with override_forward_context(context):
        outputs = layers(*states)
        assert get_forward_context() is context

    rows = torch.tensor(REPLAY_ROWS, device=DEVICE)
    assert torch.equal(seen["hidden_states"], hidden[rows])
    assert seen["attn_metadata"] is sub and seen["is_padding"] is None
    assert torch.equal(topk[: len(REPLAY_ROWS)], topk_before[rows])
    assert torch.equal(candidates[: len(REPLAY_ROWS)], candidates_before[rows])
    assert len(outputs) == 2 and outputs[0].shape == residual.shape
    assert torch.equal(outputs[0][rows], residual[rows])
    assert outputs[0][1:173].abs().sum() == 0


def test_no_replay_batch_runs_the_whole_batch():
    seen = {}

    def run_layers(hidden_states, *rest):
        seen["attn_metadata"] = get_forward_context().attn_metadata
        return (hidden_states,)

    layers = DecoderReplayLayers(WINDOW, run_layers, [])
    states = _states(NUM_TOKENS)
    full = object()
    with override_forward_context(_context(full)):
        (output,) = layers(*states)
    assert output is states[0] and seen["attn_metadata"] is full


def test_graph_break_writes_fixed_buffers():
    """Above the trim threshold, a PIECEWISE graph's replay layers write fixed
    buffers, which the graph's next segment reads on every replay."""
    layers = DecoderReplayLayers(WINDOW, lambda hidden, *rest: (hidden * 2,), [])
    layers.trim_threshold = NUM_TOKENS - 1
    context = ForwardContext(
        no_compile_layers={},
        attn_metadata=object(),
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
        batch_descriptor=BatchDescriptor(num_tokens=NUM_TOKENS),
    )
    states = _states(NUM_TOKENS)
    with override_forward_context(context):
        (first,) = layers(*states)
        (second,) = layers(states[0] * 3, *states[1:])
    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(second, states[0] * 6)


@pytest.fixture
def replay_config(monkeypatch):
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return SimpleNamespace(
        compilation_config=CompilationConfig(),
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=None,
        cache_config=SimpleNamespace(use_kda_recoverssm=False),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(hc_mult=HC, hidden_size=HIDDEN),
            dtype=torch.float32,
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
    )


@pytest.mark.parametrize(
    "max_num_seqs,main_sizes,replay_sizes,expected",
    [
        (8, [6, 1020, 2046, 8190], [], [6, 1020, 1024]),
        (128, [6, 1020, 2046, 8190], [], [6, 1020, 2046, 8190]),
        (8, [6, 510], [], [6, 510]),
        (8, [6, 1020], [2048, 256, 2048], [256, 2048]),
    ],
)
def test_decoder_replay_graph_sizes(
    replay_config, max_num_seqs, main_sizes, replay_sizes, expected
):
    """Replay captures, dispatch and buffers agree without changing model sizes."""
    cfg = replay_config
    cfg.scheduler_config.max_num_seqs = max_num_seqs
    cfg.compilation_config.cudagraph_capture_sizes = main_sizes
    cfg.compilation_config.max_cudagraph_capture_size = main_sizes[-1]
    cfg.compilation_config.decoder_replay_cudagraph_capture_sizes = replay_sizes
    layers = DecoderReplayLayers(128, lambda *args: args, [])
    manager = DecoderReplayCudaGraphManager(cfg, torch.device("cpu"), layers)
    assert (
        sorted(d.num_tokens for d in manager._capture_descs[CUDAGraphMode.PIECEWISE])
        == expected
    )
    assert all(buf.shape[0] == expected[-1] for buf in manager.inputs)
    manager._graphs_captured = True
    for tokens in (1, expected[-1] - 1, expected[-1]):
        desc = manager.dispatch(max_num_seqs, tokens, None, 0)
        assert desc.cg_mode == CUDAGraphMode.PIECEWISE
        assert desc.num_tokens == next(s for s in expected if s >= tokens)
    assert manager.dispatch(max_num_seqs, expected[-1] + 1, None, 0).cg_mode == (
        CUDAGraphMode.NONE
    )
    assert cfg.compilation_config.cudagraph_capture_sizes == main_sizes
    assert cfg.compilation_config.max_cudagraph_capture_size == main_sizes[-1]
    assert cfg.compilation_config.decoder_replay_cudagraph_capture_sizes == replay_sizes


def test_decoder_replay_graph_sizes_fit_runner_buffers(replay_config):
    cfg = replay_config
    cfg.compilation_config.decoder_replay_cudagraph_capture_sizes = [
        cfg.scheduler_config.max_num_batched_tokens + 1
    ]
    with pytest.raises(ValueError, match="must not exceed max_num_batched_tokens"):
        DecoderReplayCudaGraphManager(
            cfg, torch.device("cpu"), DecoderReplayLayers(128, lambda *args: args, [])
        )


@torch.inference_mode()
def test_bounded_replay_cuda_graph_matches_eager(replay_config, monkeypatch):
    """Changing replay rows and padding reuses captured graphs with eager results."""
    cfg = replay_config
    cfg.compilation_config.cudagraph_capture_sizes = [4, 8, 16]
    cfg.compilation_config.max_cudagraph_capture_size = 16
    cfg.scheduler_config.max_num_seqs = 2
    monkeypatch.setattr(
        gpu_cudagraph_utils,
        "graph_capture",
        lambda device: torch.cuda.stream(torch.cuda.Stream(device=device)),
    )
    monkeypatch.setattr(gpu_cudagraph_utils, "is_global_first_rank", lambda: False)

    def run_layers(hidden, positions, ids, pre, post, mix, residual):
        return residual + hidden[:, None, :] + positions[:, None, None], pre + ids[
            :, None
        ]

    layers = DecoderReplayLayers(4, run_layers, [])
    manager = DecoderReplayCudaGraphManager(cfg, DEVICE, layers)

    def prepare(desc):
        layers.replay_batch = ReplayBatch(
            torch.arange(desc.num_tokens, device=DEVICE),
            ForwardContext(
                no_compile_layers={},
                attn_metadata={},
                slot_mapping={},
                batch_descriptor=BatchDescriptor(num_tokens=desc.num_tokens),
            ),
        )

    torch.accelerator.synchronize()
    manager.capture_replay_graphs(prepare)
    torch.accelerator.synchronize()
    assert {d.num_tokens for d in manager.breakable_cg_runner.entries} == {4, 8}
    assert layers.replay_batch is None
    states = tuple(
        torch.randn(12, *buf.shape[1:], device=DEVICE).to(buf.dtype)
        for buf in manager.inputs
    )
    for row_ids in ([11, 2, 7], [1, 2, 3, 4, 5, 6, 7, 8], [9, 8, 7, 6, 5], [3]):
        rows = torch.tensor(row_ids, device=DEVICE)
        desc = manager.dispatch(2, len(row_ids), None, 0)
        context = ForwardContext(
            no_compile_layers={},
            attn_metadata={},
            slot_mapping={},
            cudagraph_runtime_mode=desc.cg_mode,
            batch_descriptor=BatchDescriptor(num_tokens=desc.num_tokens),
        )
        with override_forward_context(context):
            actual = manager.run(rows, states)
            expected = run_layers(*(t[rows] for t in states))
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=0, atol=0)
        states[0].add_(1)
    assert len(manager.breakable_cg_runner.entries) == 2

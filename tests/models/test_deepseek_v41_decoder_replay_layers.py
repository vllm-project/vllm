# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DecoderReplayLayers runs the replay layers on the forward's replay batch."""

import pytest
import torch

from vllm.compilation.breakable_cudagraph import (
    BreakableCUDAGraphCapture,
    eager_break_during_capture,
)
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    override_forward_context,
)
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")
NUM_TOKENS = 401
GRAPH_SIZE = 512
HC, HIDDEN = 2, 3
# decode (1 token), trimmed prefill (300 of 300), untrimmed prefill (100 of 300)
WINDOW = 128
REPLAY_ROWS = [0, *range(301 - WINDOW, 301), *range(301, 401)]


def _context(metadata, cudagraph_mode=CUDAGraphMode.NONE):
    return ForwardContext(
        no_compile_layers={},
        attn_metadata=metadata,
        slot_mapping={},
        is_padding=torch.zeros(GRAPH_SIZE, dtype=torch.bool, device=DEVICE),
        cudagraph_runtime_mode=cudagraph_mode,
    )


def _set_replay_batch(layers, rows, metadata):
    layers.rows = torch.tensor(rows, device=DEVICE)
    layers.forward_context = ForwardContext(
        no_compile_layers={}, attn_metadata=metadata, slot_mapping={}
    )


def _states(num_tokens: int, seed: int | None = None):
    """(hidden_states, positions, input_ids, pre_mix, post_mix, res_mix, residual),
    shaped as the model's: hidden states [T, H], mixes [T, HC], residual [T, HC, H]."""
    if seed is None:
        hidden = torch.arange(num_tokens, device=DEVICE, dtype=torch.float32)
        hidden = hidden[:, None].expand(num_tokens, HIDDEN).contiguous()
    else:
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        hidden = torch.randn(num_tokens, HIDDEN, device=DEVICE, generator=g)
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


def _new_outputs(hidden, positions, input_ids, pre_mix, post_mix, res_mix, residual):
    """The model's outputs: collapsed hidden states like the residual, pre_mix."""
    return torch.zeros_like(residual), torch.zeros_like(pre_mix)


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

    layers = DecoderReplayLayers(WINDOW, run_layers, _new_outputs, [topk, candidates])
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

    layers = DecoderReplayLayers(WINDOW, run_layers, _new_outputs, [])
    states = _states(NUM_TOKENS)
    full = object()
    with override_forward_context(_context(full)):
        (output,) = layers(*states)
    assert output is states[0] and seen["attn_metadata"] is full


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
    """Reads the current metadata, like the real attention kernels."""
    md = get_forward_context().attn_metadata
    n = md.num_tokens
    out[:n] = x[:n] + md.token_to_req_indices[:n].to(x.dtype)[:, None, None]


def _fake_layers(attention):
    def run_layers(hidden, positions, _, pre_mix, post_mix, res_mix, residual):
        # Like the window KV insert, this op reads the metadata's slot mapping.
        slots = get_forward_context().attn_metadata.slot_mapping
        x = residual * hidden[:, None, :] + slots[: hidden.shape[0], None, None].to(
            hidden.dtype
        )
        out = torch.empty_like(x)
        attention(x, out)
        return (out + positions[:, None, None].to(out.dtype), pre_mix + post_mix)

    return run_layers


def test_piecewise_graph_replays_eagerly_into_its_outputs(monkeypatch):
    """Captured as an eager break of a piecewise graph on a dummy batch, the
    replay runs on every replay of that graph with the step's replay batch, into
    the outputs the graph allocated."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import _current_stream_tls

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    # The attention kernels are eager breaks of their own; inside the replay's
    # break they simply run.
    run_layers = _fake_layers(eager_break_during_capture(_fake_attention))
    graphed = DecoderReplayLayers(WINDOW, run_layers, _new_outputs, [])
    eager = DecoderReplayLayers(WINDOW, run_layers, _new_outputs, [])
    metadata = _Metadata()
    states = _states(GRAPH_SIZE, seed=0)
    all_rows = list(range(GRAPH_SIZE))

    prev_stream = getattr(_current_stream_tls, "value", None)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(stream):
            with override_forward_context(
                _context(metadata.fill(all_rows), CUDAGraphMode.PIECEWISE)
            ):
                capture = BreakableCUDAGraphCapture(
                    current_platform.get_global_graph_pool()
                )
                with capture:
                    hidden_out, pre_mix_out = graphed(*states)
            assert capture._num_eager_breaks == 1
            # A later step: new inputs in the graph's buffers, a trimming batch.
            for dst, src in zip(states, _states(GRAPH_SIZE, seed=1)):
                if dst is not None:
                    dst.copy_(src)
            _set_replay_batch(graphed, REPLAY_ROWS, metadata.fill(REPLAY_ROWS))
            with override_forward_context(_context(None, CUDAGraphMode.PIECEWISE)):
                capture.replay()
            _set_replay_batch(eager, REPLAY_ROWS, metadata.fill(REPLAY_ROWS))
            with override_forward_context(_context(None)):
                expected = eager(*states)
            torch.accelerator.synchronize()
    finally:
        torch.cuda.current_stream().wait_stream(stream)
        _current_stream_tls.value = prev_stream
    assert torch.equal(hidden_out, expected[0])
    assert torch.equal(pre_mix_out, expected[1])
    assert hidden_out[1:173].abs().sum() == 0

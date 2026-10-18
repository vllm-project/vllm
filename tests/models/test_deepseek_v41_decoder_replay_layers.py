# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DecoderReplayLayers runs the replay layers on the forward's replay batch."""

from functools import partial
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
    ReplayStage,
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


def test_stages_run_on_shrinking_rows():
    """A later stage runs its rows of the previous stage's outputs, with its own
    batch positions; outputs and each stage's aux land on that stage's rows.
    The buffers layers share move to the next stage's rows before it runs."""
    seen = {}
    shared = []

    def run_stage(i, hidden, positions, input_ids, pre_mix, post, res, residual):
        seen[i] = (hidden.clone(), positions.clone(), get_forward_context())
        if i == 0:
            return hidden + 1, pre_mix, post, res, residual, hidden * 10
        return hidden * 2, pre_mix, hidden + 100

    layers = DecoderReplayLayers(
        WINDOW,
        MagicMock(),
        MagicMock(),
        set(),
        "swa",
        kept_rows=200,
        stage_rows=[200, WINDOW],
        run_stage=run_stage,
        stage_metadata_prefixes=[set(), set()],
        select_shared_rows=lambda select: shared.append((len(seen), select)),
    )
    # The trimmed prefill keeps 200 rows, then 128; the others keep theirs.
    rows0 = [0, *range(101, 301), *range(301, 401)]
    rows1 = [0, *range(173, 301), *range(301, 401)]
    select = [rows0.index(row) for row in rows1]
    contexts = [
        ForwardContext(no_compile_layers={}, attn_metadata=m, slot_mapping={})
        for m in ("stage0", "stage1")
    ]
    layers.replay_batch = ReplayBatch(
        torch.tensor(rows0, device=DEVICE),
        contexts[0],
        stages=[
            ReplayStage(
                torch.tensor(rows1, device=DEVICE),
                torch.tensor(select, device=DEVICE),
                contexts[1],
            )
        ],
    )
    states = _states(NUM_TOKENS)
    hidden, positions = states[0], states[1]
    with override_forward_context(_context(object())):
        out, pre_mix, aux0, aux1 = layers(*states)
    r0 = torch.tensor(rows0, device=DEVICE)
    r1 = torch.tensor(rows1, device=DEVICE)
    assert torch.equal(seen[0][0], hidden[r0]) and seen[0][2] is contexts[0]
    assert torch.equal(seen[1][0], hidden[r1] + 1) and seen[1][2] is contexts[1]
    assert torch.equal(seen[1][1], positions[r1])
    assert torch.equal(out[r1], (hidden[r1] + 1) * 2)
    assert out[1:173].abs().sum() == 0
    assert torch.equal(pre_mix[r1], states[3][r1])
    assert torch.equal(aux0[r0], hidden[r0] * 10) and aux0[1:101].abs().sum() == 0
    assert torch.equal(aux1[r1], hidden[r1] + 101)
    assert len(shared) == 1 and shared[0][0] == 1
    assert shared[0][1].tolist() == select


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


def _graph_config(monkeypatch, moe_backend="auto"):
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
    return cfg


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
    cfg = _graph_config(monkeypatch, moe_backend)

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
        for b in manager.stage_input_buffers[0]
    ]
    states[2] = torch.arange(12, device=DEVICE, dtype=ids_dtype)  # input_ids
    rows = torch.tensor([11, 2, 7], device=DEVICE)
    desc = manager.dispatch(2, 3, None, 0)
    with override_forward_context(_graph_context(desc.num_tokens)):
        actual = manager.run(rows, states)
    for a, e in zip(actual, run_layers(*(t[rows] for t in states))):
        torch.testing.assert_close(a, e, rtol=0, atol=0)


def test_staged_replay_graphs_match_eager(monkeypatch):
    """One capture holds every stage's graph; each stage replays on its rows of
    the previous stage's outputs."""
    cfg = _graph_config(monkeypatch)

    def run_stage(i, hidden, positions, ids, pre, post, mix, residual):
        if i == 0:
            return hidden + positions[:, None], pre, post, mix, residual
        return residual + hidden[:, None], pre + ids[:, None]

    layers = DecoderReplayLayers(
        4,
        MagicMock(),
        MagicMock(),
        set(),
        "swa",
        kept_rows=4,
        stage_rows=[4, 2],
        run_stage=run_stage,
        stage_metadata_prefixes=[set(), set()],
    )
    manager = DecoderReplayCudaGraphManager(cfg, DEVICE, layers)

    def replay_batch(rows, select, num_tokens, run_graph=None):
        return ReplayBatch(
            torch.tensor(rows, device=DEVICE),
            _graph_context(num_tokens),
            run_graph and partial(run_graph, stage=0),
            [
                ReplayStage(
                    torch.tensor([rows[j] for j in select], device=DEVICE),
                    torch.tensor(select, device=DEVICE),
                    _graph_context(num_tokens),
                    run_graph and partial(run_graph, stage=1),
                )
            ],
        )

    def prepare(desc):
        rows = list(range(desc.num_tokens))
        layers.replay_batch = replay_batch(rows, rows[-2:], desc.num_tokens)

    manager.capture_replay_graphs(prepare)
    states = list(_states(NUM_TOKENS))
    states[2] = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int32)
    outputs = []
    for run_graph in (None, manager.run):
        layers.replay_batch = replay_batch([2, 7, 9], [1, 2], 4, run_graph)
        with override_forward_context(_context(object())):
            outputs.append(layers(*states))
    for graph, eager in zip(*outputs, strict=True):
        torch.testing.assert_close(graph, eager, rtol=0, atol=0)
    assert outputs[0][0][[7, 9]].abs().sum() > 0 and outputs[0][0][2].sum() == 0

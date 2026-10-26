# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model Runner V2's low-batch speculative decode paths.

The fused kernels match the sequences they replace bitwise on random batches
(padded rows, prefill-tail rows): the draft prefill's fused IO and the final
draft step's store into the runner's table.
Also the post-forward CUDA graph: its opt-in, per-step conditions and replay.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.worker.gpu import post_forward_cudagraph as pf_module
from vllm.v1.worker.gpu.cudagraph_utils import copy_small
from vllm.v1.worker.gpu.input_batch import (
    InputBuffers,
)
from vllm.v1.worker.gpu.post_forward_cudagraph import (
    PostForwardCudaGraphs,
    maybe_create_post_forward_cudagraphs,
)
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    _draft_prefill_store_kernel,
    prepare_prefill_inputs,
    update_draft_inputs,
)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

K = 3  # speculative tokens
MAX_REQS = 8
HIDDEN = 3000
I64 = torch.int64


def _ri(g, lo, hi, shape, dtype=torch.int32):
    """Random integers in [lo, hi) of ``shape`` (an int or a tuple)."""
    shape = shape if isinstance(shape, tuple) else (shape,)
    return torch.randint(lo, hi, shape, generator=g, dtype=dtype)


def _map(x, fn):
    """Apply ``fn`` to every tensor in nested tuples, namespaces and dicts."""
    if isinstance(x, torch.Tensor):
        return fn(x)
    if isinstance(x, tuple):
        return tuple(_map(v, fn) for v in x)
    if isinstance(x, SimpleNamespace):
        return SimpleNamespace(**{k: _map(v, fn) for k, v in vars(x).items()})
    if isinstance(x, dict):
        return {k: _map(v, fn) for k, v in x.items()}
    return x


@needs_cuda
@pytest.mark.parametrize("seed", range(6))
def test_draft_prefill_positions_and_row_store(seed: int):
    """prepare_prefill_inputs also writes each request's sampled-row position
    and +1 (padded rows: position 0's), and _draft_prefill_store_kernel stores
    the sampled hidden rows and those positions as the gathers they replace."""
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cuda")
    num_reqs = int(torch.randint(1, MAX_REQS + 1, (1,), generator=g))
    num_tokens = num_reqs * (K + 1)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        idx_mapping=torch.randperm(MAX_REQS, generator=g)[:num_reqs].int(),
        input_ids=_ri(g, 0, 99, num_tokens),
        positions=_ri(g, 0, 999, num_tokens, I64),
        query_start_loc=torch.arange(num_reqs + 1, dtype=torch.int32) * (K + 1),
        seq_lens=_ri(g, 5, 99, num_reqs),
    )
    num_sampled = _ri(g, 1, K + 2, num_reqs)
    last_sampled = _ri(g, 0, 99, (MAX_REQS, 1), I64)
    next_prefill = _ri(g, 0, 99, MAX_REQS)
    src = torch.randn(num_tokens, HIDDEN, generator=g).bfloat16()
    batch, num_sampled, last_sampled, next_prefill, src = _map(
        (batch, num_sampled, last_sampled, next_prefill, src), torch.Tensor.cuda
    )
    last_token_indices = torch.zeros(MAX_REQS, dtype=I64, device=dev)
    prefill_positions = torch.full((MAX_REQS,), -7, dtype=I64, device=dev)
    prefill_sample_src = torch.full_like(prefill_positions, -7)
    buffers = InputBuffers(MAX_REQS, num_tokens + 4, dev)
    prepare_prefill_inputs(
        last_token_indices,
        torch.zeros((), dtype=I64, device=dev),
        buffers,
        batch,
        num_sampled,
        K + 1 - num_sampled,
        last_sampled,
        next_prefill,
        MAX_REQS,
        prefill_positions,
        prefill_sample_src,
    )
    gathered = buffers.positions[last_token_indices]
    assert torch.equal(prefill_positions, gathered)
    assert torch.equal(prefill_sample_src, gathered + 1)

    dst = torch.zeros(MAX_REQS, HIDDEN, dtype=torch.bfloat16, device=dev)
    dst_pos, dst_src = torch.zeros(2, MAX_REQS, dtype=I64, device=dev)
    _draft_prefill_store_kernel[(num_reqs, triton.cdiv(HIDDEN, 1024))](
        src,
        src.stride(0),
        last_token_indices,
        dst,
        dst.stride(0),
        prefill_positions,
        dst_pos,
        prefill_sample_src,
        dst_src,
        HIDDEN,
        BLOCK=1024,
    )
    assert torch.equal(dst[:num_reqs], src[last_token_indices[:num_reqs]])
    assert torch.equal(dst_pos[:num_reqs], prefill_positions[:num_reqs])
    assert torch.equal(dst_src[:num_reqs], prefill_sample_src[:num_reqs])


@needs_cuda
@pytest.mark.parametrize("seed", range(4))
def test_final_draft_step_stores_into_the_sink(seed: int):
    """update_draft_inputs with a sink: the final step stores each request's
    drafts into sink[idx_mapping[r]] (the runner's draft-token table) exactly as
    the runner's store would, and skips rows whose idx_mapping is -1."""
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cuda")
    num_reqs = int(torch.randint(1, MAX_REQS + 1, (1,), generator=g))
    hidden = torch.randn(num_reqs, HIDDEN, generator=g).bfloat16().to(dev)
    output = _ri(g, 0, 99, (MAX_REQS, K), I64).to(dev)
    sink = torch.full((MAX_REQS, K), -5, dtype=I64, device=dev)
    idx_mapping = torch.randperm(MAX_REQS, generator=g)[:num_reqs].int().to(dev)
    if num_reqs > 1:
        idx_mapping[-1] = -1  # a padded row
    drafts = _ri(g, 0, 99, num_reqs, I64).to(dev)
    update_draft_inputs(
        drafts,
        torch.full((), K - 1, dtype=I64, device=dev),
        hidden,
        output,
        torch.zeros(MAX_REQS, HIDDEN, dtype=torch.bfloat16, device=dev),
        InputBuffers(MAX_REQS, MAX_REQS * (K + 1), dev),
        torch.zeros(MAX_REQS, dtype=I64, device=dev),
        num_reqs,
        1 << 20,
        K,
        idx_mapping=idx_mapping,
        sink=sink,
    )
    expected = torch.full_like(sink, -5)
    for row, target in enumerate(idx_mapping.tolist()):
        if target >= 0:
            expected[target] = output[row]
    assert torch.equal(output[:num_reqs, K - 1], drafts)
    assert torch.equal(sink, expected)


@pytest.fixture
def fake_cudagraphs(monkeypatch):
    """Record captures, replays and stream waits in place of CUDA graphs: a
    capture runs its launches once, eagerly, and a replay runs nothing."""
    rec = SimpleNamespace(graphs=[], events=[])

    class FakeGraph:
        def __init__(self):
            self.replays = 0
            rec.graphs.append(self)

        def replay(self):
            self.replays += 1
            rec.events.append("replay")

    @contextmanager
    def capture(graph, pool=None, stream=None, capture_error_mode="global"):
        rec.events.append("capture")
        yield

    stream = SimpleNamespace(wait_event=lambda e: rec.events.append(("wait", e)))
    platform = SimpleNamespace(graph_pool_handle=lambda: 0)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeGraph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(pf_module, "current_stream", lambda: stream)
    monkeypatch.setattr(pf_module, "current_platform", platform)
    return rec


def test_post_forward_warm_up_capture_replay_and_release(fake_cudagraphs):
    events = fake_cudagraphs.events
    graphs = PostForwardCudaGraphs(None, 4, 32, torch.device("cpu"))
    calls: list = []

    def fn():
        calls.append(None)
        return (object(), len(calls))

    assert graphs._run("a", fn)[1] == 1 and not events  # warm-up: eager
    captured = graphs._run("a", fn)
    assert events == ["capture", "replay"]
    graphs.set_output_copy_event("copied")
    assert graphs._run("a", fn) is captured and len(calls) == 2
    assert events[2:] == [("wait", "copied"), "replay"]
    graphs.release()
    events.clear()
    graphs._run("a", fn)
    assert len(calls) == 3 and not events  # warm-up again


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r, b: setattr(b, "has_prefill", True),
        lambda r, b: setattr(b, "has_structured_output_reqs", True),
        lambda r, b: setattr(b, "num_draft_tokens", 5),
        lambda r, b: setattr(b, "num_scheduled_tokens", np.array([K + 1, K])),
        lambda r, b: r.sampler.needs_logits_processing.__setitem__(1, True),
        lambda r, b: setattr(r.sampler.sampling_states, "max_num_logprobs", len),
        lambda r, b: r.prompt_logprobs_worker.uses_prompt_logprobs.__setitem__(0, 1),
        lambda r, b: setattr(r.speculator, "acceptance_estimator", object()),
    ],
)
def test_post_forward_step_eligible_conditions(mutate):
    # A runner and a uniform two-request decode batch that meet every condition.
    runner = SimpleNamespace(
        num_speculative_steps=K,
        vocab_size=1000,
        sampler=SimpleNamespace(
            needs_logits_processing=np.zeros(4, dtype=bool),
            sampling_states=SimpleNamespace(max_num_logprobs=lambda idx: NO_LOGPROBS),
        ),
        speculator=SimpleNamespace(acceptance_estimator=None),
        prompt_logprobs_worker=SimpleNamespace(
            uses_prompt_logprobs=np.zeros(4, dtype=bool), token_id_scores={}
        ),
    )
    batch = SimpleNamespace(
        num_reqs=2,
        has_prefill=False,
        has_structured_output_reqs=False,
        num_draft_tokens=2 * K,
        num_scheduled_tokens=np.full(2, K + 1, dtype=np.int32),
        num_tokens=2 * (K + 1),
        idx_mapping_np=np.arange(2, dtype=np.int32),
    )
    graphs = PostForwardCudaGraphs(runner.speculator, 4, 32, torch.device("cpu"))
    assert graphs.step_eligible(runner, batch, None, full_cudagraph=True)
    assert not graphs.step_eligible(runner, batch, None, full_cudagraph=False)
    assert not graphs.step_eligible(runner, batch, object(), full_cudagraph=True)
    mutate(runner, batch)
    assert not graphs.step_eligible(runner, batch, None, full_cudagraph=True)


def test_post_forward_opt_in_and_unsupported_configuration(monkeypatch):
    warnings: list = []
    monkeypatch.setattr(pf_module.logger, "warning", lambda *a: warnings.append(a))
    unset = """sampler rejection_sampler pcp_manager batch_sharder
        adaptive_verification lora_config aux_output_connector fast_prefill"""
    runner = SimpleNamespace(
        model_state=SimpleNamespace(low_latency_decode=False),
        speculator=object(),
        use_pp=True,
        dcp_size=1,
        **dict.fromkeys(unset.split()),
    )
    # Off unless the model state opts in: no graphs and no warning.
    assert maybe_create_post_forward_cudagraphs(runner) is None and not warnings
    runner.model_state.low_latency_decode = True
    assert maybe_create_post_forward_cudagraphs(runner) is None
    reasons = warnings[0][-1]
    assert "pipeline parallelism" in reasons and "LoRA" not in reasons


@needs_cuda
def test_post_forward_captured_region_replays_like_eager():
    """A region reading a persistent input through copy_small (a kernel copy
    under capture, bitwise, NaN payloads and -0.0 included) replays exactly what
    the eager region computes as the input changes."""
    device = torch.device("cuda")
    graphs = PostForwardCudaGraphs(None, 4, 32, device)
    src = torch.zeros(4, 512, dtype=torch.bfloat16, device=device)
    dst = torch.empty_like(src)

    def region():
        copy_small(dst, src)
        return (dst.view(torch.int16) ^ 0x5A5A, dst.float().sum(dim=-1))

    gen = torch.Generator(device=device).manual_seed(0)
    with torch.cuda.stream(torch.cuda.Stream()):
        for step in range(5):
            src.normal_(generator=gen)
            src[0, :2] = torch.tensor([float("nan"), -0.0])
            src.view(torch.int16)[1, 0] |= 1
            want = tuple(t.clone() for t in region())
            got = graphs._run("key", region)
            torch.accelerator.synchronize()
            for a, b in zip(got, want):
                assert torch.equal(a.view(torch.int8), b.view(torch.int8)), step
    assert graphs.num_replays == 4

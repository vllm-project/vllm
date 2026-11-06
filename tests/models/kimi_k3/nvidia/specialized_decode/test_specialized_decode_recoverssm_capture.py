# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred RecoverSSM under the server's FULL CUDA-graph capture.

A DCP deployment captures its FULL graphs with KV context
(``set_dcp_capture_context``): the dummy requests get block ids 0, 1, ..., so
their KDA state slots are live, while the runner binds no request slots
(``DeferredRecoverSSMState.bind(None)``), so every record row is -1. The capture runs
each graph once eagerly (warmup), then captures and replays it.

Every verify of such a step must stay inside the records allocation: the stock
Triton verify and each PreparedKda policy (flush + regular core, KdaFusedWS,
KdaFused) at the server's capture sizes 5..320 (q = 5, TP8: 12 heads), eager
and graphed. Their writes for rows without a record slot go to the scratch row
-1 (``recoverssm.context.allocate_records``); the records of real slots, the
pending counts and the checkpoints stay untouched, and the outputs equal the same
step's with valid record slots."""

from functools import partial

import pytest
import torch

from vllm.platforms import current_platform

if not (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip("The specialized decode needs SM10x", allow_module_level=True)
pytest.importorskip("cutlass")

from tests.models.kimi_k3.nvidia.specialized_decode.test_specialized_decode_kda_fused import (  # noqa: E402, E501
    DEVICE,
    LOWER_BOUND,
    D,
    _layer,
    _pitch,
    _step_inputs,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda import (  # noqa: E402
    prepared,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.kda.prepared import (  # noqa: E402
    PreparedKda,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.context import (  # noqa: E402
    KDADeferredCommitContext,
    KDARecoverSSMDeferred,
    allocate_records,
)
from vllm.models.kimi_k3.nvidia.specialized_decode.recoverssm.kernels import (  # noqa: E402
    kda_recoverssm_verify,
)

HEADS, Q, MAX_REQS = 12, 5, 64  # TP8 rank, DSpark k = 4, max-num-seqs 64


def _records(gen):
    records = allocate_records(MAX_REQS, HEADS, Q, D, torch.bfloat16, DEVICE)
    records.correction.normal_(generator=gen)
    records.key_gate.copy_(
        torch.randn(records.key_gate.shape, device=DEVICE, generator=gen)
    )
    return records


def _snapshot(records):
    return records.correction.clone(), records.key_gate.clone()


def _assert_records_untouched(records, saved):
    torch.testing.assert_close(records.correction, saved[0], atol=0, rtol=0)
    torch.testing.assert_close(records.key_gate, saved[1], atol=0, rtol=0)


def _capture_slots(requests, blocks):
    """The DCP capture context's state slots: block ids 0, 1, ... (row 0 null)."""
    return torch.arange(requests, dtype=torch.int32, device=DEVICE) % blocks


def _graphed(fn, reset=None):
    """The server's capture of one graph: an eager warmup, capture, replay
    (after ``reset``: restore what the warmup wrote, e.g. the conv windows)."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    if reset is not None:
        reset()
    graph.replay()
    torch.cuda.synchronize()


def test_records_scratch_row():
    records = allocate_records(4, 2, 3, 8, torch.bfloat16, DEVICE)
    for tensor in (records.correction, records.key_gate):
        base = tensor._base
        assert base is not None and base.shape[0] == 5
        assert tensor.shape[0] == 4
        assert tensor.data_ptr() == base[1].data_ptr()


@torch.inference_mode()
def test_stock_verify_capture_rows_without_record_slots():
    """The stock Triton RecoverSSM verify (the step's stock KDA module) at every
    server capture size 5..320: record index -1 with live state slots."""
    gen = torch.Generator(device=DEVICE).manual_seed(3)
    dim, blocks = HEADS * D, MAX_REQS
    state = torch.randn(blocks, HEADS, D, D, device=DEVICE, generator=gen) * 0.1
    records = _records(gen)
    rand = lambda *s: torch.randn(*s, device=DEVICE, generator=gen)  # noqa: E731
    params = (rand(HEADS) * 0.5, rand(dim) * 0.5)
    none = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=DEVICE)
    valid = torch.arange(MAX_REQS, dtype=torch.int32, device=DEVICE)
    for requests in range(1, MAX_REQS + 1):
        n = requests * Q
        q, k, v, g = (rand(1, n, HEADS, D).bfloat16() for _ in range(4))
        beta = rand(1, n, HEADS).bfloat16()
        starts = torch.arange(0, n + 1, Q, dtype=torch.int32, device=DEVICE)
        table = _capture_slots(requests, blocks)
        args = (q, k, v, g, beta, *params, LOWER_BOUND, state, records, starts)
        verify = partial(kda_recoverssm_verify, *args, table, Q)
        saved, state_saved = _snapshot(records), state.clone()
        outs = [torch.empty(1, n, HEADS, D, device=DEVICE, dtype=torch.bfloat16)]
        _graphed(partial(verify, out=outs[0], record_indices=none[:requests]))
        _assert_records_untouched(records, saved)
        outs.append(verify(record_indices=valid[:requests]))
        torch.testing.assert_close(outs[0], outs[1], atol=0, rtol=0)
        torch.testing.assert_close(state, state_saved, atol=0, rtol=0)


# Server capture sizes 5..320 at q = 5 (requests 1..64) that cover each policy of
# the deferred step at 12 heads: flush + regular core below FUSED_ITEMS (192 items),
# KdaFusedWS from there; the 768-item maximum.
CAPTURE_REQUESTS = (1, 2, 8, 15, 16, 17, 38, 63, 64)


@pytest.mark.parametrize("requests", CAPTURE_REQUESTS)
@torch.inference_mode()
def test_specialized_capture_rows_without_record_slots(requests):
    gen = torch.Generator(device=DEVICE).manual_seed(requests)
    dim, blocks = HEADS * D, MAX_REQS
    conv = torch.randn(blocks, 3 * dim, Q + 2, device=DEVICE, generator=gen)
    state = torch.randn(blocks, HEADS, D, D, device=DEVICE, generator=gen) * 0.1
    caches = (conv, state)
    records = _records(gen)
    layer = _layer(HEADS, requests, records, caches)
    kda = PreparedKda(layer, {(requests, Q): (_pitch(HEADS), 1)}, True)
    kda.prepare_caches(caches)
    assert kda.fused
    context = KDADeferredCommitContext.create(
        [layer], spec_query_len=Q, max_num_reqs=MAX_REQS
    )
    projected = _step_inputs(HEADS, requests, requests * Q)
    table = _capture_slots(requests, blocks)
    accepted = torch.ones(MAX_REQS, dtype=torch.int32, device=DEVICE)
    pending = torch.zeros(MAX_REQS, dtype=torch.int32, device=DEVICE)
    none = torch.full((MAX_REQS,), -1, dtype=torch.int32, device=DEVICE)
    valid = torch.arange(MAX_REQS, dtype=torch.int32, device=DEVICE)

    def step(rows, out):
        # A fresh metadata object per forward, as the builder makes one per step.
        deferred = KDARecoverSSMDeferred(
            rows, pending, None, context=context, state_indices=table
        )
        kda.run(
            projected, out, caches=caches, slots=table.view(-1, 1),
            accepted=accepted, q=Q, deferred=deferred,
        )  # fmt: skip

    saved = _snapshot(records)
    cache_saved = (conv.clone(), state.clone())
    captured = torch.empty(requests * Q, dim, device=DEVICE, dtype=torch.bfloat16)
    _graphed(lambda: step(none, captured), reset=lambda: conv.copy_(cache_saved[0]))
    _assert_records_untouched(records, saved)
    assert not pending.any()
    torch.testing.assert_close(state, cache_saved[1], atol=0, rtol=0)
    # The same step with record slots: the same outputs and conv windows (pending
    # 0: no replay).
    windows = conv.clone()
    conv.copy_(cache_saved[0])
    expected = torch.empty_like(captured)
    step(valid, expected)
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    torch.testing.assert_close(windows, conv, atol=0, rtol=0)


def test_capture_requests_cover_each_policy():
    paths = {prepared.replays(HEADS, requests, True) for requests in CAPTURE_REQUESTS}
    assert paths == {False, True}

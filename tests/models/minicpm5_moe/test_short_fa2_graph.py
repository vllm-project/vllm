# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace as NS
from weakref import ref

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.models.minicpm5_moe.xpu.model_state import _active_xpu_capture_stream
from vllm.models.minicpm5_moe.xpu.specialized_decode import short_fa2_graph as short
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def test_xpu_capture_uses_active_stream_after_profiling(monkeypatch):
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4)
    profiling, capture = object(), object()
    active = iter([profiling, capture])
    monkeypatch.setattr(torch.xpu, "current_stream", lambda: next(active))
    assert _active_xpu_capture_stream(None, descriptor) is profiling
    assert _active_xpu_capture_stream(None, descriptor) is capture


def test_xpu_microbatch_capture_keeps_its_designated_stream():
    stream = object()
    descriptor = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4, num_ubatches=2)
    assert _active_xpu_capture_stream(stream, descriptor) is stream


def batch(lengths):
    return NS(
        num_reqs=5,
        num_tokens=5,
        num_reqs_after_padding=8,
        num_tokens_after_padding=8,
        has_prefill=False,
        num_draft_tokens=0,
        is_prefilling_np=np.zeros(5, dtype=bool),
        prefill_len_np=np.full(5, 1024),
        num_scheduled_tokens=np.ones(5, dtype=np.int32),
        seq_lens_cpu_upper_bound=torch.tensor(lengths + [0, 0, 0]),
    )


def test_live_cpu_lengths_exclude_padding_and_capture():
    selection = short.ShortFa2Selection()
    candidate = batch([1024, 1025, 1030, 1150, 1152])
    selection.prepare(candidate, CUDAGraphMode.FULL, False)
    assert selection.max_kv == 1152
    candidate.seq_lens_cpu_upper_bound[2] = 32768
    selection.prepare(candidate, CUDAGraphMode.FULL, False)
    assert selection.max_kv == 32768
    selection.prepare(candidate, CUDAGraphMode.FULL, True)
    assert selection.max_kv is None
    candidate.prefill_len_np.fill(0)
    selection.prepare(candidate, CUDAGraphMode.FULL, False)
    assert selection.max_kv is None
    candidate.prefill_len_np.fill(1024)
    candidate.is_prefilling_np[0] = True
    selection.prepare(candidate, CUDAGraphMode.FULL, False)
    assert selection.max_kv is None
    candidate.is_prefilling_np[0] = False
    # A GPU tensor, if a future caller ever supplies one, must be ignored
    # before any scalar read. The fake has no tensor methods.
    candidate.seq_lens_cpu_upper_bound = NS(device=NS(type="xpu"))
    selection.prepare(candidate, CUDAGraphMode.FULL, False)
    assert selection.max_kv is None


def test_variant_capture_replay_and_profile_manager_lifetime(monkeypatch):
    rows8 = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=8, num_reqs=8, uniform_token_count=1
    )
    rows4 = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL, num_tokens=4, num_reqs=4, uniform_token_count=1
    )
    wrappers = [NS(force_one=False, calls=0) for _ in range(27)]
    calls = []
    primary_pool, private_pool = object(), object()
    input_buffers, kv_buffers = object(), object()

    class Manager:
        def __init__(self, config, device, mode, queries):
            self.vllm_config, self.device, self.cudagraph_mode = config, device, mode
            self._capture_descs = {CUDAGraphMode.FULL: [rows8, rows4]}
            self.pool = primary_pool
            self.capture_stream = NS(synchronize=lambda: None)
            stream = self.capture_stream
            self._capture_stream = lambda desc: stream

        def capture(self, inputs, kv):
            assert inputs is input_buffers and kv is kv_buffers
            choices = self._capture_descs[CUDAGraphMode.FULL]
            calls.append(("capture", self.pool, choices))
            for wrapper in wrappers:
                assert wrapper.force_one == (self.pool is private_pool)
                if wrapper.force_one:
                    wrapper.calls += 1

        def run_fullgraph(self, desc):
            calls.append(("replay", self.pool, desc))
            return self.pool

    monkeypatch.setattr(short, "ModelCudaGraphManager", Manager)
    monkeypatch.setattr(torch.xpu, "graph_pool_handle", lambda: private_pool)
    selection = short.ShortFa2Selection()
    parent = Manager(None, torch.device("cpu"), CUDAGraphMode.FULL_DECODE_ONLY, {1: 8})
    decode = NS(
        layers={i: NS(dense=False, pre_o_fa2=w) for i, w in enumerate(wrappers)}
    )
    short.attach_short_fa2_graph(parent, decode, selection)
    parent.capture(input_buffers, kv_buffers)
    assert calls == [
        ("capture", primary_pool, [rows8, rows4]),
        ("capture", private_pool, [rows8]),
    ]
    assert not any(w.force_one for w in wrappers)

    for length, desc, expected in [
        (1152, rows8, private_pool),
        (1153, rows8, primary_pool),
        (32768, rows8, primary_pool),
        (1152, rows4, primary_pool),
    ]:
        selection.prepare(
            batch([1024, 1025, 1030, 1100, length]), CUDAGraphMode.FULL, False
        )
        assert parent.run_fullgraph(desc) is expected
        assert selection.max_kv is None
    assert parent.run_fullgraph(rows8) is primary_pool
    assert parent.short_fa2_graphs.replays == {"auto": 3, "short": 1}
    assert parent.short_fa2_graphs.max_kv == {"auto": 32768, "short": 1152}

    parent_ref = ref(parent)
    child_ref = ref(parent.short_fa2_graphs.short)
    del parent
    assert parent_ref() is None and child_ref() is None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from weakref import ref

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.minicpm5_moe.xpu.model_state import MiniCPM5XpuModelState
from vllm.models.minicpm5_moe.xpu.specialized_decode import decode as runtime
from vllm.models.minicpm5_moe.xpu.specialized_decode.config import DecodeConfig
from vllm.models.minicpm5_moe.xpu.specialized_decode.draft import DraftMono
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.utils import clear_layer_kv_caches
from vllm.v1.worker.xpu_model_runner import XPUModelRunner


@pytest.mark.parametrize(
    "has_prefill,real_rows,captured_rows,query_len,spec_tokens,bucket,expected_mode",
    [
        (True, 1, 1, 1, 0, CUDAGraphMode.FULL, CUDAGraphMode.NONE),
        (True, 3, 4, 1, 0, CUDAGraphMode.FULL, CUDAGraphMode.NONE),
        (True, 16, 16, 8, 7, CUDAGraphMode.FULL, CUDAGraphMode.NONE),
        (True, 1, 1, None, 0, CUDAGraphMode.FULL, CUDAGraphMode.NONE),
        (True, 9, 16, None, 0, CUDAGraphMode.FULL, CUDAGraphMode.FULL),
        (True, 3, 4, None, 0, CUDAGraphMode.PIECEWISE, CUDAGraphMode.PIECEWISE),
        (False, 3, 4, 1, 0, CUDAGraphMode.FULL, CUDAGraphMode.FULL),
        (False, 16, 16, 8, 7, CUDAGraphMode.FULL, CUDAGraphMode.FULL),
    ],
)
def test_xpu_prefill_does_not_replay_native_decode_graph(
    monkeypatch,
    has_prefill,
    real_rows,
    captured_rows,
    query_len,
    spec_tokens,
    bucket,
    expected_mode,
):
    def gather(*args):
        return NS(has_prefill=has_prefill), query_len

    monkeypatch.setattr(GPUModelRunner, "gather_batch_req_state", gather)
    model_state = MiniCPM5XpuModelState.__new__(MiniCPM5XpuModelState)
    model_state.decode = NS(
        config=DecodeConfig(tp=1, speculative_tokens=spec_tokens),
        rows=frozenset(range(1, 9)),
    )
    runner = XPUModelRunner.__new__(XPUModelRunner)
    runner.model_state = model_state
    real_reqs = real_rows // (query_len or 1)
    graph = BatchExecutionDescriptor(
        cg_mode=bucket,
        num_tokens=captured_rows,
        num_reqs=captured_rows // (query_len or 1),
        uniform_token_count=query_len,
    )

    class Manager:
        def dispatch(self, *args, **kwargs):
            return graph

    manager = Manager()
    model_state._guard_mono_graph_dispatch(manager)
    scheduler_output = NS()
    _, uniform_tokens = runner.gather_batch_req_state(scheduler_output, False)
    desc, _ = dispatch_cg_and_sync_dp(
        manager, real_reqs, real_rows, uniform_tokens, 1, 0
    )
    assert desc.cg_mode == expected_mode
    if expected_mode == CUDAGraphMode.NONE:
        assert desc.num_tokens == real_rows
        assert desc.num_reqs == real_reqs
    else:
        assert desc is graph

    # A subsequent pure decode step can replay the same captured shape.
    runner.model_state.prepare_xpu_graph_batch(None)
    assert manager.dispatch(real_reqs, real_rows, uniform_tokens, 0) is graph

    # The wrapper must not pin the profiling manager after teardown.
    lifetime = ref(manager)
    dispatch = manager.dispatch
    del manager
    assert lifetime() is None
    del dispatch


def test_target_does_not_own_profiled_kv_cache(monkeypatch):
    attention = NS(
        layer_name="layer.0",
        kv_cache=torch.zeros((1, 1, 64, 256), dtype=torch.float16),
    )
    monkeypatch.setattr(runtime, "layer_weights", lambda layer: [torch.empty(1)])
    ops = NS(control_words=lambda: 1, validate=lambda *args: None)
    layer = runtime.MonoLayer(
        NS(mlp=NS(), self_attn=NS(attn=attention)),
        0,
        DecodeConfig(tp=1),
        None,
        None,
        ops,
    )
    lifetime = ref(attention.kv_cache)
    layer.bind()
    assert layer.cache is attention.kv_cache
    assert layer.key == layer.cache_key(attention.kv_cache)
    clear_layer_kv_caches([attention])
    assert lifetime() is None

    attention.kv_cache = torch.zeros((2, 1, 64, 256), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="cache moved"):
        layer.launch(None, None, None, None)
    layer.bind()
    assert layer.cache is attention.kv_cache
    assert layer.key == layer.cache_key(attention.kv_cache)


def test_native_draft_does_not_own_profiled_kv_cache():
    attention = NS(kv_cache=torch.zeros((1, 1, 64, 256), dtype=torch.float16))
    rotary = NS(
        _match_cos_sin_cache_dtype=lambda weight: weight.new_empty((64, 128)),
    )
    draft = DraftMono.__new__(DraftMono)
    draft.target = NS(ops=NS(draft_tensor_parallel_sizes=lambda: (2,)), draft_slots=1)
    draft.tp = 2
    draft.max_requests = 1
    draft.max_context = 64
    draft.layers = (NS(self_attn=NS(rotary_emb=rotary)),)
    draft.attentions = (attention,)
    draft.weights = ((torch.empty(1, dtype=torch.float16),),)
    draft.controls = (torch.zeros(1, dtype=torch.int32),)
    draft._all_ranks_support = bool

    lifetime = ref(attention.kv_cache)
    draft.bind_caches()
    assert draft.caches[0]() is attention.kv_cache
    clear_layer_kv_caches([attention])
    assert lifetime() is None

    attention.kv_cache = torch.zeros((2, 1, 64, 256), dtype=torch.float16)
    draft.bind_caches()
    assert draft.caches[0]() is attention.kv_cache

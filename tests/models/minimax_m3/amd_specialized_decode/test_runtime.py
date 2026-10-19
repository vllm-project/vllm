# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.models.minimax_m3.amd.specialized_decode.runtime import decode as runtime
from vllm.models.minimax_m3.amd.specialized_decode.runtime.caches import CacheBinding
from vllm.models.minimax_m3.amd.specialized_decode.runtime.spec import DecodeConfig
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseMetadata,
)


def _metadata() -> MiniMaxM3SparseMetadata:
    return MiniMaxM3SparseMetadata(
        seq_lens=torch.ones(1, dtype=torch.int32),
        max_seq_len=1,
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        num_actual_tokens=1,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=MiniMaxM3SparseDecodeMetadata(
            seq_lens=torch.ones(1, dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            decode_query_len=1,
        ),
    )


class _Executor:
    def __init__(self):
        self.binds: list[tuple[int, CacheBinding]] = []
        self.bind_failure: Exception | None = None
        self.layer_calls: list[int] = []

    def layer(self, layer_id: int):
        self.layer_calls.append(layer_id)
        return lambda _positions, hidden, residual: (hidden, residual)

    def bind(self, layer_id: int, binding: CacheBinding):
        if self.bind_failure is not None:
            raise self.bind_failure
        self.binds.append((layer_id, binding))

    def health(self):
        return None

    def after_step(self):
        return None

    def abort(self):
        return None

    def close(self):
        return None


def _binding(*, replacement: torch.Tensor | None = None) -> CacheBinding:
    tensor = replacement if replacement is not None else torch.empty(1)
    return CacheBinding(
        main=tensor,
        key_pages=torch.empty(1),
        value_pages=torch.empty(1),
        k_scale=torch.empty(1),
        v_scale=torch.empty(1),
        index_key=torch.empty(1),
        sparse_table=torch.empty(1),
        sparse_context=torch.empty(1),
    )


@pytest.fixture
def decode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(runtime, "checked_weights", lambda model, config: {})
    layers = [
        SimpleNamespace(layer_id=i, self_attn=SimpleNamespace(layer_name=f"layer.{i}"))
        for i in (3, 4)
    ]
    model = SimpleNamespace(layers=layers)
    executor = _Executor()
    instance = runtime.SpecializedDecode(
        model, DecodeConfig(last_layer=4), executor=executor
    )
    monkeypatch.setattr(
        runtime, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    return instance, executor


def test_plan_is_once_per_forward_and_returns_immutable_layer_map(
    decode, monkeypatch: pytest.MonkeyPatch
):
    instance, executor = decode
    context = SimpleNamespace(
        attn_metadata={"layer.3": _metadata()},
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    monkeypatch.setattr(runtime, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context)

    step = instance.plan(torch.zeros(1, dtype=torch.int64), None)
    assert step is not None and step.launched
    assert executor.layer_calls == [3, 4]
    assert step.layer(3) is not None
    assert step.layer(4) is not None
    assert instance.mono_steps == 1
    assert instance.stock_steps == 0


def test_executor_is_built_after_weight_checks(monkeypatch: pytest.MonkeyPatch):
    order = []
    monkeypatch.setattr(
        runtime, "checked_weights", lambda model, config: order.append("weights")
    )
    model = SimpleNamespace(layers=[])

    def factory(decode):
        order.append("executor")
        return _Executor()

    instance = runtime.SpecializedDecode(model, DecodeConfig(), factory)
    assert order == ["weights", "executor"]
    assert isinstance(instance.executor, _Executor)


def test_cache_bind_is_identity_stable_and_rebinds_replacements(
    decode, monkeypatch: pytest.MonkeyPatch
):
    instance, executor = decode
    first = _binding()
    bindings = {3: first}
    monkeypatch.setattr(runtime, "checked_cache_bindings", lambda layers: bindings)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    instance.bind_caches()
    instance.bind_caches()
    assert [layer_id for layer_id, _ in executor.binds] == [3]

    bindings = {3: _binding(replacement=torch.empty(1))}
    monkeypatch.setattr(runtime, "checked_cache_bindings", lambda layers: bindings)
    instance.bind_caches()
    assert [layer_id for layer_id, _ in executor.binds] == [3, 3]


def test_failed_cache_bind_does_not_publish_identity(
    decode, monkeypatch: pytest.MonkeyPatch
):
    instance, executor = decode
    binding = _binding()
    monkeypatch.setattr(runtime, "checked_cache_bindings", lambda layers: {3: binding})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    executor.bind_failure = ValueError("injected")
    with pytest.raises(RuntimeError, match="rank-uniform.*cache bind") as error:
        instance.bind_caches()
    assert isinstance(error.value.__cause__, ValueError)
    assert instance._cache_identities == {}

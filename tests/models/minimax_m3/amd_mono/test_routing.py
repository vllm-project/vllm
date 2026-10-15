# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.models.minimax_m3.amd.mono.routing import decide_step, tensor_identity
from vllm.models.minimax_m3.amd.mono.spec import MONO_SPEC
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseMetadata,
)


def _metadata(rows: int, *, decode_tokens: int | None = None, prefills: int = 0):
    decode_tokens = rows if decode_tokens is None else decode_tokens
    return MiniMaxM3SparseMetadata(
        seq_lens=torch.ones(rows, dtype=torch.int32),
        max_seq_len=1,
        slot_mapping=torch.arange(rows),
        num_actual_tokens=decode_tokens,
        num_decodes=decode_tokens,
        num_decode_tokens=decode_tokens,
        num_prefills=prefills,
        num_prefill_tokens=prefills,
        decode=MiniMaxM3SparseDecodeMetadata(
            seq_lens=torch.ones(decode_tokens, dtype=torch.int32),
            block_table=torch.zeros(decode_tokens, 1, dtype=torch.int32),
            decode_query_len=1,
        ),
    )


@pytest.mark.parametrize("width", MONO_SPEC.widths)
@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_decode_widths_route_uniformly(width: int, mode: CUDAGraphMode):
    left = decide_step(MONO_SPEC, width, mode, _metadata(width))
    right = decide_step(MONO_SPEC, width, mode, _metadata(width))
    assert left == right
    assert left.use_mono


@pytest.mark.parametrize(
    ("rows", "mode", "metadata"),
    [
        (MONO_SPEC.max_rows + 1, CUDAGraphMode.NONE, _metadata(1)),
        (1, CUDAGraphMode.PIECEWISE, _metadata(1)),
        (1, CUDAGraphMode.NONE, None),
        (1, CUDAGraphMode.NONE, _metadata(1, prefills=1)),
        (1, CUDAGraphMode.NONE, _metadata(1, decode_tokens=2)),
    ],
)
def test_unsupported_steps_fall_back(rows, mode, metadata):
    assert not decide_step(MONO_SPEC, rows, mode, metadata).use_mono


def test_padding_slots_are_eligible():
    metadata = _metadata(4, decode_tokens=2)
    metadata.slot_mapping[2:] = -1
    assert decide_step(MONO_SPEC, 4, CUDAGraphMode.FULL, metadata).use_mono


def test_speculative_verification_falls_back():
    metadata = _metadata(4)
    metadata.decode.decode_query_len = 4
    assert not decide_step(MONO_SPEC, 4, CUDAGraphMode.FULL, metadata).use_mono


def test_cache_identity_detects_replacement():
    first = torch.empty(8)
    replacement = torch.empty(8)
    assert tensor_identity(first) != tensor_identity(replacement)
    assert tensor_identity(first) == tensor_identity(first)


def _config(**overrides):
    values = dict(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_eplb=False,
    )
    values.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**values),
        cache_config=SimpleNamespace(
            cache_dtype="fp8", block_size=128, enable_prefix_caching=False
        ),
        attention_config=SimpleNamespace(
            resolve_indexer_kv_dtype=lambda _default: "fp8"
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 32]),
        lora_config=None,
        kv_transfer_config=SimpleNamespace(kv_connector=None),
        use_v2_model_runner=True,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(use_index_cache=False)
        ),
    )


def test_static_refusals_report_all_reasons():
    config = _config(
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=True,
    )
    config.cache_config.enable_prefix_caching = True
    reasons = MONO_SPEC.refusal_reasons(config)
    assert len(reasons) == 4
    assert any("tensor parallel" in reason for reason in reasons)
    assert any("prefix caching" in reason for reason in reasons)


def test_graph_widths_extend_instead_of_truncate():
    assert MONO_SPEC.extend_final_graph_widths([1, 32]) == [
        1,
        2,
        4,
        8,
        16,
        32,
    ]


def test_attach_does_not_change_parameter_tree(monkeypatch):
    from vllm.models.minimax_m3.amd.mono import layer as mono_layer

    class FakeSparseLayer(nn.Module):
        def __init__(self, layer_id: int):
            super().__init__()
            self.layer_id = layer_id
            self.is_moe_layer = layer_id >= MONO_SPEC.first_layer
            self.weight = nn.Parameter(torch.randn(2))
            self.self_attn = SimpleNamespace(indexer=object())
            self.mono = None

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(FakeSparseLayer(i) for i in range(6))

    monkeypatch.setattr(mono_layer, "_runtime_refusals", lambda _spec: [])
    model = FakeModel()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    controller = mono_layer.MiniMaxM3MonoLayer(model, _config())
    controller.attach()
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert all(model.layers[i].mono is None for i in range(3))
    assert all(model.layers[i].mono is controller for i in range(3, 6))

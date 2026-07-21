# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.warmup import fa4_cutedsl_warmup as warmup_module


class _CustomMLAAttention(MLAAttention):
    pass


def _make_worker(attention_layer: object):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(use_mla=True),
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0.self_attn": attention_layer}
        ),
    )
    runner = SimpleNamespace(is_pooling_model=False, vllm_config=vllm_config)
    return SimpleNamespace(model_runner=runner)


def test_fa4_warmup_skips_custom_mla_attention(monkeypatch):
    get_backend = Mock()
    monkeypatch.setattr(warmup_module, "get_mla_prefill_backend", get_backend)

    custom_mla_attention = object.__new__(_CustomMLAAttention)
    warmup_module.fa4_cutedsl_warmup(_make_worker(custom_mla_attention))

    get_backend.assert_not_called()


def test_fa4_warmup_selects_backend_for_generic_mla_attention(monkeypatch):
    backend = Mock()
    backend.get_name.return_value = "TRITON_MLA"
    get_backend = Mock(return_value=backend)
    monkeypatch.setattr(warmup_module, "get_mla_prefill_backend", get_backend)

    generic_mla_attention = object.__new__(MLAAttention)
    worker = _make_worker(generic_mla_attention)
    warmup_module.fa4_cutedsl_warmup(worker)

    get_backend.assert_called_once_with(worker.model_runner.vllm_config)

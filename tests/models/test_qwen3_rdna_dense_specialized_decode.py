# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_amd.specialized_decode.config import (
    DecodeConfig,
    specialized_decode_config,
)
from vllm.model_executor.models.qwen3_amd.specialized_decode.planning import (
    decide_step,
)


def _make_vllm_config(
    *,
    monokernel: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    quant_config=None,
    cache_dtype: str = "auto",
    speculative_config=None,
    lora_config=None,
    is_sleep: bool = False,
    qk_norm: bool = True,
    hidden_act: str = "silu",
    tp: int = 1,
):
    text_config = SimpleNamespace(qk_norm=qk_norm, hidden_act=hidden_act)
    hf_config = SimpleNamespace(get_text_config=lambda: text_config)
    model_config = SimpleNamespace(
        dtype=dtype,
        enable_sleep_mode=is_sleep,
        hf_config=hf_config,
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=tp,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        use_ubatching=False,
        enable_dbo=False,
        enable_expert_parallel=False,
        enable_eplb=False,
    )
    cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    kernel_config = SimpleNamespace(monokernel=monokernel)
    return SimpleNamespace(
        kernel_config=kernel_config,
        model_config=model_config,
        parallel_config=parallel,
        cache_config=cache_config,
        quant_config=quant_config,
        speculative_config=speculative_config,
        lora_config=lora_config,
    )


def test_specialized_decode_config_disabled_when_monokernel_off():
    cfg = _make_vllm_config(monokernel=False)
    assert specialized_decode_config(cfg) is None


def test_specialized_decode_config_accepts_qualified_rdna(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna", lambda: True)
    cfg = _make_vllm_config()
    assert specialized_decode_config(cfg) == DecodeConfig()


def test_specialized_decode_config_rejects_non_rdna(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna", lambda: False)
    cfg = _make_vllm_config()
    with pytest.raises(ValueError, match="requires RDNA"):
        specialized_decode_config(cfg)


def test_specialized_decode_config_rejects_quantized_weights(monkeypatch):
    monkeypatch.setattr("vllm.platforms.current_platform.is_rocm", lambda: True)
    monkeypatch.setattr("vllm.platforms.rocm.on_rdna", lambda: True)
    cfg = _make_vllm_config(quant_config=object())
    with pytest.raises(ValueError, match="does not support quantized weights"):
        specialized_decode_config(cfg)


def test_decide_step_accepts_one_token_decode():
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=0,
        block_table=torch.zeros((1, 8), dtype=torch.int32),
    )
    decision = decide_step(
        input_ids=torch.zeros((1,), dtype=torch.int32),
        intermediate_tensors=None,
        inputs_embeds=None,
        metadata=metadata,
        layer_name=None,
    )
    assert decision.use_specialized
    assert decision.block_table is metadata.block_table


def test_decide_step_rejects_bad_block_table():
    metadata = SimpleNamespace(
        num_decodes=1,
        num_prefills=0,
        block_table=torch.zeros((1, 8), dtype=torch.int64),
    )
    decision = decide_step(
        input_ids=torch.zeros((1,), dtype=torch.int32),
        intermediate_tensors=None,
        inputs_embeds=None,
        metadata=metadata,
        layer_name=None,
    )
    assert not decision.use_specialized
    assert decision.reason == "block_table_invalid"

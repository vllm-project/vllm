# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from types import SimpleNamespace

import pytest
from transformers.models.qwen3_5 import Qwen3_5Config

from vllm.model_executor.models.config import Qwen3_5ForSlotDecisionConfig
from vllm.model_executor.models.qwen3_5_slot import Qwen3_5SlotDecisionModelState
from vllm.sampling_params import PREFIX_LM_LEN, SamplingParams
from vllm.transformers_utils.model_arch_config_convertor import (
    Qwen3_5ModelArchConfigConvertor,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = pytest.mark.cpu_test

OVERRIDES = {
    "attention": {
        "full_attention": "block_bidirectional",
        "linear_attention": "causal",
    },
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}


def make_config(overrides=OVERRIDES) -> Qwen3_5Config:
    config = Qwen3_5Config()
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    config.modeling_overrides = overrides
    return config


def test_detected_from_modeling_overrides():
    config = make_config()
    arch = Qwen3_5ModelArchConfigConvertor(config, config.text_config).convert()
    assert arch.architectures == ["Qwen3_5ForSlotDecision"]


@pytest.mark.parametrize("head_dtype,expected", [(None, "float32"), ("model", "model")])
def test_model_config_defaults_to_an_fp32_head(head_dtype, expected):
    config = make_config()
    config.head_dtype = head_dtype
    Qwen3_5ForSlotDecisionConfig.verify_and_update_model_config(
        SimpleNamespace(hf_config=config)
    )
    assert config.head_dtype == expected


@pytest.mark.parametrize(
    "change",
    [
        {"attention": {"full_attention": "causal"}},
        {"temperature": 2.0},
        {"n_slots": 300},
        {"n_slots": True},
    ],
)
def test_model_config_rejects_unsupported_overrides(change):
    model_config = SimpleNamespace(hf_config=make_config({**OVERRIDES, **change}))
    with pytest.raises(ValueError, match="modeling_overrides"):
        Qwen3_5ForSlotDecisionConfig.verify_and_update_model_config(model_config)


def make_vllm_config(**changes):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            architecture="Qwen3_5ForSlotDecision",
            max_model_len=32768,
            hf_text_config=SimpleNamespace(mamba_ssm_dtype="float32"),
        ),
        cache_config=SimpleNamespace(
            enable_prefix_caching=True,
            mamba_ssm_cache_dtype="auto",
            prefix_match_unit=None,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=2048,
            max_num_batched_tokens=32768,
        ),
        attention_config=SimpleNamespace(backend=None),
        kv_transfer_config=None,
        speculative_config=None,
    )
    for path, value in changes.items():
        *parents, name = path.split(".")
        setattr(functools.reduce(getattr, parents, config), name, value)
    return config


def test_config_prefills_each_prompt_in_one_step():
    config = make_vllm_config(**{"scheduler_config.enable_chunked_prefill": False})
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
    assert config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    # Prefix caching stays on, with the chunked prefill its Mamba cache needs.
    assert config.cache_config.enable_prefix_caching
    assert config.scheduler_config.enable_chunked_prefill
    assert config.scheduler_config.prefill_whole_prompt
    assert config.scheduler_config.long_prefill_token_threshold == 0
    # Qwen3.5's own hook still caches the GatedDeltaNet state in fp32.
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


@pytest.mark.parametrize(
    "change,match",
    [
        (
            {"attention_config.backend": AttentionBackendEnum.FLEX_ATTENTION},
            "TRITON_ATTN",
        ),
        ({"scheduler_config.max_num_batched_tokens": 8192}, "max-num-batched"),
        ({"cache_config.prefix_match_unit": 32}, "prefix-match-unit"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV transfer"),
    ],
)
def test_config_rejects_unsupported_features(change, match):
    with pytest.raises(ValueError, match=match):
        Qwen3_5ForSlotDecisionConfig.verify_and_update_config(
            make_vllm_config(**change)
        )


def test_model_state_ranges(monkeypatch):
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(MambaHybridModelState, "remove_request", lambda *args: None)
    state = object.__new__(Qwen3_5SlotDecisionModelState)
    state.blocks = {}

    def add(req_id: str, prefix_len, length: int = 10) -> None:
        extra_args = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
        state.add_request(
            0,
            SimpleNamespace(
                req_id=req_id,
                sampling_params=SamplingParams(extra_args=extra_args),
                prompt_token_ids=list(range(length)),
            ),
        )

    add("a", 4)
    add("causal", None)
    for invalid in (0, 11, 4.0, True):
        add(f"invalid {invalid}", invalid)
    add("b", 1)
    # A prompt that is all prefix, as the strategy's warm-up sends.
    add("prefix", 10)
    batch = SimpleNamespace(req_ids=["causal", "b", "invalid 0", "a", "prefix"])
    assert state.get_mm_prefix_ranges(batch) == {
        1: [(0, 0), (1, 9)],
        3: [(0, 3), (4, 9)],
        4: [(0, 9)],
    }
    for req_id in ("a", "b", "prefix"):
        state.remove_request(req_id)
    assert state.get_mm_prefix_ranges(batch) is None

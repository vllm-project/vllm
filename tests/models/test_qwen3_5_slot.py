# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from types import SimpleNamespace

import numpy as np
import pytest
from transformers.models.qwen3_5 import Qwen3_5Config

from vllm.model_executor.models.config import Qwen3_5ForSlotDecisionConfig
from vllm.model_executor.models.qwen3_5_slot import (
    PREFIX_LM_LEN,
    Qwen3_5SlotDecisionModelState,
)
from vllm.sampling_params import SamplingParams
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
            kv_offloading_size=None,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            long_prefill_token_threshold=2048,
            max_num_batched_tokens=33024,
            user_specified_max_num_batched_tokens=True,
            max_num_scheduled_tokens=None,
            max_num_seqs=256,
        ),
        attention_config=SimpleNamespace(backend=None, backend_per_kind={}),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
    )
    for path, value in changes.items():
        *parents, name = path.split(".")
        setattr(functools.reduce(getattr, parents, config), name, value)
    return config


def test_config_prefills_each_prompt_in_one_step():
    config = make_vllm_config()
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
    assert config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    assert not config.cache_config.enable_prefix_caching
    assert not config.scheduler_config.enable_chunked_prefill
    assert config.scheduler_config.long_prefill_token_threshold == 0
    # Qwen3.5's own hook still caches the GatedDeltaNet state in fp32.
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


@pytest.mark.parametrize(
    "budget,user_specified,expected,warns",
    [
        # A default budget is raised to --max-model-len plus --max-num-seqs.
        (8192, False, 33024, False),
        (33023, False, 33024, False),
        (40000, False, 40000, False),
        # A user's budget is kept, with a warning when it leaves no headroom.
        (33024, True, 33024, False),
        (33023, True, 33023, True),
        (40000, True, 40000, False),
    ],
)
def test_config_defaults_to_headroom_for_running_requests(
    budget, user_specified, expected, warns, monkeypatch
):
    warnings = []
    monkeypatch.setattr(
        "vllm.model_executor.models.config.logger.warning",
        lambda *args: warnings.append(args),
    )
    config = make_vllm_config(
        **{
            "scheduler_config.max_num_batched_tokens": budget,
            "scheduler_config.user_specified_max_num_batched_tokens": user_specified,
        }
    )
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
    assert config.scheduler_config.max_num_batched_tokens == expected
    assert config.scheduler_config.max_num_scheduled_tokens is None
    assert bool(warnings) == warns


def test_config_keeps_a_user_scheduled_budget(monkeypatch):
    warnings = []
    monkeypatch.setattr(
        "vllm.model_executor.models.config.logger.warning",
        lambda *args: warnings.append(args),
    )
    config = make_vllm_config(
        **{
            "scheduler_config.max_num_batched_tokens": 8192,
            "scheduler_config.user_specified_max_num_batched_tokens": False,
            "scheduler_config.max_num_scheduled_tokens": 32768,
        }
    )
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
    assert config.scheduler_config.max_num_batched_tokens == 33024
    assert config.scheduler_config.max_num_scheduled_tokens == 32768
    assert warnings


def test_config_accepts_triton_and_a_full_scheduled_budget():
    config = make_vllm_config(
        **{
            "attention_config.backend_per_kind": {
                "full_attention": AttentionBackendEnum.TRITON_ATTN,
                "sliding_window": AttentionBackendEnum.FLASH_ATTN,
            },
            "scheduler_config.max_num_scheduled_tokens": 32768,
        }
    )
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)


@pytest.mark.parametrize(
    "change,match",
    [
        (
            {"attention_config.backend": AttentionBackendEnum.FLEX_ATTENTION},
            "TRITON_ATTN",
        ),
        (
            {
                "attention_config.backend_per_kind": {
                    "full_attention": AttentionBackendEnum.FLASH_ATTN
                }
            },
            "TRITON_ATTN attention backend for full attention, not FLASH_ATTN",
        ),
        ({"scheduler_config.max_num_batched_tokens": 8192}, "max-num-batched"),
        ({"scheduler_config.max_num_scheduled_tokens": 8192}, "max-num-scheduled"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV transfer"),
        ({"cache_config.kv_offloading_size": 4.0}, "kv-offloading-size"),
        ({"parallel_config.prefill_context_parallel_size": 2}, "context parallel"),
        ({"parallel_config.decode_context_parallel_size": 2}, "context parallel"),
        ({"parallel_config.use_ubatching": True}, "micro-batching"),
    ],
)
def test_config_rejects_unsupported_features(change, match):
    with pytest.raises(ValueError, match=match):
        Qwen3_5ForSlotDecisionConfig.verify_and_update_config(
            make_vllm_config(**change)
        )


def make_model_state(monkeypatch):
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(MambaHybridModelState, "remove_request", lambda *args: None)
    state = object.__new__(Qwen3_5SlotDecisionModelState)
    state.blocks = {}

    def add(req_id: str, prefix_len, length: int = 10, mm_features=()) -> None:
        extra_args = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
        state.add_request(
            0,
            SimpleNamespace(
                req_id=req_id,
                sampling_params=SamplingParams(extra_args=extra_args),
                prompt_token_ids=list(range(length)),
                mm_features=list(mm_features),
            ),
        )

    return state, add


def make_batch(req_ids, computed=None, scheduled=None):
    """A batch of decode rows, or of prefill rows with these token counts."""
    n = len(req_ids)
    computed = np.array(computed if computed is not None else [10] * n)
    return SimpleNamespace(
        req_ids=req_ids,
        num_computed_prefill_tokens_np=computed,
        num_scheduled_tokens=np.array(scheduled if scheduled is not None else [1] * n),
        is_prefilling_np=computed < 10,
    )


def test_model_state_ranges(monkeypatch):
    state, add = make_model_state(monkeypatch)
    add("a", 4)
    add("causal", None)
    for invalid in (0, 11, 4.0, True):
        add(f"invalid {invalid}", invalid)
    add("b", 1)
    # A prompt that is all prefix, as the strategy's warm-up sends.
    add("prefix", 10)
    # Encoder budgets can end a step inside the prompt.
    add("image", 4, mm_features=[object()])
    batch = make_batch(["causal", "b", "invalid 0", "a", "prefix", "image"])
    assert state.get_mm_prefix_ranges(batch) == {
        1: [(0, 0), (1, 9)],
        3: [(0, 3), (4, 9)],
        4: [(0, 9)],
    }
    for req_id in ("a", "b", "prefix"):
        state.remove_request(req_id)
    assert state.get_mm_prefix_ranges(batch) is None


@pytest.mark.parametrize(
    "computed,scheduled,ok",
    [
        (0, 10, True),  # the whole prompt
        (0, 4, True),  # the prefix
        (4, 6, True),  # the question after a hit at the prefix
        (0, 3, False),  # inside the prefix
        (0, 6, False),  # inside the question
        (4, 2, False),
    ],
)
def test_model_state_refuses_a_step_inside_a_block(
    monkeypatch, computed, scheduled, ok
):
    state, add = make_model_state(monkeypatch)
    add("a", 4)
    batch = make_batch(["a"], [computed], [scheduled])
    if ok:
        assert state.get_mm_prefix_ranges(batch) == {0: [(0, 3), (4, 9)]}
    else:
        with pytest.raises(RuntimeError, match="inside one of its blocks"):
            state.get_mm_prefix_ranges(batch)

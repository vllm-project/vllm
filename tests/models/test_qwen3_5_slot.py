# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from transformers.models.qwen3_5 import Qwen3_5Config

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.models.config import Qwen3_5ForSlotDecisionConfig
from vllm.model_executor.models.qwen3_5_slot import Qwen3_5SlotDecisionModelState
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.model_arch_config_convertor import (
    Qwen3_5ModelArchConfigConvertor,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.prefix_lm_scheduler import PREFIX_LM_LEN
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = pytest.mark.cpu_test

PREFIX_LM = "vllm.v1.core.sched.prefix_lm_scheduler.PrefixLM"
TRITON, FLASH = AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLASH_ATTN


@pytest.fixture(autouse=True)
def gdn_prefill_backend(monkeypatch):
    """The hook sees the GDN prefill backend that make_vllm_config names."""
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "_resolve_gdn_prefill_backend",
        lambda vllm_config: ("auto", vllm_config.gdn_prefill_backend),
    )


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
    arch = ["Qwen3_5ForConditionalGeneration"]
    return Qwen3_5Config(architectures=arch, modeling_overrides=overrides)


@pytest.mark.parametrize("head_dtype,expected", [(None, "float32"), ("model", "model")])
def test_detected_with_an_fp32_head_by_default(head_dtype, expected):
    config = make_config()
    arch = Qwen3_5ModelArchConfigConvertor(config, config.text_config).convert()
    assert arch.architectures == ["Qwen3_5ForSlotDecision"]
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
            kv_offloading_size=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=33024,
            user_specified_max_num_batched_tokens=True,
            max_num_scheduled_tokens=None,
            max_num_seqs=256,
            scheduler_cls=None,
        ),
        attention_config=SimpleNamespace(backend=None, backend_per_kind={}),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        kv_transfer_config=None,
        speculative_config=None,
        gdn_prefill_backend="flashinfer",
    )
    for path, value in changes.items():
        *parents, name = path.split(".")
        setattr(functools.reduce(getattr, parents, config), name, value)
    return config


@pytest.mark.parametrize(
    "change,prefix_caching",
    [
        ({}, True),
        ({"cache_config.enable_prefix_caching": False}, False),
        ({"gdn_prefill_backend": "triton"}, False),
        (
            {
                "attention_config.backend_per_kind": {
                    "full_attention": TRITON,
                    "sliding_window": FLASH,
                },
                # As the hook and VllmConfig set it, for a second run of the hook.
                "scheduler_config.scheduler_cls": f"{PREFIX_LM}AsyncScheduler",
            },
            True,
        ),
    ],
)
def test_config_prefills_each_prompt_in_one_step(change, prefix_caching):
    config = make_vllm_config(
        **{"scheduler_config.enable_chunked_prefill": not prefix_caching}, **change
    )
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
    scheduler_config = config.scheduler_config
    assert config.attention_config.backend == TRITON
    assert config.cache_config.enable_prefix_caching == prefix_caching
    assert scheduler_config.enable_chunked_prefill == prefix_caching
    assert scheduler_config.scheduler_cls == (
        f"{PREFIX_LM}Scheduler" if prefix_caching else None
    )
    assert scheduler_config.long_prefill_token_threshold == 0
    assert config.cache_config.mamba_ssm_cache_dtype == "float32"


@pytest.mark.parametrize(
    "budget,user_specified,scheduled,expected,warns",
    [
        # A default budget is raised to --max-model-len plus --max-num-seqs.
        (8192, False, None, 33024, False),
        (40000, False, None, 40000, False),
        # A user's budget is kept, with a warning when it leaves no headroom.
        (33023, True, None, 33023, True),
        (8192, False, 32768, 33024, True),
    ],
)
def test_config_defaults_to_headroom_for_running_requests(
    budget, user_specified, scheduled, expected, warns
):
    config = make_vllm_config()
    scheduler_config = config.scheduler_config
    scheduler_config.max_num_batched_tokens = budget
    scheduler_config.user_specified_max_num_batched_tokens = user_specified
    scheduler_config.max_num_scheduled_tokens = scheduled
    with patch("vllm.model_executor.models.config.logger.warning") as warning:
        Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
    assert scheduler_config.max_num_batched_tokens == expected
    assert warning.called == warns


@pytest.mark.parametrize(
    "change,match",
    [
        ({"attention_config.backend": FLASH}, "TRITON_ATTN"),
        (
            {"attention_config.backend_per_kind": {"full_attention": FLASH}},
            "TRITON_ATTN attention backend for full attention, not FLASH_ATTN",
        ),
        ({"scheduler_config.max_num_batched_tokens": 8192}, "max-num-batched"),
        ({"scheduler_config.max_num_scheduled_tokens": 8192}, "max-num-scheduled"),
        ({"cache_config.prefix_match_unit": 32}, "prefix-match-unit"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": object()}, "KV transfer"),
        ({"cache_config.kv_offloading_size": 4.0}, "kv-offloading-size"),
        ({"parallel_config.prefill_context_parallel_size": 2}, "context parallel"),
        ({"parallel_config.decode_context_parallel_size": 2}, "context parallel"),
        ({"parallel_config.use_ubatching": True}, "micro-batching"),
        ({"scheduler_config.scheduler_cls": "my.Scheduler"}, "scheduler-cls"),
    ],
)
def test_config_rejects_unsupported_features(change, match):
    config = make_vllm_config(**change)
    with pytest.raises(ValueError, match=match):
        Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)


def make_model_state(monkeypatch):
    monkeypatch.setattr(MambaHybridModelState, "add_request", lambda *args: None)
    monkeypatch.setattr(MambaHybridModelState, "remove_request", lambda *args: None)
    state = object.__new__(Qwen3_5SlotDecisionModelState)
    state.blocks = {}

    def add(req_id, prefix_len, mm_features=()):
        extra_args = None if prefix_len is None else {PREFIX_LM_LEN: prefix_len}
        state.add_request(
            0,
            SimpleNamespace(
                req_id=req_id,
                sampling_params=SamplingParams(extra_args=extra_args),
                prompt_token_ids=list(range(10)),
                mm_features=mm_features,
            ),
        )

    return state, add


def make_batch(req_ids, computed, scheduled):
    """A batch of 10-token prompts; a row with 10 computed tokens decodes."""
    computed = np.array(computed)
    return SimpleNamespace(
        req_ids=req_ids,
        num_computed_prefill_tokens_np=computed,
        num_scheduled_tokens=np.array(scheduled),
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
    add("image", 4, mm_features=[object()])
    # b prefills its prefix, a its question after a hit at its prefix.
    batch = make_batch(
        ["causal", "b", "invalid 0", "a", "prefix", "image"],
        [10, 0, 10, 4, 10, 10],
        [1, 1, 1, 6, 1, 1],
    )
    assert state.get_mm_prefix_ranges(batch) == {
        1: [(0, 0), (1, 9)],
        3: [(0, 3), (4, 9)],
        4: [(0, 9)],
    }
    for req_id in ("a", "b", "prefix"):
        state.remove_request(req_id)
    assert state.get_mm_prefix_ranges(batch) is None


@pytest.mark.parametrize("scheduled", [3, 6])  # Ends inside the prefix, question.
def test_model_state_refuses_a_step_inside_a_block(monkeypatch, scheduled):
    state, add = make_model_state(monkeypatch)
    add("a", 4)
    with pytest.raises(RuntimeError, match="inside one of its blocks"):
        state.get_mm_prefix_ranges(make_batch(["a"], [0], [scheduled]))

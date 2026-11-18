# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.qwen3_5 import Qwen3_5Config

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.models.qwen3_5_slot import Qwen3_5SlotDecisionModelState
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.prefix_lm_scheduler import PREFIX_LM_LEN
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

pytestmark = pytest.mark.cpu_test

PREFIX_LM = "vllm.v1.core.sched.prefix_lm_scheduler.PrefixLM"
TRITON, FLASH = AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLASH_ATTN

OVERRIDES = {
    "attention": {
        "full_attention": "block_bidirectional",
        "linear_attention": "causal",
    },
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}


@pytest.fixture(autouse=True)
def gdn_prefill_backend(monkeypatch):
    """Report FlashInfer's GDN prefill, which this CPU host lacks, unless a test
    names another backend."""
    backend = {"name": "flashinfer"}
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "_resolve_gdn_prefill_backend",
        lambda vllm_config: ("auto", backend["name"]),
    )
    return backend


def make_vllm_config(
    tmp_path, overrides=OVERRIDES, head_dtype=None, **engine_args
) -> VllmConfig:
    """The engine config of a checkpoint with only a config.json."""
    Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        modeling_overrides=overrides,
        head_dtype=head_dtype,
        # As in the checkpoints, far above the trained length.
        text_config={"max_position_embeddings": 262144},
    ).save_pretrained(tmp_path)
    return EngineArgs(
        model=str(tmp_path), skip_tokenizer_init=True, **engine_args
    ).create_engine_config()


@pytest.mark.parametrize("head_dtype,fp32", [(None, True), ("model", False)])
def test_detected_with_an_fp32_head_by_default(tmp_path, head_dtype, fp32):
    model_config = make_vllm_config(tmp_path, head_dtype=head_dtype).model_config
    assert model_config.architecture == "Qwen3_5ForSlotDecision"
    assert (model_config.head_dtype == torch.float32) == fp32


@pytest.mark.parametrize("max_model_len,expected", [(None, 16384), (65536, 65536)])
def test_model_config_defaults_to_the_trained_length(tmp_path, max_model_len, expected):
    config = make_vllm_config(tmp_path, max_model_len=max_model_len)
    assert config.model_config.max_model_len == expected


@pytest.mark.parametrize(
    "change",
    [
        {"attention": {"full_attention": "causal"}},
        {"n_slots": 300},
        {"n_slots": True},
    ],
)
def test_model_config_rejects_unsupported_overrides(tmp_path, change):
    with pytest.raises(ValueError, match="modeling_overrides"):
        make_vllm_config(tmp_path, overrides={**OVERRIDES, **change})


@pytest.mark.parametrize(
    "engine_args,gdn,prefix_caching",
    [
        ({}, "flashinfer", True),
        ({}, "triton", False),
        (
            {
                "attention_config": {
                    "backend_per_kind": {
                        "full_attention": "TRITON_ATTN",
                        "sliding_window": "FLASH_ATTN",
                    }
                },
                # As the hook and VllmConfig set it, for a second run of the hook.
                "scheduler_cls": f"{PREFIX_LM}AsyncScheduler",
            },
            "flashinfer",
            True,
        ),
        # A per-kind Triton backend leaves the global one as it is.
        (
            {
                "attention_config": {
                    "backend": "FLASH_ATTN",
                    "backend_per_kind": {"full_attention": "TRITON_ATTN"},
                }
            },
            "flashinfer",
            True,
        ),
    ],
)
def test_config_prefills_each_prompt_in_one_step(
    tmp_path, gdn_prefill_backend, engine_args, gdn, prefix_caching
):
    gdn_prefill_backend["name"] = gdn
    config = make_vllm_config(tmp_path, enable_chunked_prefill=True, **engine_args)
    scheduler_config = config.scheduler_config
    backend = engine_args.get("attention_config", {}).get("backend", "TRITON_ATTN")
    assert config.attention_config.backend == AttentionBackendEnum[backend]
    assert config.cache_config.enable_prefix_caching == prefix_caching
    assert not scheduler_config.enable_chunked_prefill
    assert scheduler_config.scheduler_cls == f"{PREFIX_LM}Scheduler"
    assert scheduler_config.long_prefill_token_threshold == 0


@pytest.mark.parametrize(
    "engine_args,match",
    [
        ({"attention_backend": "FLASH_ATTN"}, "TRITON_ATTN"),
        (
            {
                "attention_config": {
                    "backend_per_kind": {"full_attention": "FLASH_ATTN"}
                }
            },
            "TRITON_ATTN attention backend for full attention, not FLASH_ATTN",
        ),
        ({"max_num_scheduled_tokens": 8192}, "max-num-scheduled"),
        ({"prefix_match_unit": 32}, "prefix-match-unit"),
        ({"kv_offloading_size": 1.0}, "kv-offloading-size"),
        (
            {"speculative_config": {"method": "ngram", "num_speculative_tokens": 2}},
            "speculative decoding",
        ),
        ({"prefill_context_parallel_size": 2}, "context parallel"),
        ({"enable_dbo": True}, "micro-batching"),
        ({"scheduler_cls": "my.Scheduler"}, "scheduler-cls"),
    ],
)
def test_config_rejects_unsupported_features(tmp_path, engine_args, match):
    with pytest.raises(ValueError, match=match):
        make_vllm_config(tmp_path, **engine_args)


@pytest.mark.parametrize("budget,expected", [(8192, 32768), (33024, 33024)])
def test_config_raises_a_budget_below_max_model_len(tmp_path, budget, expected):
    config = make_vllm_config(
        tmp_path, max_model_len=32768, max_num_batched_tokens=budget
    )
    assert config.scheduler_config.max_num_batched_tokens == expected


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
    # b prefills its prefix, a its question after a hit at its prefix, and
    # prefix its whole prompt; done decodes, which needs no ranges.
    add("done", 4)
    batch = make_batch(
        ["causal", "b", "invalid 0", "a", "prefix", "image", "done"],
        [10, 0, 10, 4, 0, 10, 10],
        [1, 1, 1, 6, 10, 1, 1],
    )
    assert state._block_ranges(batch) == {
        1: [(0, 0), (1, 9)],
        3: [(0, 3), (4, 9)],
        4: [(0, 9)],
    }
    for req_id in ("a", "b", "prefix"):
        state.remove_request(req_id)
    assert state._block_ranges(batch) is None


def test_model_state_refuses_a_step_inside_a_block(monkeypatch):
    state, add = make_model_state(monkeypatch)
    add("a", 4)
    with pytest.raises(RuntimeError, match="inside one of its blocks"):
        state._block_ranges(make_batch(["a"], [0], [6]))

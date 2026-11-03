# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.qwen3_5 import Qwen3_5Config

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.models import qwen3_5_slot
from vllm.model_executor.models.config import Qwen3_5ForSlotDecisionConfig
from vllm.model_executor.models.qwen3_5_slot import Qwen3_5SlotDecisionModelState
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.model_arch_config_convertor import (
    Qwen3_5ModelArchConfigConvertor,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.prefix_lm_scheduler import PREFIX_LM_LEN
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
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


def make_model_config(hf_config, max_model_len=None):
    return SimpleNamespace(
        hf_config=hf_config,
        original_max_model_len=max_model_len,
        max_model_len=max_model_len or 262144,
    )


@pytest.mark.parametrize("head_dtype,expected", [(None, "float32"), ("model", "model")])
def test_detected_with_an_fp32_head_by_default(head_dtype, expected):
    config = make_config()
    arch = Qwen3_5ModelArchConfigConvertor(config, config.text_config).convert()
    assert arch.architectures == ["Qwen3_5ForSlotDecision"]
    config.head_dtype = head_dtype
    Qwen3_5ForSlotDecisionConfig.verify_and_update_model_config(
        make_model_config(config)
    )
    assert config.head_dtype == expected


@pytest.mark.parametrize("user_len,expected", [(None, 16384), (65536, 65536)])
def test_model_config_defaults_to_the_trained_length(user_len, expected):
    model_config = make_model_config(make_config(), user_len)
    Qwen3_5ForSlotDecisionConfig.verify_and_update_model_config(model_config)
    assert model_config.max_model_len == expected


@pytest.mark.parametrize(
    "change",
    [
        {"attention": {"full_attention": "causal"}},
        {"n_slots": 300},
        {"n_slots": True},
    ],
)
def test_model_config_rejects_unsupported_overrides(change):
    model_config = make_model_config(make_config({**OVERRIDES, **change}))
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
        use_v2_model_runner=True,
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
        # A per-kind Triton backend leaves the global one as it is.
        (
            {
                "attention_config.backend": FLASH,
                "attention_config.backend_per_kind": {"full_attention": TRITON},
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
    backend = change.get("attention_config.backend", TRITON)
    assert config.attention_config.backend == backend
    assert config.cache_config.enable_prefix_caching == prefix_caching
    assert scheduler_config.enable_chunked_prefill == prefix_caching
    assert scheduler_config.scheduler_cls == f"{PREFIX_LM}Scheduler"
    assert scheduler_config.long_prefill_token_threshold == 0


@pytest.mark.parametrize(
    "change,match",
    [
        ({"attention_config.backend": FLASH}, "TRITON_ATTN"),
        (
            {"attention_config.backend_per_kind": {"full_attention": FLASH}},
            "TRITON_ATTN attention backend for full attention, not FLASH_ATTN",
        ),
        ({"scheduler_config.max_num_scheduled_tokens": 8192}, "max-num-scheduled"),
        ({"cache_config.prefix_match_unit": 32}, "prefix-match-unit"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"parallel_config.prefill_context_parallel_size": 2}, "context parallel"),
        ({"scheduler_config.scheduler_cls": "my.Scheduler"}, "scheduler-cls"),
        ({"use_v2_model_runner": False}, "Model Runner V2"),
    ],
)
def test_config_rejects_unsupported_features(change, match):
    config = make_vllm_config(**change)
    with pytest.raises(ValueError, match=match):
        Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)


@pytest.mark.parametrize("budget,expected", [(8192, 32768), (33024, 33024)])
def test_config_raises_a_budget_below_max_model_len(budget, expected):
    config = make_vllm_config(**{"scheduler_config.max_num_batched_tokens": budget})
    Qwen3_5ForSlotDecisionConfig.verify_and_update_config(config)
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


def test_model_state_builds_the_full_attention_with_the_ranges(monkeypatch):
    state, add = make_model_state(monkeypatch)
    hybrid = lambda self, batch, mode, tables, slots, groups, config: {"gdn": groups}
    monkeypatch.setattr(MambaHybridModelState, "prepare_attn", hybrid)
    monkeypatch.setattr(qwen3_5_slot, "build_attn_metadata", lambda **kw: {"attn": kw})
    attn, gdn = [object.__new__(FullAttentionSpec)], [object.__new__(MambaSpec)]
    groups = [attn, gdn]
    config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=s[0]) for s in (attn, gdn)]
    )
    add("a", 4)
    batch = make_batch(["causal", "a"], [10, 4], [1, 6])
    batch.__dict__.update(
        num_reqs=2,
        num_tokens=7,
        max_query_len=6,
        query_start_loc=None,
        query_start_loc_np=np.array([0, 1, 7]),
        seq_lens=None,
        seq_lens_cpu_upper_bound=torch.tensor([11, 10]),
        prompt_lens=None,
    )
    # A FULL graph step runs one-token queries and gets no ranges.
    full = state.prepare_attn(batch, CUDAGraphMode.FULL, (), None, groups, config)
    assert full == {"gdn": groups}
    metadata = state.prepare_attn(batch, CUDAGraphMode.NONE, (), None, groups, config)
    assert metadata["gdn"] == [[], gdn]
    assert metadata["attn"]["attn_groups"] == [attn, []]
    assert metadata["attn"]["mm_req_doc_ranges"] == {1: [(0, 3), (4, 9)]}
    assert metadata["attn"]["max_seq_len"] == 11

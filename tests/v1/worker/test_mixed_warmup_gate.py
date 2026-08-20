# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for scheduler-realistic attention warmup."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from vllm.model_executor.warmup import fa4_cutedsl_warmup as fa4_warmup
from vllm.v1.worker.gpu import warmup as gpu_warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [0, 1])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_mixed_warmup_skipped_at_exact_multi_decode_capacity():
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=3,
        max_model_len=128,
        model_state=SimpleNamespace(max_encoder_len=0),
        vllm_config=SimpleNamespace(num_lookahead_tokens=1),
        kv_cache_config=SimpleNamespace(
            num_blocks=6,
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=4))
            ],
        ),
    )

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=9,
            decode_prompt_len=3,
            num_decode_reqs=2,
        )
        is False
    )


def test_v2_fa4_dense_warmup_covers_causal_query_lengths(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(
                        vllm_flash_attn_version=4, num_queries_per_kv=8
                    )
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )
    mixed_warmup = MagicMock(return_value=True)
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert mixed_warmup.call_args_list == [
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=3,
            decode_prompt_len=2,
            decode_scheduled_tokens=1,
            req_id_prefix="_fa4_warmup_8192_2",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=3,
            decode_prompt_len=9,
            decode_scheduled_tokens=1,
            req_id_prefix="_fa4_warmup_8192_9",
        ),
        call(
            runner,
            worker.execute_model,
            worker.sample_tokens,
            num_tokens=3,
            decode_prompt_len=4096,
            decode_scheduled_tokens=1,
            req_id_prefix="_fa4_warmup_8192_4096",
        ),
    ]


def _legacy_dense_worker(
    max_tokens: int, max_num_seqs: int, num_queries_per_kv: int = 8
):
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(
                        vllm_flash_attn_version=4,
                        num_queries_per_kv=num_queries_per_kv,
                    )
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=max_tokens,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        _dummy_run=MagicMock(),
    )
    return SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=False,
    )


def test_v1_fa4_dense_warmup_covers_causal_query_lengths(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    worker = _legacy_dense_worker(8192, 2)

    fa4_warmup.fa4_cutedsl_warmup(worker)

    calls = worker.model_runner._dummy_run.call_args_list
    assert [call.args[0] for call in calls] == [2, 3, 9, 3, 4096, 3]
    assert [call.kwargs["profile_seq_lens"] for call in calls] == [
        2,
        [3, 2],
        9,
        [10, 2],
        4096,
        [4097, 2],
    ]
    assert ["create_mixed_batch" in call.kwargs for call in calls] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    for call_args in calls:
        assert call_args.kwargs["force_attention"]
        assert call_args.kwargs["is_profile"]
        assert call_args.kwargs["skip_eplb"]


def test_v2_fa4_dense_warmup_skips_local_fa2_attention(monkeypatch):
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(vllm_flash_attn_version=2)
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )
    mixed_warmup = MagicMock()
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    mixed_warmup.assert_not_called()


def test_v2_fa4_dense_warmup_seeds_batch_one_when_mixed_is_infeasible(
    monkeypatch,
):
    max_tokens = 3
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: True,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "attention": SimpleNamespace(
                    impl=SimpleNamespace(
                        vllm_flash_attn_version=4, num_queries_per_kv=32
                    )
                )
            }
        ),
        model_config=SimpleNamespace(
            use_mla=False,
            max_model_len=max_tokens,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=True,
        execute_model=MagicMock(),
        sample_tokens=MagicMock(),
    )
    mixed_warmup = MagicMock(return_value=False)
    monkeypatch.setattr(
        gpu_warmup,
        "run_mixed_prefill_decode_warmup",
        mixed_warmup,
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert runner._dummy_run.call_args_list == [
        call(query_len, skip_eplb=True, is_profile=True, num_reqs=1)
        for query_len in (2, 3)
    ]
    mixed_warmup.assert_called_once()


def test_v1_fa4_mla_warmup_covers_mixed_and_batch_one(monkeypatch):
    is_sm90 = False
    monkeypatch.setattr(
        fa4_warmup.current_platform,
        "is_device_capability",
        lambda _: is_sm90,
    )
    kernel = MagicMock()
    kernel.get_warmup_keys.return_value = [object()]
    flash_attn = ModuleType("vllm.v1.attention.backends.mla.prefill.flash_attn")
    flash_attn.FA4_MLA_PREFILL_KERNEL = kernel
    monkeypatch.setitem(sys.modules, flash_attn.__name__, flash_attn)
    monkeypatch.setattr(
        fa4_warmup,
        "get_mla_prefill_backend",
        lambda _: SimpleNamespace(get_name=lambda: "FLASH_ATTN"),
    )

    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        model_config=SimpleNamespace(
            use_mla=True,
            max_model_len=8192,
            hf_config=SimpleNamespace(model_type="test"),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    runner = SimpleNamespace(
        is_pooling_model=False,
        vllm_config=config,
        _dummy_run=MagicMock(),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=config,
        use_v2_model_runner=False,
    )

    # Preserve the pre-project MLA prefill warmup on non-SM90 architectures.
    config.attention_config.flash_attn_version = 3
    fa4_warmup.fa4_cutedsl_warmup(worker)
    kernel.warmup.assert_called_once_with(config)
    runner._dummy_run.assert_not_called()

    kernel.reset_mock()
    is_sm90 = True
    config.attention_config.flash_attn_version = 4
    fa4_warmup.fa4_cutedsl_warmup(worker)

    kernel.warmup.assert_called_once_with(config)
    assert runner._dummy_run.call_args_list == [
        call(
            513,
            force_attention=True,
            is_profile=True,
            create_mixed_batch=True,
            skip_eplb=True,
            profile_seq_lens=4096,
        ),
        call(
            2,
            force_attention=True,
            is_profile=True,
            skip_eplb=True,
            profile_seq_lens=128,
            num_reqs=1,
        ),
        call(
            2,
            force_attention=True,
            is_profile=True,
            skip_eplb=True,
            profile_seq_lens=4096,
            num_reqs=1,
        ),
    ]


def test_fa4_warmup_runs_helpers_in_order(monkeypatch):
    helper_calls = []
    worker = object()
    monkeypatch.setattr(
        fa4_warmup,
        "_warm_fa4_mla_prefill",
        lambda actual_worker: helper_calls.append(("mla", actual_worker)),
    )
    monkeypatch.setattr(
        fa4_warmup,
        "_warm_fa4_runtime_attention",
        lambda actual_worker: helper_calls.append(("runtime", actual_worker)),
    )
    monkeypatch.setattr(
        fa4_warmup,
        "_warm_inkling_fa4_rel_attention",
        lambda actual_worker: helper_calls.append(("inkling", actual_worker)),
    )

    fa4_warmup.fa4_cutedsl_warmup(worker)

    assert helper_calls == [
        ("mla", worker),
        ("runtime", worker),
        ("inkling", worker),
    ]


def test_fa4_mla_prefill_sm90_warmup_key_contract(monkeypatch):
    import torch

    import vllm.envs as envs
    from vllm.model_executor.layers.attention import mla_attention
    from vllm.v1.attention.backends.mla.prefill import flash_attn

    mla_dims = SimpleNamespace(
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    monkeypatch.setattr(mla_attention, "get_mla_dims", lambda _: mla_dims)
    monkeypatch.setattr(
        flash_attn.current_platform,
        "is_device_capability",
        lambda capability: capability == 90,
    )
    monkeypatch.setattr(
        flash_attn.current_platform,
        "is_device_capability_family",
        lambda _: False,
    )
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        get_num_attention_heads=lambda _: 16,
    )
    config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=4),
        model_config=model_config,
        parallel_config=SimpleNamespace(),
    )

    keys = flash_attn.FA4_MLA_PREFILL_KERNEL.get_warmup_keys(config)

    intended_shape_pairs = {
        (1, 128),
        (129, 512),
        (1, 4096),
        (129, 4096),
    }
    actual_axes = {
        (
            key.cu_seqlens_q_shape[0] - 1,
            (key.max_seqlen_q, key.max_seqlen_k),
            key.causal,
            key.return_softmax_lse,
        )
        for key in keys
    }
    expected_axes = {
        (batch_size, shape_pair, causal, return_lse)
        for batch_size in (1, 2)
        for shape_pair in intended_shape_pairs
        for causal in (False, True)
        for return_lse in (False, True)
    }
    assert len(keys) == 32
    assert actual_axes == expected_axes

    config.attention_config.flash_attn_version = 3
    assert flash_attn.FA4_MLA_PREFILL_KERNEL.get_warmup_keys(config) == []

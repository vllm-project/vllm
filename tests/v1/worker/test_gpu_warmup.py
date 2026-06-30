# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, TQFullAttentionSpec
from vllm.v1.worker.gpu import warmup as gpu_warmup


class _KVConnector:
    def __init__(self) -> None:
        self.disabled: list[bool] = []

    def set_disabled(self, disabled: bool) -> None:
        self.disabled.append(disabled)


def _make_warmup_model_runner(
    kv_cache_spec,
    *,
    max_model_len: int = 1024,
    max_num_seqs: int = 2,
    max_num_batched_tokens: int = 512,
    num_blocks: int = 128,
):
    return SimpleNamespace(
        num_speculative_steps=0,
        decode_query_len=1,
        is_pooling_model=False,
        is_last_pp_rank=True,
        max_model_len=max_model_len,
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=kv_cache_spec)],
            num_blocks=num_blocks,
        ),
        kv_connector=_KVConnector(),
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
    )


def test_v1_attention_warmup_uses_forced_attention_dummy_runs(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gpu_warmup.torch.accelerator,
        "synchronize",
        lambda: None,
    )

    dummy_runs = []
    model_runner = SimpleNamespace(
        _dummy_run=lambda **kwargs: dummy_runs.append(kwargs),
        is_pooling_model=False,
        attn_groups=[[object()]],
        max_num_tokens=256,
        max_model_len=4096,
        uniform_decode_query_len=1,
        scheduler_config=SimpleNamespace(
            max_num_seqs=64,
            max_num_batched_tokens=128,
        ),
    )

    gpu_warmup.warmup_v1_attention_kernels(model_runner)

    assert dummy_runs == [
        {
            "num_tokens": 64,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": True,
            "profile_seq_lens": 4096,
        },
        {
            "num_tokens": 64,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": False,
            "num_reqs_override": 1,
        },
        {
            "num_tokens": 128,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": False,
            "num_reqs_override": 1,
            "profile_seq_lens": 4096,
            "profile_as_cached_prefill": True,
        },
        {
            "num_tokens": 128,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": False,
            "num_reqs_override": 3,
            "profile_seq_lens": 4096,
            "profile_as_cached_prefill": True,
        },
        {
            "num_tokens": 128,
            "skip_eplb": True,
            "is_profile": True,
            "force_attention": True,
            "uniform_decode": False,
            "num_reqs_override": 16,
            "profile_seq_lens": 4096,
            "profile_as_cached_prefill": True,
        },
    ]


def test_v1_attention_warmup_skips_without_attention_groups(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gpu_warmup.torch.accelerator,
        "synchronize",
        lambda: None,
    )

    dummy_run = Mock()
    model_runner = SimpleNamespace(
        _dummy_run=dummy_run,
        is_pooling_model=False,
        attn_groups=[],
        max_num_tokens=256,
        max_model_len=4096,
        uniform_decode_query_len=1,
        scheduler_config=SimpleNamespace(
            max_num_seqs=3,
            max_num_batched_tokens=128,
        ),
    )

    gpu_warmup.warmup_v1_attention_kernels(model_runner)

    dummy_run.assert_not_called()


def test_warmup_kernels_runs_turboquant_continuation_prefill(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gpu_warmup.torch.accelerator,
        "synchronize",
        lambda: None,
    )

    tq_spec = TQFullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.uint8,
        tq_slot_size=128,
    )
    model_runner = _make_warmup_model_runner(
        tq_spec,
        max_num_seqs=16,
        max_num_batched_tokens=2496,
        num_blocks=256,
    )
    execute_outputs: list[Any] = []
    sample_outputs: list[Any] = []

    gpu_warmup.warmup_kernels(
        model_runner,
        execute_outputs.append,
        sample_outputs.append,
    )

    assert model_runner.kv_connector.disabled == [True, False]
    assert len(execute_outputs) == 6
    assert len(sample_outputs) == 2

    tq_prefill = execute_outputs[3]
    assert [req.req_id for req in tq_prefill.scheduled_new_reqs] == ["_warmup_tq_"]
    assert tq_prefill.num_scheduled_tokens == {"_warmup_tq_": 256}
    tq_new_req = tq_prefill.scheduled_new_reqs[0]
    assert len(tq_new_req.prompt_token_ids) == 512
    assert len(tq_new_req.prefill_token_ids) == 512

    tq_continuation = execute_outputs[4]
    assert tq_continuation.num_scheduled_tokens == {"_warmup_tq_": 256}
    assert tq_continuation.scheduled_cached_reqs.req_ids == ["_warmup_tq_"]
    assert tq_continuation.scheduled_cached_reqs.num_computed_tokens == [256]
    assert tq_continuation.scheduled_cached_reqs.num_output_tokens == [0]
    assert tq_continuation.scheduled_cached_reqs.new_block_ids != [None]

    assert execute_outputs[5].finished_req_ids == {"_warmup_tq_"}


def test_warmup_kernels_skips_turboquant_warmup_for_non_tq_cache(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        gpu_warmup.torch.accelerator,
        "synchronize",
        lambda: None,
    )

    model_runner = _make_warmup_model_runner(
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
        )
    )
    execute_outputs: list[Any] = []

    gpu_warmup.warmup_kernels(
        model_runner,
        execute_outputs.append,
        lambda _: None,
    )

    assert len(execute_outputs) == 3
    assert all(
        "_warmup_tq_"
        not in {
            *output.num_scheduled_tokens.keys(),
            *output.finished_req_ids,
        }
        for output in execute_outputs
    )

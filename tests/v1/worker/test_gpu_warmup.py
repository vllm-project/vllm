# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.worker.gpu import warmup as gpu_warmup


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
            max_num_seqs=3,
            max_num_batched_tokens=128,
        ),
    )

    gpu_warmup.warmup_v1_attention_kernels(model_runner)

    assert dummy_runs == [
        {
            "num_tokens": 3,
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
            "profile_seq_lens": 4096,
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

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.model_executor.warmup import attention_warmup
from vllm.model_executor.warmup.attention_warmup import mixed_batch_attention_warmup


def _worker(*, use_v2: bool, max_num_batched_tokens: int, max_model_len: int = 4096):
    calls = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(_dummy_run=lambda **kw: calls.append(kw)),
        use_v2_model_runner=use_v2,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_num_batched_tokens),
        model_config=SimpleNamespace(max_model_len=max_model_len),
        execute_model=object(),
        sample_tokens=object(),
    )
    return worker, calls


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "max_model_len", "expected"),
    [(8192, 4096, 16), (6, 4096, 6), (8192, 8, 8)],
)
def test_v1_runs_mixed_dummy_batch_within_limits(
    max_num_batched_tokens, max_model_len, expected
):
    worker, calls = _worker(
        use_v2=False,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
    )
    mixed_batch_attention_warmup(worker)
    assert calls == [
        dict(
            num_tokens=expected,
            skip_eplb=True,
            is_profile=True,
            force_attention=True,
            create_mixed_batch=True,
        )
    ]


@pytest.mark.parametrize(("max_num_batched_tokens", "expected"), [(8192, 16), (6, 6)])
@pytest.mark.parametrize("ran", [True, False])
def test_v2_runs_mixed_prefill_decode_step(
    monkeypatch, max_num_batched_tokens, expected, ran
):
    worker, dummy_calls = _worker(
        use_v2=True, max_num_batched_tokens=max_num_batched_tokens
    )
    mixed_calls = []

    def fake_mixed_warmup(runner, execute_model, sample_tokens, num_tokens):
        mixed_calls.append((runner, execute_model, sample_tokens, num_tokens))
        return ran

    monkeypatch.setattr(
        attention_warmup, "run_mixed_prefill_decode_warmup", fake_mixed_warmup
    )
    mixed_batch_attention_warmup(worker)

    assert mixed_calls == [
        (worker.model_runner, worker.execute_model, worker.sample_tokens, expected)
    ]
    # No dummy-run fallback when the mixed step is skipped.
    assert dummy_calls == []

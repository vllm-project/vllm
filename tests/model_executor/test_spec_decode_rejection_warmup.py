# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig, derive_watermark_key
from vllm.model_executor.warmup.spec_decode_rejection_warmup import (
    spec_decode_rejection_warmup,
)

MASTER_KEY = 20250910


def _worker(
    watermark_config: WatermarkConfig | None,
    draft_sample_method: str = "greedy",
) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=2,
                rejection_sample_method="standard",
                draft_sample_method=draft_sample_method,
            ),
            model_config=SimpleNamespace(
                get_vocab_size=lambda: 128, dtype=torch.bfloat16
            ),
            watermark_config=watermark_config,
        ),
    )


def _capture_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_rejection_sample(**kwargs):
        calls.append(kwargs)
        return None, None

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils.rejection_sample",
        fake_rejection_sample,
    )
    return calls


def test_warmup_token_id_dtypes_match_the_runtime_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None))

    assert calls
    for kwargs in calls:
        assert kwargs["draft_sampled"].dtype == torch.int32
        assert kwargs["pos"].dtype == torch.int64


def test_warmup_index_mapping_dtypes_match_the_runtime_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None))

    assert calls
    for kwargs in calls:
        assert kwargs["idx_mapping"].dtype == torch.int64
        assert kwargs["expanded_idx_mapping"].dtype == torch.int64
        assert kwargs["expanded_local_pos"].dtype == torch.int32


def test_warmup_watermark_args_match_the_runtime_kernel_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    watermark_config = WatermarkConfig(
        key=MASTER_KEY, algorithm="dual_key_gumbel", context_width=3
    )
    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(
        _worker(watermark_config, draft_sample_method="probabilistic")
    )

    assert calls
    key_b = derive_watermark_key(MASTER_KEY, b"key_b")
    for kwargs in calls:
        contexts = kwargs["contexts"]
        assert contexts.dtype == torch.int32
        assert contexts.shape == (3, 3) and contexts.stride() == (3, 1)
        assert kwargs["watermarking"].dtype == torch.bool
        assert kwargs["watermark_key"] == key_b
        assert key_b & 0xFFFFFFFF >= 2**31 and key_b >> 32 >= 2**31


def test_warmup_is_skipped_when_the_recovery_key_cannot_be_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    monkeypatch.setattr(
        "vllm.v1.watermarking.spec_decode.speculative_target_watermark_key",
        lambda _: (_ for _ in ()).throw(NotImplementedError("no philox")),
    )
    spec_decode_rejection_warmup(_worker(WatermarkConfig(key=MASTER_KEY)))

    assert calls == []


def test_warmup_draft_logits_match_the_runtime_draft_sample_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None))
    assert calls
    assert all(kwargs["draft_logits"] is None for kwargs in calls)

    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None, draft_sample_method="probabilistic"))
    assert calls
    assert all(kwargs["draft_logits"] is not None for kwargs in calls)

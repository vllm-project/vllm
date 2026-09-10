# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The warmup must launch the specialization the engine will launch.

Triton keys a specialization on tensor pointer dtypes and on the i32/i64 type it
infers from an integer argument's magnitude, so warming a variant the runtime
never uses leaves the first request paying the compile this module exists to
remove.
"""

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
    """``draft_sampled`` slices ``InputBatch.input_ids``, which is int32.

    ``pos`` slices ``InputBatch.positions``, which is int64. Warming int64
    ``draft_sampled`` compiles a specialization no engine launches.
    """
    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None))

    assert calls
    for kwargs in calls:
        assert kwargs["draft_sampled"].dtype == torch.int32
        assert kwargs["pos"].dtype == torch.int64


def test_warmup_index_mapping_dtypes_match_the_runtime_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``idx_mapping`` and ``expanded_idx_mapping`` are int64 at runtime.

    ``GPUModelRunner`` builds ``idx_mapping`` from an ``np.intp`` array (and
    ``InputBatch.make_dummy`` from an explicitly int64 ``torch.arange``), and
    ``expanded_idx_mapping`` is either that same tensor or
    ``idx_mapping.new_empty(...)``. ``expanded_local_pos`` is separately
    allocated as int32.
    """
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
    """Contexts are int32 and the key is the real key-B, not a placeholder.

    ``GPUWatermarkSampler._get_contexts`` reads int32 request-state token ids,
    and both 32-bit halves of a derived key exceed 2**31, so a key of 0 is typed
    i32 where the real key is typed i64.
    """
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
    """``Speculator`` allocates ``draft_logits`` only for probabilistic drafts.

    With the default ``draft_sample_method="greedy"`` the runtime calls
    ``rejection_sample(..., draft_logits=None)``, i.e. the
    ``HAS_DRAFT_LOGITS=False`` specialization.
    """
    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None))
    assert calls
    assert all(kwargs["draft_logits"] is None for kwargs in calls)

    calls = _capture_calls(monkeypatch)
    spec_decode_rejection_warmup(_worker(None, draft_sample_method="probabilistic"))
    assert calls
    assert all(kwargs["draft_logits"] is not None for kwargs in calls)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The warmup must cover every specialization the sampler will launch.

``_philox_gumbel_kernel`` is keyed on the Philox key halves, on the logits
dtype (model dtype when the sampler passes logits through, fp32 when logits
processing copies them) and on whether a skip mask is passed, so a missing
combination leaves a request paying the compile this module exists to remove.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm.config.watermarking import WatermarkConfig, derive_watermark_key
from vllm.model_executor.warmup.watermark_sample_warmup import watermark_sample_warmup
from vllm.platforms import current_platform
from vllm.v1.watermarking.factory import create_watermarker
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.spec_decode import create_speculative_target_watermarker

MASTER_KEY = 20250910
KEY_A = derive_watermark_key(MASTER_KEY, b"key_a")
KEY_B = derive_watermark_key(MASTER_KEY, b"key_b")


@pytest.fixture(autouse=True)
def cuda_alike_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warmup returns early off CUDA, so pin the platform it reads.

    Everything here runs on the fake worker's cpu device; without this the
    tests below would pass only on a CUDA-alike host and silently warm nothing
    anywhere else.
    """
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: True)


def _sampler(
    watermark_config: WatermarkConfig,
    speculative: bool,
    use_fp64_gumbel: bool,
) -> GPUWatermarkSampler:
    """The sampler ``GPUModelRunner`` builds for this config.

    Mirrors ``vllm/v1/worker/gpu/model_runner.py:473-486``. ``__new__`` skips
    ``Sampler.__init__``, which allocates UVA-backed device buffers; the warmup
    only reads the three attributes set here.
    """
    watermarker = create_watermarker(watermark_config)
    if speculative:
        watermarker = create_speculative_target_watermarker(watermarker)
    sampler = GPUWatermarkSampler.__new__(GPUWatermarkSampler)
    sampler.watermarker = watermarker
    sampler.deduplicate_contexts = watermark_config.deduplicate_contexts
    sampler.use_fp64_gumbel = use_fp64_gumbel
    return sampler


def _worker(
    watermark_config: WatermarkConfig | None,
    speculative: bool = False,
    use_fp64_gumbel: bool = False,
    with_sampler: bool = True,
) -> SimpleNamespace:
    sampler = (
        _sampler(watermark_config, speculative, use_fp64_gumbel)
        if watermark_config is not None and with_sampler
        else None
    )
    return SimpleNamespace(
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(sampler=sampler),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                get_vocab_size=lambda: 128,
                dtype=torch.bfloat16,
            ),
            watermark_config=watermark_config,
        ),
    )


def _capture_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_philox_gumbel_sample(logits, contexts, key, **kwargs):
        calls.append({"logits": logits, "contexts": contexts, "key": key, **kwargs})
        return torch.zeros(logits.shape[0], dtype=torch.int64)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.watermark.philox_gumbel_sample",
        fake_philox_gumbel_sample,
    )
    return calls


def _variants(calls: list[dict]) -> set[tuple[int, torch.dtype, bool]]:
    return {
        (call["key"], call["logits"].dtype, call.get("skip_mask") is not None)
        for call in calls
    }


def test_no_warmup_without_a_watermark_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(_worker(None))

    assert calls == []


def test_no_warmup_without_a_watermarked_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank that does not sample launches nothing to warm.

    Only the final pipeline stage of a generative model holds a
    ``GPUWatermarkSampler``; on any other rank the kernel is never launched.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(WatermarkConfig(key=MASTER_KEY), with_sampler=False)
    )

    assert calls == []


def test_no_warmup_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GumbelWatermarker`` launches the kernel only on a CUDA-alike device.

    Elsewhere -- an XPU worker inherits ``kernel_warmup`` from the CUDA worker
    -- the sampler takes the torch path, so warming would compile a kernel
    nothing launches.
    """
    calls = _capture_calls(monkeypatch)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
    watermark_sample_warmup(_worker(WatermarkConfig(key=MASTER_KEY)))

    assert calls == []


def test_single_key_warms_both_logits_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(_worker(WatermarkConfig(key=MASTER_KEY)))

    assert _variants(calls) == {
        (MASTER_KEY, torch.bfloat16, True),
        (MASTER_KEY, torch.float32, True),
    }
    assert len(calls) == len(_variants(calls))


def test_dual_key_warms_both_derived_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DualKeyGumbelWatermarker.sample`` launches the kernel once per key.

    The halves of a derived key exceed 2**31, so each key is typed i64/i32 by
    magnitude and a wrong key compiles a specialization the engine never uses.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(WatermarkConfig(key=MASTER_KEY, algorithm="dual_key_gumbel"))
    )

    assert _variants(calls) == {
        (KEY_A, torch.bfloat16, True),
        (KEY_A, torch.float32, True),
        (KEY_B, torch.bfloat16, True),
        (KEY_B, torch.float32, True),
    }


@pytest.mark.parametrize("alpha,key", [(0.0, KEY_A), (1.0, KEY_B)])
def test_dual_key_alpha_bounds_warm_one_key(
    monkeypatch: pytest.MonkeyPatch, alpha: float, key: int
) -> None:
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(key=MASTER_KEY, algorithm="dual_key_gumbel", alpha=alpha)
        )
    )

    assert _variants(calls) == {
        (key, torch.bfloat16, True),
        (key, torch.float32, True),
    }


def test_speculative_config_warms_the_target_role_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under spec decode the sampler holds ``create_target_watermarker()``.

    Rows without draft tokens sample through it, so the sampler path launches
    key B alone; the draft key runs in the speculator's own sampler.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(key=MASTER_KEY, algorithm="dual_key_gumbel"),
            speculative=True,
        )
    )

    assert _variants(calls) == {
        (KEY_B, torch.bfloat16, True),
        (KEY_B, torch.float32, True),
    }


def test_dedup_none_also_warms_the_mask_free_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without deduplication an all-watermarked batch passes no skip mask.

    A later batch mixing in an opted-out request passes one, so both
    specializations are reachable and both must be warmed.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(WatermarkConfig(key=MASTER_KEY, deduplicate_contexts="none"))
    )

    assert _variants(calls) == {
        (MASTER_KEY, torch.bfloat16, True),
        (MASTER_KEY, torch.float32, True),
        (MASTER_KEY, torch.bfloat16, False),
        (MASTER_KEY, torch.float32, False),
    }
    for call in calls:
        if call.get("skip_mask") is None:
            assert "temperatures" not in call


def test_warmup_argument_dtypes_match_the_runtime_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contexts are int32, the sampling state int64 indices and fp32 temperatures.

    ``GPUWatermarkSampler._get_contexts`` gathers from the int32 request-state
    token ids, ``expanded_idx_mapping``/``positions``/``seeds`` are int64 and
    ``SamplingStates.temperature`` is fp32; warming other dtypes compiles
    specializations no engine launches.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(_worker(WatermarkConfig(key=MASTER_KEY, context_width=3)))

    assert calls
    for call in calls:
        num_tokens = call["logits"].shape[0]
        assert call["contexts"].dtype == torch.int32
        assert call["contexts"].shape == (num_tokens, 3)
        assert call["skip_mask"].dtype == torch.bool
        assert call["expanded_idx_mapping"].dtype == torch.int64
        assert call["positions"].dtype == torch.int64
        assert call["seeds"].dtype == torch.int64
        assert call["temperatures"].dtype == torch.float32


def test_use_fp64_follows_the_path_that_forwards_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the mixed path forwards the sampler's ``use_fp64_gumbel``.

    ``GumbelWatermarker._sample_watermarked`` launches the mask-free variant
    with the ``use_fp64=False`` default, so warming it with fp64 would compile
    a specialization the engine never launches and miss the one it does.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(key=MASTER_KEY, deduplicate_contexts="none"),
            use_fp64_gumbel=True,
        )
    )

    assert calls
    for call in calls:
        assert call["use_fp64"] is (call.get("skip_mask") is not None)


def test_warmup_failure_does_not_abort_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_philox_gumbel_sample(*args, **kwargs):
        raise RuntimeError("no triton")

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.watermark.philox_gumbel_sample",
        failing_philox_gumbel_sample,
    )
    with caplog.at_level(logging.WARNING):
        watermark_sample_warmup(_worker(WatermarkConfig(key=MASTER_KEY)))

    assert "Skipping watermark sampler warmup." in caplog.text

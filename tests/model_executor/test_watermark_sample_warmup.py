# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The warmup must cover every specialization the sampler will launch.

Watermark kernels are keyed on the logits dtype (model dtype when the sampler
passes logits through, fp32 when logits processing copies them), on whether a
skip mask is passed and on algorithm arguments such as the Philox key, so a
missing combination leaves a request paying the compile this module exists to
remove. The warmup samples through the sampler's watermarker: the cpu tests
check which samples it requests, the CUDA tests which kernels those launch.
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
from vllm.v1.watermarking.synthid import _STREAM_DOMAIN as SYNTHID_STREAM
from vllm.v1.watermarking.watermarker import WatermarkSample

MASTER_KEY = 20250910
KEY_A = derive_watermark_key(MASTER_KEY, b"key_a")
KEY_B = derive_watermark_key(MASTER_KEY, b"key_b")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="watermark kernels launch only on CUDA"
)


@pytest.fixture(autouse=True)
def cuda_alike_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warmup returns early off CUDA, so pin the platform it reads.

    The sampling tests run on the fake worker's cpu device; without this they
    would pass only on a CUDA-alike host and silently warm nothing anywhere
    else.
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
    device: str = "cpu",
) -> SimpleNamespace:
    sampler = (
        _sampler(watermark_config, speculative, use_fp64_gumbel)
        if watermark_config is not None and with_sampler
        else None
    )
    return SimpleNamespace(
        device=torch.device(device),
        model_runner=SimpleNamespace(sampler=sampler),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                get_vocab_size=lambda: 128,
                dtype=torch.bfloat16,
            ),
            watermark_config=watermark_config,
        ),
    )


def _capture_samples(
    monkeypatch: pytest.MonkeyPatch, worker: SimpleNamespace
) -> list[dict]:
    calls: list[dict] = []

    def fake_sample(logits, contexts, random_sampler=None, skip_mask=None):
        calls.append(
            {
                "logits": logits,
                "contexts": contexts,
                "random_sampler": random_sampler,
                "skip_mask": skip_mask,
            }
        )
        return WatermarkSample(torch.zeros(logits.shape[0], dtype=torch.int64), logits)

    monkeypatch.setattr(worker.model_runner.sampler.watermarker, "sample", fake_sample)
    return calls


def _capture_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_philox_gumbel_sample(logits, contexts, key, **kwargs):
        calls.append({"logits": logits, "contexts": contexts, "key": key, **kwargs})
        return torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.watermark.philox_gumbel_sample",
        fake_philox_gumbel_sample,
    )
    return calls


def _capture_synthid_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_synthid_watermark_logits(logits, contexts, key, depth, **kwargs):
        calls.append(
            {"logits": logits, "contexts": contexts, "key": key, "depth": depth}
            | kwargs
        )
        return logits.float()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.sample.watermark.synthid_watermark_logits",
        fake_synthid_watermark_logits,
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
    worker = _worker(WatermarkConfig(key=MASTER_KEY))
    calls = _capture_samples(monkeypatch, worker)
    worker.vllm_config.watermark_config = None
    watermark_sample_warmup(worker)

    assert calls == []


def test_no_warmup_without_a_watermarked_sampler(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rank that does not sample launches nothing to warm.

    Only the final pipeline stage of a generative model holds a
    ``GPUWatermarkSampler``; on any other rank the kernel is never launched.
    """
    with caplog.at_level(logging.INFO):
        watermark_sample_warmup(
            _worker(WatermarkConfig(key=MASTER_KEY), with_sampler=False)
        )

    assert "Warming up watermark sampler" not in caplog.text


def test_no_warmup_off_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Watermarkers launch their kernels only on a CUDA-alike device.

    Elsewhere -- an XPU worker inherits ``kernel_warmup`` from the CUDA worker
    -- the sampler takes the torch path, so warming would compile a kernel
    nothing launches.
    """
    worker = _worker(WatermarkConfig(key=MASTER_KEY))
    calls = _capture_samples(monkeypatch, worker)
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: False)
    watermark_sample_warmup(worker)

    assert calls == []


@pytest.mark.parametrize("algorithm", ["gumbel", "dual_key_gumbel", "synthid_text"])
@pytest.mark.parametrize(
    "deduplicate_contexts,skip_masks",
    [("single_turn", {True}), ("none", {True, False})],
)
def test_samples_every_reachable_dtype_and_skip_mask(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    deduplicate_contexts: str,
    skip_masks: set[bool],
) -> None:
    """Without deduplication an all-watermarked batch passes no skip mask.

    A later batch mixing in an opted-out request passes one, so both variants
    are reachable and both must be warmed, for both logits dtypes.
    """
    worker = _worker(
        WatermarkConfig(
            key=MASTER_KEY,
            algorithm=algorithm,
            deduplicate_contexts=deduplicate_contexts,
        )
    )
    calls = _capture_samples(monkeypatch, worker)
    watermark_sample_warmup(worker)

    variants = [(call["logits"].dtype, call["skip_mask"] is not None) for call in calls]
    assert set(variants) == {
        (dtype, skip)
        for dtype in (torch.bfloat16, torch.float32)
        for skip in skip_masks
    }
    assert len(variants) == len(set(variants))


def test_warmup_argument_dtypes_match_the_runtime_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contexts are int32, the sampling state int64 indices and fp32 temperatures.

    ``GPUWatermarkSampler._get_contexts`` gathers from the int32 request-state
    token ids, ``expanded_idx_mapping``/``positions``/``seeds`` are int64 and
    ``SamplingStates.temperature`` is fp32; warming other dtypes compiles
    specializations no engine launches.
    """
    worker = _worker(WatermarkConfig(key=MASTER_KEY, context_width=3))
    calls = _capture_samples(monkeypatch, worker)
    watermark_sample_warmup(worker)

    assert calls
    for call in calls:
        num_tokens = call["logits"].shape[0]
        random_sampler = call["random_sampler"]
        assert call["contexts"].dtype == torch.int32
        assert call["contexts"].shape == (num_tokens, 3)
        assert call["skip_mask"].dtype == torch.bool
        assert random_sampler.expanded_idx_mapping.dtype == torch.int64
        assert random_sampler.positions.dtype == torch.int64
        assert random_sampler.seeds.dtype == torch.int64
        assert random_sampler.temperatures.dtype == torch.float32


@pytest.mark.parametrize("use_fp64_gumbel", [False, True])
def test_random_sampler_follows_the_samplers_fp64_setting(
    monkeypatch: pytest.MonkeyPatch, use_fp64_gumbel: bool
) -> None:
    worker = _worker(WatermarkConfig(key=MASTER_KEY), use_fp64_gumbel=use_fp64_gumbel)
    calls = _capture_samples(monkeypatch, worker)
    watermark_sample_warmup(worker)

    assert calls
    for call in calls:
        assert call["random_sampler"].use_fp64 is use_fp64_gumbel


def test_warmup_failure_does_not_abort_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_sample(*args, **kwargs):
        raise RuntimeError("no triton")

    worker = _worker(WatermarkConfig(key=MASTER_KEY))
    monkeypatch.setattr(
        worker.model_runner.sampler.watermarker, "sample", failing_sample
    )
    with caplog.at_level(logging.WARNING):
        watermark_sample_warmup(worker)

    assert "Skipping watermark sampler warmup." in caplog.text


@requires_cuda
def test_single_key_warms_both_logits_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(_worker(WatermarkConfig(key=MASTER_KEY), device="cuda"))

    assert _variants(calls) == {
        (MASTER_KEY, torch.bfloat16, True),
        (MASTER_KEY, torch.float32, True),
    }
    assert len(calls) == len(_variants(calls))


@requires_cuda
def test_dual_key_warms_both_derived_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DualKeyGumbelWatermarker.sample`` launches the kernel once per key.

    The halves of a derived key exceed 2**31, so each key is typed i64/i32 by
    magnitude and a wrong key compiles a specialization the engine never uses.
    """
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(key=MASTER_KEY, algorithm="dual_key_gumbel"),
            device="cuda",
        )
    )

    assert _variants(calls) == {
        (KEY_A, torch.bfloat16, True),
        (KEY_A, torch.float32, True),
        (KEY_B, torch.bfloat16, True),
        (KEY_B, torch.float32, True),
    }


@requires_cuda
@pytest.mark.parametrize("alpha,key", [(0.0, KEY_A), (1.0, KEY_B)])
def test_dual_key_alpha_bounds_warm_one_key(
    monkeypatch: pytest.MonkeyPatch, alpha: float, key: int
) -> None:
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(key=MASTER_KEY, algorithm="dual_key_gumbel", alpha=alpha),
            device="cuda",
        )
    )

    assert _variants(calls) == {
        (key, torch.bfloat16, True),
        (key, torch.float32, True),
    }


@requires_cuda
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
            device="cuda",
        )
    )

    assert _variants(calls) == {
        (KEY_B, torch.bfloat16, True),
        (KEY_B, torch.float32, True),
    }


@requires_cuda
def test_dedup_none_also_warms_the_mask_free_specialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _capture_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(key=MASTER_KEY, deduplicate_contexts="none"),
            device="cuda",
        )
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


@requires_cuda
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
            device="cuda",
        )
    )

    assert calls
    for call in calls:
        assert call.get("use_fp64", False) is (call.get("skip_mask") is not None)


@requires_cuda
@pytest.mark.parametrize(
    "deduplicate_contexts,skip_masks",
    [("single_turn", {True}), ("none", {True, False})],
)
def test_synthid_warms_every_reachable_specialization(
    monkeypatch: pytest.MonkeyPatch,
    deduplicate_contexts: str,
    skip_masks: set[bool],
) -> None:
    """``SynthIDWatermarker.sample`` launches its kernel, not the Gumbel one.

    It passes the configured key, depth and SynthID-Text Philox stream, the
    model-dtype or fp32 logits and the sampler's skip mask, so the warmup has
    to compile the same combinations.
    """
    gumbel_calls = _capture_calls(monkeypatch)
    calls = _capture_synthid_calls(monkeypatch)
    watermark_sample_warmup(
        _worker(
            WatermarkConfig(
                key=MASTER_KEY,
                algorithm="synthid_text",
                context_width=3,
                depth=24,
                deduplicate_contexts=deduplicate_contexts,
            ),
            device="cuda",
        )
    )

    assert gumbel_calls == []
    assert {
        (call["logits"].dtype, call["skip_mask"] is not None) for call in calls
    } == {
        (dtype, skip)
        for dtype in (torch.bfloat16, torch.float32)
        for skip in skip_masks
    }
    assert len(calls) == 2 * len(skip_masks)
    for call in calls:
        assert call["key"] == MASTER_KEY
        assert call["depth"] == 24
        assert call["stream"] == SYNTHID_STREAM
        assert call["contexts"].dtype == torch.int32
        assert call["contexts"].shape == (call["logits"].shape[0], 3)

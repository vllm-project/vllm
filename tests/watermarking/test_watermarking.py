# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.config.watermarking import WatermarkConfig, WatermarkPRFName
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.watermarker import WatermarkSample
from vllm.v1.worker.gpu.sample.sampler import Sampler


@pytest.mark.parametrize("algorithm", ["gumbel"])
@pytest.mark.parametrize("prf_name", ["philox", "hmac_sha256"])
def test_watermarker_contract(algorithm: str, prf_name: WatermarkPRFName):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4, prf=prf_name)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    random_sample = lambda sample_logits: sample_logits.argmax(dim=-1)

    first = watermarker.sample(logits, contexts, random_sample)
    second = watermarker.sample(logits, contexts, random_sample)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)


def test_large_context_width_warns_but_is_allowed():
    config = WatermarkConfig(key=42, context_width=17)

    with pytest.warns(UserWarning, match="reduce robustness to edits"):
        watermarker = create_watermarker(config)

    assert watermarker.context_width == 17


def test_sampling_params_can_disable_watermarking():
    assert SamplingParams().watermarking
    assert not SamplingParams.from_optional(watermarking=False).watermarking


def test_gpu_sampler_warns_when_watermarking_is_enabled_for_greedy(monkeypatch):
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarking = SimpleNamespace(np=np.ones(1, dtype=bool))
    messages: list[str] = []
    monkeypatch.setattr(Sampler, "add_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.logger.warning_once", messages.append
    )

    sampler.add_request(0, 1, SamplingParams(temperature=0))
    sampler.add_request(0, 1, SamplingParams(temperature=1))
    sampler.add_request(0, 1, SamplingParams(temperature=0, watermarking=False))

    assert messages == [
        "Watermarking is enabled, but greedy decoding (temperature=0) cannot be "
        "watermarked. This request will use ordinary greedy sampling."
    ]


def test_gpu_sampler_respects_mixed_request_watermarking(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            return WatermarkSample(torch.tensor([7, 7]), logits + 10)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, False]), gpu=torch.tensor([True, False])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=torch.ones(2)),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler._get_contexts = lambda expanded_idx_mapping: torch.zeros(
        2, 1, dtype=torch.int64
    )
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )
    logits = torch.zeros(2, 8)

    sampled, output_logits = sampler._sample_random(
        logits,
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
        False,
    )

    assert torch.equal(sampled, torch.tensor([7, 4]))
    assert torch.equal(output_logits[0], torch.full((8,), 10.0))
    assert torch.equal(output_logits[1], logits[1])


def test_gpu_sampler_skips_watermarking_for_greedy_batch(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            raise AssertionError("watermarker should not run for greedy requests")

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, True]), gpu=torch.tensor([True, True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.zeros(2), gpu=torch.zeros(2)),
    )
    expected = (torch.tensor([3, 4]), torch.zeros(2, 8))
    monkeypatch.setattr(
        "vllm.v1.watermarking.gpu_sampler.Sampler._sample_random",
        lambda *args, **kwargs: expected,
    )

    actual = sampler._sample_random(
        torch.zeros(2, 8),
        torch.tensor([0, 1]),
        np.array([0, 1]),
        torch.zeros(2, dtype=torch.int64),
        None,
        None,
        False,
        False,
    )

    assert actual is expected

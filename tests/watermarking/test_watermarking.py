# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import create_watermarker
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.watermarker import WatermarkSample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.watermark import repeated_context_mask


@pytest.mark.parametrize("algorithm", ["gumbel"])
def test_watermarker_contract(algorithm: str):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])
    random_sample = lambda sample_logits: sample_logits.argmax(dim=-1)

    first = watermarker.sample(logits, contexts, random_sample)
    second = watermarker.sample(logits, contexts, random_sample)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)


def test_gumbel_config_warns_when_context_deduplication_is_disabled(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(
        "vllm.config.watermarking.logger.warning_once",
        lambda message, *, scope: messages.append(message),
    )

    WatermarkConfig(key=42)
    WatermarkConfig(key=42, deduplicate_contexts=False)

    assert messages == [
        (
            "Single-key Gumbel-max watermarking with deduplicate_contexts=False "
            "may increase the frequency of degenerate generations, including "
            "repetition loops; keep deduplicate_contexts=True to mitigate this."
        )
    ]


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
        (
            "Watermarking is enabled, but greedy decoding (temperature=0) cannot be "
            "watermarked. This request will use ordinary greedy sampling."
        )
    ]


def test_gpu_sampler_respects_mixed_request_watermarking(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            return WatermarkSample(torch.tensor([7, 7]), logits + 10)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = True
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
    sampler._get_repeated_contexts = lambda expanded_idx_mapping, contexts: torch.zeros(
        2, dtype=torch.bool
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
    )

    assert torch.equal(sampled, torch.tensor([7, 4]))
    assert torch.equal(output_logits[0], torch.full((8,), 10.0))
    assert torch.equal(output_logits[1], logits[1])


def test_gpu_sampler_skips_watermarking_for_repeated_contexts(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            return WatermarkSample(torch.tensor([7, 7]), logits + 10)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = True
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, True]), gpu=torch.tensor([True, True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=torch.ones(2)),
        seeds=SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
    )
    sampler.use_fp64_gumbel = False
    sampler._get_contexts = lambda expanded_idx_mapping: torch.zeros(
        2, 1, dtype=torch.int64
    )
    sampler._get_repeated_contexts = lambda expanded_idx_mapping, contexts: (
        torch.tensor([True, False])
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
    )

    assert torch.equal(sampled, torch.tensor([3, 7]))
    assert torch.equal(output_logits[0], logits[0])
    assert torch.equal(output_logits[1], torch.full((8,), 10.0))


def test_gpu_sampler_can_disable_context_deduplication(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sample):
            return WatermarkSample(torch.tensor([7, 7]), logits + 10)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = False
    sampler.watermarking = SimpleNamespace(
        np=np.array([True, True]), gpu=torch.tensor([True, True])
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=torch.ones(2)),
    )
    sampler._get_contexts = lambda expanded_idx_mapping: torch.zeros(
        2, 1, dtype=torch.int64
    )
    sampler._get_repeated_contexts = lambda *args: pytest.fail(
        "context deduplication should not run"
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
    )

    assert torch.equal(sampled, torch.tensor([7, 7]))
    assert torch.equal(output_logits, torch.full((2, 8), 10.0))


def test_repeated_context_mask_ignores_prompt_tokens():
    all_token_ids = torch.tensor(
        [
            [8, 9, 1, 2, 1, 2],
            [3, 4, 1, 2, 3, 4],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    req_indices = torch.tensor([0, 1, -1])
    prompt_lens = torch.tensor([2, 2, 0])
    total_lens = torch.tensor([6, 6, 0])
    contexts = torch.tensor([[1, 2], [3, 4], [-1, -1]])

    repeated = repeated_context_mask(
        all_token_ids,
        req_indices,
        prompt_lens,
        total_lens,
        contexts,
    )

    assert torch.equal(repeated, torch.tensor([True, False, False]))


def test_repeated_context_mask_respects_max_history():
    all_token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    req_indices = torch.tensor([0])
    prompt_lens = torch.tensor([0])
    total_lens = torch.tensor([10])
    contexts = torch.tensor([[1, 2, 3, 4]])

    full_history = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts
    )
    last_six = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts, max_history=6
    )
    last_five = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts, max_history=5
    )

    assert full_history.item()
    assert last_six.item()
    assert not last_five.item()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("max_history", [5, 6])
def test_repeated_context_mask_max_history_accelerator_parity(max_history: int):
    inputs = (
        torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([10]),
        torch.tensor([[1, 2, 3, 4]]),
    )

    expected = repeated_context_mask(*inputs, max_history=max_history)
    actual = repeated_context_mask(
        *(value.cuda() for value in inputs), max_history=max_history
    ).cpu()

    assert torch.equal(actual, expected)


def _repeated_context_inputs(
    context_width: int, device: str = "cpu"
) -> tuple[torch.Tensor, ...]:
    """Build repeated, unique, and padded request rows for mask tests."""
    prompt = [101, 102, 103, 104]
    repeated_output = [*range(1, context_width + 1)] * 2
    unique_output = [*range(1, context_width + 2)]
    max_len = len(prompt) + len(repeated_output)
    rows = [
        prompt + repeated_output,
        prompt + unique_output,
        [],
    ]
    all_token_ids = torch.zeros((3, max_len), dtype=torch.int32, device=device)
    for row, token_ids in enumerate(rows):
        all_token_ids[row, : len(token_ids)] = torch.tensor(
            token_ids, dtype=torch.int32, device=device
        )
    return (
        all_token_ids,
        torch.tensor([0, 1, -1], dtype=torch.int32, device=device),
        torch.tensor([len(prompt), len(prompt), 0], dtype=torch.int32, device=device),
        torch.tensor([len(rows[0]), len(rows[1]), 0], dtype=torch.int32, device=device),
        torch.tensor(
            [
                repeated_output[-context_width:],
                unique_output[-context_width:],
                [-1] * context_width,
            ],
            dtype=torch.int64,
            device=device,
        ),
    )


@pytest.mark.parametrize("context_width", [1, 3, 4, 16, 17])
def test_repeated_context_mask(context_width: int):
    repeated = repeated_context_mask(*_repeated_context_inputs(context_width))

    assert torch.equal(repeated, torch.tensor([True, False, False]))


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("context_width", [1, 3, 4, 16, 17])
def test_repeated_context_mask_accelerator_parity(context_width: int):
    cpu_inputs = _repeated_context_inputs(context_width)
    accelerator_inputs = list(_repeated_context_inputs(context_width, "cuda"))
    contexts = accelerator_inputs[-1]
    storage = torch.empty(
        contexts.shape[0], contexts.shape[1] * 2, dtype=contexts.dtype, device="cuda"
    )
    storage[:, ::2] = contexts
    accelerator_inputs[-1] = storage[:, ::2]

    expected = repeated_context_mask(*cpu_inputs)
    actual = repeated_context_mask(*accelerator_inputs).cpu()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_repeated_context_mask_scans_multiple_blocks():
    context = [1, 2, 3, 4]
    repeated_output = [*range(10_000, 11_030), *context, 99, *context]
    unique_output = list(range(20_000, 21_039))
    prompt = [101, 102, 103, 104]
    rows = [prompt + repeated_output, prompt + unique_output]
    all_token_ids = torch.zeros((2, len(rows[0])), dtype=torch.int32)
    for row, token_ids in enumerate(rows):
        all_token_ids[row, : len(token_ids)] = torch.tensor(token_ids)
    inputs = (
        all_token_ids,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([len(prompt), len(prompt)], dtype=torch.int32),
        torch.tensor([len(rows[0]), len(rows[1])], dtype=torch.int32),
        torch.tensor([context, unique_output[-4:]], dtype=torch.int64),
    )

    expected = repeated_context_mask(*inputs)
    actual = repeated_context_mask(*(value.cuda() for value in inputs)).cpu()

    assert torch.equal(expected, torch.tensor([True, False]))
    assert torch.equal(actual, expected)


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
    )

    assert actual is expected

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm import SamplingParams
from vllm.config.watermarking import WatermarkConfig
from vllm.platforms import current_platform
from vllm.v1.watermarking import (
    DualKeyGumbelWatermarker,
    SupportsSpeculativeDecoding,
    create_watermarker,
    derive_watermark_key,
)
from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
from vllm.v1.watermarking.gumbel import GumbelWatermarker
from vllm.v1.watermarking.spec_decode import (
    DraftWatermarker,
    create_speculative_draft_watermarker,
)
from vllm.v1.watermarking.watermarker import Watermarker, WatermarkSample
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.watermark import (
    philox_gumbel_sample,
    repeated_context_mask,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


class StubWatermarker(Watermarker):
    context_width = 1

    def _sample_watermarked(self, logits, contexts):
        return WatermarkSample(torch.tensor([7, 7]), logits + 10)


@pytest.mark.parametrize("algorithm", ["gumbel", "dual_key_gumbel"])
def test_watermarker_contract(algorithm: str):
    watermarker = create_watermarker(
        WatermarkConfig(algorithm=algorithm, key=42, context_width=4)
    )
    logits = torch.zeros(2, 128)
    contexts = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]])

    first = watermarker.sample(logits, contexts)
    second = watermarker.sample(logits, contexts)

    assert first.token_ids.shape == (2,)
    assert first.logits.shape == logits.shape
    assert torch.equal(first.token_ids, second.token_ids)


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"deduplicate_contexts": "none"},
        {"deduplicate_contexts_max_history": 1023},
    ],
)
def test_gumbel_config_warns_when_context_deduplication_is_weak(
    monkeypatch, config_overrides
):
    messages: list[str] = []
    monkeypatch.setattr(
        "vllm.config.watermarking.logger.warning_once",
        lambda message, *, scope: messages.append(message),
    )

    WatermarkConfig(key=42)
    WatermarkConfig(key=42, deduplicate_contexts_max_history=1024)
    WatermarkConfig(key=42, deduplicate_contexts_max_history=None)
    WatermarkConfig(key=42, **config_overrides)

    assert messages == [
        (
            "Single-key Gumbel-max watermarking with context deduplication disabled "
            "or limited to fewer than 1024 positions may increase the frequency of "
            "degenerate generations, including repetition loops. Use "
            "deduplicate_contexts='single_turn' or 'all' with "
            "deduplicate_contexts_max_history at least 1024 or null to mitigate this."
        )
    ]


def test_context_deduplication_history_can_be_unbounded():
    config = WatermarkConfig(key=42, deduplicate_contexts_max_history=None)

    assert config.deduplicate_contexts_max_history is None


def test_large_context_width_warns_but_is_allowed():
    config = WatermarkConfig(key=42, context_width=17)

    with pytest.warns(UserWarning, match="reduce robustness to edits"):
        watermarker = create_watermarker(config)

    assert watermarker.context_width == 17


def test_dual_key_watermarker_uses_domain_separated_keys():
    config = WatermarkConfig(algorithm="dual_key_gumbel", key=42)

    target = create_watermarker(config)
    assert isinstance(target, SupportsSpeculativeDecoding)
    draft = target.create_draft_watermarker()

    assert isinstance(target, DualKeyGumbelWatermarker)
    assert target.prf.key == derive_watermark_key(42, b"target")
    assert draft.prf.key == derive_watermark_key(42, b"draft")
    assert target.prf.key != draft.prf.key


def test_target_only_speculative_watermarking_skips_draft_watermarker():
    watermarker = create_watermarker(WatermarkConfig(algorithm="gumbel", key=42))

    assert not isinstance(watermarker, SupportsSpeculativeDecoding)
    with pytest.raises(ValueError, match="does not support speculative decoding"):
        create_speculative_draft_watermarker(
            watermarker,
            max_num_reqs=1,
            device=torch.device("cpu"),
            allow_target_only=False,
        )
    assert (
        create_speculative_draft_watermarker(
            watermarker,
            max_num_reqs=1,
            device=torch.device("cpu"),
            allow_target_only=True,
        )
        is None
    )


def test_dual_key_derivation_is_stable():
    assert derive_watermark_key(32, b"target") == 7484172436829796191
    assert derive_watermark_key(32, b"draft") == 18284270469433393546


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
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = "single_turn"
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
        "vllm.v1.watermarking.watermarker.gumbel_sample",
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
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = "single_turn"
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
        "vllm.v1.watermarking.watermarker.gumbel_sample",
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
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = StubWatermarker()
    sampler.deduplicate_contexts = "none"
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


def test_repeated_context_mask_can_include_prompt_tokens():
    all_token_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
    req_indices = torch.tensor([0])
    prompt_lens = torch.tensor([4])
    total_lens = torch.tensor([10])
    contexts = torch.tensor([[1, 2, 3, 4]])

    single_turn = repeated_context_mask(
        all_token_ids, req_indices, prompt_lens, total_lens, contexts
    )
    all_history = repeated_context_mask(
        all_token_ids,
        req_indices,
        prompt_lens,
        total_lens,
        contexts,
        include_prompt=True,
    )

    assert not single_turn.item()
    assert all_history.item()


def test_repeated_context_mask_can_skip_partial_contexts():
    all_token_ids = torch.tensor([[10, 11, 12, 13, 1, 2, 3, 4]], dtype=torch.int32)
    req_indices = torch.tensor([0])
    prompt_lens = torch.tensor([4])
    contexts = torch.tensor([[-1, -1, -1, 1]])

    partial = repeated_context_mask(
        all_token_ids,
        req_indices,
        prompt_lens,
        torch.tensor([5]),
        contexts,
        include_prompt=True,
        skip_partial_context=True,
    )
    complete = repeated_context_mask(
        all_token_ids,
        req_indices,
        prompt_lens,
        torch.tensor([8]),
        torch.tensor([[1, 2, 3, 4]]),
        include_prompt=True,
        skip_partial_context=True,
    )

    assert partial.item()
    assert not complete.item()


def test_gpu_sampler_contexts_are_completion_local_for_all_scopes():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=4)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[10, 11, 12, 13, 1, 2]])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([4])),
        total_len=SimpleNamespace(gpu=torch.tensor([6])),
    )
    request_indices = torch.tensor([0])

    sampler.deduplicate_contexts = "single_turn"
    single_turn = sampler._get_contexts(request_indices)
    sampler.deduplicate_contexts = "all"
    all_history = sampler._get_contexts(request_indices)

    assert torch.equal(single_turn, torch.tensor([[-1, -1, 1, 2]]))
    assert torch.equal(all_history, single_turn)


def test_gpu_sampler_all_history_skips_partial_context():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=4)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[10, 11, 12, 13, 1, 2]])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([4])),
        total_len=SimpleNamespace(gpu=torch.tensor([6])),
    )
    sampler.deduplicate_contexts_max_history = None
    request_indices = torch.tensor([0])
    contexts = sampler._get_contexts(request_indices)

    sampler.deduplicate_contexts = "single_turn"
    single_turn = sampler._get_repeated_contexts(request_indices, contexts)
    sampler.deduplicate_contexts = "all"
    all_history = sampler._get_repeated_contexts(request_indices, contexts)

    assert not single_turn.item()
    assert all_history.item()


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


@pytest.mark.parametrize("max_history", [0, -1])
def test_repeated_context_mask_rejects_non_positive_max_history(max_history: int):
    with pytest.raises(ValueError, match="max_history must be positive or None"):
        repeated_context_mask(
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            max_history=max_history,
        )


def test_gpu_sampler_wires_request_history_scope():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]])
        ),
        prompt_len=SimpleNamespace(gpu=torch.tensor([4])),
        total_len=SimpleNamespace(gpu=torch.tensor([10])),
    )
    sampler.deduplicate_contexts_max_history = None
    request_indices = torch.tensor([0])
    contexts = torch.tensor([[1, 2, 3, 4]])

    sampler.deduplicate_contexts = "single_turn"
    single_turn = sampler._get_repeated_contexts(request_indices, contexts)
    sampler.deduplicate_contexts = "all"
    all_history = sampler._get_repeated_contexts(request_indices, contexts)

    assert not single_turn.item()
    assert all_history.item()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize("max_history", [5, 6])
@pytest.mark.parametrize("include_prompt", [False, True])
def test_repeated_context_mask_max_history_accelerator_parity(
    max_history: int, include_prompt: bool
):
    inputs = (
        torch.tensor([[1, 2, 3, 4, 5, 6, 1, 2, 3, 4]]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([10]),
        torch.tensor([[1, 2, 3, 4]]),
    )

    expected = repeated_context_mask(
        *inputs, max_history=max_history, include_prompt=include_prompt
    )
    actual = repeated_context_mask(
        *(value.cuda() for value in inputs),
        max_history=max_history,
        include_prompt=include_prompt,
    ).cpu()

    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_repeated_context_mask_partial_context_accelerator_parity():
    inputs = (
        torch.tensor([[10, 11, 12, 13, 1]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([[-1, -1, -1, 1]], dtype=torch.int64),
    )

    expected = repeated_context_mask(
        *inputs, include_prompt=True, skip_partial_context=True
    )
    actual = repeated_context_mask(
        *(value.cuda() for value in inputs),
        include_prompt=True,
        skip_partial_context=True,
    ).cpu()

    assert torch.equal(actual, expected)


def _unaligned_max_history_inputs() -> tuple[torch.Tensor, ...]:
    """One row whose only repeated context ends at position 680.

    680 is not a multiple of the kernel's 512-position block, so the scan window
    selected by ``max_history`` starts in the middle of a block.
    """
    history = list(range(1_000, 2_200))
    history[676:680] = [1, 2, 3, 4]
    return (
        torch.tensor([history], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([len(history)], dtype=torch.int32),
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
@pytest.mark.parametrize(
    "max_history,expected", [(1200 - 680, True), (1200 - 680 - 1, False)]
)
def test_repeated_context_mask_unaligned_max_history_accelerator_parity(
    max_history: int, expected: bool
):
    inputs = _unaligned_max_history_inputs()

    reference = repeated_context_mask(*inputs, max_history=max_history)
    actual = repeated_context_mask(
        *(value.cuda() for value in inputs), max_history=max_history
    ).cpu()

    assert torch.equal(reference, torch.tensor([expected]))
    assert torch.equal(actual, reference)


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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_repeated_context_mask_last_request_at_capacity():
    max_model_len = 1024
    all_token_ids = torch.arange(
        2 * max_model_len, dtype=torch.int32, device="cuda"
    ).reshape(2, max_model_len)
    inputs = (
        all_token_ids,
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        torch.tensor([0, 0], dtype=torch.int32, device="cuda"),
        torch.tensor([max_model_len, max_model_len], dtype=torch.int32, device="cuda"),
        all_token_ids[1, -4:].to(torch.int64).unsqueeze(0),
    )

    repeated = repeated_context_mask(*inputs)

    assert not repeated.item()


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like accelerator"
)
def test_gpu_sampler_uses_fused_gumbel_for_repeated_contexts():
    device = torch.device("cuda")
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]], device=device)
    all_token_ids = torch.tensor(
        [[101, 102, 1, 2, 1, 2], [101, 102, 3, 4, 5, 6]],
        dtype=torch.int32,
        device=device,
    )
    request_indices = torch.tensor([0, 1], dtype=torch.int64, device=device)
    request_indices_np = np.array([0, 1])
    temperatures = torch.ones(2, device=device)
    seeds = torch.tensor([11, 22], dtype=torch.int64, device=device)
    positions = torch.tensor([6, 6], dtype=torch.int64, device=device)

    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = GumbelWatermarker(key=42, context_width=1)
    sampler.deduplicate_contexts = "single_turn"
    sampler.deduplicate_contexts_max_history = None
    sampler.watermarking = SimpleNamespace(
        np=np.ones(2, dtype=bool), gpu=torch.ones(2, dtype=torch.bool, device=device)
    )
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.ones(2), gpu=temperatures),
        seeds=SimpleNamespace(gpu=seeds),
    )
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=all_token_ids),
        prompt_len=SimpleNamespace(
            gpu=torch.tensor([2, 2], dtype=torch.int32, device=device)
        ),
        total_len=SimpleNamespace(
            gpu=torch.tensor([6, 6], dtype=torch.int32, device=device)
        ),
    )
    sampler.use_fp64_gumbel = False

    contexts = sampler._get_contexts(request_indices)
    repeated = sampler._get_repeated_contexts(request_indices, contexts)
    expected = philox_gumbel_sample(
        logits,
        contexts,
        42,
        skip_mask=repeated,
        expanded_idx_mapping=request_indices,
        temperatures=temperatures,
        seeds=seeds,
        positions=positions,
    )

    actual, output_logits = sampler._sample_random(
        logits,
        request_indices,
        request_indices_np,
        positions,
        None,
        None,
        False,
    )

    assert torch.equal(repeated, torch.tensor([True, False], device=device))
    assert torch.equal(actual, expected)
    assert output_logits is logits


def test_gpu_sampler_skips_watermarking_for_greedy_batch(monkeypatch):
    class StubWatermarker:
        context_width = 1

        def sample(self, logits, contexts, random_sampler=None, skip_mask=None):
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


def test_gpu_sampler_builds_speculative_contexts_from_drafts():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=2)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[100, 101, 10, 11, 0, 0]])),
        prompt_len=SimpleNamespace(gpu=torch.tensor([2])),
        total_len=SimpleNamespace(gpu=torch.tensor([4])),
    )

    contexts = sampler._get_contexts(
        torch.tensor([0, 0, 0]),
        torch.tensor([0, 1, 2]),
        torch.tensor([11, 20, 21]),
    )

    assert torch.equal(contexts, torch.tensor([[10, 11], [11, 20], [20, 21]]))


def test_gpu_sampler_builds_chunked_multi_request_speculative_contexts():
    sampler = object.__new__(GPUWatermarkSampler)
    sampler.watermarker = SimpleNamespace(context_width=2)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor(
                [
                    [100, 101, 10, 11, 0, 0],
                    [200, 201, 30, 31, 0, 0],
                    [300, 301, 50, 51, 0, 0],
                ]
            )
        ),
        prompt_len=SimpleNamespace(gpu=torch.tensor([2, 2, 2])),
        total_len=SimpleNamespace(gpu=torch.tensor([4, 4, 4])),
    )

    contexts = sampler._get_contexts(
        torch.tensor([2, 2, 1, 1]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([51, 60, 31, 40]),
    )

    assert torch.equal(
        contexts,
        torch.tensor([[50, 51], [51, 60], [30, 31], [31, 40]]),
    )


def test_draft_sampler_uses_draft_key_and_advances_context(monkeypatch):
    class StubSpeculator(DraftModelSpeculator):
        def capture(self): ...

        def init_cudagraph_manager(self, cudagraph_mode): ...

        def load_draft_model(self, target_model, target_attn_layer_names): ...

        def propose(self, *args, **kwargs): ...

    class StubModel:
        @staticmethod
        def compute_logits(hidden_states):
            return torch.zeros(hidden_states.shape[0], 8)

    class StubWatermarker:
        @staticmethod
        def sample(logits, contexts, random_sample):
            return WatermarkSample(torch.tensor([7, 7]), logits)

    speculator = object.__new__(StubSpeculator)
    speculator.model = StubModel()
    speculator.use_fp64_gumbel = False
    draft_watermarker = object.__new__(DraftWatermarker)
    draft_watermarker.watermarker = StubWatermarker()
    draft_watermarker.contexts = torch.tensor([[1, 2], [3, 4]])
    draft_watermarker.enabled = torch.tensor([True, False])
    speculator.draft_watermarker = draft_watermarker
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.speculator.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )

    sampled = speculator.sample_draft(
        hidden_states=torch.zeros(2, 4),
        sample_src_positions=torch.zeros(2, dtype=torch.int64),
        idx_mapping=torch.tensor([0, 1]),
        temperature=torch.ones(2),
        seeds=torch.zeros(2, dtype=torch.int64),
        draft_step=torch.tensor(0),
        draft_logits=torch.zeros(2, 1, 8),
    )

    assert torch.equal(sampled, torch.tensor([7, 4]))
    assert torch.equal(draft_watermarker.contexts, torch.tensor([[2, 7], [4, 4]]))


def test_dspark_reduced_vocab_draft_sampler_applies_watermarking(monkeypatch):
    speculator = object.__new__(DSparkSpeculator)
    speculator.draft_logits = torch.zeros(2, 1, 8)
    speculator._d2t_scatter_index = torch.tensor([1, 5])
    speculator._draft_scatter_buf = torch.full((2, 8), float("-inf"))
    speculator.temperature = torch.ones(2)
    speculator.seeds = torch.zeros(2, dtype=torch.int64)
    speculator._step_cols = torch.tensor([0])
    speculator.use_fp64_gumbel = False
    watermark_logits: list[torch.Tensor] = []

    def sample(logits, sampled, idx_map, temperature):
        watermark_logits.append(logits.clone())
        return sampled + 1

    speculator.draft_watermarker = SimpleNamespace(sample=sample)
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )

    sampled = speculator._sample_logits(
        torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        torch.tensor([0, 1]),
        torch.tensor([1, 1]),
        0,
    )

    assert torch.equal(sampled, torch.tensor([4, 5]))
    assert torch.equal(
        watermark_logits[0][:, [1, 5]], torch.tensor([[10, 20], [30, 40]])
    )
    assert torch.isneginf(watermark_logits[0][:, [0, 2, 3, 4, 6, 7]]).all()


def test_dspark_target_only_watermarking_leaves_drafts_unwatermarked(monkeypatch):
    speculator = object.__new__(DSparkSpeculator)
    speculator.draft_logits = torch.zeros(2, 1, 8)
    speculator._d2t_scatter_index = None
    speculator.temperature = torch.ones(2)
    speculator.seeds = torch.zeros(2, dtype=torch.int64)
    speculator._step_cols = torch.tensor([0])
    speculator.use_fp64_gumbel = False
    speculator.draft_watermarker = None
    monkeypatch.setattr(
        "vllm.v1.worker.gpu.spec_decode.dspark.speculator.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([3, 4]),
    )

    sampled = speculator._sample_logits(
        torch.zeros(2, 8),
        torch.tensor([0, 1]),
        torch.tensor([1, 1]),
        0,
    )

    assert torch.equal(sampled, torch.tensor([3, 4]))

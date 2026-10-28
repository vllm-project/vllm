# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptLinearMethod,
    resolve,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
    dequantize_to_dtype,
    nvfp4_gathered_bias,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator


def _markov_head(weight: torch.Tensor) -> DSparkMarkovHead:
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = nn.Linear(
        weight.shape[1], weight.shape[0], bias=False, dtype=weight.dtype
    )
    head.markov_w2.weight.data.copy_(weight)
    head.markov_w2._retain_weight_for_gather = False
    head.markov_w2.is_w4a16_nvfp4 = False
    return head


def test_gathered_markov_bias_overwrites_dense_logits():
    weight = torch.arange(21, dtype=torch.float32).view(7, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]])
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ]
    )
    values, index = logits.topk(3, dim=-1)
    values = torch.stack((values, torch.zeros_like(values)), dim=1)[:, 0]
    expected = values + torch.bmm(weight[index], markov_embed.unsqueeze(-1)).squeeze(-1)
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index
    )

    assert result is logits
    torch.testing.assert_close(result.gather(1, index), expected)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


def test_gathered_markov_bias_matches_dense_at_full_vocab():
    weight = torch.arange(15, dtype=torch.float32).view(5, 3) / 10
    markov_embed = torch.tensor([[0.5, -1.0, 0.25]])
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.3, 0.0]])
    original = logits.clone()
    values, index = logits.topk(logits.shape[-1], dim=-1)
    scale = 0.5
    logits.fill_(float("-inf"))

    result = _markov_head(weight).apply_bias_gathered(
        markov_embed, logits, values, index, scale
    )

    expected = original + markov_embed @ weight.T * scale
    torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_fp64", [False, True])
def test_dspark_revisited_position_uses_step_seed(use_fp64):
    """A cached draft at a revisited position must use its current step."""
    device = torch.device("cuda")
    batch = 4096
    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.acceptance_estimator = None
    spec._d2t_scatter_index = None
    spec.draft_watermarker = None
    spec.use_fp64_gumbel = use_fp64
    spec.temperature = torch.ones(batch, device=device)
    spec.seeds = torch.arange(batch, dtype=torch.int64, device=device)
    spec._step_cols = torch.arange(2, dtype=torch.int32, device=device)
    spec.draft_logits = torch.empty(batch, 2, 2, device=device)
    spec.draft_logits_num_shards = 1
    spec.draft_logits_shard_idx = 0
    logits = torch.tensor([0.7, 0.3], device=device).log().expand(batch, -1)
    mapping = torch.arange(batch, device=device)
    positions = torch.full((batch,), 22, dtype=torch.int64, device=device)

    first = spec._sample_logits(logits, mapping, positions, 1)
    repeated = spec._sample_logits(logits, mapping, positions, 1)
    next_round = spec._sample_logits(logits, mapping, positions, 0)
    torch.testing.assert_close(first, repeated)
    expected = gumbel_sample(
        logits,
        mapping,
        spec.temperature,
        spec.seeds,
        positions - 1,
        apply_temperature=True,
        is_drafting=True,
        logits_cache=spec.draft_logits,
        logits_cache_col=spec._step_cols[0],
        use_fp64=use_fp64,
    )
    torch.testing.assert_close(next_round, expected)
    mismatch_rate = (first != next_round).float().mean().item()
    assert 0.35 < mismatch_rate < 0.5


@pytest.mark.parametrize("cuda_alike_platform", [False, True])
def test_gathered_markov_bias_dequantizes_selected_w4a16_rows(
    monkeypatch, cuda_alike_platform
):
    packed_weight = torch.tensor(
        [[0x00] * 8, [0x22] * 8, [0xAA] * 8, [0x31] * 8], dtype=torch.uint8
    )
    weight_scale = torch.ones((4, 1), dtype=torch.float8_e4m3fn)

    layer = nn.Module()
    layer.weight = nn.Parameter(packed_weight, requires_grad=False)
    layer.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
    layer.weight_scale_2 = nn.Parameter(torch.ones(1), requires_grad=False)
    layer._retain_weight_for_gather = True

    spec, ctx, fmt = resolve("W4A16_NVFP4", type("Config", (), {"group_size": 16}), "")
    method = ModelOptLinearMethod(spec, ctx, fmt)

    class RepackingKernel:
        @staticmethod
        def process_weights_after_loading(layer):
            layer.weight = nn.Parameter(
                torch.zeros((1, 1), dtype=torch.int32), requires_grad=False
            )
            layer.weight_scale = nn.Parameter(
                torch.zeros((1, 1), dtype=torch.float32), requires_grad=False
            )
            layer.weight_global_scale = nn.Parameter(
                torch.zeros(1, dtype=torch.float32), requires_grad=False
            )

    method.kernel = RepackingKernel()
    method.process_weights_after_loading(layer)
    assert layer.is_w4a16_nvfp4

    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = layer
    markov_embed = torch.tensor([[1.0] * 16, [0.5, -0.5] * 8], dtype=torch.float32)
    index = torch.tensor([[1, 3], [2, 0]])
    values = torch.tensor([[0.25, -0.5], [1.0, 2.0]])
    original_values = values.clone()
    logits = torch.full((2, 4), float("-inf"))

    # CPU tensors must use the fallback even on CUDA/ROCm hosts.
    monkeypatch.setattr(current_platform, "is_cuda_alike", lambda: cuda_alike_platform)
    result = head.apply_bias_gathered(markov_embed, logits, values, index)

    dense_weight = torch.tensor([[0.0] * 16, [1.0] * 16, [-1.0] * 16, [0.5, 1.5] * 8])
    expected = original_values + torch.bmm(
        dense_weight[index], markov_embed.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(result.gather(1, index), expected)
    torch.testing.assert_close(values, original_values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_nvfp4_gathered_bias_rejects_non_16_group_size():
    device = torch.device("cuda")
    rank = 32
    vocab_size = 2

    with pytest.raises(ValueError, match="group size of 16"):
        nvfp4_gathered_bias(
            torch.zeros((1, rank), dtype=torch.bfloat16, device=device),
            torch.zeros((vocab_size, rank // 2), dtype=torch.uint8, device=device),
            torch.ones(
                (vocab_size, rank // 32), dtype=torch.float8_e4m3fn, device=device
            ),
            torch.ones(1, dtype=torch.float32, device=device),
            torch.zeros((1, 1), dtype=torch.bfloat16, device=device),
            torch.zeros((1, 1), dtype=torch.int64, device=device),
            torch.full(
                (1, vocab_size), float("-inf"), dtype=torch.bfloat16, device=device
            ),
            1.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_nvfp4_gathered_markov_bias_kernel_matches_reference(batch_size: int):
    torch.manual_seed(0)
    device = torch.device("cuda")
    topk = 256
    rank = 512
    vocab_size = 512
    alpha = 0.75

    packed_weight = torch.randint(
        0, 256, (vocab_size, rank // 2), dtype=torch.uint8, device=device
    )
    weight_scale = torch.full(
        (vocab_size, rank // 16),
        0.5,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    global_scale = torch.tensor([0.25], dtype=torch.float32, device=device)

    markov_storage = torch.randn(
        batch_size, 2, rank, dtype=torch.bfloat16, device=device
    )
    values_storage = torch.randn(
        batch_size, 2, topk, dtype=torch.bfloat16, device=device
    )
    index_storage = torch.empty(batch_size, 2, topk, dtype=torch.int64, device=device)
    for batch_idx in range(batch_size):
        index_storage[batch_idx, 0] = (
            torch.arange(topk, device=device) + batch_idx
        ) % vocab_size
    logits_storage = torch.full(
        (batch_size, 2, vocab_size),
        float("-inf"),
        dtype=torch.bfloat16,
        device=device,
    )

    markov_embed = markov_storage[:, 0]
    values = values_storage[:, 0]
    index = index_storage[:, 0]
    logits = logits_storage[:, 0]

    layer = nn.Module()
    layer.register_buffer("_nvfp4_weight_for_gather", packed_weight)
    layer.register_buffer("_nvfp4_weight_scale_for_gather", weight_scale)
    layer.register_buffer("_nvfp4_weight_global_scale_for_gather", global_scale)
    layer._nvfp4_group_size_for_gather = 16
    layer._retain_weight_for_gather = True
    layer.is_w4a16_nvfp4 = True
    head = DSparkMarkovHead.__new__(DSparkMarkovHead)
    nn.Module.__init__(head)
    head.markov_w2 = layer

    flat_index = index.reshape(-1)
    selected_weight = dequantize_to_dtype(
        packed_weight.index_select(0, flat_index),
        weight_scale.index_select(0, flat_index),
        global_scale,
        dtype=torch.bfloat16,
        block_size=16,
        swizzle=False,
    ).view(batch_size, topk, rank)
    expected = (
        values
        + torch.bmm(selected_weight, markov_embed.unsqueeze(-1)).squeeze(-1) * alpha
    )

    result = head.apply_bias_gathered(markov_embed, logits, values, index, alpha)

    torch.testing.assert_close(
        result.gather(1, index), expected, rtol=1e-2, atol=6.25e-2
    )
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()

    if batch_size == 1:
        logits.fill_(float("-inf"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = head.apply_bias_gathered(
                markov_embed, logits, values, index, alpha
            )
        logits.fill_(float("-inf"))
        graph.replay()
        assert captured is logits
        torch.testing.assert_close(
            logits.gather(1, index), expected, rtol=1e-2, atol=6.25e-2
        )


def _candidate_speculator(device, reduced_vocab, batch=1, topk=2):
    vocab, steps = 5, 2
    target_ids = torch.arange(vocab, device=device) * (2 if reduced_vocab else 1)
    target_vocab = vocab * (2 if reduced_vocab else 1)
    base = torch.tensor([5.0, 4.0, 0.0, -1.0, -2.0], device=device)
    base = base.repeat(batch, steps, 1)
    transitions = -torch.arange(vocab, device=device).float().repeat(target_vocab, 1)
    transitions[1] = torch.tensor([0.0, 1.0, 10.0, -2.0, -3.0], device=device)
    transitions[2 * (2 if reduced_vocab else 1)] = torch.tensor(
        [0.0, 1.0, -2.0, 12.0, -3.0], device=device
    )
    head = _markov_head(torch.eye(vocab)).to(device)
    spec = object.__new__(DSparkSpeculator)
    spec.model = SimpleNamespace(
        logits_processor=SimpleNamespace(org_vocab_size=vocab),
        compute_draft_logits=lambda hidden: base.flatten(0, 1).clone(),
        markov_embed=lambda ids: transitions[ids],
        markov_bias=lambda embeddings: embeddings,
        apply_markov_bias_gathered=head.apply_bias_gathered,
        map_draft_to_target=lambda ids: target_ids[ids],
    )
    spec._draft_topk = topk
    spec._markov_candidate_ids = None
    spec.vocab_size = target_vocab
    spec.device = torch.device(device)
    spec.num_speculative_steps = steps
    spec.sample_indices = torch.arange(batch * steps, device=device)
    spec.sample_idx_mapping = torch.arange(batch, device=device).repeat_interleave(
        steps
    )
    spec.sample_pos = torch.arange(batch * steps, device=device) + 1
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.ones(batch, dtype=torch.int64, device=device)
    )
    spec._anchor_idx = torch.arange(batch, device=device)
    spec._step_cols = torch.arange(steps, device=device, dtype=torch.int32)
    spec.draft_tokens = torch.empty(batch, steps, dtype=torch.int64, device=device)
    spec.draft_logits = None
    spec._d2t_scatter_index = target_ids if reduced_vocab else None
    spec._draft_scatter_buf = torch.full(
        (batch, target_vocab), -torch.inf, device=device
    )
    spec.use_confidence_head = False
    spec.acceptance_estimator = None
    spec._init_markov_candidates(spec.model, chunk_size=2)
    return spec, base, transitions, target_ids


@pytest.mark.parametrize("reduced_vocab", [False, True])
@torch.inference_mode()
def test_markov_candidates_rescue_tokens_and_follow_target_ids(
    monkeypatch, reduced_vocab
):
    """Markov-only winners extend the chain without double-biasing overlaps."""
    spec, base, transitions, target_ids = _candidate_speculator("cpu", reduced_vocab)
    observed = []

    def sample(logits, *args):
        observed.append(logits.clone())
        return DSparkSpeculator._sample_logits(spec, logits, *args)

    monkeypatch.setattr(spec, "_sample_logits", sample)
    spec._sample_sequential_topk(1, torch.zeros(2, 1))

    torch.testing.assert_close(spec.draft_tokens, target_ids[torch.tensor([[2, 3]])])
    torch.testing.assert_close(
        observed[0], torch.tensor([[5.0, 5.0, 10.0, -torch.inf, -torch.inf]])
    )
    torch.testing.assert_close(
        observed[1], torch.tensor([[5.0, 5.0, -torch.inf, 11.0, -torch.inf]])
    )
    torch.testing.assert_close(
        spec._markov_candidate_ids, transitions.topk(2, dim=-1).indices
    )


@torch.inference_mode()
def test_full_vocabulary_candidates_use_dense_sampling():
    """Selecting every token must not allocate a quadratic candidate table."""
    spec, _, _, _ = _candidate_speculator("cpu", False, topk=5)

    assert spec._draft_topk is None
    assert spec._markov_candidate_ids is None
    spec._sample_sequential(1, torch.zeros(2, 1))
    torch.testing.assert_close(spec.draft_tokens, torch.tensor([[2, 3]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("temperature", [0.0, 0.7])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@torch.inference_mode()
def test_markov_candidates_cache_matches_sampling_after_graph_replay(
    temperature, reduced_vocab
):
    """Replaying a changed union replaces the proposal cached for rejection."""
    spec, base, transitions, target_ids = _candidate_speculator(
        "cuda", reduced_vocab, 2
    )
    mapping = torch.tensor([2, 0], device="cuda")
    spec.sample_idx_mapping = mapping.repeat_interleave(2)
    spec.temperature = torch.full((3,), temperature, device="cuda")
    spec.seeds = torch.tensor([7, 11, 19], device="cuda")
    spec.use_fp64_gumbel = False
    spec.draft_watermarker = None
    spec.draft_logits = torch.full((3, 2, spec.vocab_size), -torch.inf, device="cuda")
    hidden = torch.zeros(4, 1, device="cuda")
    spec._sample_sequential_topk(2, hidden)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        spec._sample_sequential_topk(2, hidden)

    original = base.clone()
    reference_cache = torch.empty_like(spec.draft_logits)
    for shift in (0, 2, 0):
        base.copy_(original.roll(shift, dims=-1))
        spec.input_buffers.input_ids.fill_(1 + shift)
        graph.replay()
        expected_cache = torch.full_like(spec.draft_logits, -torch.inf)
        prev = spec.input_buffers.input_ids
        for step in range(2):
            bias = transitions[prev]
            selected = torch.zeros_like(bias, dtype=torch.bool)
            selected.scatter_(1, base[:, step].topk(2, dim=-1).indices, True)
            selected.scatter_(1, bias.topk(2, dim=-1).indices, True)
            logits = (base[:, step] + bias).masked_fill(~selected, -torch.inf)
            target_logits = torch.full((2, spec.vocab_size), -torch.inf, device="cuda")
            target_logits[:, target_ids] = logits
            expected_cache[mapping, step] = target_logits
            prev = gumbel_sample(
                target_logits,
                mapping,
                spec.temperature,
                spec.seeds,
                spec.sample_pos.view(2, 2)[:, step] - 1,
                apply_temperature=True,
                is_drafting=True,
                logits_cache=reference_cache,
                logits_cache_col=spec._step_cols[step],
            )
            torch.testing.assert_close(spec.draft_tokens[:, step], prev)
        torch.testing.assert_close(spec.draft_logits, expected_cache)

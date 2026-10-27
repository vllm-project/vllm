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
from vllm.v1.worker.gpu.spec_decode.dspark.topk_walk import TopkMarkovWalk


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


class _TopkDraftModel:
    def __init__(self, head: DSparkMarkovHead, base_logits: torch.Tensor, scale):
        self.head = head
        self.base_logits = base_logits
        self.scale = scale

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.base_logits.clone()

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.head.markov_w1(token_ids)

    def apply_markov_bias_gathered(self, markov_embed, logits, values, index):
        return self.head.apply_bias_gathered(
            markov_embed, logits, values, index, self.scale
        )

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def compute_confidence(self, head_hidden, markov_embed):
        return markov_embed.float().sum(-1)


def _topk_speculator(
    batch: int,
    temperature: float,
    use_fp64: bool,
    fused: bool,
    scale: float = 0.5,
    num_shards: int = 1,
    probabilistic: bool = True,
) -> DSparkSpeculator:
    """A top-k DSpark sampler whose Markov products and sums are exact in fp32,
    so the fused walk and the per-step loop must agree bit for bit."""
    device = torch.device("cuda")
    n_spec, vocab_size, rank, topk = 4, 4096, 64, 12
    max_num_reqs = batch + 2
    gen = torch.Generator(device=device).manual_seed(0)

    def small(*shape):
        return (
            torch.randint(-2, 3, shape, generator=gen, device=device).to(torch.bfloat16)
            / 8
        )

    head = _markov_head(small(vocab_size, rank))
    head.markov_w1 = nn.Embedding.from_pretrained(small(vocab_size, rank))
    head.to(device)
    base_logits = (
        torch.randn(batch * n_spec, vocab_size, generator=gen, device=device) * 3
    ).to(torch.bfloat16)

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = _TopkDraftModel(head, base_logits, scale)
    spec.num_speculative_steps = n_spec
    spec._draft_topk = topk
    spec.use_confidence_head = False
    spec.acceptance_estimator = None
    spec.draft_watermarker = None
    spec._d2t_scatter_index = None
    spec.use_fp64_gumbel = use_fp64
    spec.draft_logits_num_shards = num_shards
    spec.draft_logits_shard_idx = num_shards - 1
    spec.draft_logits = None
    if probabilistic:
        spec.draft_logits = torch.full(
            ((max_num_reqs + num_shards - 1) // num_shards, n_spec, vocab_size),
            float("-inf"),
            dtype=torch.bfloat16,
            device=device,
        )
    spec.temperature = torch.full((max_num_reqs,), temperature, device=device)
    spec.seeds = torch.randint(0, 2**62, (max_num_reqs,), generator=gen, device=device)
    spec._step_cols = torch.arange(n_spec, dtype=torch.int32, device=device)
    spec.sample_indices = torch.arange(batch * n_spec, device=device)
    # Request states in reverse order, and the last row is CUDA graph padding.
    states = torch.arange(batch, dtype=torch.int32, device=device).flip(0) + 1
    states[-1] = -1
    spec.sample_idx_mapping = states.repeat_interleave(n_spec)
    context = torch.randint(100, 8000, (batch,), generator=gen, device=device)
    spec.sample_pos = (
        context[:, None] + 1 + torch.arange(n_spec, device=device)
    ).flatten()
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.randint(
            0, vocab_size, (batch * n_spec,), generator=gen, device=device
        )
    )
    spec._anchor_idx = torch.arange(batch, device=device) * n_spec
    spec.draft_tokens = torch.zeros(
        max_num_reqs, n_spec, dtype=torch.int64, device=device
    )
    spec.draft_token_confidence_probs = torch.zeros(
        max_num_reqs, n_spec, dtype=torch.float32, device=device
    )
    spec._topk_walk = None
    if fused:
        spec._topk_walk = TopkMarkovWalk(
            head.markov_w1,
            head.markov_w2,
            scale,
            max_num_reqs,
            n_spec,
            topk,
            device,
        )
    return spec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0])
@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("num_shards", [1, 2])
def test_fused_topk_walk_matches_per_step_loop(temperature, use_fp64, num_shards):
    """Same draft tokens and the same cached proposal q as the per-step loop,
    also when the cache holds only this rank's batch-sharded rows."""
    batch = 5
    reference = _topk_speculator(
        batch, temperature, use_fp64, fused=False, num_shards=num_shards
    )
    fused = _topk_speculator(
        batch, temperature, use_fp64, fused=True, num_shards=num_shards
    )
    hidden = torch.zeros(batch * reference.num_speculative_steps, 1, device="cuda")

    reference._sample_sequential(batch, hidden)
    fused._sample_sequential(batch, hidden)

    torch.testing.assert_close(fused.draft_tokens, reference.draft_tokens)
    torch.testing.assert_close(
        fused.draft_logits, reference.draft_logits, atol=0, rtol=0
    )
    assert fused.draft_logits is not None
    assert torch.isfinite(fused.draft_logits).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_topk_walk_clears_previous_candidates():
    """A slot's cached row holds only the current candidates, even when the
    previous walk on that slot chose overlapping ones, also under replay."""
    batch = 5
    fused = _topk_speculator(batch, 1.0, False, fused=True)
    reference = _topk_speculator(batch, 1.0, False, fused=False)
    hidden = torch.zeros(batch * fused.num_speculative_steps, 1, device="cuda")
    fused._sample_sequential(batch, hidden)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused._sample_sequential(batch, hidden)
    for model in (fused.model, reference.model):
        logits = model.base_logits
        # Keep half of each row's candidates and move the rest.
        logits[:, ::2] = logits[:, ::2].roll(1, dims=0)
    graph.replay()
    reference._sample_sequential(batch, hidden)

    torch.testing.assert_close(fused.draft_tokens, reference.draft_tokens)
    torch.testing.assert_close(
        fused.draft_logits, reference.draft_logits, atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_topk_walk_greedy_with_confidence_head():
    """Greedy drafting (no draft logits cache) picks the same tokens, and both
    paths feed the confidence head each step's previous token."""
    batch = 5
    paths = []
    for use_walk in (False, True):
        spec = _topk_speculator(batch, 1.0, False, fused=use_walk, probabilistic=False)
        spec.use_confidence_head = True
        hidden = torch.zeros(batch * spec.num_speculative_steps, 1, device="cuda")
        spec._sample_sequential(batch, hidden)
        paths.append(spec)
    reference, fused = paths

    torch.testing.assert_close(fused.draft_tokens, reference.draft_tokens)
    anchors = reference.input_buffers.input_ids[reference._anchor_idx]
    prev = torch.cat([anchors[:, None], reference.draft_tokens[:batch, :-1]], dim=1)
    expected = reference.model.markov_embed(prev).float().sum(-1)
    for spec in paths:
        torch.testing.assert_close(spec.draft_token_confidence_probs[:batch], expected)

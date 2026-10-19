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
from vllm.model_executor.models.qwen3_dspark import (
    DSparkMarkovHead,
    Qwen3DSparkForCausalLM,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import (
    cache_markov_candidates,
    compute_markov_bias_top_ids,
    markov_walk_topk,
)

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires CUDA or ROCm"
)


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


@requires_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduced_vocab", [False, True])
def test_candidate_walk_matches_dense_greedy_chain(dtype, reduced_vocab):
    """Full candidates preserve the dense chain, including draft-to-target IDs."""
    torch.manual_seed(42)
    batch, steps, vocab, rank = 3, 4, 37, 33
    target_vocab = 2 * vocab if reduced_vocab else vocab
    base = torch.randn(batch, steps, vocab, device="cuda", dtype=dtype)
    w1 = torch.randn(target_vocab, rank, device="cuda", dtype=dtype) / rank
    w2 = torch.randn(vocab, rank, device="cuda", dtype=dtype) / rank
    d2t = torch.arange(vocab, device="cuda") if reduced_vocab else None
    anchor = torch.arange(batch, device="cuda")
    values, ids = base.topk(vocab, dim=-1)
    tokens = torch.empty(batch, steps, dtype=torch.int64, device="cuda")
    scores = torch.empty_like(values, dtype=torch.float32)
    markov_walk_topk(
        num_reqs=batch,
        cand_values=values,
        cand_ids=ids,
        w1=w1,
        w2=w2,
        scale=0.75,
        draft_tokens=tokens,
        input_ids=anchor,
        anchor_indices=anchor,
        sample_pos=torch.arange(batch * steps, device="cuda"),
        sample_idx_mapping=anchor.repeat_interleave(steps),
        temperature=torch.zeros(batch, device="cuda"),
        seeds=anchor,
        realized_scores=scores,
        d2t=d2t,
    )
    prev = anchor
    for step in range(steps):
        logits = base[:, step].float() + 0.75 * (w1[prev].float() @ w2.float().T)
        expected = logits.argmax(-1)
        if d2t is not None:
            expected += d2t[expected]
        torch.testing.assert_close(tokens[:, step], expected)
        torch.testing.assert_close(
            scores[:, step], logits.gather(1, ids[:, step]), atol=1e-5, rtol=1e-5
        )
        prev = expected


@requires_gpu
@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.5])
@pytest.mark.parametrize("reduced_vocab", [False, True])
def test_candidate_union_samples_the_cached_distribution(temperature, reduced_vocab):
    """Overlapping candidates use their maximum score in both sampling and cache."""
    torch.manual_seed(7)
    batch, steps, vocab, rank = 32, 3, 17, 8
    target_vocab = 2 * vocab if reduced_vocab else vocab
    base = torch.randn(batch, steps, vocab, device="cuda")
    w1 = torch.zeros(target_vocab, rank, device="cuda")
    w2 = torch.zeros(vocab, rank, device="cuda")
    d2t = torch.arange(vocab, device="cuda") if reduced_vocab else None
    values, ids = base.topk(vocab, dim=-1)
    # Model unequal rounding in the two projections, in both directions.
    static_ids = torch.tensor([1, 3], device="cuda").repeat(target_vocab, 1)
    static_biases = torch.tensor([0.25, -0.25], device="cuda").repeat(target_vocab, 1)
    union_ids = torch.empty(batch, steps, vocab + 2, dtype=torch.int64, device="cuda")
    scores = torch.empty_like(union_ids, dtype=torch.float32)
    tokens = torch.empty(batch, steps, dtype=torch.int64, device="cuda")
    rows = torch.arange(batch, device="cuda")
    mapping = rows.repeat_interleave(steps)
    positions = torch.arange(batch * steps, device="cuda") + 1
    temperatures = torch.full((batch,), temperature, device="cuda")
    seeds = rows * 104729 + 7
    markov_walk_topk(
        num_reqs=batch,
        cand_values=values,
        cand_ids=ids,
        static_ids=static_ids,
        static_biases=static_biases,
        base_logits=base,
        union_ids=union_ids,
        w1=w1,
        w2=w2,
        scale=1.0,
        draft_tokens=tokens,
        input_ids=torch.zeros(batch, dtype=torch.int64, device="cuda"),
        anchor_indices=rows,
        sample_pos=positions,
        sample_idx_mapping=mapping,
        temperature=temperatures,
        seeds=seeds,
        realized_scores=scores,
        d2t=d2t,
        probabilistic=True,
    )
    expected = base.clone()
    expected[:, :, 1] += 0.25
    reference = gumbel_sample(
        expected.flatten(0, 1),
        mapping,
        temperatures,
        seeds,
        positions - 1,
        apply_temperature=True,
        is_drafting=True,
    ).view(batch, steps)
    if d2t is not None:
        reference += d2t[reference]
    torch.testing.assert_close(tokens, reference)

    cache = torch.full((batch, steps, target_vocab), -float("inf"), device="cuda")
    cache_markov_candidates(
        draft_logits=cache,
        cached_ids=torch.zeros_like(union_ids),
        cand_ids=union_ids,
        realized_scores=scores,
        sample_idx_mapping=mapping,
        d2t=d2t,
    )
    target_ids = torch.arange(vocab, device="cuda")
    if d2t is not None:
        target_ids += d2t
    expected_cache = torch.full_like(cache, -float("inf"))
    expected_cache[:, :, target_ids] = expected
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)


@requires_gpu
@pytest.mark.parametrize("num_warps", [1, 4])
def test_candidate_cache_rewrite_under_cuda_graph(num_warps):
    """Replay clears stale candidates, merges duplicates and ignores padded rows."""
    cache = torch.full((3, 2, 128), -float("inf"), device="cuda")
    ids = torch.zeros(2, 2, 64, dtype=torch.int64, device="cuda")
    scores = torch.arange(ids.numel(), device="cuda", dtype=torch.float32).view_as(ids)
    cached_ids = torch.zeros(3, 2, 64, dtype=torch.int64, device="cuda")
    mapping = torch.tensor([2, 2, -1, -1], device="cuda")

    def update():
        cache_markov_candidates(
            draft_logits=cache,
            cached_ids=cached_ids,
            cand_ids=ids,
            realized_scores=scores,
            sample_idx_mapping=mapping,
            num_warps=num_warps,
        )

    update()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()
    for shift in (0, 8, 64, 0):
        ids.copy_((torch.arange(64, device="cuda").flip(0) % 16 + shift).expand_as(ids))
        graph.replay()
        expected = torch.full_like(cache, -float("inf"))
        expected[2].scatter_reduce_(1, ids[0], scores[0], reduce="amax")
        torch.testing.assert_close(cache, expected, atol=0, rtol=0)


@pytest.mark.parametrize("scale", [1.0, -1.0])
def test_bigram_candidates_match_dense_projection(scale):
    torch.manual_seed(42)
    w1 = torch.randn(19, 7)
    w2 = torch.randn(17, 7)
    ids, values = compute_markov_bias_top_ids(w1, w2, 3, scale, chunk=5)
    expected = (w1 @ w2.T).topk(3, dim=-1, largest=scale >= 0)
    torch.testing.assert_close(ids.long(), expected.indices)
    torch.testing.assert_close(values, expected.values)


def test_bigram_candidates_exclude_padded_vocabulary():
    """Padded W2 rows must never index base logits or the draft-to-target map."""
    weight = torch.arange(16, dtype=torch.float32).view(8, 2)
    head = _markov_head(weight)
    head.markov_w1 = nn.Embedding.from_pretrained(torch.ones(5, 2))
    model = SimpleNamespace(
        model=SimpleNamespace(markov_head=head),
        logits_processor=SimpleNamespace(org_vocab_size=3, scale=1.0),
    )
    w1, w2, scale = Qwen3DSparkForCausalLM.markov_walk_inputs(model)
    ids, values = compute_markov_bias_top_ids(w1, w2, 2, scale)
    expected = (w1 @ weight[:3].T).topk(2, dim=-1)
    torch.testing.assert_close(ids.long(), expected.indices)
    torch.testing.assert_close(values, expected.values)


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

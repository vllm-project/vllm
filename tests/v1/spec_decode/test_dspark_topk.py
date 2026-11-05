# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
    DSparkSpeculator,
    _argmax_candidates,
)
from vllm.v1.worker.gpu.spec_decode.dspark.topk_markov import markov_walk_topk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("reduced_vocab", [False, True])
@pytest.mark.parametrize("topk,rank,vocab", [(3, 65, 17), (32, 256, 67)])
@torch.inference_mode()
def test_fused_markov_walk_matches_sequential_after_graph_replay(
    dtype, reduced_vocab, topk, rank, vocab
):
    """Eight dependent steps preserve mapping and reread anchors on replay."""
    torch.manual_seed(0)
    batch, steps, scale = 3, 8, 0.5
    target_vocab = vocab * (2 if reduced_vocab else 1)
    d2t = torch.arange(vocab, device="cuda") if reduced_vocab else None
    # Dyadic weights make the reference's dot products exact in FP32.
    w1 = torch.randint(-2, 3, (target_vocab, rank + 1), device="cuda").to(dtype)
    w2 = torch.randint(-2, 3, (vocab, rank + 1), device="cuda").to(dtype)
    w1.div_(16)
    w2.div_(16)
    w1, w2 = w1[:, 1:], w2[:, 1:]
    biases = (w1.float() @ w2.float().T * scale).to(dtype)
    markov_biases, markov_ids = biases.topk(topk, dim=-1)
    storage = torch.randint(-16, 17, (batch, steps, vocab + 1), device="cuda")
    base = storage.to(dtype)[:, :, 1:]
    original = base.clone()
    anchors = torch.arange(batch, device="cuda") * 2 + 1
    input_ids = torch.zeros(batch * 2 + 1, dtype=torch.int32, device="cuda")
    output = torch.full((batch, steps + 1), -1, dtype=torch.int64, device="cuda")

    def run():
        values, ids = base.topk(topk, dim=-1)
        markov_walk_topk(
            base_logits=base,
            base_values=values,
            base_ids=ids,
            markov_ids=markov_ids,
            markov_biases=markov_biases,
            w1=w1,
            w2=w2,
            input_ids=input_ids,
            anchor_indices=anchors,
            draft_tokens=output[:, 1:],
            d2t=d2t,
            scale=scale,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for shift in (0, 3):
        base.copy_(original.roll(shift, dims=-1))
        input_ids[anchors] = (
            torch.arange(batch, dtype=torch.int32, device="cuda") + shift
        )
        graph.replay()
        values, ids = base.topk(topk, dim=-1)
        prev = input_ids[anchors]
        expected = torch.empty_like(output[:, 1:])
        for step in range(steps):
            corrected = values[:, step].unsqueeze(-1).clone()
            corrected.baddbmm_(w2[ids[:, step]], w1[prev].unsqueeze(-1), alpha=scale)
            logits = torch.full_like(base[:, step], -torch.inf)
            logits.scatter_(1, ids[:, step], corrected.squeeze(-1))
            cached_ids = markov_ids[prev]
            logits.scatter_(
                1,
                cached_ids,
                base[:, step].gather(1, cached_ids) + markov_biases[prev],
            )
            prev = logits.argmax(dim=-1)
            if d2t is not None:
                prev = prev + d2t[prev]
            expected[:, step] = prev
        torch.testing.assert_close(output[:, 1:], expected, rtol=0, atol=0)
        assert (output[:, 0] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_markov_walk_preserves_cached_overwrite_and_argmax_rules(dtype):
    """Cached duplicates override base scores before tie and NaN selection."""
    base_ids = torch.tensor([[[8, 2, 5]]], device="cuda").repeat(4, 1, 1)
    values = torch.tensor(
        [[9, 7, 6], [9, 7, 8], [-torch.inf] * 3, [9, torch.nan, 6]],
        dtype=dtype,
        device="cuda",
    ).unsqueeze(1)
    base = torch.full((4, 1, 10), -100.0, dtype=dtype, device="cuda")
    base.scatter_(2, base_ids, values)
    markov_ids = torch.tensor([[8, 9, 1]], device="cuda").repeat(10, 1)
    biases = torch.zeros((10, 3), dtype=dtype, device="cuda")
    biases[:4] = torch.tensor(
        [
            [-7, 107, 107],
            [-7, 107, 107],
            [0, -torch.inf, -torch.inf],
            [torch.nan, 107, 107],
        ],
        dtype=dtype,
        device="cuda",
    )
    tokens = torch.empty((4, 1), dtype=torch.int64, device="cuda")
    markov_walk_topk(
        base_logits=base,
        base_values=values,
        base_ids=base_ids,
        markov_ids=markov_ids,
        markov_biases=biases,
        w1=torch.ones((10, 1), dtype=dtype, device="cuda"),
        w2=torch.zeros((10, 1), dtype=dtype, device="cuda"),
        input_ids=torch.arange(4, device="cuda"),
        anchor_indices=torch.arange(4, device="cuda"),
        draft_tokens=tokens,
        d2t=None,
        scale=1.0,
    )
    expected = torch.tensor([[1], [5], [0], [2]], device="cuda")
    torch.testing.assert_close(tokens, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_candidate_argmax_matches_dense_selection(dtype):
    """Ignore non-candidates and preserve overwrite and vocabulary tie rules."""
    base_ids = torch.tensor([[8, 2, 5]]).expand(4, -1)
    bias_ids = torch.tensor([[8, 9, 1]]).expand(4, -1)
    base_scores = torch.tensor(
        [[9, 7, 6], [9, 7, 8], [-torch.inf] * 3, [9, torch.nan, 6]], dtype=dtype
    )
    bias_scores = torch.tensor(
        [[2, 7, 7], [2, 7, 7], [-torch.inf] * 3, [torch.nan, 7, 7]], dtype=dtype
    )
    logits = torch.full((4, 10), 100.0, dtype=dtype)
    reference = torch.full_like(logits, -torch.inf)
    for scores in (logits, reference):
        scores.scatter_(1, base_ids, base_scores)
        scores.scatter_(1, bias_ids, bias_scores)

    actual = _argmax_candidates(logits, torch.cat((base_ids, bias_ids), dim=-1))
    torch.testing.assert_close(reference.argmax(dim=-1), torch.tensor([1, 5, 0, 2]))
    torch.testing.assert_close(actual, reference.argmax(dim=-1))


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

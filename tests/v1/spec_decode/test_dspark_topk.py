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
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache
from vllm.v1.worker.gpu.spec_decode.dspark import speculator as speculator_module
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


@pytest.mark.parametrize("head_dtype", [torch.bfloat16, torch.float32])
def test_gathered_markov_bias_preserves_head_dtype(default_vllm_config, head_dtype):
    """An FP32 output head must accept BF16 Markov weights and embeddings."""
    torch.manual_seed(17)
    head = DSparkMarkovHead(64, 64, 16, prefix="markov_head").to(torch.bfloat16)
    with torch.no_grad():
        head.markov_w1.weight.normal_(std=0.25)
        head.markov_w2.weight.normal_(std=0.25)
    markov_embed = head.embed(torch.tensor([1, 3]))
    logits = torch.randn(2, 64, dtype=head_dtype)
    values, index = logits.topk(8, dim=-1)
    original_values = values.clone()
    dense_bias = markov_embed.float() @ head.markov_w2.weight.float().T
    expected = (values.float() + 0.5 * dense_bias.gather(1, index)).to(head_dtype)
    logits.fill_(float("-inf"))

    result = head.apply_bias_gathered(markov_embed, logits, values, index, scale=0.5)

    assert result.dtype == head_dtype
    torch.testing.assert_close(result.gather(1, index), expected)
    torch.testing.assert_close(values, original_values)
    selected = torch.zeros_like(result, dtype=torch.bool).scatter_(1, index, True)
    assert torch.isneginf(result.masked_select(~selected)).all()


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
    spec.draft_logits = DraftLogitsCache(
        batch, 2, 2, dtype=torch.float32, fill=0.0, device=device
    )
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
        logits_cache=spec.draft_logits.values,
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


class _TransitionDraft:
    """Draft model whose Markov bias is a fixed row per previous token."""

    def __init__(self, transitions: torch.Tensor, transform: bool):
        self.transitions = transitions
        if transform:
            self.apply_output_logit_transform = lambda logits: logits * 0.5

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.transitions[token_ids]

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return markov_embed

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("logits_dtype", "bias_dtype", "greedy", "reduced_vocab", "transform", "fused"),
    [
        pytest.param(torch.bfloat16, torch.bfloat16, 0, 0, 0, True, id="bf16"),
        pytest.param(torch.float16, torch.float16, 0, 0, 0, True, id="fp16"),
        pytest.param(torch.float32, torch.float32, 0, 0, 0, True, id="fp32"),
        pytest.param(torch.bfloat16, torch.float32, 0, 0, 0, False, id="mixed-dtype"),
        pytest.param(torch.float32, torch.float32, 1, 0, 0, False, id="greedy"),
        pytest.param(torch.float32, torch.float32, 0, 1, 0, False, id="reduced-vocab"),
        pytest.param(torch.float32, torch.float32, 0, 0, 1, False, id="transform"),
    ],
)
def test_markov_chain_matches_adding_bias_first(
    monkeypatch, logits_dtype, bias_dtype, greedy, reduced_vocab, transform, fused
):
    """Whether or not the Gumbel kernel adds the Markov bias, the chain draws the
    same tokens and caches the same logits as adding the bias before sampling."""
    device = torch.device("cuda")
    # The last row is a padding slot; 2049 leaves a partial last block. Rounding
    # the sum to the logits dtype only shows on near-tied draws, hence the rows.
    num_reqs, num_steps, vocab = 1024, 3, 2049
    target_vocab = vocab + 100 if reduced_vocab else vocab
    torch.manual_seed(0)
    base_logits = torch.randn(
        num_reqs, num_steps, vocab, device=device, dtype=logits_dtype
    )
    transitions = torch.randn(target_vocab, vocab, device=device, dtype=bias_dtype)
    model = _TransitionDraft(transitions, bool(transform))
    prev = torch.randint(target_vocab, (num_reqs,), device=device)
    idx_map = torch.arange(num_reqs, device=device).unsqueeze(1).repeat(1, num_steps)
    idx_map[-1] = -1
    sample_pos = 100 + torch.arange(num_steps, device=device).expand(num_reqs, -1)
    cache_dtype = torch.promote_types(logits_dtype, bias_dtype)

    def speculator() -> DSparkSpeculator:
        spec = DSparkSpeculator.__new__(DSparkSpeculator)
        spec.model = model
        spec.num_speculative_steps = num_steps
        spec.use_confidence_head = False
        spec.acceptance_estimator = None
        spec.draft_watermarker = None
        spec.use_fp64_gumbel = False
        spec.temperature = torch.tensor([0.0, 0.6, 1.0, 1.3], device=device).repeat(
            num_reqs // 4
        )
        spec.seeds = torch.arange(num_reqs, dtype=torch.int64, device=device)
        spec._step_cols = torch.arange(num_steps, dtype=torch.int32, device=device)
        spec.draft_logits = None
        if not greedy:
            spec.draft_logits = DraftLogitsCache(
                num_reqs,
                num_steps,
                target_vocab,
                dtype=cache_dtype,
                fill=0.0,
                device=device,
            )
        spec._d2t_scatter_index = None
        if reduced_vocab:
            spec._d2t_scatter_index = torch.arange(vocab, device=device) + 100
            spec._draft_scatter_buf = torch.full(
                (num_reqs, target_vocab),
                float("-inf"),
                dtype=cache_dtype,
                device=device,
            )
        # As load_model sets it; the mixed-dtype case falls back in the loop.
        spec._fuse_markov_bias = not (greedy or reduced_vocab or transform)
        return spec

    fused_steps = []

    def spy(*args, **kwargs):
        fused_steps.append(kwargs.get("bias") is not None)
        return gumbel_sample(*args, **kwargs)

    monkeypatch.setattr(speculator_module, "gumbel_sample", spy)
    spec = speculator()
    tokens = torch.empty(num_reqs, num_steps, dtype=torch.int64, device=device)
    spec._sample_markov_chain(
        base_logits,
        torch.zeros(num_reqs * num_steps, 1, device=device),
        prev,
        idx_map,
        sample_pos,
        tokens,
    )
    assert sum(fused_steps) == (num_steps if fused else 0)

    reference = speculator()
    expected = tokens.new_empty(tokens.shape)
    for i in range(num_steps):
        logits = base_logits[:, i] + transitions[prev]
        if transform:
            logits = model.apply_output_logit_transform(logits)
        prev = reference._sample_logits(logits, idx_map[:, i], sample_pos[:, i], i)
        expected[:, i] = prev

    assert torch.equal(tokens, expected)
    if not greedy:
        assert spec.draft_logits is not None and reference.draft_logits is not None
        assert spec.draft_logits.values.any()
        assert torch.equal(spec.draft_logits.values, reference.draft_logits.values)

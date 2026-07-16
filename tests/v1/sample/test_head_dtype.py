# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for running the generation lm_head in fp32 via ``head_dtype``.

An fp32 head lets rollout logits match a trainer that computes the lm_head in
fp32, which is required for RL training-inference consistency.
"""

import math

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)


class _FakeLmHead:
    def __init__(
        self,
        weight: torch.Tensor,
        quantized: bool = False,
        shard_indices: object | None = None,
    ):
        self.weight = weight
        self.quant_method = object() if quantized else UnquantizedEmbeddingMethod()
        self.shard_indices = shard_indices


def _build_processor(vocab_size: int) -> LogitsProcessor:
    lp = LogitsProcessor(vocab_size)
    # The TP gather is orthogonal to the dtype behavior under test.
    lp._gather_logits = lambda logits: logits
    return lp


def test_fp32_head_runs_projection_in_fp32(default_vllm_config):
    vocab_size, hidden_size, num_tokens = 64, 16, 4
    lp = _build_processor(vocab_size)
    lp.head_dtype = torch.float32

    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)

    logits = lp._get_logits(hidden_states, _FakeLmHead(weight), None)

    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())
    torch.testing.assert_close(logits, expected)


def test_non_fp32_head_dtype_uses_cast_path(default_vllm_config):
    # head_dtype != fp32 must not hit the CUDA out_dtype-mm fast path
    # (torch.mm only supports fp32 out for fp16/bf16 inputs); the cast path
    # handles any dtype.
    vocab_size, hidden_size = 64, 16
    lp = _build_processor(vocab_size)
    lp.head_dtype = torch.float16

    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)

    logits = lp._get_logits(hidden_states, _FakeLmHead(weight), None)

    assert logits.dtype == torch.float16
    expected = torch.nn.functional.linear(hidden_states.half(), weight.half())
    torch.testing.assert_close(logits, expected)


def test_head_dtype_equal_to_model_dtype_uses_quant_method(default_vllm_config):
    from unittest import mock

    vocab_size, hidden_size = 64, 16
    lp = _build_processor(vocab_size)
    lp.head_dtype = torch.bfloat16

    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lm_head = _FakeLmHead(weight)

    with mock.patch.object(
        lm_head.quant_method,
        "apply",
        side_effect=lambda layer, x, bias=None: torch.nn.functional.linear(
            x, layer.weight, bias
        ),
    ) as apply_mock:
        logits = lp._get_logits(hidden_states, lm_head, None)

    apply_mock.assert_called_once()
    assert logits.dtype == torch.bfloat16


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Exercises the torch.mm(out_dtype=...) device fast path, "
    "available on CUDA and ROCm.",
)
def test_fp32_head_uses_mm_fast_path_on_device(default_vllm_config):
    # On ROCm, current_platform.is_cuda() is False, so this previously fell
    # through to the cast path (F.linear) instead of torch.mm(out_dtype=...),
    # even though ROCm supports the out_dtype mm via its non-Lt GEMM path.
    from unittest import mock

    vocab_size, hidden_size, num_tokens = 64, 16, 4
    lp = _build_processor(vocab_size)
    lp.head_dtype = torch.float32

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device="cuda")

    with mock.patch(
        "vllm.model_executor.layers.logits_processor.F.linear"
    ) as linear_mock:
        logits = lp._get_logits(hidden_states, _FakeLmHead(weight), None)

    linear_mock.assert_not_called()
    assert logits.dtype == torch.float32
    expected = torch.nn.functional.linear(hidden_states.float(), weight.float())
    torch.testing.assert_close(logits, expected)


def test_fp32_head_rejects_quantized_lm_head(default_vllm_config):
    lp = _build_processor(64)
    lp.head_dtype = torch.float32
    lm_head = _FakeLmHead(torch.randn(64, 16, dtype=torch.bfloat16), quantized=True)

    with pytest.raises(ValueError, match="unquantized"):
        lp._get_logits(torch.randn(4, 16, dtype=torch.bfloat16), lm_head, None)


def test_get_top_tokens_honors_head_dtype(default_vllm_config):
    # The spec-decode local-argmax path (get_top_tokens) must run the lm_head
    # in head_dtype too, not just _get_logits.
    import types
    from unittest import mock

    vocab_size, hidden_size = 64, 16
    lp = _build_processor(vocab_size)
    lp.head_dtype = torch.float32

    hidden_states = torch.randn(4, hidden_size, dtype=torch.bfloat16)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16)
    lm_head = _FakeLmHead(
        weight,
        shard_indices=types.SimpleNamespace(
            num_org_vocab_padding=0, org_vocab_start_index=0
        ),
    )

    with mock.patch(
        "vllm.model_executor.layers.logits_processor."
        "get_tensor_model_parallel_world_size",
        return_value=1,
    ):
        top = lp.get_top_tokens(lm_head, hidden_states, None)

    expected = torch.nn.functional.linear(hidden_states.float(), weight.float()).argmax(
        dim=-1
    )
    assert torch.equal(top, expected)


@pytest.mark.core_model
def test_fp32_head_e2e_no_nan():
    """An fp32 head produces finite logprobs end-to-end.

    Runs on the default (v2) model runner and exercises the
    processed_logprobs path, which forces the native sampler and is where a
    non-contiguous fp32 logits row previously produced NaN.
    """
    llm = LLM(
        model="facebook/opt-125m",
        hf_overrides={"head_dtype": "float32"},
        logprobs_mode="processed_logprobs",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )
    sampling_params = SamplingParams(
        temperature=1.0, top_p=0.95, top_k=50, max_tokens=32, logprobs=5, seed=0
    )
    outputs = llm.generate(
        ["The capital of France is", "Once upon a time,"], sampling_params
    )

    for output in outputs:
        for completion in output.outputs:
            for token_id, position in zip(completion.token_ids, completion.logprobs):
                # The sampled token survived filtering, so its logprob is finite.
                assert math.isfinite(position[token_id].logprob)
                # No returned logprob is NaN.
                assert not any(math.isnan(lp.logprob) for lp in position.values())

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.v1.worker.gpu.spec_decode.draft_lm_head import (
    E2M1_VALUES,
    NVFP4_GROUP_SIZE,
    QuantizedDraftLMHead,
    quantize_nvfp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Marlin needs CUDA"
)

# Not a multiple of the vocab padding, so the head has padded rows.
VOCAB, HIDDEN = 4000, 512


def _head(seed: int = 0) -> ParallelLMHead:
    head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(seed)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
    return head.to("cuda", torch.bfloat16)


def _dequantize(weight: torch.Tensor, quantization: str) -> torch.Tensor:
    if quantization == "fp8":
        qweight, scale = ops.scaled_fp8_quant(weight, use_per_token_if_dynamic=True)
        return qweight.float() * scale.bfloat16().float()
    packed, scales, global_scale = quantize_nvfp4(weight)
    codes = torch.stack([packed & 0xF, packed >> 4], -1).view(weight.shape[0], -1)
    values = torch.tensor(E2M1_VALUES, device=weight.device)[(codes & 7).long()]
    values = torch.where(codes >= 8, -values, values)
    step = scales.float().repeat_interleave(NVFP4_GROUP_SIZE, 1) * global_scale
    return values * step


@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_matches_dequantized_rows(default_vllm_config, quantization):
    head = _head()
    draft_head = QuantizedDraftLMHead(head, quantization)
    hidden = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    out = draft_head.quant_method.apply(draft_head, hidden).float()
    ref = hidden.float() @ _dequantize(head.weight, quantization).t()
    torch.testing.assert_close(out, ref, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize(
    ("quantization", "max_rel_err", "min_top1"),
    [("fp8", 0.05, 0.9), ("nvfp4", 0.15, 0.75)],
)
def test_tracks_bf16_head_through_logits_processor(
    default_vllm_config, quantization, max_rel_err, min_top1
):
    head = _head(1)
    draft_head = QuantizedDraftLMHead(head, quantization)
    logits_processor = LogitsProcessor(VOCAB)
    hidden = torch.randn(2048, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = logits_processor(head, hidden).float()
    out = logits_processor(draft_head, hidden).float()
    assert out.shape == ref.shape == (2048, VOCAB)
    rel_err = ((out - ref).norm() / ref.norm()).item()
    top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert rel_err < max_rel_err
    assert top1 > min_top1


def test_rejects_quantized_or_biased_head(default_vllm_config):
    head = _head()
    head.quant_method = object()
    with pytest.raises(ValueError, match="unquantized"):
        QuantizedDraftLMHead(head, "fp8")
    head = ParallelLMHead(VOCAB, HIDDEN, bias=True, disable_tp=True)
    with pytest.raises(ValueError, match="without bias"):
        QuantizedDraftLMHead(head.to("cuda", torch.bfloat16), "fp8")

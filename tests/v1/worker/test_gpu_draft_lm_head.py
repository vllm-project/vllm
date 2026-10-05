# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.lm_head import (
    quantized_lm_head_copy,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Marlin needs CUDA"
)

# Not a multiple of the vocab padding, so the head has padded rows.
VOCAB, HIDDEN = 4000, 512


@pytest.mark.parametrize(
    ("quantization", "max_rel_err", "min_top1"),
    [("fp8", 0.05, 0.9), ("nvfp4", 0.15, 0.75)],
)
def test_tracks_bf16_head_through_logits_processor(
    default_vllm_config, dist_init, monkeypatch, quantization, max_rel_err, min_top1
):
    # Online methods read the model dtype from the current config.
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
    head = head.to("cuda", torch.bfloat16)
    target_weight = head.weight.detach().clone()
    draft_head = quantized_lm_head_copy(head, quantization)

    assert isinstance(draft_head, ParallelLMHead)
    # The source head keeps its bf16 weight, and no bf16 copy is kept around.
    assert torch.equal(head.weight, target_weight)
    assert draft_head.weight.dtype != torch.bfloat16

    logits_processor = LogitsProcessor(VOCAB)
    hidden = torch.randn(2048, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ref = logits_processor(head, hidden).float()
    out = logits_processor(draft_head, hidden).float()
    assert out.shape == ref.shape == (2048, VOCAB)
    rel_err = ((out - ref).norm() / ref.norm()).item()
    top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
    assert rel_err < max_rel_err
    assert top1 > min_top1


def test_rejects_biased_head(default_vllm_config):
    head = ParallelLMHead(VOCAB, HIDDEN, bias=True, disable_tp=True).to(
        "cuda", torch.bfloat16
    )
    with pytest.raises(ValueError, match="without bias"):
        quantized_lm_head_copy(head, "fp8")

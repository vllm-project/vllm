# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.lm_head import (
    quantized_lm_head_copy,
    refresh_quantized_lm_heads,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)

# Not a multiple of the vocab padding, so the head has padded rows.
VOCAB, HIDDEN = 4000, 512
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Marlin needs CUDA"
)


def _target_head(monkeypatch, seed: int = 0) -> ParallelLMHead:
    # Online methods read the model dtype from the current config.
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.online.fp8.get_current_vllm_config",
        lambda: SimpleNamespace(model_config=model_config),
    )
    head = ParallelLMHead(VOCAB, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(seed)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen) * 0.02)
    return head.to("cuda", torch.bfloat16)


def _tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(module.named_parameters()) | dict(module.named_buffers())


@requires_cuda
@pytest.mark.parametrize(
    ("quantization", "max_rel_err", "min_top1"),
    [("fp8", 0.05, 0.9), ("nvfp4", 0.15, 0.75)],
)
def test_tracks_bf16_head_through_logits_processor(
    default_vllm_config, dist_init, monkeypatch, quantization, max_rel_err, min_top1
):
    head = _target_head(monkeypatch)
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


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_refresh_follows_target_update(
    default_vllm_config, dist_init, monkeypatch, quantization
):
    """After the target's weights change, the copy is re-derived in place: it
    equals a fresh quantization of the new target and keeps its storage."""
    head = _target_head(monkeypatch)
    drafter = torch.nn.Module()
    drafter.lm_head = quantized_lm_head_copy(head, quantization)
    ptrs = {k: t.data_ptr() for k, t in _tensors(drafter.lm_head).items()}

    new_weight = _target_head(monkeypatch, seed=1).weight.detach()
    head.weight.data.copy_(new_weight)
    refresh_quantized_lm_heads(drafter)

    expected = _tensors(quantized_lm_head_copy(head, quantization))
    for name, tensor in _tensors(drafter.lm_head).items():
        assert tensor.data_ptr() == ptrs[name]
        if name != "workspace":
            assert torch.equal(tensor, expected[name]), name
    assert torch.equal(head.weight, new_weight)


@requires_cuda
def test_layerwise_reload_of_drafter_keeps_copy(
    default_vllm_config, dist_init, monkeypatch
):
    """A drafter weight update reloads the drafter layerwise; the derived copy
    must come back unchanged, not be re-quantized."""
    head = _target_head(monkeypatch)
    drafter = torch.nn.Module()
    record_metadata_for_reloading(drafter)
    drafter.lm_head = quantized_lm_head_copy(head, "nvfp4")
    before = {k: t.clone() for k, t in _tensors(drafter.lm_head).items()}

    def fail(*args, **kwargs):
        raise AssertionError("the copy was re-quantized")

    monkeypatch.setattr(
        drafter.lm_head.quant_method, "process_weights_after_loading", fail
    )
    initialize_layerwise_reload(drafter)
    finalize_layerwise_reload(drafter, None)

    after = _tensors(drafter.lm_head)
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        assert torch.equal(after[name], tensor), name

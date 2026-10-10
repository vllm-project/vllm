# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
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
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.v1.worker.gpu_worker import Worker

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
    equals a fresh quantization of the new target, keeps its storage, and
    frees the temporary copy without waiting for the cycle collector."""
    head = _target_head(monkeypatch)
    drafter = torch.nn.Module()
    drafter.lm_head = quantized_lm_head_copy(head, quantization)
    ptrs = {k: t.data_ptr() for k, t in _tensors(drafter.lm_head).items()}

    new_weight = _target_head(monkeypatch, seed=1).weight.detach()
    head.weight.data.copy_(new_weight)
    gc.disable()
    try:
        allocated = torch.accelerator.memory_allocated()
        refresh_quantized_lm_heads(drafter)
        assert torch.accelerator.memory_allocated() == allocated
    finally:
        gc.enable()

    expected = _tensors(quantized_lm_head_copy(head, quantization))
    for name, tensor in _tensors(drafter.lm_head).items():
        assert tensor.data_ptr() == ptrs[name]
        if name != "workspace":
            assert torch.equal(tensor, expected[name]), name
    assert torch.equal(head.weight, new_weight)


def _draft_update_worker(vllm_config, drafter: torch.nn.Module, **engine) -> Worker:
    worker = object.__new__(Worker)
    worker.vllm_config = vllm_config
    worker.weight_transfer_engine = SimpleNamespace(
        supports_draft_weight_update=True,
        reset_weight_update_target=lambda: None,
        **engine,
    )
    worker.model_runner = SimpleNamespace(get_draft_model=lambda: drafter)
    worker._weight_update_active = False
    worker._set_draft_weight_update_target = lambda: None
    return worker


@requires_cuda
def test_draft_weight_update_refreshes_copy(
    default_vllm_config, dist_init, monkeypatch
):
    """A draft weight update can write a target weight the drafter shares, such
    as an embedding tied to the lm_head; the copy must follow it."""
    head = _target_head(monkeypatch)
    drafter = torch.nn.Module()
    drafter.lm_head = quantized_lm_head_copy(head, "fp8")
    worker = _draft_update_worker(
        default_vllm_config,
        drafter,
        start_weight_update=lambda: None,
        finish_weight_update=lambda: None,
    )

    worker.start_draft_weight_update()
    head.weight.data.copy_(_target_head(monkeypatch, seed=1).weight)
    worker.finish_weight_update()

    expected = _tensors(quantized_lm_head_copy(head, "fp8"))
    for name, tensor in _tensors(drafter.lm_head).items():
        assert torch.equal(tensor, expected[name]), name


@requires_cuda
@pytest.mark.parametrize("quantization", ["fp8", "nvfp4"])
def test_draft_weight_update_loads_lm_head(
    default_vllm_config, dist_init, monkeypatch, quantization
):
    """A draft update carrying lm_head.weight loads the shared target head, as
    without quantization, and the copy is re-derived from it in place."""
    head = _target_head(monkeypatch)
    record_metadata_for_reloading(head)
    drafter = torch.nn.Module()
    draft_head = quantized_lm_head_copy(head, quantization)
    drafter.lm_head = draft_head
    ptrs = {k: t.data_ptr() for k, t in _tensors(draft_head).items()}
    new_weight = torch.randn(VOCAB, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def load(update_info):
        param = drafter.get_parameter("lm_head.weight")
        getattr(param, "weight_loader", default_weight_loader)(param, new_weight)

    worker = _draft_update_worker(
        default_vllm_config,
        drafter,
        start_weight_update=lambda: initialize_layerwise_reload(drafter),
        update_weights=load,
        finish_weight_update=lambda: finalize_layerwise_reload(drafter, None),
    )
    worker.start_draft_weight_update()
    worker.update_weights({})
    worker.finish_weight_update()

    assert drafter.lm_head is draft_head
    assert torch.equal(head.weight[:VOCAB], new_weight)
    expected = _tensors(quantized_lm_head_copy(head, quantization))
    for name, tensor in _tensors(draft_head).items():
        assert tensor.data_ptr() == ptrs[name]
        if name != "workspace":
            assert torch.equal(tensor, expected[name]), name
    hidden = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    assert LogitsProcessor(VOCAB)(draft_head, hidden).isfinite().all()


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

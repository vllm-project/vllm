# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selective token adapters must match replacing the corresponding base rows.

These small layer-operation tests exercise batching, slot reuse, and graph
replay without loading a model; model parity is covered separately.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm.lora.ops.trainable_tokens import (
    TrainableTokensBuffer,
    replace_token_embeddings,
    replace_token_logits,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
pytestmark = pytest.mark.skip_global_cleanup


def _buffer(device, dtype=torch.float32):
    buffer = TrainableTokensBuffer(2, 3, 33, dtype, torch.device(device))
    buffer.set(0, torch.tensor([2, 7]), torch.arange(66).view(2, 33) / 100)
    buffer.set(1, torch.tensor([2]), -torch.arange(33).view(1, 33) / 100)
    return buffer


@torch.inference_mode()
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("layer_kind", ["embedding", "head"])
def test_bfloat16_replacement_rows_keep_their_range_with_float16_lora(
    device, layer_kind, monkeypatch
):
    """A finite BF16 learned row must not overflow in FP16 LoRA storage."""
    from vllm.config.lora import LoRAConfig
    from vllm.lora.layers import (
        LogitsProcessorWithLoRA,
        VocabParallelEmbeddingWithLoRA,
    )

    base = torch.nn.Embedding(11, 33, device=device, dtype=torch.bfloat16)
    base.weight.zero_()
    config = LoRAConfig(
        max_lora_rank=8,
        lora_dtype=torch.float16,
        max_lora_trainable_tokens=1,
    )
    indices = torch.zeros(1, dtype=torch.long, device=device)
    no_lora = SimpleNamespace(
        _embeddings_indices=indices,
        _token_lora_indices=indices,
        _sampler_indices=indices,
        add_lora_embedding=lambda output, *args, **kwargs: output,
        add_lora_logits=lambda output, *args, **kwargs: output,
    )
    if layer_kind == "embedding":
        base.org_vocab_size = 11
        layer = VocabParallelEmbeddingWithLoRA(base)
    else:
        monkeypatch.setattr(
            "vllm.lora.layers.logits_processor.get_tensor_model_parallel_rank",
            lambda: 0,
        )
        monkeypatch.setattr(
            "vllm.lora.layers.logits_processor.get_tensor_model_parallel_world_size",
            lambda: 1,
        )
        processor = SimpleNamespace(
            vocab_size=11,
            _apply_head=lambda head, hidden, bias: F.linear(
                hidden.float(), head.weight.float(), bias
            ),
            _gather_logits=lambda logits: logits,
        )
        layer = LogitsProcessorWithLoRA(
            processor, 33, torch.bfloat16, torch.device(device), None
        )
    layer.create_lora_weights(1, config)
    layer.set_mapping(no_lora)
    saved_row = torch.full((1, 33), 1e5, dtype=torch.bfloat16, device=device)
    layer.set_trainable_tokens(0, torch.tensor([2]), saved_row)

    if layer_kind == "embedding":
        output = layer(torch.tensor([2], device=device))
        expected = saved_row
    else:
        hidden = torch.ones((1, 33), dtype=torch.bfloat16, device=device)
        output = layer._get_logits(hidden, base)
        expected = torch.zeros((1, 11), device=device)
        expected[0, 2] = saved_row.float().sum()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_embedding_rows_are_replaced_per_adapter_and_survive_slot_reuse(device):
    """The same token uses each adapter's absolute row, while base rows survive."""
    buffer = _buffer(device)
    tokens = torch.tensor([2, 7, 2, 2, 5, 7], device=device)
    adapters = torch.tensor([0, 0, 1, -1, 1, 1], device=device)
    output = torch.full((6, 66), 9.0, device=device)[:, ::2]
    expected = output.clone()
    expected[0] = buffer.weights[0, 0]
    expected[1] = buffer.weights[0, 1]
    expected[2] = buffer.weights[1, 0]

    replace_token_embeddings(output, tokens, adapters, buffer.token_ids, buffer.weights)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    buffer.reset(0)
    buffer.set(1, torch.tensor([7]), torch.full((1, 33), 4.0))
    output.fill_(9)
    expected.fill_(9)
    expected[5].fill_(4)
    replace_token_embeddings(output, tokens, adapters, buffer.token_ids, buffer.weights)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_logits_match_projection_through_replaced_rows(device, dtype, with_bias):
    """Only selected columns change, with independent sampler adapter metadata."""
    torch.manual_seed(17)
    buffer = _buffer(device, dtype)
    hidden = torch.randn((6, 66), device=device, dtype=dtype)[:, ::2]
    base = torch.randn((11, 33), device=device, dtype=dtype)
    bias = torch.randn(11, device=device, dtype=dtype) if with_bias else None
    adapters = torch.tensor([1, -1, 0, 1, 0, -1], device=device)
    output = torch.empty((6, 22), device=device, dtype=dtype)[:, ::2]
    output.copy_(
        F.linear(
            hidden.float(), base.float(), bias.float() if bias is not None else None
        )
    )
    expected = output.clone()
    for row, adapter in enumerate(adapters.tolist()):
        if adapter < 0:
            continue
        valid = buffer.token_ids[adapter] >= 0
        ids = buffer.token_ids[adapter, valid]
        replaced = base.clone()
        replaced[ids] = buffer.weights[adapter, valid]
        expected[row] = F.linear(
            hidden[row].float(),
            replaced.float(),
            bias.float() if bias is not None else None,
        ).to(dtype)

    replace_token_logits(
        output, hidden, adapters, buffer.token_ids, buffer.weights, bias
    )
    torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_padding_adapter_indices_do_not_access_replacement_buffers(device):
    buffer = _buffer(device)
    adapters = torch.tensor([-1, 2, 2**32], device=device)
    tokens = torch.tensor([2, 2, 2], device=device)
    embeddings = torch.full((3, 33), 4.0, device=device)
    hidden = torch.ones((3, 33), device=device)
    logits = torch.full((3, 11), 5.0, device=device)
    replace_token_embeddings(
        embeddings, tokens, adapters, buffer.token_ids, buffer.weights
    )
    replace_token_logits(
        logits, hidden, adapters, buffer.token_ids, buffer.weights, None
    )
    torch.testing.assert_close(embeddings, torch.full_like(embeddings, 4))
    torch.testing.assert_close(logits, torch.full_like(logits, 5))


@pytest.mark.parametrize("device", DEVICES)
def test_empty_batch_is_supported(device):
    buffer = _buffer(device)
    empty_indices = torch.empty(0, dtype=torch.long, device=device)
    embeddings = torch.empty((0, 33), device=device)
    logits = torch.empty((0, 11), device=device)
    replace_token_embeddings(
        embeddings, empty_indices, empty_indices, buffer.token_ids, buffer.weights
    )
    replace_token_logits(
        logits, embeddings, empty_indices, buffer.token_ids, buffer.weights, None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs require CUDA")
def test_graph_replay_reads_updated_adapters_and_replacement_rows():
    """Graph capture must not bake in the active adapter or its saved rows."""
    buffer = _buffer("cuda")
    tokens = torch.tensor([2, 7], device="cuda")
    adapters = torch.tensor([-1, -1], device="cuda")
    embeddings = torch.zeros((2, 33), device="cuda")
    hidden = torch.ones((2, 33), device="cuda")
    logits = torch.zeros((2, 11), device="cuda")

    def run():
        replace_token_embeddings(
            embeddings, tokens, adapters, buffer.token_ids, buffer.weights
        )
        replace_token_logits(
            logits, hidden, adapters, buffer.token_ids, buffer.weights, None
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    adapters.copy_(torch.tensor([1, 0], device="cuda"))
    graph.replay()
    torch.testing.assert_close(embeddings[0], buffer.weights[1, 0])
    torch.testing.assert_close(embeddings[1], buffer.weights[0, 1])
    torch.testing.assert_close(logits[0, 2], buffer.weights[1, 0].sum())

    buffer.reset(0)
    buffer.set(1, torch.tensor([2]), torch.full((1, 33), 3.0))
    embeddings.zero_()
    logits.zero_()
    graph.replay()
    torch.testing.assert_close(embeddings[0], torch.full((33,), 3.0, device="cuda"))
    torch.testing.assert_close(embeddings[1], torch.zeros(33, device="cuda"))
    torch.testing.assert_close(logits[0, 2], torch.tensor(99.0, device="cuda"))
    torch.testing.assert_close(logits[1], torch.zeros(11, device="cuda"))

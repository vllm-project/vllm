# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 VLM non-materialized generator streaming."""

import torch


def simulate_stream_reordered(weights_stream):
    """Simulate _stream_reordered generator logic from
    DeepseekV41ForCausalLM.load_weights."""
    non_lm_weights: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights_stream:
        if name.startswith("language_model."):
            yield name, tensor
        else:
            non_lm_weights.append((name, tensor.clone()))

    yield from non_lm_weights


def test_stream_reordered_lazy_consumption():
    """Verify pulling the first LM weight does not eagerly consume all input
    weights."""
    consumed_count = 0

    def generating_weights():
        nonlocal consumed_count
        weights = [
            ("vision.patch_embed.weight", torch.randn(4, 4)),
            ("language_model.layers.0.weight", torch.randn(4, 4)),
            ("vision.encoder.layers.0.weight", torch.randn(4, 4)),
            ("language_model.layers.1.weight", torch.randn(4, 4)),
            ("aligner.proj.weight", torch.randn(4, 4)),
            ("language_model.layers.2.weight", torch.randn(4, 4)),
        ]
        for item in weights:
            consumed_count += 1
            yield item

    stream = simulate_stream_reordered(generating_weights())

    # Pull first item
    first_name, _ = next(stream)

    # First yielded item must be the first LM layer encountered
    assert first_name == "language_model.layers.0.weight"
    # It only needed to consume up to item 2 (vision.patch_embed and
    # language_model.layers.0)
    assert consumed_count == 2


def test_stream_reordered_ordering_guarantee():
    """Verify all language_model weights precede non-LM weights while
    preserving relative order."""
    input_weights = [
        ("vision.patch_embed.weight", torch.tensor([1.0])),
        ("language_model.layers.0.weight", torch.tensor([2.0])),
        ("aligner.proj.weight", torch.tensor([3.0])),
        ("language_model.layers.1.weight", torch.tensor([4.0])),
        ("image_embed.weight", torch.tensor([5.0])),
        ("language_model.layers.2.weight", torch.tensor([6.0])),
    ]

    reordered = list(simulate_stream_reordered(input_weights))
    names = [name for name, _ in reordered]

    expected_order = [
        "language_model.layers.0.weight",
        "language_model.layers.1.weight",
        "language_model.layers.2.weight",
        "vision.patch_embed.weight",
        "aligner.proj.weight",
        "image_embed.weight",
    ]

    assert names == expected_order


def test_stream_reordered_tensor_cloning():
    """Verify non-LM tensors are cloned to isolate against in-place upstream
    mutations."""
    mutable_tensor = torch.tensor([10.0, 20.0])
    input_weights = [
        ("vision.head.weight", mutable_tensor),
        ("language_model.head.weight", torch.tensor([30.0])),
    ]

    stream = simulate_stream_reordered(input_weights)

    # Consume language_model weight
    name_lm, _ = next(stream)
    assert name_lm == "language_model.head.weight"

    # Mutate the input tensor in place before consuming the buffered non-LM weight
    mutable_tensor.add_(100.0)

    # Consume buffered non-LM weight
    name_non_lm, tensor_non_lm = next(stream)
    assert name_non_lm == "vision.head.weight"

    # The cloned tensor must retain original values
    assert torch.equal(tensor_non_lm, torch.tensor([10.0, 20.0]))

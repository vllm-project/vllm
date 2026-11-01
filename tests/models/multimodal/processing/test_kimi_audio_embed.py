# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.kimi_audio import KimiAudioForConditionalGeneration


@pytest.mark.parametrize("container", [list, tuple, torch.stack])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_audio_matches_individual_requests(container, dtype):
    """Every request's audio must survive batching and chunked prefill."""
    embedding = torch.nn.Embedding(8, 4, dtype=dtype)
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding))
    )
    embed = KimiAudioForConditionalGeneration.embed_input_ids
    lengths = [2, 2] if container is torch.stack else [2, 3]
    audio = [torch.full((n, 4), float(i + 1)) for i, n in enumerate(lengths)]
    ids = [torch.arange(n + 2) for n in lengths]
    masks = [torch.tensor([False] + [True] * n + [False]) for n in lengths]
    expected = torch.cat(
        [
            embed(model, tokens, [features], is_multimodal=mask)
            for tokens, features, mask in zip(ids, audio, masks)
        ]
    )
    all_ids, all_masks = torch.cat(ids), torch.cat(masks)
    actual = embed(model, all_ids, container(audio), is_multimodal=all_masks)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[~all_masks], embedding(all_ids)[~all_masks])
    fused = (embedding(all_ids)[all_masks] + torch.cat(audio).to(dtype)) * (2**0.5)
    torch.testing.assert_close(actual[all_masks], fused)

    # The runner passes only the audio rows in the scheduled token window.
    split = 2
    first_count = int(all_masks[:split].sum())
    chunks = [
        embed(
            model,
            all_ids[:split],
            [audio[0][:first_count]],
            is_multimodal=all_masks[:split],
        ),
        embed(
            model,
            all_ids[split:],
            [audio[0][first_count:], audio[1]],
            is_multimodal=all_masks[split:],
        ),
    ]
    torch.testing.assert_close(torch.cat(chunks), expected)


@pytest.mark.parametrize("audio", [None, []])
def test_text_only_embeddings_unchanged(audio):
    embedding = torch.nn.Embedding(8, 4)
    model = SimpleNamespace(
        language_model=SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding))
    )
    ids = torch.arange(4)
    actual = KimiAudioForConditionalGeneration.embed_input_ids(model, ids, audio)
    torch.testing.assert_close(actual, embedding(ids))

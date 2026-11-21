# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MTP head drafts from the target's input stream, where image and video
positions hold the vision embeddings."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import supports_multimodal_embeddings
from vllm.models.nemotron_shortcut.nvidia.mtp import NemotronShortcutMTP


def test_mtp_merges_multimodal_embeddings():
    assert supports_multimodal_embeddings(NemotronShortcutMTP)
    table = torch.arange(8, dtype=torch.float32).repeat(3, 1).T  # row i holds i
    drafter = SimpleNamespace(get_input_embeddings=lambda input_ids: table[input_ids])
    input_ids = torch.tensor([1, 5, 6, 2])

    text_only = NemotronShortcutMTP.embed_input_ids(drafter, input_ids)
    assert torch.equal(text_only, table[input_ids])

    mm_embeds = [torch.full((2, 3), -1.0)]
    is_multimodal = torch.tensor([False, True, True, False])
    merged = NemotronShortcutMTP.embed_input_ids(
        drafter, input_ids, mm_embeds, is_multimodal=is_multimodal
    )
    assert torch.equal(merged[[0, 3]], table[[1, 2]])
    assert torch.equal(merged[[1, 2]], mm_embeds[0])

    with pytest.raises(ValueError, match="is_multimodal"):
        NemotronShortcutMTP.embed_input_ids(drafter, input_ids, mm_embeds)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.clip import dual_encoder_has_text_tokens

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_text_only_batch_runs_text_encoder():
    assert dual_encoder_has_text_tokens(False, None)
    assert dual_encoder_has_text_tokens(False, torch.zeros(4, dtype=torch.bool))


def test_vision_only_batch_skips_text_encoder():
    is_multimodal = torch.ones(8, dtype=torch.bool)
    assert not dual_encoder_has_text_tokens(True, is_multimodal)


def test_mixed_batch_runs_text_encoder():
    # Image tokens followed by a text sequence: the old batch-wide flag
    # treated this as vision-only and skipped the text encoder (#53091).
    is_multimodal = torch.tensor([True, True, True, False, False, False])
    assert dual_encoder_has_text_tokens(True, is_multimodal)


def test_missing_token_mask_with_mm_embeddings_is_vision_only():
    assert not dual_encoder_has_text_tokens(True, None)

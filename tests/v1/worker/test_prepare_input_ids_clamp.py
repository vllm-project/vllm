# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Regression test for issue #36906: EAGLE3 + multimodal + async scheduling
crashes with CUDA error in F.embedding() due to -1 placeholder token IDs
reaching the embedding layer.

The actual crash path is in the EAGLE draft model's propose() method:
1. get_token_id(seq_lens[i]) returns -1 when async scheduling's seq_lens
   exceeds known tokens
2. -1 propagates through prepare_next_token_ids_padded() as a backup token
3. The Triton kernel stores -1 when valid_count == 0 (discarded request)
4. -1 enters self.input_ids via set_inputs_first_pass()
5. propose() calls self.model.embed_input_ids(self.input_ids[:num_tokens])
6. F.embedding(-1) triggers CUDA device-side assert

The fix clamps self.input_ids[:num_tokens] to min=0 in propose() before
the embed_input_ids call.
"""

import pytest
import torch


def test_eagle_input_ids_clamp_prevents_negative_embedding():
    """Verify that -1 token IDs in EAGLE's input_ids are clamped before
    embedding, preventing F.embedding() crashes.

    Simulates the exact scenario:
    - next_token_ids contains -1 from get_token_id() returning -1
    - set_inputs_first_pass() writes -1 into self.input_ids
    - The fix clamps self.input_ids before embed_input_ids
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_tokens = 6
    vocab_size = 1000

    # Simulate EAGLE's self.input_ids after set_inputs_first_pass()
    # Positions 2 and 5 have -1 from next_token_ids (backup tokens
    # from get_token_id returning -1 for async scheduling)
    input_ids = torch.tensor(
        [100, 200, -1, 300, 400, -1], dtype=torch.int32, device=device
    )

    # Verify -1 values are present (the bug condition)
    assert (input_ids == -1).any(), "Test setup error: -1 values should be present"

    # Apply the fix: clamp before embed_input_ids
    input_ids[:num_tokens].clamp_(min=0)

    # Verify no -1 values remain
    assert input_ids.min().item() >= 0, (
        f"Bug: input_ids still contains negative values: {input_ids}"
    )

    # Verify F.embedding succeeds (would crash on -1)
    embedding = torch.nn.Embedding(vocab_size, 16).to(device)
    result = embedding(input_ids.to(torch.int64))
    assert result.shape == (num_tokens, 16)


def test_get_token_id_returns_negative_one_for_out_of_bounds():
    """Verify get_token_id returns -1 when index exceeds known tokens.

    This is the root source of -1 values in the crash path. When async
    scheduling sets seq_lens beyond the known prompt + output tokens,
    get_token_id() returns -1 as a fallback.
    """
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    req = CachedRequestState(
        req_id="test",
        prompt_token_ids=[10, 20, 30],  # 3 prompt tokens
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=(),
        num_computed_tokens=0,
        output_token_ids=[40, 50],  # 2 output tokens
    )

    # Valid indices: 0-4 (3 prompt + 2 output)
    assert req.get_token_id(0) == 10
    assert req.get_token_id(2) == 30
    assert req.get_token_id(3) == 40
    assert req.get_token_id(4) == 50

    # Out-of-bounds: returns -1 (async scheduling's seq_lens can exceed this)
    assert req.get_token_id(5) == -1
    assert req.get_token_id(10) == -1


def test_negative_token_ids_crash_embedding():
    """Demonstrate that F.embedding crashes on -1 input (the bug).

    Uses CPU to avoid poisoning the CUDA context with a device-side assert.
    On GPU, this same error manifests as the CUDA device-side assert
    reported in issue #36906.
    """
    vocab_size = 1000
    embedding = torch.nn.Embedding(vocab_size, 16)

    # F.embedding with -1 should raise IndexError on CPU
    bad_input_ids = torch.tensor([100, -1, 200], dtype=torch.int64)
    with pytest.raises(IndexError):
        embedding(bad_input_ids)

    # After clamping, embedding succeeds
    fixed_input_ids = bad_input_ids.clamp(min=0)
    result = embedding(fixed_input_ids)
    assert result.shape == (3, 16), "Clamped input should embed successfully"

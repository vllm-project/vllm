# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the buffer-based decode-only steering implementation.

The steering math in each decoder layer is:
    hidden_states + steering_decode_mask[:N] * steering_vector

where ``steering_decode_mask`` is a persistent buffer updated in-place by
the model runner before each forward pass (1.0 for decode positions, 0.0
for prefill/padding).
"""

import torch


class TestBufferBasedSteering:
    """Tests verify the decode mask + steering vector math directly."""

    def test_steers_only_decode_tokens(self):
        """Only the first num_decode_tokens rows are modified."""
        batch_size, hidden_size = 6, 4
        hidden = torch.zeros(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size)
        mask = torch.zeros(batch_size, 1)
        mask[:3] = 1.0

        result = hidden + mask[:hidden.shape[0]] * steering_vec

        expected = torch.zeros(batch_size, hidden_size)
        expected[:3] = 1.0
        assert torch.allclose(result, expected), (
            f"Expected first 3 rows steered.\nGot:\n{result}"
        )

    def test_zero_mask_steers_nothing(self):
        """All-zero mask (prefill-only or no-context) produces no change."""
        batch_size, hidden_size = 4, 4
        hidden = torch.ones(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size) * 5.0
        mask = torch.zeros(batch_size, 1)

        result = hidden + mask[:hidden.shape[0]] * steering_vec
        assert torch.allclose(result, hidden)

    def test_all_decode_tokens(self):
        """When all tokens are decodes, all rows are steered."""
        batch_size, hidden_size = 4, 4
        hidden = torch.zeros(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size) * 2.0
        mask = torch.ones(batch_size, 1)

        result = hidden + mask[:hidden.shape[0]] * steering_vec

        expected = torch.ones(batch_size, hidden_size) * 2.0
        assert torch.allclose(result, expected)

    def test_zero_vector_noop(self):
        """A zero steering vector produces no change regardless of mask."""
        batch_size, hidden_size = 4, 4
        hidden = torch.randn(batch_size, hidden_size)
        steering_vec = torch.zeros(1, hidden_size)
        mask = torch.ones(batch_size, 1)

        result = hidden + mask[:hidden.shape[0]] * steering_vec
        assert torch.allclose(result, hidden)

    def test_output_shape_and_dtype(self):
        """Output shape and dtype must match input."""
        batch_size, hidden_size = 6, 8
        hidden = torch.randn(batch_size, hidden_size, dtype=torch.float32)
        steering_vec = torch.ones(1, hidden_size, dtype=torch.float32)
        mask = torch.zeros(10, 1)
        mask[:3] = 1.0

        result = hidden + mask[:hidden.shape[0]] * steering_vec
        assert result.shape == hidden.shape
        assert result.dtype == hidden.dtype

    def test_mask_larger_than_batch(self):
        """Mask buffer can be larger than actual batch (sliced to fit)."""
        batch_size, hidden_size = 4, 4
        hidden = torch.zeros(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size)
        mask = torch.zeros(100, 1)
        mask[:2] = 1.0

        result = hidden + mask[:hidden.shape[0]] * steering_vec
        expected = torch.zeros(batch_size, hidden_size)
        expected[:2] = 1.0
        assert torch.allclose(result, expected)

    def test_inplace_mask_update_visible(self):
        """In-place mask updates (simulating model runner) are visible."""
        batch_size, hidden_size = 4, 4
        hidden = torch.zeros(batch_size, hidden_size)
        steering_vec = torch.ones(1, hidden_size)
        mask = torch.zeros(10, 1)

        # First forward: no decode tokens
        result1 = hidden + mask[:hidden.shape[0]] * steering_vec
        assert torch.allclose(result1, hidden)

        # Simulate model runner updating mask in-place
        mask.zero_()
        mask[:3].fill_(1.0)

        # Second forward: 3 decode tokens steered
        result2 = hidden + mask[:hidden.shape[0]] * steering_vec
        expected = torch.zeros(batch_size, hidden_size)
        expected[:3] = 1.0
        assert torch.allclose(result2, expected)

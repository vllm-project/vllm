# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import torch

from vllm.compilation.backends import make_copy_and_call


class TestMakeCopyAndCall:
    """Tests for make_copy_and_call.

    In production, callable_fn is split_gm (a torch.fx.GraphModule) and
    input_buffers are pre-allocated at max shape via example_inputs[x].clone().
    sym_tensor_indices are the positions of tensors with symbolic (dynamic) dim 0.
    """

    MAX_NUM_TOKENS = 16
    HIDDEN_SIZE = 64

    @pytest.mark.parametrize(
        "num_tokens",
        [MAX_NUM_TOKENS - 1, MAX_NUM_TOKENS],
        ids=["partial", "full_buffer"],
    )
    def test_sym_tensor_copied_as_buffer_view(self, num_tokens: int):
        """Sym tensor is copied into buffer; callable_fn receives a view of it.
        Covers num_tokens < buffer_size and num_tokens == buffer_size (boundary)."""

        input_buffer = torch.randn(self.MAX_NUM_TOKENS, self.HIDDEN_SIZE)
        callable_fn = MagicMock()
        copy_and_call = make_copy_and_call(
            sym_tensor_indices=[0],
            input_buffers=[input_buffer],
            callable_fn=callable_fn,
        )

        runtime_tensor = torch.randn(num_tokens, self.HIDDEN_SIZE)
        copy_and_call(runtime_tensor)

        # Check: runtime_tensor was correctly copied into the buffer
        assert torch.allclose(input_buffer[:num_tokens], runtime_tensor)

        # Check: callable_fn got a view into the buffer, not a fresh allocation
        assert (
            callable_fn.call_args_list[0].args[0].data_ptr() == input_buffer.data_ptr()
        )

    def test_oversize_input_raises(self):
        """Sequence longer than max buffer size must raise AssertionError."""

        input_buffer = torch.randn(self.MAX_NUM_TOKENS, self.HIDDEN_SIZE)
        callable_fn = MagicMock()
        copy_and_call = make_copy_and_call(
            sym_tensor_indices=[0],
            input_buffers=[input_buffer],
            callable_fn=callable_fn,
        )

        # Check: input with num_tokens > MAX_NUM_TOKENS raises an error
        with pytest.raises(AssertionError, match="exceeds static buffer size"):
            copy_and_call(torch.randn(self.MAX_NUM_TOKENS + 1, self.HIDDEN_SIZE))

        # Check: callable_fn was never reached
        callable_fn.assert_not_called()

    def test_second_call_overwrites_buffer(self):
        """Second call overwrites the buffer with new data."""

        input_buffer = torch.randn(self.MAX_NUM_TOKENS, self.HIDDEN_SIZE)
        callable_fn = MagicMock()
        copy_and_call = make_copy_and_call(
            sym_tensor_indices=[0],
            input_buffers=[input_buffer],
            callable_fn=callable_fn,
        )

        runtime_tensor_0 = torch.randn(8, self.HIDDEN_SIZE)
        runtime_tensor_1 = torch.randn(4, self.HIDDEN_SIZE)
        copy_and_call(runtime_tensor_0)
        copy_and_call(runtime_tensor_1)

        # Check: second call overwrote the first 4 rows with new data
        assert torch.allclose(input_buffer[:4], runtime_tensor_1)

        # Check: rows 4:8 still contain leftover data from the first call
        assert torch.allclose(input_buffer[4:8], runtime_tensor_0[4:8])

    def test_multiple_sym_tensors_with_static_arg(self):
        """Non-contiguous sym_tensor_indices: two sym tensors at positions 0 and 2,
        with a static tensor at position 1 that must pass through unchanged."""

        input_buffer_0 = torch.randn(self.MAX_NUM_TOKENS, self.HIDDEN_SIZE)
        input_buffer_2 = torch.randn(self.MAX_NUM_TOKENS, self.HIDDEN_SIZE)
        callable_fn = MagicMock()
        copy_and_call = make_copy_and_call(
            sym_tensor_indices=[0, 2],
            input_buffers=[input_buffer_0, input_buffer_2],
            callable_fn=callable_fn,
        )

        runtime_tensor_0 = torch.randn(4, self.HIDDEN_SIZE)
        passthrough_1 = torch.ones(self.MAX_NUM_TOKENS, self.MAX_NUM_TOKENS)
        runtime_tensor_2 = torch.randn(6, self.HIDDEN_SIZE)
        copy_and_call(runtime_tensor_0, passthrough_1, runtime_tensor_2)

        # Check: each runtime_tensor was copied into its respective buffer
        assert torch.allclose(input_buffer_0[:4], runtime_tensor_0)
        assert torch.allclose(input_buffer_2[:6], runtime_tensor_2)

        # Check: callable_fn received buffer views, not the original runtime_tensors
        assert (
            callable_fn.call_args_list[0].args[0].data_ptr()
            == input_buffer_0.data_ptr()
        )
        assert (
            callable_fn.call_args_list[0].args[2].data_ptr()
            == input_buffer_2.data_ptr()
        )

        # Check: passthrough arg at index 1 is passed through unchanged
        assert callable_fn.call_args_list[0].args[1] is passthrough_1

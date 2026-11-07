# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual batch overlap (DBO) in the V1 model runner's `UBatchWrapper`."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker import gpu_ubatch_wrapper as legacy_gpu_ubatch_wrapper
from vllm.v1.worker.ubatch_utils import UBatchSlice


@pytest.mark.parametrize(
    ("cudagraph_runtime_mode", "execution_method"),
    [
        (CUDAGraphMode.FULL, "_capture_ubatches"),
        (CUDAGraphMode.NONE, "_run_ubatches"),
    ],
)
def test_legacy_ubatch_contexts_slice_padding_mask(
    cudagraph_runtime_mode: CUDAGraphMode, execution_method: str
):
    """Replacement contexts preserve each microbatch's padding rows."""
    ubatch_slices = [
        UBatchSlice(slice(0, 2), slice(0, 3)),
        UBatchSlice(slice(2, 4), slice(3, 6)),
    ]
    is_padding = torch.tensor([False, True, False, True, False, True])
    forward_context = SimpleNamespace(
        batch_descriptor=object(),
        ubatch_slices=ubatch_slices,
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        attn_metadata=[{}, {}],
        slot_mapping={},
        dp_metadata=object(),
        is_padding=is_padding,
    )

    wrapper = object.__new__(legacy_gpu_ubatch_wrapper.UBatchWrapper)
    wrapper.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=2)
    )
    wrapper.comm_stream = object()
    wrapper.ready_barrier = object()
    wrapper.cudagraphs = {}
    wrapper.sm_control = nullcontext()
    wrapper.runnable = object()

    def make_contexts(*, forward_contexts, **kwargs):
        return [
            SimpleNamespace(id=i, forward_context=context)
            for i, context in enumerate(forward_contexts)
        ]

    def create_context(attn_metadata, vllm_config, **kwargs):
        return SimpleNamespace(attn_metadata=attn_metadata, **kwargs)

    model_inputs = {
        "input_ids": torch.arange(6),
        "positions": torch.arange(6),
        "inputs_embeds": None,
        "intermediate_tensors": None,
    }
    expected = object()
    with (
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "get_forward_context",
            return_value=forward_context,
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper.torch.cuda,
            "current_stream",
            return_value=object(),
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper.DPMetadata, "make", return_value=object()
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "create_forward_context",
            side_effect=create_context,
        ),
        patch.object(
            legacy_gpu_ubatch_wrapper,
            "make_ubatch_contexts",
            side_effect=make_contexts,
        ),
        patch.object(wrapper, execution_method, return_value=expected) as execute,
    ):
        result = wrapper(**model_inputs)

    assert result is expected
    contexts = [
        metadata.context.forward_context for metadata in execute.call_args.args[0]
    ]
    torch.testing.assert_close(contexts[0].is_padding, is_padding[:3])
    torch.testing.assert_close(contexts[1].is_padding, is_padding[3:])

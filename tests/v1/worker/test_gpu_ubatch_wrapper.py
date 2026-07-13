# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.config import CUDAGraphMode
from vllm.v1.worker import gpu_ubatch_wrapper
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.ubatch_utils import UBatchSlice


def test_ubatch_contexts_preserve_additional_forward_context(monkeypatch) -> None:
    additional_kwargs = {"cudagraph_warmup_mode": CUDAGraphMode.FULL}
    monkeypatch.setattr(
        gpu_ubatch_wrapper,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs=additional_kwargs),
    )

    def create_forward_context(*args, additional_kwargs, **kwargs):
        return SimpleNamespace(additional_kwargs=additional_kwargs)

    monkeypatch.setattr(
        gpu_ubatch_wrapper, "create_forward_context", create_forward_context
    )
    monkeypatch.setattr(
        gpu_ubatch_wrapper,
        "make_ubatch_contexts",
        lambda **kwargs: kwargs["forward_contexts"],
    )

    wrapper = object.__new__(UBatchWrapper)
    wrapper.vllm_config = SimpleNamespace()
    wrapper.comm_stream = SimpleNamespace()
    wrapper.ready_barrier = SimpleNamespace()
    ubatch_slices = [
        UBatchSlice(slice(0, 1), slice(0, 1)),
        UBatchSlice(slice(1, 2), slice(1, 2)),
    ]

    metadata = wrapper._make_ubatch_metadata(
        ubatch_slices=ubatch_slices,
        attn_metadata=None,
        slot_mapping=None,
        input_ids=torch.arange(2),
        positions=torch.arange(2),
        inputs_embeds=None,
        intermediate_tensors=None,
        compute_stream=SimpleNamespace(),
        dp_metadata=[None, None],
        batch_descriptor=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    contexts = [item.context for item in metadata]
    assert all(context.additional_kwargs == additional_kwargs for context in contexts)
    assert contexts[0].additional_kwargs is not contexts[1].additional_kwargs

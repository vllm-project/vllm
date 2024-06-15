import dataclasses
from typing import List, Tuple, Type

import torch

from vllm.attention import AttentionMetadata
from vllm.attention.backends.abstract import AttentionBackend
from vllm.model_executor import SamplingMetadata
from vllm.worker.model_input import GPUModelInputWithSamplingMetadata


class MockAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        pass

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        pass


def test_gpu_model_input():
    sampling_metadata = SamplingMetadata(
        ["seq_group"],
        "selected_token_indices",
        "categorized_sample_indices",
        "num_prompts",
    )
    attn_metadata = AttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decode_tokens=3,
        slot_mapping=torch.zeros(1),
    )
    model_input = GPUModelInputWithSamplingMetadata.new(
        num_seq_groups=10,
        sampling_metadata=sampling_metadata,
        attn_metadata=attn_metadata)

    assert isinstance(model_input, GPUModelInputWithSamplingMetadata)

    # Test round trip serialization.
    tensor_dict = model_input.as_broadcastable_tensor_dict()
    attn_backend = MockAttentionBackend()
    received_model_input = GPUModelInputWithSamplingMetadata.new(
        attn_backend=attn_backend, **tensor_dict)
    assert isinstance(received_model_input, GPUModelInputWithSamplingMetadata)

    # Broadcast should not contain empty values.
    for field in dataclasses.fields(model_input):
        if getattr(model_input, field.name) is None:
            assert field.name not in tensor_dict
    # Broadcast should contain all non-empty fields defined by the developer
    # for this input type.
    for field_name in model_input.broadcastable_fields:
        if getattr(model_input, field_name, None) is not None:
            assert field_name in tensor_dict

    # Check that received copy has correct values.
    for field in dataclasses.fields(AttentionMetadata):
        assert getattr(received_model_input.attn_metadata, field.name,
                       None) == getattr(attn_metadata, field.name, None)
    # For sampling metadata, only selected_token_indices is copied.
    assert (received_model_input.sampling_metadata.selected_token_indices ==
            sampling_metadata.selected_token_indices)
    assert received_model_input.sampling_metadata.seq_groups is None

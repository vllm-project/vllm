import dataclasses
from typing import List, Tuple, Type

import torch

from vllm.attention import AttentionMetadata, AttentionMetadataBuilder
from vllm.attention.backends.abstract import AttentionBackend
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.worker.embedding_model_runner import (
    ModelInputForGPUWithPoolingMetadata)
from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata


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
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        raise AttentionMetadataBuilder

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


def test_model_runner_input():
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
    model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=torch.ones(10),
        input_positions=torch.ones(10),
        sampling_metadata=sampling_metadata,
        attn_metadata=attn_metadata)

    assert isinstance(model_input, ModelInputForGPUWithSamplingMetadata)

    # Test round trip serialization.
    tensor_dict = model_input.as_broadcastable_tensor_dict()
    attn_backend = MockAttentionBackend()
    received_model_input = (
        ModelInputForGPUWithSamplingMetadata.from_broadcasted_tensor_dict(
            tensor_dict, attn_backend=attn_backend))
    # Check that received copy has correct values.
    assert isinstance(received_model_input,
                      ModelInputForGPUWithSamplingMetadata)
    assert received_model_input.input_tokens is not None
    assert (
        received_model_input.input_tokens == model_input.input_tokens).all()
    assert received_model_input.input_positions is not None
    assert (received_model_input.input_positions == model_input.input_positions
            ).all()
    assert received_model_input.multi_modal_kwargs is None
    assert (received_model_input.multi_modal_kwargs ==
            model_input.multi_modal_kwargs)
    assert received_model_input.lora_requests is None
    assert received_model_input.lora_requests == model_input.lora_requests
    assert received_model_input.lora_mapping is None
    assert received_model_input.lora_mapping == model_input.lora_mapping
    for field in dataclasses.fields(AttentionMetadata):
        assert getattr(received_model_input.attn_metadata, field.name,
                       None) == getattr(attn_metadata, field.name, None)
    # For sampling metadata, only selected_token_indices is copied.
    assert (received_model_input.sampling_metadata.selected_token_indices ==
            sampling_metadata.selected_token_indices)
    assert received_model_input.sampling_metadata.seq_groups is None


def test_embedding_model_runner_input():
    pooling_metadata = PoolingMetadata(
        seq_groups=[[0]],
        seq_data={},
        prompt_lens=[1],
    )
    attn_metadata = AttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decode_tokens=3,
        slot_mapping=torch.zeros(1),
    )
    model_input = ModelInputForGPUWithPoolingMetadata(
        input_tokens=torch.ones(10),
        input_positions=torch.ones(10),
        pooling_metadata=pooling_metadata,
        attn_metadata=attn_metadata)

    assert isinstance(model_input, ModelInputForGPUWithPoolingMetadata)

    # Test round trip serialization.
    tensor_dict = model_input.as_broadcastable_tensor_dict()
    attn_backend = MockAttentionBackend()
    received_model_input = (
        ModelInputForGPUWithPoolingMetadata.from_broadcasted_tensor_dict(
            tensor_dict, attn_backend=attn_backend))
    # Check that received copy has correct values.
    assert isinstance(received_model_input,
                      ModelInputForGPUWithPoolingMetadata)
    assert received_model_input.input_tokens is not None
    assert (
        received_model_input.input_tokens == model_input.input_tokens).all()
    assert received_model_input.input_positions is not None
    assert (received_model_input.input_positions == model_input.input_positions
            ).all()
    assert received_model_input.multi_modal_kwargs is None
    assert (received_model_input.multi_modal_kwargs ==
            model_input.multi_modal_kwargs)
    assert received_model_input.lora_requests is None
    assert received_model_input.lora_requests == model_input.lora_requests
    assert received_model_input.lora_mapping is None
    assert received_model_input.lora_mapping == model_input.lora_mapping
    for field in dataclasses.fields(AttentionMetadata):
        assert getattr(received_model_input.attn_metadata, field.name,
                       None) == getattr(attn_metadata, field.name, None)
    # Pooling metadata is not broadcast.
    assert received_model_input.pooling_metadata is None

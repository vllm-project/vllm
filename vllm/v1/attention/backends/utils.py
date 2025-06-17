# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import abc
import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.utils import (
    get_kv_connector_cache_layout)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    """

    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""
    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    num_reqs: int
    """Number of requests"""
    num_actual_tokens: int
    """Total number of tokens in batch"""
    max_query_len: int
    """Longest query in batch"""


M = TypeVar("M")


class AttentionMetadataBuilder(abc.ABC, Generic[M]):
    # Does this backend/builder support CUDA Graphs for attention.
    full_cudagraph_supported: ClassVar[bool] = False

    @abstractmethod
    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata) -> M:
        """
        Central method that builds attention metadata.
        Some builders (MLA) require reorder_batch to be called prior to build.
        """
        raise NotImplementedError

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        """
        Can this batch (with given metadata) use CUDA Graphs for attention.
        """
        return False

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata) -> M:
        """
        Build attention metadata for CUDA graph capture. Uses build by default.
        Subclasses that override this method should call self.build or
        super().build_for_cudagraph_capture.
        """
        return self.build(common_prefix_len=0,
                          common_attn_metadata=common_attn_metadata)

    def use_cascade_attention(
        self,
        common_prefix_len: int,
        query_lens: np.ndarray,
        num_query_heads: int,
        num_kv_heads: int,
        use_alibi: bool,
        use_sliding_window: bool,
        num_sms: int,
    ) -> bool:
        return False

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        """
        This method can reorder the batch if desired by the backend.
        :return: Has the batch been reordered (default False).
        """
        return False


def validate_kv_sharing_target(current_layer_name, target_layer_name,
                               static_forward_context):
    error_msg = (f"Specified KV sharing target layer for {current_layer_name} "
                 f"is not valid: target layer {target_layer_name} ")

    if current_layer_name == target_layer_name:
        raise ValueError(error_msg +
                         "cannot be the same as the current layer.")

    if target_layer_name not in static_forward_context:
        from vllm.model_executor.models.utils import extract_layer_index

        # If target layer name is not in the static fwd context, it means either
        # a) the target layer does not come BEFORE the current layer, or
        # b) the target layer is not an Attention layer that exists in the model
        current_layer_idx = extract_layer_index(current_layer_name)
        target_layer_idx = extract_layer_index(target_layer_name)
        if current_layer_idx <= target_layer_idx:
            raise ValueError(error_msg + "must come before the current layer.")
        else:
            raise ValueError(error_msg +
                             "is not a valid Attention layer in the model.")

    # Currently KV sharing is only supported between layers of the same type
    target_layer_attn_type = static_forward_context[
        target_layer_name].attn_type
    expected = static_forward_context[current_layer_name].attn_type
    if target_layer_attn_type != expected:
        raise ValueError(
            error_msg +
            f"must be the same type as the current layer ({expected}).")


@functools.lru_cache
def get_kv_cache_layout():
    # Override with format specified by the user.
    cache_layout = envs.VLLM_KV_CACHE_LAYOUT
    if cache_layout is None:
        cache_layout = get_kv_connector_cache_layout()
    else:
        logger.info_once("`FLASHINFER_KV_CACHE_LAYOUT` environment variable " \
        "detected. Setting KV cache layout to %s.", cache_layout)

    return cache_layout

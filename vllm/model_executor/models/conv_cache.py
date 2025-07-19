# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import VllmConfig
from vllm.model_executor.models.constant_size_cache import ConstantSizeCache


@dataclass
class ConvCacheParams:
    conv_state: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return ConvCacheParams(self.conv_state[layer_idx],
                               self.state_indices_tensor)


class ConvCacheManager(ConstantSizeCache):

    def __init__(self, vllm_config: VllmConfig, dtype: torch.dtype,
                 num_conv_layers: int, conv_state_shape: tuple[int, int]):

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        if not vllm_config.model_config.enforce_eager:
            max_batch_size = vllm_config.pad_for_cudagraph(max_batch_size)

        # Initialize parent class
        super().__init__(max_batch_size)

        # Note(pp): this is for the V0 runner.
        # assume conv_state = (dim, state_len).
        assert conv_state_shape[0] > conv_state_shape[1]
        conv_state = torch.empty(size=(num_conv_layers, max_batch_size) +
                                 (conv_state_shape[1], conv_state_shape[0]),
                                 dtype=dtype,
                                 device="cuda").transpose(-1, -2)
        self._lfm2_cache = conv_state

    @property
    def cache(self):
        return self._lfm2_cache

    def _copy_cache(self, from_index: int, to_index: int):
        for cache_t in self.cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index],
                                       non_blocking=True)

    def current_run_tensors(self, **kwargs) -> ConvCacheParams:
        """
        Return the tensors for the current run's conv state.
        """
        cache_tensor, state_indices_tensor = super().current_run_tensors(
            **kwargs)
        return ConvCacheParams(cache_tensor, state_indices_tensor)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the Lfm2 Cache during the CUDA graph
        replay runs.
        """
        return self._lfm2_cache, torch.as_tensor([PAD_SLOT_ID] * batch_size,
                                                 dtype=torch.int32,
                                                 device="cuda")

# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import TYPE_CHECKING, List, Optional

# Third Party
import habana_frameworks.torch as htorch
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import (
    get_block_size,
    get_dtype,
    get_head_size,
    get_hidden_dim_size,
    get_num_blocks,
    get_num_heads,
    get_num_layers,
    get_page_buffer_size,
    is_mla,
    normalize_kv_and_discover_format,
)
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class VLLMPagedMemHPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]
    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        use_gpu: bool = False,
        **kwargs,
    ):
        self._attributes_initialized = False
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.use_gpu = use_gpu

    @classmethod
    def from_metadata(
        cls,
        metadata: "LMCacheMetadata",
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemHPUConnectorV2":
        """Create a connector from LMCacheMetadata.
        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
        Returns:
            A new instance of VLLMPagedMemHPUConnectorV2.
        """
        return cls(
            use_gpu=use_gpu,
        )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        # Flush the HPU lazy-mode op graph so the slot_mapping slice is
        # materialized before downstream ops consume it. This also keeps
        # LMCache's transfer ops decoupled from vLLM's HPU compute graph,
        # which issues its own mark_step() calls at forward-pass boundaries.
        htorch.core.mark_step()

        if self.use_mla:
            tmp = memory_obj.tensor[0].to(slot_mapping.device)
            total_blocks = self.num_blocks * self.block_size
            for i, kvcache in enumerate(self.kvcaches):
                kvcache.view(total_blocks, self.head_size).index_copy_(
                    0, slices, tmp[i]
                )
                htorch.core.mark_step()
        else:
            tmp_k = memory_obj.tensor[0].to(slot_mapping.device)
            tmp_v = memory_obj.tensor[1].to(slot_mapping.device)
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            for i, (kcache, vcache) in enumerate(self.kvcaches):
                kcache.view(total_blocks, d).index_copy_(0, slices, tmp_k[i])
                vcache.view(total_blocks, d).index_copy_(0, slices, tmp_v[i])
                htorch.core.mark_step()

        torch.hpu.synchronize()

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_MLA_FMT
        if use_mla is True.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        slices = slot_mapping[start:end]
        self._initialize_attributes(self.kvcaches)
        self._validate_memory_format(memory_obj)

        htorch.core.mark_step()

        if self.use_mla:
            total_blocks = self.num_blocks * self.block_size
            tmp = torch.stack(
                [
                    kvcache.view(total_blocks, self.head_size).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
        else:
            total_blocks = self.num_blocks * self.block_size
            d = self.num_heads * self.head_size
            tmp_k = torch.stack(
                [
                    kvcache[0].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp_v = torch.stack(
                [
                    kvcache[1].view(total_blocks, d).index_select(0, slices)
                    for kvcache in self.kvcaches
                ]
            )
            tmp = torch.stack([tmp_k, tmp_v])
        memory_obj.tensor.copy_(tmp, non_blocking=True)

        htorch.core.mark_step()
        torch.hpu.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens.

        Args:
            num_tokens: The number of tokens in the data.

        Returns:
            The shape of the KV cache data.

        Raises:
            RuntimeError: If attributes have not been initialized yet
                (i.e., no kv_caches have been seen).
        """
        if not self._attributes_initialized:
            raise RuntimeError(
                "Cannot determine shape before attributes are initialized. "
                "Call to_gpu or from_gpu first so that _initialize_attributes "
                "can discover the KV cache layout."
            )
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])

    def initialize_kvcaches_ptr(self, **kwargs) -> None:
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]

    def _validate_memory_format(self, memory_obj: MemoryObj) -> None:
        """Validate that the memory object has the expected format.

        Args:
            memory_obj: The memory object to validate.

        Raises:
            ValueError: If the memory format does not match the expected
                format based on whether MLA is in use.
        """
        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemHPUConnectorV2"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemHPUConnectorV2"
                )

    def _initialize_attributes(self, kv_caches: List[torch.Tensor]):
        if self._attributes_initialized or not kv_caches:
            return

        first = kv_caches[0]
        if isinstance(first, torch.Tensor):
            self.device = first.device
        else:
            self.device = first[0].device
        assert self.device.type == "hpu", "The device should be HPU."

        # HPU vLLM provides kv_caches as List[TensorTuple(k_tensor, v_tensor)],
        # where each TensorTuple contains two 4D tensors of shape
        # (num_blocks, block_size, num_heads, head_size).
        # We create a lightweight proxy List[Tensor(2, ...)] to match the
        # standard vLLM format (NL_X_TWO_NB_BS_NH_HS) for format discovery.
        if (
            isinstance(kv_caches, (list, tuple))
            and len(kv_caches) > 0
            and len(kv_caches[0]) == 2
            and not isinstance(kv_caches[0], torch.Tensor)
            and isinstance(kv_caches[0][0], torch.Tensor)
            and isinstance(kv_caches[0][1], torch.Tensor)
        ):
            # kv_caches[i][0].shape = (num_blocks, block_size, num_heads, head_size)
            # We need shape (2, num_blocks, block_size, num_heads, head_size)
            inner_shape = kv_caches[0][0].shape
            fake_shape = (2, *inner_shape)
            kv_caches = [
                torch.empty(fake_shape, dtype=kv_caches[0][0].dtype, device="meta")
                for _ in range(len(kv_caches))
            ]
            logger.info(
                "HPU: created lightweight kv_caches proxy with shape %s "
                "for format discovery",
                fake_shape,
            )

        self.engine_kv_format, kv_caches = normalize_kv_and_discover_format(
            kv_caches, EngineType.VLLM
        )
        self.num_layers = get_num_layers(kv_caches, self.engine_kv_format)
        self.num_blocks = get_num_blocks(kv_caches, self.engine_kv_format)
        self.block_size = get_block_size(kv_caches, self.engine_kv_format)
        self.page_buffer_size = get_page_buffer_size(kv_caches, self.engine_kv_format)
        self.hidden_dim_size = get_hidden_dim_size(kv_caches, self.engine_kv_format)
        self.head_size = get_head_size(kv_caches, self.engine_kv_format)
        self.use_mla = is_mla(self.engine_kv_format)
        self.dtype = get_dtype(kv_caches, self.engine_kv_format)
        self.num_heads = (
            1 if self.use_mla else get_num_heads(kv_caches, self.engine_kv_format)
        )

        self._attributes_initialized = True
        logger.info(
            "HPU: attributes initialized - format: %s, "
            "num_layers: %d, num_blocks: %d, block_size: %d, "
            "page_buffer_size: %d, hidden_dim_size: %d, head_size: %d, "
            "use_mla: %s, dtype: %s, num_heads: %d",
            self.engine_kv_format,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.page_buffer_size,
            self.hidden_dim_size,
            self.head_size,
            self.use_mla,
            self.dtype,
            self.num_heads,
        )

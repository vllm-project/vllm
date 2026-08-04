# SPDX-License-Identifier: Apache-2.0
"""Mock GPU connector for testing and standalone mode without real GPU."""

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector import GPUConnectorInterface


class MockGPUConnector(GPUConnectorInterface):
    """Mock GPU connector for testing without real GPU.

    This connector provides a no-op implementation of GPUConnectorInterface
    that can be used in standalone mode or testing environments where no
    actual GPU operations are needed.
    """

    def __init__(self, kv_shape: tuple):
        """Initialize mock GPU connector.

        Args:
            kv_shape: KV cache shape tuple (num_layers, kv_dim,
            num_blocks, num_heads, head_size)
        """
        self.kv_shape = kv_shape
        self.num_layers = kv_shape[0]
        self.stored_data: dict = {}
        self.kvcaches = None

    def from_gpu(self, memory_obj, start: int, end: int, **kwargs):
        """Mock from_gpu operation."""
        pass

    def to_gpu(self, memory_obj, start: int, end: int, **kwargs):
        """Mock to_gpu operation."""
        pass

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        """Mock batched_from_gpu operation."""
        pass

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        """Mock batched_to_gpu operation."""
        pass

    def get_shape(self, num_tokens=None):
        """Mock get_shape operation.

        Returns the shape based on the initialized kv_shape.
        """
        if num_tokens is None:
            num_tokens = self.kv_shape[2]
        return torch.Size(
            [
                self.num_layers,
                self.kv_shape[1],
                num_tokens,
                self.kv_shape[3],
                self.kv_shape[4],
            ]
        )

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]

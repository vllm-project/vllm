import torch
from typing import Dict, Tuple


class KVBuffer:
    """
    A class which is the key-value buffer for the model.
    A loose analogy is that this buffer is like an L1 cache and the conventional
    KV-cache is like an L2 cache
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.device = device
        self.dtype = dtype
        self.k_buffers: Dict[int, torch.Tensor] = {}
        self.v_buffers: Dict[int, torch.Tensor] = {}
        self.buffer_indices: Dict[int, int] = {}

    def add_request(self, seq_id: int, prompt_length: int) -> None:
        assert seq_id not in self.k_buffers \
            and seq_id not in self.v_buffers \
            and seq_id not in self.buffer_indices

        self.k_buffers[seq_id] = torch.zeros(
            (prompt_length, self.num_kv_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_buffers[seq_id] = torch.zeros(
            (prompt_length, self.num_kv_heads, self.head_size),
            dtype=self.dtype,
            device=self.device,
        )

        self.buffer_indices[seq_id] = 0

    def add_request_with_kv_tensors(self, seq_id: int, k: torch.Tensor,
                                    v: torch.Tensor) -> None:
        self.add_request(seq_id, k.shape[0])
        self.k_buffers[seq_id] = k
        self.v_buffers[seq_id] = v
        self.buffer_indices[seq_id] = k.shape[0]

    def free_request(self, seq_id: int) -> None:
        assert seq_id in self.k_buffers \
            and seq_id in self.v_buffers \
            and seq_id in self.buffer_indices
        del self.k_buffers[seq_id]
        del self.v_buffers[seq_id]
        del self.buffer_indices[seq_id]

    def get_kv_tensors(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert seq_id in self.k_buffers \
            and seq_id in self.v_buffers \
            and seq_id in self.buffer_indices
        offset = self.buffer_indices[seq_id]
        return (
            self.k_buffers[seq_id][:offset],
            self.v_buffers[seq_id][:offset],
        )

    def extend(self, seq_id: int, key: torch.Tensor,
               value: torch.Tensor) -> None:
        assert key.shape == value.shape
        offset = self.buffer_indices[seq_id]
        # Ideally we need to assert the following
        # assert self.k_buffers[seq_id][offset : offset + key.shape[0]].size(dim=0) != 0
        # assert self.v_buffers[seq_id][offset : offset + value.shape[0]].size(dim=0) != 0
        self.k_buffers[seq_id][offset:offset + key.shape[0]].copy_(key)
        self.v_buffers[seq_id][offset:offset + value.shape[0]].copy_(value)
        self.buffer_indices[seq_id] += key.shape[0]

    def set_offset(self, seq_id: int, length: int) -> None:
        # WARN: This method is used only for constructing a mock scenario
        # in the memory profiling phase
        self.buffer_indices[seq_id] = length

    def reset(self) -> None:
        self.k_buffers = {}
        self.v_buffers = {}
        self.buffer_indices = {}

    def get_offset(self, seq_id: int) -> int:
        return self.buffer_indices[seq_id]

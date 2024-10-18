from abc import ABC, abstractmethod
import torch

from vllm.attention import AttentionMetadata


class KVCacheTransporterBase(ABC):

    @abstractmethod
    def save_kv_cache(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_idx: int,
        kv_cache: torch.Tensor,
    ):
        """
        Save the key-value cache for a specific layer.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attn_metadata (AttentionMetadata): Metadata related to attention.
            layer_idx (int): The index of the layer.
            kv_cache (torch.Tensor): The key-value cache tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def read_kv_cache(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        layer_idx: int,
        kv_cache: torch.Tensor,
    ):
        """
        Read the key-value cache.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attn_metadata (AttentionMetadata): Metadata related to attention.
            kv_cache (torch.Tensor): The key-value cache tensor to be populated.
        """
        raise NotImplementedError

    @abstractmethod
    def save_hidden_states(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
    ):
        """
        Save the hidden states.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attn_metadata (AttentionMetadata): Metadata related to attention.
            hidden_states (torch.Tensor): The hidden states tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def read_hidden_states(
        self,
        input_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        hidden_states: torch.Tensor,
    ):
        """
        read the hidden states.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            attn_metadata (AttentionMetadata): Metadata related to attention.
            hidden_states (torch.Tensor): The hidden states tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def synchronize(self):
        """Synchronize any asynchronous operations."""
        raise NotImplementedError

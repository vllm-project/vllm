from abc import ABC, abstractmethod
import torch
from typing import List, Tuple


class KVCacheTransporterBase(ABC):

    @abstractmethod
    def save_kv_cache(self, prompt_token_page_hashes: List[str],
                      offsets: List[Tuple[int, int]], layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        raise NotImplementedError

    @abstractmethod
    def read_kv_cache(self, prompt_token_page_hashes: List[str],
                      prompt_seq_lengths: List[int], offsets: List[Tuple[int,
                                                                         int]],
                      layer_idx: int, kv_cache: torch.Tensor) -> None:

        raise NotImplementedError

    @abstractmethod
    def save_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor) -> None:

        raise NotImplementedError

    @abstractmethod
    def read_hidden_states(self, prompt_token_page_hashes: List[str],
                           prompt_seq_lengths: List[int],
                           hidden_states: torch.Tensor):

        raise NotImplementedError

    def get_hidden_states_cache_key(self, page_hash: str) -> str:
        raise NotImplementedError

    def get_kv_cache_key(self, page_hash: str,
                         layer_idx: int) -> Tuple[str, str]:
        raise NotImplementedError

    @abstractmethod
    def key_exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_match_last_index(self, keys: List[str]) -> int:
        raise NotImplementedError

    @abstractmethod
    def synchronize(self):

        raise NotImplementedError

    @abstractmethod
    def publish_kv_cache_prefill_done(self, input_token_hashes: List[str],
                                      seq_lens: List[int],
                                      layer_idx: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def check_kv_cache_ready(self, hash: str) -> bool:
        raise NotImplementedError

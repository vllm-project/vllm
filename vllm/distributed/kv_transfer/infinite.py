import math
import hashlib
import logging
from typing import Dict, List, Tuple
import torch
import os

from infinity import InfinityConnection
from vllm.attention import AttentionMetadata
from vllm.distributed.kv_transfer.base import KVCacheTransporterBase

logger = logging.getLogger(__name__)

Default_Infinite_Server = "127.0.0.1"

class InfiniStoreKVCacheTransporter(KVCacheTransporterBase):

    def __init__(self, model: str, tokens_per_page=16) -> None:
        if not model:
            raise ValueError("model cannot be empty.")
        if tokens_per_page <= 0:
            raise ValueError("tokens_per_page must be greater than 0.")

        self.model = model
        self.tokens_per_page = tokens_per_page
        self.conn: InfinityConnection = InfinityConnection()
       
        infinite_server = os.environ.get("INFINITE_STORE_SERVER", Default_Infinite_Server)
        print("~~~~~~~~~~~~~connecting to infinite store server: ", infinite_server)
        self.conn.connect(infinite_server)

    def _compute_kv_cache_block_offsets(
            self, input_ids: torch.Tensor, attn_metadata: AttentionMetadata,
            seq_index: int, seq_length: int, layer_idx: int,
            kv_cache: torch.Tensor) -> Tuple[List[Tuple[str, int]], int]:

        seq_tokens = input_ids[seq_index:seq_index + seq_length].cpu().numpy()
        num_pages = math.ceil(seq_length / self.tokens_per_page)
        block_offsets: List[Tuple[str, int]] = []
        prev_hash = ""
        page_size = kv_cache[0][0].numel()  # Number of elements in one page
        k_or_v_cache_size = kv_cache[0].numel(
        )  # Size of key or value cache per token

        for page_num in range(num_pages):
            # Calculate token indices for the current page
            start_token = page_num * self.tokens_per_page
            end_token = min((page_num + 1) * self.tokens_per_page, seq_length)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute the hash for the current page
            tokens_bytes = tokens_in_page.tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            # Generate cache keys using the current hash
            k_cache_key = f"{self.model}_{current_hash}_layer_{layer_idx}_k"
            v_cache_key = f"{self.model}_{current_hash}_layer_{layer_idx}_v"

            # Calculate the offset in the kv_cache for the current page
            try:
                slot_index = page_num * self.tokens_per_page
                slot_mapping_value = attn_metadata.slot_mapping[
                    seq_index + slot_index].item()
                page_offset = (slot_mapping_value //
                               self.tokens_per_page) * page_size
            except IndexError as e:
                logger.error("Invalid slot mapping index %s: %s", slot_index,
                             e)
                raise

            block_offsets.append((k_cache_key, page_offset))
            block_offsets.append(
                (v_cache_key, page_offset + k_or_v_cache_size))

            # Update the previous hash for the next page
            prev_hash = current_hash

            logger.debug(
                "Computed kv_cache block offsets: layer %s, page %s, "
                "k_cache_key %s, v_cache_key %s", layer_idx, page_num,
                k_cache_key, v_cache_key)

        return block_offsets, page_size

    def _compute_hidden_states_block_offsets(
            self, input_ids: torch.Tensor, attn_metadata: AttentionMetadata,
            seq_index: int, seq_length: int,
            hidden_states: torch.Tensor) -> Dict[int, List[Tuple[str, int]]]:

        seq_tokens = input_ids[seq_index:seq_index + seq_length].cpu().numpy()
        num_pages = math.ceil(seq_length / self.tokens_per_page)
        block_offsets: Dict[int, List[Tuple[str, int]]] = {}
        prev_hash = ""
        hidden_size = hidden_states.size(-1)

        for page_num in range(num_pages):
            # Calculate token indices for the current page
            start_token = page_num * self.tokens_per_page
            end_token = min((page_num + 1) * self.tokens_per_page, seq_length)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute the hash for the current page
            tokens_bytes = tokens_in_page.tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            # Generate cache key using the current hash
            cache_key = f"{self.model}_{current_hash}_hidden_states"

            # Calculate cache size and offset
            cache_size = hidden_size * (end_token - start_token)
            offset = (seq_index + start_token) * hidden_size

            if cache_size not in block_offsets:
                block_offsets[cache_size] = []
            block_offsets[cache_size].append((cache_key, offset))

            # Update the previous hash for the next page
            prev_hash = current_hash

            logger.debug(
                "Computed hidden_states block offsets: page %s, cache_key %s",
                page_num, cache_key)

        return block_offsets

    def save_kv_cache(self, input_ids: torch.Tensor,
                      attn_metadata: AttentionMetadata, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        seq_index = 0

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets, page_size = self._compute_kv_cache_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, layer_idx,
                kv_cache)

            # Write to cache
            try:
                self.conn.write_cache(kv_cache, block_offsets, page_size)
            except Exception as e:
                logger.error("Failed to write kv_cache: %s", e)
                raise

            seq_index += seq_length

        logger.debug("Saved kv_cache for layer %s", layer_idx)

    def read_kv_cache(self, input_ids: torch.Tensor,
                      attn_metadata: AttentionMetadata, layer_idx: int,
                      kv_cache: torch.Tensor) -> None:

        seq_index = 0

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets, page_size = self._compute_kv_cache_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, layer_idx,
                kv_cache)

            # Read from cache
            try:
                self.conn.read_cache(kv_cache, block_offsets, page_size)
            except Exception as e:
                logger.error("Failed to read kv_cache: %s", e)
                raise

            seq_index += seq_length

        logger.debug("Loaded kv_cache for layer %s", layer_idx)

    def save_hidden_states(self, input_ids: torch.Tensor,
                           attn_metadata: AttentionMetadata,
                           hidden_states: torch.Tensor) -> None:

        seq_index = 0

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets = self._compute_hidden_states_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, hidden_states)

            # Write to cache
            try:
                for cache_size, offsets in block_offsets.items():
                    self.conn.write_cache(hidden_states, offsets, cache_size)
            except Exception as e:
                logger.error("Failed to write hidden_states: %s", e)
                raise

            seq_index += seq_length

        logger.debug("Saved hidden_states")

    def read_hidden_states(self, input_ids: torch.Tensor,
                           attn_metadata: AttentionMetadata,
                           hidden_states: torch.Tensor) -> None:

        seq_index = 0

        for seq_length_tensor in attn_metadata.seq_lens_tensor:
            seq_length = seq_length_tensor.item()
            block_offsets = self._compute_hidden_states_block_offsets(
                input_ids, attn_metadata, seq_index, seq_length, hidden_states)

            # Read from cache
            try:
                for cache_size, offsets in block_offsets.items():
                    self.conn.read_cache(hidden_states, offsets, cache_size)
            except Exception as e:
                logger.error("Failed to read hidden_states: %s", e)
                raise

            seq_index += seq_length

        logger.debug("Loaded hidden_states")

    def synchronize(self) -> None:
        try:
            self.conn.sync()
            logger.debug("Synchronized with Infinity service")
        except Exception as e:
            logger.error("Failed to synchronize: %s", e)
            raise

from enum import Enum
import math
import os
import torch
from typing import List
import hashlib

from vllm.attention import AttentionMetadata

PAGE_SIZE = 16

class pd_separate_stage(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class ForwardPassType(Enum):
    PREFILL = "prefill_pass"
    FIRST_DECODE = "first_decode_pass"
    REGULAR = "regular_pass"


def get_forward_pass_type(input_ids: torch.Tensor, attn_metadata: AttentionMetadata):
    pd_stage = os.environ.get("PD_SEPARATE_STAGE", "").lower()
    is_profile_run = torch.any(input_ids == 0).item()
    if pd_stage not in pd_separate_stage._value2member_map_ or is_profile_run:
        return ForwardPassType.REGULAR

    if pd_stage == "prefill":
        return ForwardPassType.PREFILL
    else:
        if (attn_metadata.prefill_metadata is not None
            and attn_metadata.decode_metadata is None):
            return ForwardPassType.FIRST_DECODE

    return ForwardPassType.REGULAR


def prepare_kv_cache_transport(input_ids, attn_metadata, cache_config):

    fp_type = get_forward_pass_type(input_ids, attn_metadata)

    input_token_hashes = []
    offsets = []
    kv_cache_transporter = cache_config.kv_cache_transporter
    if fp_type in (ForwardPassType.PREFILL, ForwardPassType.FIRST_DECODE):
        input_token_hashes = compute_token_page_hashes(input_ids,
                                                       attn_metadata.seq_lens,
                                                       kv_cache_transporter.tokens_per_page)

        # Compute block ids for each page in the input sequence
        assert kv_cache_transporter is not None
        seq_start_index = 0
        page_start_index = 0
        tokens_per_page = kv_cache_transporter.tokens_per_page
        page_size = kv_cache_transporter.page_size
        k_or_v_total_size = kv_cache_transporter.k_or_v_total_size
        for seq_length in attn_metadata.seq_lens:
            num_pages = math.ceil(seq_length / tokens_per_page)

            for page_num in range(num_pages):
                start_token_idx = page_num * tokens_per_page

                slot_mapping_value = attn_metadata.slot_mapping[seq_start_index +
                                                      start_token_idx].item()

                block_id = slot_mapping_value //tokens_per_page
                k_offset = block_id * page_size
                offsets.append((k_offset, k_offset + k_or_v_total_size))

            seq_start_index += seq_length
            page_start_index += num_pages

        assert len(offsets) == len(input_token_hashes)

    return fp_type, kv_cache_transporter, input_token_hashes, offsets


def finalize_kv_cache_transport(fp_type, kv_cache_transporter,
                                input_token_hashes, attn_metadata,
                                hidden_states):
    if fp_type == ForwardPassType.PREFILL:
        kv_cache_transporter.save_hidden_states(input_token_hashes,
                                                attn_metadata.seq_lens,
                                                hidden_states)

        kv_cache_transporter.synchronize()

    return

def compute_token_page_hashes(prompt_token_ids: torch.Tensor,
                              prompt_seq_lengths: List[int],
                              tokens_per_page=PAGE_SIZE) -> List[str]:

    hashes = []
    seq_index = 0

    prompt_ids = prompt_token_ids.cpu().numpy()

    for seq_len in prompt_seq_lengths:
        seq_tokens = prompt_ids[seq_index:seq_index + seq_len]
        num_pages = math.ceil(seq_len / tokens_per_page)
        prev_hash = ""

        # Loop over each page within the current sequence
        for page_num in range(num_pages):
            start_token = page_num * tokens_per_page
            end_token = min((page_num + 1) * tokens_per_page, seq_len)
            tokens_in_page = seq_tokens[start_token:end_token]

            # Compute hash for the current page
            tokens_bytes = tokens_in_page.tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            hashes.append(current_hash)

        seq_index += seq_len

    return hashes

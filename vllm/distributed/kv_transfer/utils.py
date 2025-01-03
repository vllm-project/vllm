from enum import Enum
import math
import os
from typing import List
import hashlib
import array

from vllm.attention import AttentionMetadata

PAGE_SIZE = 16


class PDDisaggStage(Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    NONE = "none"


class ForwardPassType(Enum):
    PREFILL = "prefill_pass"
    FIRST_DECODE = "first_decode_pass"
    FOLLOWING_DECODE = "following_decode_pass"
    REGULAR = "regular_pass"


def get_pd_stage() -> PDDisaggStage:
    pd_stage = os.environ.get("PD_SEPARATE_STAGE", "").lower()

    if pd_stage == "prefill":
        return PDDisaggStage.PREFILL
    elif pd_stage == "decode":
        return PDDisaggStage.DECODE
    else:
        return PDDisaggStage.NONE


def get_forward_pass_type(attn_metadata: AttentionMetadata,
                          cache_config) -> ForwardPassType:
    pd_stage = get_pd_stage()
    is_profile_run = cache_config.kv_cache_transporter is None
    if pd_stage == PDDisaggStage.NONE or is_profile_run:
        return ForwardPassType.REGULAR

    if pd_stage == PDDisaggStage.PREFILL:
        return ForwardPassType.PREFILL
    else:
        if (attn_metadata.prefill_metadata is not None
                and attn_metadata.decode_metadata is None):
            return ForwardPassType.FIRST_DECODE
        else:
            return ForwardPassType.FOLLOWING_DECODE


def prepare_kv_cache_transport(attn_metadata, cache_config):

    fp_type = get_forward_pass_type(attn_metadata, cache_config)

    input_token_hashes = []
    offsets = []
    kv_cache_transporter = cache_config.kv_cache_transporter
    if fp_type in (ForwardPassType.PREFILL, ForwardPassType.FIRST_DECODE):
        input_token_hashes = getattr(attn_metadata, "token_hashes", None)
        assert input_token_hashes is not None

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

                slot_mapping_value = attn_metadata.slot_mapping[
                    seq_start_index + start_token_idx].item()

                block_id = slot_mapping_value // tokens_per_page
                k_offset = block_id * page_size
                offsets.append((k_offset, k_offset + k_or_v_total_size))

            seq_start_index += seq_length
            page_start_index += num_pages

        assert len(offsets) == len(input_token_hashes)

    return fp_type, kv_cache_transporter, input_token_hashes, offsets


def finalize_kv_cache_transport(fp_type, kv_cache_transporter,
                                input_token_hashes, attn_metadata,
                                hidden_states) -> None:
    if fp_type == ForwardPassType.PREFILL and kv_cache_transporter.tp_rank == 0:
        kv_cache_transporter.save_hidden_states(input_token_hashes,
                                                attn_metadata.seq_lens,
                                                hidden_states)

        kv_cache_transporter.synchronize()

    return


def compute_token_page_hashes(prompt_ids: List[int],
                              prompt_seq_lengths: List[int]) -> List[str]:

    global PAGE_SIZE
    tokens_per_page = PAGE_SIZE

    hashes = []
    seq_index = 0

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
            tokens_bytes = array.array('l', tokens_in_page).tobytes()
            hash_input = prev_hash.encode('utf-8') + tokens_bytes
            current_hash = hashlib.sha256(hash_input).hexdigest()

            prev_hash = current_hash

            hashes.append(current_hash)

        seq_index += seq_len

    return hashes

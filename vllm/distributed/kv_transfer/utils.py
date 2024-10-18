import os
import torch

from vllm.attention import AttentionMetadata


def _get_pd_sep_stage():
    return os.environ.get("PD_SEPARATE_STAGE", "").lower()


def _is_profile_run(input_ids: torch.Tensor):
    # profile_run will send in an all-zero input_ids tensor
    return torch.any(input_ids == 0).item()


def is_first_decode_pass(input_ids: torch.tensor,
                         attn_metadata: AttentionMetadata):
    if _get_pd_sep_stage() != "decode":
        return False

    if _is_profile_run(input_ids):
        return False

    return (attn_metadata.prefill_metadata is not None
            and attn_metadata.decode_metadata is None)


def is_prefill_run(input_ids: torch.Tensor):
    if _get_pd_sep_stage() != "prefill":
        return False

    return not _is_profile_run(input_ids)

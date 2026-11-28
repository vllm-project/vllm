# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU memory required to build sampler state before the KV cache exists."""

from vllm.utils.math_utils import cdiv
from vllm.utils.mem_utils import format_gib

# PenaltiesState stores these as int32.
_INT32_BYTES = 4


def penalty_state_bytes(max_num_reqs: int, vocab_size: int) -> int:
    """Bytes allocated by PenaltiesState before any request is added.

    `output_bin_counts` is one int32 per request slot per vocab entry.
    `prompt_bin_mask` is the bitset allocated beside it. Both are created
    while the sampler is constructed, which is before KV-cache planning.
    """
    bin_counts = max_num_reqs * vocab_size * _INT32_BYTES
    prompt_mask = max_num_reqs * cdiv(vocab_size, 32) * _INT32_BYTES
    return bin_counts + prompt_mask


def check_room_for_sampler_state(
    *,
    model_memory_bytes: int,
    free_bytes: int,
    penalty_bytes: int,
) -> None:
    """Raise when the weights leave no room for the sampler penalty state.

    A raw CUDA OOM from this allocation hides the cause: the KV-cache check
    has not run yet, and PyTorch suggests a fragmentation setting that does
    not apply when the device is simply full.
    """
    if free_bytes >= penalty_bytes:
        return
    raise ValueError(
        "Not enough GPU memory to finish loading. "
        f"The weights took {format_gib(model_memory_bytes)} GiB and "
        f"{format_gib(free_bytes)} GiB is free, which is less than the "
        f"{format_gib(penalty_bytes)} GiB sampler state "
        "(one int32 per request slot per vocab entry). "
        "The KV cache is sized after this. "
        "Use a smaller or quantized checkpoint, a GPU with more memory, "
        "tensor_parallel_size > 1, or a lower max_num_seqs. "
        "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
        "for more details."
    )

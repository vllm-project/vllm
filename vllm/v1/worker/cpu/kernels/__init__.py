# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU implementations of the Model Runner V2 Triton kernels, for CPUs without
Triton: numba ports of the per-request bookkeeping kernels and torch for the
vocab-sized sampling math. Registered through `vllm.triton_utils.dispatcher`."""

from collections.abc import Callable
from typing import Any


def get_kernel_overrides() -> dict[str, Callable[..., Any]]:
    from vllm.v1.worker.cpu.kernels import block_table, input_batch, sample

    gpu = "vllm.v1.worker.gpu"
    return {
        f"{gpu}.mm.rope._prepare_rope_positions_kernel": (
            input_batch.prepare_rope_positions
        ),
        f"{gpu}.model_states.prompt_embeds._apply_prompt_embeds_kernel": (
            input_batch.apply_prompt_embeds
        ),
        f"{gpu}.sample.logprob._topk_log_softmax_kernel": sample.topk_log_softmax,
        f"{gpu}.sample.logprob._ranks_kernel": sample.ranks,
        f"{gpu}.sample.logprob._fill_logprob_token_ids_kernel": (
            sample.fill_logprob_token_ids
        ),
        f"{gpu}.sample.prompt_logprob._prompt_logprobs_token_ids_kernel": (
            sample.prompt_logprobs_token_ids
        ),
        f"{gpu}.structured_outputs._apply_grammar_bitmask_kernel": (
            sample.apply_grammar_bitmask
        ),
        f"{gpu}.metrics.logits._num_nans_kernel": sample.num_nans,
        f"{gpu}.input_batch._get_num_sampled_and_rejected_kernel": (
            input_batch.get_num_sampled_and_rejected
        ),
        f"{gpu}.input_batch._post_update_kernel": input_batch.post_update,
        f"{gpu}.input_batch._post_update_num_computed_tokens_kernel": (
            input_batch.post_update_num_computed_tokens
        ),
        f"{gpu}.buffer_utils._apply_write_kernel": block_table.apply_write,
        f"{gpu}.cp_utils._dcp_local_seq_lens_kernel": block_table.dcp_local_seq_lens,
        "vllm.v1.worker.utils._zero_kv_blocks_kernel": block_table.zero_kv_blocks,
    }

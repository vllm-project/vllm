# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu_input_batch import InputBatch


def is_spec_decode_supported(req_id: str, input_batch: InputBatch) -> bool:
    if req_id in input_batch.min_p_reqs:
        # Spec decode doesn't support min_p sampling.
        return False
    elif (req_id in input_batch.frequency_penalties_reqs
          or req_id in input_batch.presence_penalties_reqs
          or req_id in input_batch.repetition_penalties_reqs):
        # Spec decode doesn't support penalties.
        return False
    elif req_id in input_batch.num_logprobs:
        # Spec decode doesn't support logprobs.
        return False

    return True


@triton.jit
def prepare_eagle_input_kernel(
    out_ptr,
    cu_query_lens_ptr,
    cu_num_tokens_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # [start_pos, end_pos)
    start_pos = tl.load(cu_num_tokens_ptr + pid)
    end_pos = tl.load(cu_num_tokens_ptr + pid + 1)
    num_tokens = end_pos - start_pos

    index_start = tl.load(cu_query_lens_ptr + pid)

    num_blocks = tl.cdiv(num_tokens, BLOCK_SIZE)
    for i in tl.range(num_blocks):
        offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            out_ptr + start_pos + offset,
            index_start + offset,
            mask=offset < num_tokens,
        )

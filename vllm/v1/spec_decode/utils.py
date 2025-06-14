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


@triton.jit
def advance_state_kernel(
    draft_token_ids_ptr,
    positions_ptr,

    # === Model input buffers to be updated ===
    model_input_ids_ptr,
    model_positions_ptr,

    # === Metadata tensors ===
    seq_lens_ptr,
    block_table_ptr,
    slot_mapping_ptr,

    # === Scalar configuration ===
    model_max_len: int,
    model_block_size: int,
    model_block_stride: int,

    # === Execution control ===
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
    PADDING_SLOT_ID: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    draft_token_list_last = tl.load(draft_token_ids_ptr + offsets, mask=mask)
    position = tl.load(positions_ptr + offsets, mask=mask)
    seq_lens = tl.load(seq_lens_ptr + offsets, mask=mask)

    # Update the inputs.
    # cast to int32 is crucial when eagle model is compiled.
    # tensor.argmax() returns int64 by default.
    input_id = draft_token_list_last.cast(tl.int32)
    position = position + 1

    # NOTE(woosuk): We should handle the case where the draft model
    # generates tokens beyond the max model length. Since it is complex
    # to remove such requests from the batch, we keep them in the batch
    # but adjust the position ids and slot mappings to avoid the
    # out-of-range access during the model execution. The draft tokens
    # generated with this adjustment should be ignored.
    exceeds_max_model_len = position >= model_max_len
    # Mask out the position ids that exceed the max model length.
    # Otherwise, we may get out-of-range error in RoPE.
    clamped_position = tl.where(exceeds_max_model_len, 0, position)

    # For the requests that exceed the max model length, we set the
    # sequence length to 1 to minimize their overheads in attention.
    seq_lens += 1
    seq_lens = tl.where(exceeds_max_model_len, 1, seq_lens)

    block_numbers = clamped_position // model_block_size
    block_offsets = clamped_position % model_block_size

    block_ids = tl.load(block_table_ptr + model_block_stride * offsets +
                        block_numbers,
                        mask=mask)

    # Compute slot mapping
    slot_mapping = block_ids * model_block_size + block_offsets

    # Mask out the slot mappings that exceed the max model length.
    # Otherwise, the KV cache will be inadvertently updated with the
    # padding tokens.
    slot_mapping = tl.where(exceeds_max_model_len, PADDING_SLOT_ID,
                            slot_mapping)

    tl.store(model_input_ids_ptr + offsets, input_id, mask=mask)
    tl.store(positions_ptr + offsets, position, mask=mask)
    tl.store(model_positions_ptr + offsets, clamped_position, mask=mask)
    tl.store(seq_lens_ptr + offsets, seq_lens, mask=mask)
    tl.store(slot_mapping_ptr + offsets, slot_mapping, mask=mask)
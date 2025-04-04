import torch
import triton
import triton.language as tl

@triton.jit
def _update_input_tokens(
    sample_output,
    seq_ids,
    input_tokens,
    input_seq_ids,
    BATCH_SIZE1,
    BATCH_SIZE2,
):
    pid = tl.program_id(0)
    if pid >= BATCH_SIZE2:
        return

    output_token = tl.load(input_tokens + pid)
    _input_seq_id = tl.load(input_seq_ids + pid)
    for i in range(BATCH_SIZE1):
        _seq_ids = tl.load(seq_ids + i)
        if _seq_ids == _input_seq_id:
            output_token = tl.load(sample_output + i)
    tl.store(input_tokens + pid, output_token)

def UpdateInputTokens(input_tokens, input_seq_ids, last_sample, last_ids):
    grid = [input_seq_ids.shape[0], 1, 1]
    _update_input_tokens[grid](last_sample, last_ids, input_tokens, input_seq_ids, last_ids.shape[0], input_seq_ids.shape[0])
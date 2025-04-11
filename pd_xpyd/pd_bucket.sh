#! /bin/bash

set_bucketing(){
    max_num_batched_tokens=${max_num_batched_tokens:-8192}
    max_num_seqs=${max_num_seqs:-128}
    input_min=${input_min:-1024}
    input_max=${input_max:-1024}
    output_max=${output_max:-2048}
    block_size=${block_size:-128}

    prompt_bs_step=1
    prompt_bs_min=1
    prompt_bs_max=$(( $max_num_batched_tokens / $input_min ))
    # prompt_bs_max = min(prompt_bs_max, max_num_seqs)
    prompt_bs_max=$(( $prompt_bs_max > $max_num_seqs ? $max_num_seqs : $prompt_bs_max ))
    # prompt_bs_max = CEILING.MATH(prompt_bs_max, prompt_bs_step)
    prompt_bs_max=$(( ($prompt_bs_max + $prompt_bs_step - 1) / $prompt_bs_step * $prompt_bs_step ))    
    export VLLM_PROMPT_BS_BUCKET_MIN=${VLLM_PROMPT_BS_BUCKET_MIN:-$prompt_bs_min}
    export VLLM_PROMPT_BS_BUCKET_STEP=${VLLM_PROMPT_BS_BUCKET_STEP:-$prompt_bs_step}
    export VLLM_PROMPT_BS_BUCKET_MAX=${VLLM_PROMPT_BS_BUCKET_MAX:-$prompt_bs_max}

    prompt_seq_step=128
    # prompt_seq_min = CEILING.MATH(input_min, prompt_seq_step)
    prompt_seq_min=$(( ($input_min + $prompt_seq_step -1) / $prompt_seq_step * $prompt_seq_step ))
    # prompt_seq_max = CEILING.MATH(input_max, prompt_seq_step) + prompt_seq_step
    prompt_seq_max=$(( (($input_max + $prompt_seq_step -1) / $prompt_seq_step) * $prompt_seq_step ))
    export VLLM_PROMPT_SEQ_BUCKET_MIN=${VLLM_PROMPT_SEQ_BUCKET_MIN:-$prompt_seq_min}
    export VLLM_PROMPT_SEQ_BUCKET_STEP=${VLLM_PROMPT_SEQ_BUCKET_STEP:-$prompt_seq_step}
    export VLLM_PROMPT_SEQ_BUCKET_MAX=${VLLM_PROMPT_SEQ_BUCKET_MAX:-$prompt_seq_max}

    # decode_bs_step = ROUNDUP(max_num_seqs / 16, 0)
    decode_bs_step=$(( ($max_num_seqs + 15) / 16 ))
    decode_bs_min=1
    # decode_bs_max = CEILING.MATH(max_num_seqs, decode_bs_step)
    decode_bs_max=$(( ($max_num_seqs + $decode_bs_step -1) / $decode_bs_step * $decode_bs_step ))
    export VLLM_DECODE_BS_BUCKET_MIN=${VLLM_DECODE_BS_BUCKET_MIN:-$decode_bs_min}
    export VLLM_DECODE_BS_BUCKET_STEP=${VLLM_DECODE_BS_BUCKET_STEP:-$decode_bs_step}
    export VLLM_DECODE_BS_BUCKET_MAX=${VLLM_DECODE_BS_BUCKET_MAX:-$decode_bs_max}

    decode_block_step=$decode_bs_max
    # decode_block_min = ROUNDUP(input_min / block_size, 0)
    decode_block_min=$(( ($input_min + $block_size - 1) / $block_size ))
    # decode_block_min = CEILING.MATH(decode_block_min, decode_block_step)
    decode_block_min=$(( ($decode_block_min + $decode_block_step) / $decode_block_step * $decode_block_step ))
    # decode_block_max = (CEILING.MATH(input_max + output_max, block_size) + decode_bs_max
    decode_block_max=$(( (($input_max + $output_max + $block_size -1) / $block_size + 1) * $decode_bs_max))
    export VLLM_DECODE_BLOCK_BUCKET_MIN=${VLLM_DECODE_BLOCK_BUCKET_MIN:-$decode_block_min}
    export VLLM_DECODE_BLOCK_BUCKET_STEP=${VLLM_DECODE_BLOCK_BUCKET_STEP:-$decode_block_step}
    export VLLM_DECODE_BLOCK_BUCKET_MAX=${VLLM_DECODE_BLOCK_BUCKET_MAX:-$decode_block_max}
}

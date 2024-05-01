"""Compare the short outputs of HF and vLLM when using greedy sampling.

pytest tests/basic_correctness/test_preemption.py`.
"""
import pytest

MODELS = [
    "facebook/opt-125m",
]


BLOCK_SIZE = 16
from vllm.transformers_utils.tokenizer_group import get_tokenizer_group


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("chunked_prefill_token_size", [16])
def test_chunked_prefill_recompute(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    chunked_prefill_token_size: int,
) -> None:
    """Ensure that chunked prefill works with preemption."""
    max_num_seqs = min(chunked_prefill_token_size, 256)
    enable_chunked_prefill = False
    max_num_batched_tokens = None
    if chunked_prefill_token_size != -1:
        enable_chunked_prefill = True
        max_num_batched_tokens = chunked_prefill_token_size

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    blocks_for_decode = max_tokens // BLOCK_SIZE + 1
    # Assume prefill requires at max 32 tokens.
    blocks_for_prefill = 32 // BLOCK_SIZE + 1

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_seqs=max_num_seqs,
        num_gpu_blocks_override=blocks_for_prefill + blocks_for_decode,
        max_model_len=(blocks_for_prefill + blocks_for_decode) * BLOCK_SIZE,
        block_size=BLOCK_SIZE
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    assert vllm_model.model.llm_engine.scheduler.total_preempted > 0
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_preemption(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """By default, recompute preemption is enabled"""

    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)
    del hf_model

    blocks_for_decode = max_tokens // BLOCK_SIZE + 1
    # Assume prefill requires at max 32 tokens.
    blocks_for_prefill = 32 // BLOCK_SIZE + 1

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        num_gpu_blocks_override=blocks_for_prefill + blocks_for_decode,
        max_model_len=(blocks_for_prefill + blocks_for_decode) * BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    assert vllm_model.model.llm_engine.scheduler.total_preempted > 0
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_str = hf_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        assert hf_output_str == vllm_output_str, (
            f"Test{i}:\nHF: {hf_output_str!r}\nvLLM: {vllm_output_str!r}")
        assert hf_output_ids == vllm_output_ids, (
            f"Test{i}:\nHF: {hf_output_ids}\nvLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("beam_width", [4])
def test_swap(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    """Use beam search enables swapping."""
    example_prompts = example_prompts[:1]
    hf_model = hf_runner(model, dtype=dtype)
    hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width,
                                               max_tokens)
    del hf_model

    blocks_for_decode = (max_tokens * beam_width) // BLOCK_SIZE - 3
    # Assume prefill requires at max 32 tokens.
    blocks_for_prefill = 32 // BLOCK_SIZE + 1
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        swap_space=10,
        num_gpu_blocks_override=blocks_for_decode + blocks_for_prefill,
        max_model_len=(blocks_for_decode + blocks_for_prefill) * BLOCK_SIZE,
        block_size=BLOCK_SIZE,
    )
    vllm_outputs = vllm_model.generate_beam_search(example_prompts, beam_width,
                                                   max_tokens)
    assert vllm_model.model.llm_engine.scheduler.total_preempted > 0
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, _ = hf_outputs[i]
        vllm_output_ids, _ = vllm_outputs[i]
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == vllm_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"vLLM: {vllm_output_ids}")

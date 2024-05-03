"""Compare the short outputs of HF and vLLM when using greedy sampling.

VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 has to be set before running this test.

Run `VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1
pytest tests/basic_correctness/test_preemption.py`.
"""
import pytest

from vllm import SamplingParams
from vllm.core.scheduler import (ARTIFICIAL_PREEMPTION_MAX_CNT,
                                 ENABLE_ARTIFICIAL_PREEMPT)

MODELS = [
    "facebook/opt-125m",
]

assert ENABLE_ARTIFICIAL_PREEMPT is True, (
    "Use an env var VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1. "
    "`VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 pytest "
    "tests/basic_correctness/test_preemption.py`")


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

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_num_seqs=max_num_seqs,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    assert (vllm_model.model.llm_engine.scheduler.artificial_preempt_cnt <
            ARTIFICIAL_PREEMPTION_MAX_CNT)
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

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
    )
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    assert (vllm_model.model.llm_engine.scheduler.artificial_preempt_cnt <
            ARTIFICIAL_PREEMPTION_MAX_CNT)
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

    vllm_model = vllm_runner(model, dtype=dtype, swap_space=10)
    vllm_outputs = vllm_model.generate_beam_search(example_prompts, beam_width,
                                                   max_tokens)
    assert (vllm_model.model.llm_engine.scheduler.artificial_preempt_cnt <
            ARTIFICIAL_PREEMPTION_MAX_CNT)
    del vllm_model

    for i in range(len(example_prompts)):
        hf_output_ids, _ = hf_outputs[i]
        vllm_output_ids, _ = vllm_outputs[i]
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == vllm_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"vLLM: {vllm_output_ids}")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
@pytest.mark.parametrize("beam_width", [4])
def test_swap_infeasible(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    """Verify infeasible swap request will be ignored."""
    BLOCK_SIZE = 16
    prefill_blocks = 2
    decode_blocks = max_tokens // BLOCK_SIZE
    example_prompts = example_prompts[:1]

    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        swap_space=10,
        block_size=BLOCK_SIZE,
        # Since beam search have more than 1 sequence, prefill + decode blocks
        # are not enough to finish.
        num_gpu_blocks_override=prefill_blocks + decode_blocks,
        max_model_len=(prefill_blocks + decode_blocks) * BLOCK_SIZE,
    )
    sampling_params = SamplingParams(n=beam_width,
                                     use_beam_search=True,
                                     temperature=0.0,
                                     max_tokens=max_tokens,
                                     ignore_eos=True)
    req_outputs = vllm_model.model.generate(
        example_prompts,
        sampling_params=sampling_params,
    )
    assert (vllm_model.model.llm_engine.scheduler.artificial_preempt_cnt <
            ARTIFICIAL_PREEMPTION_MAX_CNT)
    del vllm_model
    # Verify the request is ignored and not hang.
    assert req_outputs[0].outputs[0].finish_reason == "length"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float"])
@pytest.mark.parametrize("max_tokens", [96])
def test_preemption_infeasible(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    """Verify infeasible preemption request will be ignored."""
    BLOCK_SIZE = 16
    prefill_blocks = 2
    decode_blocks = max_tokens // BLOCK_SIZE
    vllm_model = vllm_runner(
        model,
        dtype=dtype,
        block_size=BLOCK_SIZE,
        # Not enough gpu blocks to complete a single sequence.
        # preemption should happen, and the sequence should be
        # ignored instead of hanging forever.
        num_gpu_blocks_override=prefill_blocks + decode_blocks // 2,
        max_model_len=((prefill_blocks + decode_blocks // 2) * BLOCK_SIZE),
    )
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    req_outputs = vllm_model.model.generate(
        example_prompts,
        sampling_params=sampling_params,
    )

    assert (vllm_model.model.llm_engine.scheduler.artificial_preempt_cnt <
            ARTIFICIAL_PREEMPTION_MAX_CNT)
    del vllm_model
    # Verify the request is ignored and not hang.
    for req_output in req_outputs:
        outputs = req_output.outputs
        assert len(outputs) == 1
        assert outputs[0].finish_reason == "length"

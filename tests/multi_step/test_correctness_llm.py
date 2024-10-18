# Test the LLMEngine with multi-step-decoding

import copy
from typing import Optional

import pytest

from vllm import SamplingParams
from vllm.entrypoints.utils import STR_MULTI_STEP_BEAM_SEARCH_NOT_SUPPORTED

from ..models.utils import check_logprobs_close, check_outputs_equal

MODELS = [
    "JackFram/llama-160m",
]
NUM_SCHEDULER_STEPS = [8]  # Multi-step decoding steps
NUM_PROMPTS = [10]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("enable_chunked_prefill", [False, True])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("num_logprobs", [None, 5])
def test_multi_step_llm(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    tp_size: int,
    enable_chunked_prefill: bool,
    max_tokens: int,
    enforce_eager: int,
    num_scheduler_steps: int,
    num_prompts: int,
    num_logprobs: Optional[int],
) -> None:
    """Test vLLM engine with multi-step scheduling via sync LLM Engine.

    Set up a HuggingFace (HF) transformers model as a ground-truth reference.

    Prompt them with the same example prompts.

    Validate:
    * Generated tokens match
    * Generated logprobs are all very close

    Args:
      hf_runner: HF transformers model runner fixture
      vllm_runner: vLLM model runner fixture
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      dtype: tensor datatype for engine to utilize
      tp_size: degree of tensor-parallelism
      enable_chunked_prefill: chunked-prefill on/off
      max_tokens: the maximum number of tokens to generate
      enforce_eager
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      num_prompts: number of example prompts under test
      num_logprobs: corresponds to the `logprobs` argument to the OpenAI
                    completions endpoint; `None` -> 1 logprob returned.
    """

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            enable_chunked_prefill=enable_chunked_prefill,
            num_scheduler_steps=num_scheduler_steps,
    ) as vllm_model:
        vllm_outputs = (vllm_model.generate_greedy(prompts, max_tokens)
                        if num_logprobs is None else
                        vllm_model.generate_greedy_logprobs(
                            prompts, max_tokens, num_logprobs))

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = (hf_model.generate_greedy(prompts, max_tokens)
                      if num_logprobs is None else
                      hf_model.generate_greedy_logprobs_limit(
                          prompts, max_tokens, num_logprobs))

    if num_logprobs is None:
        check_outputs_equal(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )
    else:
        check_logprobs_close(
            outputs_0_lst=hf_outputs,
            outputs_1_lst=vllm_outputs,
            name_0="hf",
            name_1="vllm",
        )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("num_logprobs,num_prompt_logprobs", [(5, 5)])
def test_multi_step_llm_w_prompt_logprobs(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    tp_size: int,
    max_tokens: int,
    enforce_eager: int,
    num_scheduler_steps: int,
    num_prompts: int,
    num_logprobs: Optional[int],
    num_prompt_logprobs: Optional[int],
) -> None:
    """Test prompt logprobs with multi-step scheduling via sync LLM Engine.

    Set up a vLLM engine instance w/ single-step scheduling as a ground-truth
    reference.

    Prompt them with the same example prompts.

    Validate:
    * All generated logprobs are all very close

    Args:
      hf_runner: HF transformers model runner fixture
      vllm_runner: vLLM model runner fixture
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      dtype: tensor datatype for engine to utilize
      tp_size: degree of tensor-parallelism
      max_tokens: the maximum number of tokens to generate
      enforce_eager
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      num_prompts: number of example prompts under test
      num_logprobs: corresponds to the `logprobs` argument to the OpenAI
                    completions endpoint; `None` -> no logprobs
      num_prompt_logprobs: number of logprobs to return for each prompt token;
                           note that this argument is not supported by the
                           OpenAI completions endpoint.
    """

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            num_scheduler_steps=num_scheduler_steps,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            prompts,
            max_tokens,
            num_logprobs,
            num_prompt_logprobs=num_prompt_logprobs)

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
    ) as vllm_model:
        single_step_vllm_outputs = vllm_model.generate_greedy_logprobs(
            prompts,
            max_tokens,
            num_logprobs,
            num_prompt_logprobs=num_prompt_logprobs)

    check_logprobs_close(
        outputs_0_lst=single_step_vllm_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="single_step_vllm",
        name_1="multi_step_vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("max_output_len", [7])
@pytest.mark.parametrize("n,best_of", [
    (1, 2),
    (2, 2),
    (2, 3),
])
@pytest.mark.parametrize("enable_chunked_prefill", [True, False])
@pytest.mark.parametrize("enable_prefix_caching", [True, False])
def test_multi_step_llm_best_of_fallback(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    tp_size: int,
    enforce_eager: int,
    num_scheduler_steps: int,
    num_prompts: int,
    max_output_len: int,
    n: int,
    best_of: int,
    enable_chunked_prefill: bool,
    enable_prefix_caching: bool,
) -> None:
    """Test vLLM engine with multi-step & best_of > 1

    Currently multi-step scheduling does not support best_of > 1 or beam search,
    however the default behavior is for the engine to fall back on single-step
    scheduling rather than failing.

    Two instantiations of the sync vLLM engine are tested, one with single-step
    and one with multi-step scheduling.

    Each instantiation of vLLM is tested in 3 phases:
    1. Batch of requests without best_of > 1
    2. Batch of requests with best_of > 1
    3. Batch of requests without best_of > 1

    For the instantiation of vLLM with multi-step scheduling, Phase 1 should use
    multi-step scheduling, Phase 2 should fall back on single-step scheduling,
    and Phase 3 should resume multi-step scheduling.

    The other instantiation should use single-step scheduling for all phases.

    Args:
      vllm_runner: vLLM model runner fixture
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      dtype: tensor datatype for engine to utilize
      tp_size: degree of tensor-parallelism
      enforce_eager
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      num_prompts: number of example prompts under test
      max_output_len: the maximum number of tokens to generate
      n: num seqs to output per :class:`SequenceGroup`
      best_of: num seqs per :class:`SequenceGroup` from which to choose
      enable_chunked_prefill
      enable_prefix_caching
    """

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    # Sampling parameters with best_of > 1 which should trigger a
    # multi-step scheduler to fall back on single-step scheduling
    sampling_params_best_of_gt_1 = SamplingParams(
        max_tokens=max_output_len,
        ignore_eos=True,
        temperature=1.0,
        n=n,
        best_of=best_of,
        seed=42,
    )

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            use_v2_block_manager=True,
            num_scheduler_steps=1,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
    ) as vllm_model:
        outputs_ss_best_of_gt_1 = vllm_model.generate(
            prompts, sampling_params_best_of_gt_1)

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            use_v2_block_manager=True,
            num_scheduler_steps=num_scheduler_steps,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
    ) as vllm_model:
        outputs_ms_best_of_gt_1 = (vllm_model.generate(
            prompts, sampling_params_best_of_gt_1))

    check_outputs_equal(
        outputs_0_lst=outputs_ss_best_of_gt_1,
        outputs_1_lst=outputs_ms_best_of_gt_1,
        name_0="outputs_ss_best_of_gt_1",
        name_1="outputs_ms_best_of_gt_1",
    )


@pytest.mark.parametrize("model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize("num_scheduler_steps", [8])
@pytest.mark.parametrize("max_output_len", [7])
def test_multi_step_beam_search_fail(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    enforce_eager: int,
    num_scheduler_steps: int,
    max_output_len: int,
) -> None:
    """Test that vLLM engine with multi-step fails if beam search is enabled.

    Beam search is not supported with multi-step.
    
    Args:
      vllm_runner: vLLM model runner fixture
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      dtype: tensor datatype for engine to utilize
      enforce_eager
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      max_output_len
    """

    with pytest.raises(ValueError,
                       match=STR_MULTI_STEP_BEAM_SEARCH_NOT_SUPPORTED), \
         vllm_runner(
             model,
             dtype=dtype,
             enforce_eager=enforce_eager,
             gpu_memory_utilization=0.7,
             tensor_parallel_size=1,
             use_v2_block_manager=True,
             num_scheduler_steps=num_scheduler_steps,
         ) as vllm_model:
        vllm_model.generate_beam_search(example_prompts, 2, max_output_len)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("enforce_eager", [True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("num_logprobs", [None, 5])
def test_multi_step_llm_chunked_prefill_prefix_cache(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    tp_size: int,
    max_tokens: int,
    enforce_eager: int,
    num_scheduler_steps: int,
    num_prompts: int,
    num_logprobs: Optional[int],
) -> None:
    """Test vLLM engine with multi-step+"single-step chunked prefill"+APC.

    Set up contrived scenario which tests for a possible failure mode of
    scheduling with multi-step+"single-step chunked prefill"+APC

    "single-step chunked prefill" here refers to the current vLLM multi-step+
    chunked-prefill implementation, which requires that a prefill may only
    be scheduled in the same step as decodes if the prefill prompt fits in a
    single chunk (note that "complete" multi-step+chunked-prefill would allow
    a prefill to span multiple chunks & multiple steps but that is not yet
    the case.)

    "APC" is short for "automatic prefix caching".

    This test creates a scenario where the scheduler must decide whether/how
    to schedule a prefill with a prompt that exceeds the available token budget.
    The correct behavior for multi-step+"single-step chunked prefill"+APC is to
    put off scheduling the prefill until a future step.

    Validate that:
    * Multi-step kernels do not raise an exception due to incorrect scheduler
      behavior
    * Generated tokens match between
      multi-step+"single-step chunked prefill"+APC and
      single-step scheduling.
    * (If logprobs are enabled) check logprobs are close enough

    Args:
      vllm_runner: vLLM model runner fixture
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      dtype: tensor datatype for engine to utilize
      tp_size: degree of tensor-parallelism
      max_tokens: the maximum number of tokens to generate
      enforce_eager
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      num_prompts: number of example prompts under test
      num_logprobs: corresponds to the `logprobs` argument to the OpenAI
                    completions endpoint; `None` -> 1 logprob returned.
    """

    # Set up contrived test for correct scheduling behavior with
    # multi-step+"single-step chunked prefill"+APC.
    #
    # Assume block_size=16
    #
    # Assume max_num_batched_tokens=48
    #   => Per-step token budget=48
    #
    # 1. Scheduler schedules 0th prompt (24 tokens)
    #      => Remaining token budget=24
    # 2. Scheduler attempts to schedule 1st prompt (30 tokens)
    #    * 30 tokens exceeds 24 token remaining budget
    #    * Correct behavior: do not schedule this prompt in this step
    #    * Incorrect behavior: schedule prompt chunk
    #      * `do_sample=False` for this prompt in this step
    #      * Chunk size = (remaining tokens // block size) * block size
    #
    # The Incorrect scheduling behavior - if it occurs - will cause an exception
    # in the model runner resulting from `do_sample=False`.
    assert len(example_prompts) >= 2
    challenge_prompts = copy.deepcopy(example_prompts)
    challenge_prompts[0] = ('vLLM is a high-throughput and memory-efficient '
                            'inference and serving engine for LLMs.\n'
                            )  # 24 tok
    challenge_prompts[1] = (
        'Briefly describe the major milestones in the '
        'development of artificial intelligence from 1950 to 2020.\n'
    )  # 30 tok

    # If necessary, adjust the length of `challenge_prompts` to match
    # `num_prompts`
    if len(challenge_prompts) < num_prompts:
        challenge_prompts = (challenge_prompts *
                             ((num_prompts // len(challenge_prompts)) + 1))
    challenge_prompts = challenge_prompts[:num_prompts]
    assert len(challenge_prompts) == num_prompts

    # Single-step scheduler baseline
    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            num_scheduler_steps=num_scheduler_steps,
            max_model_len=48,
            max_num_batched_tokens=48,
            max_num_seqs=4,
            block_size=16,
    ) as vllm_model:
        outputs_baseline = (vllm_model.generate_greedy(
            challenge_prompts, max_tokens) if num_logprobs is None else
                            vllm_model.generate_greedy_logprobs(
                                challenge_prompts, max_tokens, num_logprobs))

    # multi-step+"single-step chunked prefill"+APC
    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            enable_chunked_prefill=True,
            enable_prefix_caching=True,
            num_scheduler_steps=num_scheduler_steps,
            max_model_len=48,
            max_num_batched_tokens=48,
            max_num_seqs=4,
            block_size=16,
    ) as vllm_model:
        outputs_w_features = (vllm_model.generate_greedy(
            challenge_prompts, max_tokens) if num_logprobs is None else
                              vllm_model.generate_greedy_logprobs(
                                  challenge_prompts, max_tokens, num_logprobs))

    if num_logprobs is None:
        # No-logprobs test
        check_outputs_equal(
            outputs_0_lst=outputs_baseline,
            outputs_1_lst=outputs_w_features,
            name_0="multi-step",
            name_1="multi-step+features",
        )
    else:
        # Yes-logprobs test
        check_logprobs_close(
            outputs_0_lst=outputs_baseline,
            outputs_1_lst=outputs_w_features,
            name_0="multi-step",
            name_1="multi-step+features",
        )

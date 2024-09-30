# Test the LLMEngine with multi-step-decoding

from typing import Optional

import pytest

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
            use_v2_block_manager=True,
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
            use_v2_block_manager=True,
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
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("tp_size", [1])
@pytest.mark.parametrize("enable_chunked_prefill", [True])
@pytest.mark.parametrize("enable_prefix_caching", [True])
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
    enable_chunked_prefill: bool,
    enable_prefix_caching: bool,
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
            use_v2_block_manager=True,
            num_scheduler_steps=num_scheduler_steps,
    ) as vllm_model:
        outputs_baseline = (vllm_model.generate_greedy(prompts, max_tokens)
                            if num_logprobs is None else
                            vllm_model.generate_greedy_logprobs(
                                prompts, max_tokens, num_logprobs))

    with vllm_runner(
            model,
            dtype=dtype,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=0.7,
            tensor_parallel_size=tp_size,
            use_v2_block_manager=True,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            num_scheduler_steps=num_scheduler_steps,
    ) as vllm_model:
        outputs_w_features = (vllm_model.generate_greedy(prompts, max_tokens)
                              if num_logprobs is None else
                              vllm_model.generate_greedy_logprobs(
                                  prompts, max_tokens, num_logprobs))

    if num_logprobs is None:
        check_outputs_equal(
            outputs_0_lst=outputs_baseline,
            outputs_1_lst=outputs_w_features,
            name_0="multi-step",
            name_1="multi-step+features",
        )
    else:
        check_logprobs_close(
            outputs_0_lst=outputs_baseline,
            outputs_1_lst=outputs_w_features,
            name_0="multi-step",
            name_1="multi-step+features",
        )


from typing import List, Optional

import pytest

from tests.kernels.utils import override_backend_env_variable

from ..models.utils import check_logprobs_close
from ..utils import (completions_with_server_args, get_client_text_generations,
                     get_client_text_logprob_generations)

DEFAULT_SERVER_ARGS: List[str] = [
    "--disable-log-requests",
    "--use-v2-block-manager",
    "--worker-use-ray",
    "--gpu-memory-utilization",
    "0.85",
    "--swap-space",
    "16",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(("tp_size, pp_size"), [
    (1, 1),
    (2, 2),
])
@pytest.mark.parametrize("eager_mode", [False, True])
@pytest.mark.parametrize("num_scheduler_steps", NUM_SCHEDULER_STEPS)
@pytest.mark.parametrize("num_prompts", NUM_PROMPTS)
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize("is_async", [True])
@pytest.mark.parametrize("attention_backend", ["FLASHINFER", "FLASH_ATTN"])
@pytest.mark.parametrize("enable_chunked_prefill", [True])
@pytest.mark.asyncio
async def test_multi_step_async(
    example_prompts,
    model: str,
    tp_size: int,
    pp_size: int,
    eager_mode: int,
    num_scheduler_steps: int,
    num_prompts: int,
    is_async: bool,
    num_logprobs: Optional[int],
    attention_backend: str,
    enable_chunked_prefill: bool,
    monkeypatch,
) -> None:
    """Test vLLM engine with multi-step scheduling in an OpenAI-protocol
    client/server environment.

    Set up an engine with single-step scheduling as a ground-truth reference.

    Send a completions API request to both engines with the same prompts.

    Validate:
    * Generated tokens match
    * Generated logprobs are all very close

    Args:
      example_prompts: test fixture providing example prompts
      model: model under test (same for single- and multi-step engines)
      tp_size: degree of tensor-parallelism
      pp_size: degree of pipeline-parallelism
      eager_mode
      num_scheduler_steps: for multi-step scheduling, GPU-side steps per
                           GPU -> CPU output transfer
      num_prompts: number of example prompts under test
      num_logprobs: corresponds to the `logprobs` argument to the OpenAI
                    completions endpoint; `None` -> no logprobs
    """

    override_backend_env_variable(monkeypatch, attention_backend)

    prompts = example_prompts
    if len(prompts) < num_prompts:
        prompts = prompts * ((num_prompts // len(prompts)) + 1)
    prompts = prompts[:num_prompts]
    assert len(prompts) == num_prompts

    server_args = DEFAULT_SERVER_ARGS + ["--enforce-eager"]
    ms_server_args = DEFAULT_SERVER_ARGS + \
        ["--num-scheduler-steps", f"{num_scheduler_steps}"]

    if not is_async:
        ms_server_args += ["--disable-async-output-proc"]

    if eager_mode:
        ms_server_args.append("--enforce-eager")

    ms_server_args.append("--enable-chunked-prefill")
    ms_server_args.append("--enable-prefix-caching")

    distributed_args = [
        "--tensor-parallel-size",
        str(tp_size),
        "--pipeline-parallel-size",
        str(pp_size),
    ]

    # Spin up client/server & issue completion API requests.
    # Default `max_wait_seconds` is 240 but was empirically
    # was raised 3x to 720 *just for this test* due to
    # observed timeouts in GHA CI
    ref_completions = await completions_with_server_args(
        prompts,
        model,
        server_args + distributed_args,
        num_logprobs,
        max_wait_seconds=5 * 240)
    test_completions = await completions_with_server_args(
        prompts,
        model,
        ms_server_args + distributed_args,
        num_logprobs,
        max_wait_seconds=5 * 240)

    # Assert multi-step scheduling produces identical tokens
    # to single-step scheduling.
    ref_generations = get_client_text_generations(ref_completions)
    test_generations = get_client_text_generations(test_completions)
    assert ref_generations == test_generations

    # Assert multi-step scheduling produces nearly-identical logprobs
    # to single-step scheduling.
    ref_text_logprobs = get_client_text_logprob_generations(ref_completions)
    test_text_logprobs = get_client_text_logprob_generations(test_completions)
    check_logprobs_close(
        outputs_0_lst=ref_text_logprobs,
        outputs_1_lst=test_text_logprobs,
        name_0="hf",
        name_1="vllm",
    )

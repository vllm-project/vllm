# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform, large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.benchmarks.datasets import InstructCoderDataset
from vllm.config.vllm import VllmConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Metric
from vllm.v1.spec_decode.draft_model import (
    create_vllm_config_for_draft_model,
    merge_toks_kernel,
)

MTP_SIMILARITY_RATE = 0.8


def _skip_if_insufficient_gpus_for_tp(tp_size: int):
    """Skip test if available GPUs < tp_size on ROCm."""
    available_gpus = torch.cuda.device_count()
    if available_gpus < tp_size:
        pytest.skip(
            f"Test requires {tp_size} GPUs, but only {available_gpus} available"
        )


Messages = list[dict[str, Any]]


def get_test_prompts(
    mm_enabled: bool, quiet: bool = False, num_prompts: int = 100
) -> list[Messages]:
    prompt_types = ["repeat", "sentence"]
    if mm_enabled:
        prompt_types.append("mm")
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    if not quiet:
        print(f"Prompt types: {random_prompt_type_choices}")

    # Generate a mixed batch of prompts, some of which can be easily
    # predicted by n-gram matching and some which likely cannot.
    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        prompt: str | list[dict[str, Any]] = ""
        if kind == "repeat":
            prompt = f"""
            please repeat the word '{word}' 10 times.
            give no other output than the word at least ten times in a row,
            in lowercase with spaces between each word and without quotes.
            """
        elif kind == "sentence":
            prompt = f"""
            please give a ten-word sentence that
            uses the word {word} at least once.
            give no other output than that simple sentence without quotes.
            """
        elif kind == "mm":
            placeholders = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"{VLLM_S3_BUCKET_URL}/{VLM_IMAGES_DIR}/stop_sign.jpg"
                    },
                }
            ]
            prompt = [
                *placeholders,
                {"type": "text", "text": "The meaning of the image is"},
            ]
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append([{"role": "user", "content": prompt}])

    return prompts


def get_instruct_coder_messages(n: int) -> list[Messages]:
    dataset = InstructCoderDataset(
        dataset_path="likaixin/InstructCoder", dataset_split="train"
    )
    prompts: Iterable[str] = dataset.sample_prompts(n=n)
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


@pytest.fixture
def sampling_config():
    return greedy_sampling()


def greedy_sampling() -> SamplingParams:
    return SamplingParams(temperature=0, max_tokens=10, ignore_eos=False)


def stochastic_sampling() -> SamplingParams:
    return SamplingParams(temperature=1.0, max_tokens=10, ignore_eos=False)


@pytest.fixture
def model_name():
    return "meta-llama/Llama-3.1-8B-Instruct"


@pytest.fixture(autouse=True)
def reset_torch_dynamo():
    """Reset torch dynamo cache before each test"""
    yield
    # Cleanup after test
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "speculative_config",
    [
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
        {
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
        },
    ],
)
def test_ngram_and_suffix_correctness(
    speculative_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    """
    Compare the outputs of an original LLM and a speculative LLM
    should be the same when using ngram speculative decoding.
    """
    test_prompts = get_test_prompts(mm_enabled=False)

    ref_llm = LLM(model=model_name, max_model_len=1024)
    ref_outputs = ref_llm.chat(test_prompts, sampling_config)
    del ref_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    spec_llm = LLM(
        model=model_name,
        speculative_config=speculative_config,
        max_model_len=1024,
    )
    spec_outputs = spec_llm.chat(test_prompts, sampling_config)
    matches = 0
    misses = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            misses += 1
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")

    # Heuristic: expect at least 66% of the prompts to match exactly
    # Upon failure, inspect the outputs to check for inaccuracy.
    assert matches >= int(0.66 * len(ref_outputs))
    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()


def test_suffix_decoding_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    """
    Check that suffix decoding caching takes effect and improves acceptance
    lengths and acceptance rates over multiple runs of the same prompts.
    """
    test_prompts = get_test_prompts(mm_enabled=False)

    spec_llm = LLM(
        model=model_name,
        speculative_config={
            "method": "suffix",
            "suffix_decoding_max_spec_factor": 2.0,
            "suffix_decoding_max_cached_requests": 1000,
        },
        max_model_len=1024,
        disable_log_stats=False,
    )

    # Run several times and check that the accepted tokens increase.
    num_draft = []
    num_accept = []
    for i in range(10):  # Run multiple times to warm up the cache.
        spec_llm.chat(test_prompts, sampling_config)
        # Collect draft and acceptance stats.
        metrics = spec_llm.get_metrics()
        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                num_draft.append(metric.value)
            if metric.name == "vllm:spec_decode_num_accepted_tokens":
                num_accept.append(metric.value)

    # Calculate the acceptance rates for the first and last runs.
    first_accept_tokens = num_accept[0]
    first_draft_tokens = num_draft[0]
    first_accept_rate = first_accept_tokens / first_draft_tokens

    # Take the diff since the stats are cumulative.
    last_accept_tokens = num_accept[-1] - num_accept[-2]
    last_draft_tokens = num_draft[-1] - num_draft[-2]
    last_accept_rate = last_accept_tokens / last_draft_tokens

    # Expect the acceptance length to improve.
    assert first_accept_tokens < last_accept_tokens

    # Expect the acceptance rate to improve.
    assert first_accept_rate < last_accept_rate

    # Heuristic: expect at least 80.0% acceptance rate at the end.
    assert last_accept_rate > 0.80

    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "model_path",
    [
        "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
        "RedHatAI/Qwen3-8B-speculator.eagle3",
    ],
    ids=["llama3_eagle3_speculator", "qwen3_eagle3_speculator"],
)
def test_speculators_model_integration(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_path: str,
):
    """
    Test that speculators models work with the simplified integration.

    This verifies the `vllm serve <speculator-model>` use case where
    speculative config is automatically detected from the model config
    without requiring explicit --speculative-config argument.

    Tests:
    1. Speculator model is correctly detected
    2. Verifier model is extracted from speculator config
    3. Speculative decoding is automatically enabled
    4. Text generation works correctly
    5. Output matches reference (non-speculative) generation
    """
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Generate test prompts
    test_prompts = get_test_prompts(mm_enabled=False)

    # First run: Direct speculator model (simplified integration)
    spec_llm = LLM(model=model_path, max_model_len=1024)
    spec_outputs = spec_llm.chat(test_prompts, sampling_config)

    # Verify speculative config was auto-detected
    assert spec_llm.llm_engine.vllm_config.speculative_config is not None, (
        f"Speculative config should be auto-detected for {model_path}"
    )

    spec_config = spec_llm.llm_engine.vllm_config.speculative_config
    assert spec_config.num_speculative_tokens > 0, (
        f"Expected positive speculative tokens, "
        f"got {spec_config.num_speculative_tokens}"
    )

    # Verify draft model is set to the speculator model
    assert spec_config.model == model_path, (
        f"Draft model should be {model_path}, got {spec_config.model}"
    )

    # Extract verifier model for reference run
    verifier_model = spec_llm.llm_engine.vllm_config.model_config.model

    del spec_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Second run: Reference without speculative decoding
    ref_llm = LLM(model=verifier_model, max_model_len=1024)
    ref_outputs = ref_llm.chat(test_prompts, sampling_config)
    del ref_llm
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    # Compare outputs
    matches = sum(
        1
        for ref, spec in zip(ref_outputs, spec_outputs)
        if ref.outputs[0].text == spec.outputs[0].text
    )

    # Heuristic: expect at least 66% of prompts to match exactly
    assert matches >= int(0.66 * len(ref_outputs)), (
        f"Only {matches}/{len(ref_outputs)} outputs matched. "
        f"Expected at least {int(0.66 * len(ref_outputs))} matches."
    )


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled", "enable_chunked_prefill", "model_impl"],
    [
        (
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "auto",
        ),
        (
            ("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1),
            False,
            False,
            "transformers",
        ),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen3-VL-8B-Instruct",
                "taobao-mnn/Qwen3-VL-8B-Instruct-Eagle3",
                1,
            ),
            False,
            False,
            "auto",
            marks=pytest.mark.skip(
                reason="architecture of its eagle3 is LlamaForCausalLMEagle3"
            ),
        ),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            False,
            "auto",
            marks=pytest.mark.skip(
                reason="Skipping due to its head_dim not being a a multiple of 32"
            ),
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            True,
            "auto",
            marks=large_gpu_mark(min_gb=40),
        ),  # works on 4x H100
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
            False,
            "auto",
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            False,
            False,
            "auto",
            marks=large_gpu_mark(min_gb=80),
        ),  # works on 4x H100
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            True,
            True,
            "auto",
            marks=large_gpu_mark(min_gb=80),
        ),  # works on 4x H100
        (
            (
                "eagle",
                "eagle618/deepseek-v3-random",
                "eagle618/eagle-deepseek-v3-random",
                1,
            ),
            False,
            False,
            "auto",
        ),
    ],
    ids=[
        "qwen3_eagle3",
        "qwen3_eagle3-transformers",
        "qwen3_vl_eagle3",
        "qwen2_5_vl_eagle3",
        "llama3_eagle",
        "llama3_eagle3",
        "llama4_eagle",
        "llama4_eagle_mm",
        "deepseek_eagle",
    ],
)
@pytest.mark.parametrize("attn_backend", get_attn_backend_list_based_on_platform())
def test_eagle_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, str, int],
    mm_enabled: bool,
    enable_chunked_prefill: bool,
    model_impl: str,
    attn_backend: str,
):
    if attn_backend == "TREE_ATTN":
        # TODO: Fix this flaky test
        pytest.skip(
            "TREE_ATTN is flaky in the test disable for now until it can be "
            "resolved (see https://github.com/vllm-project/vllm/issues/22922)"
        )
    if model_impl == "transformers":
        import transformers
        from packaging.version import Version

        installed = Version(transformers.__version__)
        required = Version("5.0.0.dev")
        if installed < required:
            pytest.skip(
                "Eagle3 with the Transformers modeling backend requires "
                f"transformers>={required}, but got {installed}"
            )

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    """
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    """
    # Determine attention config
    # Scout requires default backend selection because vision encoder has
    # head_dim 88 being incompatible with FLASH_ATTN and needs to fall back
    # to Flex Attn
    if "Llama-4-Scout" in model_setup[1] and attn_backend == "FLASH_ATTN":
        if current_platform.is_rocm():
            # TODO: Enable Flex Attn for spec_decode on ROCm
            pytest.skip("Flex Attn for spec_decode not supported on ROCm currently")
        attention_config = None  # Let it fall back to default
    else:
        attention_config = {"backend": attn_backend}

    if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
        pytest.skip(
            "TRITON_ATTN does not support "
            "multi-token eagle spec decode on current platform"
        )

    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        if attn_backend == "ROCM_AITER_FA" and current_platform.is_rocm():
            if "deepseek" in model_setup[1].lower():
                pytest.skip("ROCM_AITER_FA for deepseek not supported on ROCm platform")
            else:
                m.setenv("VLLM_ROCM_USE_AITER", "1")

        method, model_name, spec_model_name, tp_size = model_setup
        _skip_if_insufficient_gpus_for_tp(tp_size)

        max_model_len = 2048
        max_num_batched_tokens = 128 if enable_chunked_prefill else max_model_len

        ref_llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            tensor_parallel_size=tp_size,
            attention_config=attention_config,
        )
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "model": spec_model_name,
                "num_speculative_tokens": 3,
                "max_model_len": max_model_len,
            },
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            model_impl=model_impl,
            attention_config=attention_config,
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 60% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.6 * len(ref_outputs))
        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled"],
    [
        (("mtp", "XiaomiMiMo/MiMo-7B-Base", 1), False),
        (("mtp", "ZixiQi/DeepSeek-V3-4layers-MTP-FP8", 1), False),
    ],
    ids=["mimo", "deepseek"],
)
def test_mtp_correctness(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_setup: tuple[str, str, int],
    mm_enabled: bool,
):
    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    """
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using MTP speculative decoding.
    model_setup: (method, model_name, tp_size)
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_MLA_DISABLE", "1")

        method, model_name, tp_size = model_setup
        _skip_if_insufficient_gpus_for_tp(tp_size)

        ref_llm = LLM(
            model=model_name,
            max_model_len=2048,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
        )
        ref_outputs = ref_llm.chat(test_prompts, sampling_config)
        del ref_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()

        spec_llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            speculative_config={
                "method": method,
                "num_speculative_tokens": 1,
                "max_model_len": 2048,
            },
            max_model_len=2048,
        )
        spec_outputs = spec_llm.chat(test_prompts, sampling_config)
        matches = 0
        misses = 0
        for ref_output, spec_output in zip(ref_outputs, spec_outputs):
            if ref_output.outputs[0].text == spec_output.outputs[0].text:
                matches += 1
            else:
                misses += 1
                print(f"ref_output: {ref_output.outputs[0].text}")
                print(f"spec_output: {spec_output.outputs[0].text}")

        # Heuristic: expect at least 80% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(MTP_SIMILARITY_RATE * len(ref_outputs))
        del spec_llm
        torch.cuda.empty_cache()
        cleanup_dist_env_and_memory()


@dataclass
class ArgsTest:
    target_model: str
    draft_model: str
    sampling_config: SamplingParams
    num_speculative_tokens: int
    expected_acceptance_rate: float
    expected_acceptance_len: float
    # Defaults
    target_tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int = 1
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.5
    dataset: str = "test_prompts"
    num_prompts: int = 100


cases = [
    # Same model for draft and target, greedy sampling.
    ArgsTest(
        target_model="Qwen/Qwen3-0.6B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=greedy_sampling(),
        num_speculative_tokens=3,  # K
        expected_acceptance_len=3 + 1,  # K + 1
        expected_acceptance_rate=1.0,
    ),
    # Smaller draft model, stochastic sampling.
    ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=stochastic_sampling(),
        num_speculative_tokens=3,
        expected_acceptance_len=2.8 + 1,
        expected_acceptance_rate=0.9,
    ),
]


@pytest.mark.parametrize("args", cases)
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_draft_model_correctness(args: ArgsTest, enforce_eager: bool):
    assert_draft_model_correctness(args, enforce_eager)


def test_draft_model_realistic_example():
    args = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        dataset="likaixin/InstructCoder",
        num_speculative_tokens=3,
        sampling_config=greedy_sampling(),
        # values below are not derived, but just prevent a regression
        expected_acceptance_len=2.8,
        expected_acceptance_rate=0.55,
    )
    assert_draft_model_correctness(args, enforce_eager=False)


@pytest.mark.parametrize(
    "models",
    [
        # target_model,         draft_model
        ("Qwen/Qwen3-1.7B-FP8", "Qwen/Qwen3-0.6B"),  # target quantized
        ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B-FP8"),  # draft quantized
    ],
    ids=["target_quantized", "draft_quantized"],
)
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_draft_model_quantization(models: tuple[str, str], enforce_eager: bool):
    tgt_model, draft_model = models
    sd_case = ArgsTest(
        target_model=tgt_model,
        draft_model=draft_model,
        **some_high_acceptance_metrics(),
    )
    assert_draft_model_correctness(sd_case, enforce_eager)


def test_draft_model_tensor_parallelism():
    """Ensure spec decode works when running with TP > 1."""
    _skip_if_insufficient_gpus_for_tp(2)
    sd_case = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        target_tensor_parallel_size=2,
        draft_model="Qwen/Qwen3-0.6B",
        draft_tensor_parallel_size=2,
        **some_high_acceptance_metrics(),
    )
    assert_draft_model_correctness(sd_case, enforce_eager=False)


def test_draft_model_engine_args_tensor_parallelism():
    """Ensure the vllm_config for the draft model is created correctly,
    and independently of the target model (quantization, TP, etc.)"""
    _skip_if_insufficient_gpus_for_tp(2)

    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B-FP8",  # <<< tgt quantized
        tensor_parallel_size=2,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",  # <<< draft not quantized
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,  # <<< valid arg name
        },
    )
    tgt_vllm_config: VllmConfig = engine_args.create_engine_config()
    assert tgt_vllm_config.parallel_config.tensor_parallel_size == 2
    assert tgt_vllm_config.quant_config.get_name() == "fp8"

    draft_vllm_config: VllmConfig = create_vllm_config_for_draft_model(tgt_vllm_config)
    assert draft_vllm_config.parallel_config.tensor_parallel_size == 1
    assert draft_vllm_config.quant_config is None


def test_draft_model_engine_args_rejects_invalid_tp_argname():
    """The user should pass "draft_tensor_parallel_size" rather than
    "tensor_parallel_size". We enforce this with validation."""

    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=1,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "tensor_parallel_size": 1,  # <<< invalid arg name
        },
    )
    with pytest.raises(ValueError):
        engine_args.create_engine_config()


def assert_draft_model_correctness(args: ArgsTest, enforce_eager: bool):
    """Compare the outputs using and not using speculative decoding.
    In the greedy decoding case, the outputs must match EXACTLY."""
    test_prompts: list[Messages] = get_messages(
        dataset=args.dataset, n=args.num_prompts
    )

    spec_llm = LLM(
        model=args.target_model,
        speculative_config={
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "enforce_eager": enforce_eager,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "max_num_seqs": 100,  # limit cudagraph capture runtime
        },
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.target_tensor_parallel_size,
        enforce_eager=enforce_eager,
        disable_log_stats=False,  # enables get_metrics()
    )
    # we don't check the outputs, only check the metrics
    spec_llm.chat(test_prompts, args.sampling_config)
    metrics = spec_llm.get_metrics()

    acceptance_rate: float = compute_acceptance_rate(metrics)
    acceptance_len: float = compute_acceptance_len(metrics)
    del spec_llm  # CLEANUP
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    print(
        f"spec-decode: target={args.target_model}, draft={args.draft_model}, "
        f"temperature={args.sampling_config.temperature:.2f}, "
        f"acceptance_rate={acceptance_rate:.2f}, "
        f"acceptance_len={acceptance_len:.2f}, "
    )

    assert acceptance_rate >= args.expected_acceptance_rate
    assert acceptance_len >= args.expected_acceptance_len


def get_messages(dataset: str, n: int) -> list[Messages]:
    if dataset == "test_prompts":
        return get_test_prompts(mm_enabled=False, quiet=True, num_prompts=n)
    elif dataset == "likaixin/InstructCoder":
        return get_instruct_coder_messages(n=n)
    else:
        raise NotImplementedError(f"Dataset '{dataset}' not implemented")


def some_high_acceptance_metrics() -> dict:
    return {
        "sampling_config": greedy_sampling(),
        "num_speculative_tokens": 3,
        "expected_acceptance_len": 2.90 + 1,
        "expected_acceptance_rate": 0.90,
    }


def test_merge_toks_kernel():
    device = "cuda"
    merged_len = 5 + 2  # len(target_toks) = 5, batch_size = 2
    merged = torch.full((merged_len,), -100, device=device)  # -100 is arbitrary
    is_rejected_tok = torch.full((merged_len,), True, device=device)
    grid = (2,)
    merge_toks_kernel[grid](
        target_toks_ptr=torch.tensor([0, 1, 2, 0, 1], device=device),
        next_toks_ptr=torch.tensor([3, 2], device=device),
        query_start_locs_ptr=torch.tensor([0, 3], device=device),
        query_end_locs_ptr=torch.tensor([2, 4], device=device),
        out_ptr_merged_toks=merged,
        out_ptr_is_rejected_tok=is_rejected_tok,
        target_toks_size=5,
        rejected_tok_fill=-1,
    )
    expected_merged = torch.tensor([0, 1, 2, 3, 0, 1, 2], device=device)
    assert torch.allclose(merged, expected_merged)

    expected_rejected_toks = torch.tensor([False] * merged_len, device=device)
    assert torch.allclose(is_rejected_tok, expected_rejected_toks)


def test_merge_toks_kernel_with_rejected_tokens():
    device = "cuda"
    merged_size = 9 + 2  # len(target_toks) = 9, batch_size = 2
    merged = torch.full((merged_size,), -100, device=device)
    is_rejected_tok = torch.full((merged_size,), True, device=device)
    grid = (2,)
    merge_toks_kernel[grid](
        #                                       rejected tokens
        #                                       ↓   ↓   ↓         ↓
        target_toks_ptr=torch.tensor([0, 1, 2, 13, 14, 15, 0, 1, 22], device=device),
        next_toks_ptr=torch.tensor([3, 2], device=device),
        query_start_locs_ptr=torch.tensor([0, 6], device=device),
        query_end_locs_ptr=torch.tensor([2, 7], device=device),
        out_ptr_merged_toks=merged,
        out_ptr_is_rejected_tok=is_rejected_tok,
        target_toks_size=9,
        rejected_tok_fill=-1,
    )
    expected_merged = torch.tensor([0, 1, 2, 3, -1, -1, -1, 0, 1, 2, -1], device=device)
    assert torch.allclose(merged, expected_merged)

    expected_rejected_toks = torch.tensor(
        [False, False, False, False, True, True, True, False, False, False, True],
        device=device,
    )
    assert torch.allclose(is_rejected_tok, expected_rejected_toks)


def compute_acceptance_rate(metrics: list[Metric]) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    n_draft_toks = name2metric["vllm:spec_decode_num_draft_tokens"].value  # type: ignore
    if n_draft_toks == 0:
        return float("nan")
    n_accepted_toks = name2metric["vllm:spec_decode_num_accepted_tokens"].value  # type: ignore
    return n_accepted_toks / n_draft_toks


def compute_acceptance_len(metrics: list[Metric]) -> float:
    name2metric = {metric.name: metric for metric in metrics}
    n_drafts = name2metric["vllm:spec_decode_num_drafts"].value  # type: ignore
    n_accepted_toks = name2metric["vllm:spec_decode_num_accepted_tokens"].value  # type: ignore
    if n_drafts == 0:
        return 1
    return 1 + (n_accepted_toks / n_drafts)

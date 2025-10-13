# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from tests.utils import get_attn_backend_list_based_on_platform, large_gpu_mark
from vllm import LLM, SamplingParams
from vllm.assets.base import VLLM_S3_BUCKET_URL
from vllm.assets.image import VLM_IMAGES_DIR
from vllm.config.vllm import VllmConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.engine.arg_utils import EngineArgs
from vllm.outputs import RequestOutput
from vllm.platforms import current_platform
from vllm.v1.spec_decode.draft_model import create_vllm_config_for_draft_model
from vllm.v1.spec_decode.metrics import compute_acceptance_len, compute_acceptance_rate

MTP_SIMILARITY_RATE = 0.8


def get_test_prompts(mm_enabled: bool, quiet: bool = False):
    prompt_types = ["repeat", "sentence"]
    if mm_enabled:
        prompt_types.append("mm")
    num_prompts = 100
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


def test_ngram_correctness(
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
        speculative_config={
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        },
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


@pytest.mark.parametrize(
    ["model_setup", "mm_enabled"],
    [
        (("eagle3", "Qwen/Qwen3-8B", "AngelSlim/Qwen3-8B_eagle3", 1), False),
        pytest.param(
            (
                "eagle3",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Rayzl/qwen2.5-vl-7b-eagle3-sgl",
                1,
            ),
            False,
            marks=pytest.mark.skip(
                reason="Skipping due to its head_dim not being a a multiple of 32"
            ),
        ),
        (
            (
                "eagle",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
        ),
        (
            (
                "eagle3",
                "meta-llama/Llama-3.1-8B-Instruct",
                "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                1,
            ),
            False,
        ),
        pytest.param(
            (
                "eagle",
                "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                "morgendave/EAGLE-Llama-4-Scout-17B-16E-Instruct",
                4,
            ),
            False,
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
        ),
    ],
    ids=[
        "qwen3_eagle3",
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
    attn_backend: str,
):
    if attn_backend == "TREE_ATTN":
        # TODO: Fix this flaky test
        pytest.skip(
            "TREE_ATTN is flaky in the test disable for now until it can be "
            "resolved (see https://github.com/vllm-project/vllm/issues/22922)"
        )

    # Generate test prompts inside the function instead of using fixture
    test_prompts = get_test_prompts(mm_enabled)
    """
    Compare the outputs of a original LLM and a speculative LLM
    should be the same when using eagle speculative decoding.
    model_setup: (method, model_name, eagle_model_name, tp_size)
    """
    with monkeypatch.context() as m:
        if "Llama-4-Scout" in model_setup[1] and attn_backend == "FLASH_ATTN":
            # Scout requires default backend selection
            # because vision encoder has head_dim 88 being incompatible
            #  with FLASH_ATTN and needs to fall back to Flex Attn
            pass
        else:
            m.setenv("VLLM_MLA_DISABLE", "1")
            m.setenv("VLLM_ATTENTION_BACKEND", attn_backend)

        if attn_backend == "TRITON_ATTN" and not current_platform.is_rocm():
            pytest.skip(
                "TRITON_ATTN does not support "
                "multi-token eagle spec decode on current platform"
            )

        if attn_backend == "FLASH_ATTN" and current_platform.is_rocm():
            m.setenv("VLLM_ROCM_USE_AITER", "1")

        method, model_name, spec_model_name, tp_size = model_setup

        ref_llm = LLM(
            model=model_name, max_model_len=2048, tensor_parallel_size=tp_size
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

        # Heuristic: expect at least 66% of the prompts to match exactly
        # Upon failure, inspect the outputs to check for inaccuracy.
        assert matches > int(0.66 * len(ref_outputs))
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
    expected_same_output_fraction: float
    # Defaults
    target_tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int = 1
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.5


cases = [
    # Same model for draft and target, greedy sampling.
    ArgsTest(
        target_model="Qwen/Qwen3-0.6B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=greedy_sampling(),
        num_speculative_tokens=3,  # K
        expected_acceptance_len=3 + 1,  # K + 1
        expected_acceptance_rate=1.0,
        expected_same_output_fraction=1.0,
    ),
    # Smaller draft model, stochastic sampling.
    ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        draft_model="Qwen/Qwen3-0.6B",
        sampling_config=stochastic_sampling(),
        num_speculative_tokens=3,
        expected_acceptance_len=2.8 + 1,
        expected_acceptance_rate=0.9,
        expected_same_output_fraction=0.9,
    ),
]


@pytest.mark.parametrize("args", cases)
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_draft_model_correctness(args: ArgsTest, enforce_eager: bool):
    assert_draft_model_correctness(args, enforce_eager)


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
    sd_case = ArgsTest(
        target_model="Qwen/Qwen3-1.7B",
        target_tensor_parallel_size=2,
        draft_model="Qwen/Qwen3-0.6B",
        draft_tensor_parallel_size=1,
        **some_high_acceptance_metrics(),
    )
    assert_draft_model_correctness(sd_case, enforce_eager=True)


def test_draft_model_engine_args_tensor_parallelism():
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
    assert (
        tgt_vllm_config.speculative_config.draft_parallel_config.tensor_parallel_size
        == 1
    )

    draft_vllm_config: VllmConfig = create_vllm_config_for_draft_model(tgt_vllm_config)
    assert draft_vllm_config.parallel_config.tensor_parallel_size == 1
    assert draft_vllm_config.quant_config is None


def test_draft_model_engine_args_rejects_invalid_tp_argname():
    """The user should pass "draft_tensor_parallel_size", rather than
    "tensor_parallel_size". This is to catch bad syntax early."""

    engine_args = EngineArgs(
        model="Qwen/Qwen3-1.7B",
        tensor_parallel_size=2,
        speculative_config={
            "model": "Qwen/Qwen3-0.6B",
            "method": "draft_model",
            "num_speculative_tokens": 3,
            "tensor_parallel_size": 1,  # invalid arg name
        },
    )
    with pytest.raises(ValueError):
        engine_args.create_engine_config()


def assert_draft_model_correctness(args: ArgsTest, enforce_eager: bool):
    """Compare the outputs using and not using speculative decoding.
    In the greedy decoding case, the outputs must match EXACTLY."""
    test_prompts = get_test_prompts(mm_enabled=False, quiet=True)

    spec_llm = LLM(
        model=args.target_model,
        speculative_config={
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "enforce_eager": enforce_eager,
            "draft_tensor_parallel_size": args.draft_tensor_parallel_size,
            "disable_padded_drafter_batch": True,
            "max_num_seqs": 100,  # limit cudagraph capture runtime
        },
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.target_tensor_parallel_size,
        enforce_eager=enforce_eager,
        disable_log_stats=False,  # enables get_metrics()
    )
    spec_outputs = spec_llm.chat(test_prompts, args.sampling_config)
    metrics = spec_llm.get_metrics()
    acceptance_rate: float = compute_acceptance_rate(metrics)
    acceptance_len: float = compute_acceptance_len(metrics)
    del spec_llm  # CLEANUP
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    assert acceptance_rate >= args.expected_acceptance_rate
    assert acceptance_len >= args.expected_acceptance_len

    ref_llm = LLM(
        model=args.target_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.target_tensor_parallel_size,
        enforce_eager=enforce_eager,
    )
    ref_outputs = ref_llm.chat(test_prompts, args.sampling_config)
    del ref_llm  # CLEANUP
    torch.cuda.empty_cache()
    cleanup_dist_env_and_memory()

    assert len(ref_outputs) > 0
    assert len(ref_outputs) == len(spec_outputs)

    match_fraction = compute_exact_matches(ref_outputs, spec_outputs)
    assert match_fraction >= args.expected_same_output_fraction

    print(
        f"spec-decode: target={args.target_model}, draft={args.draft_model}, "
        f"temperature={args.sampling_config.temperature:.2f}, "
        f"acceptance_rate={acceptance_rate:.2f}, "
        f"acceptance_len={acceptance_len:.2f}, "
        f"match_fraction={match_fraction:.2f}"
    )


def compute_exact_matches(
    ref_outputs: list[RequestOutput], spec_outputs: list[RequestOutput]
) -> float:
    """Compute the fraction of the prompts that match exactly"""
    assert len(ref_outputs) == len(spec_outputs)
    matches = 0
    for ref_output, spec_output in zip(ref_outputs, spec_outputs):
        if ref_output.outputs[0].text == spec_output.outputs[0].text:
            matches += 1
        else:
            print(f"ref_output: {ref_output.outputs[0].text}")
            print(f"spec_output: {spec_output.outputs[0].text}")
    return matches / len(ref_outputs)


def some_high_acceptance_metrics() -> dict:
    return {
        "sampling_config": greedy_sampling(),
        "num_speculative_tokens": 3,
        "expected_acceptance_len": 2.95 + 1,
        "expected_acceptance_rate": 0.95,
        "expected_same_output_fraction": 0.95,
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.models.gemma4 import Gemma4SelfDecoderLayers
from vllm.platforms import current_platform

from ....utils import check_answers, fork_new_process_for_each_test, prep_prompts

# global seed
SEED = 42


@pytest.fixture
def test_prompts():
    """Adapted from tests/v1/e2e/spec_decode/utils.py."""
    prompt_types = ["repeat", "sentence"]
    # Setting higher num prompts increases the chance of numerics mismatch
    # due to matrix multiplication numerics depending on batch dimension
    num_prompts = 10
    prompts = []

    random.seed(0)
    random_prompt_type_choices = random.choices(prompt_types, k=num_prompts)

    for kind in random_prompt_type_choices:
        word_choices = ["test", "temp", "hello", "where"]
        word = random.choice(word_choices)
        if kind == "repeat":
            prompt = f"""please repeat the word '{word}' 10 times."""
        elif kind == "sentence":
            prompt = f"""please give a ten-word sentence that
            uses the word {word} at least once."""
        else:
            raise ValueError(f"Unknown prompt type: {kind}")
        prompts.append(prompt)

    return prompts


use_fork_for_test = (
    fork_new_process_for_each_test
    if not (current_platform.is_rocm() or current_platform.is_xpu())
    else lambda x: x
)


@use_fork_for_test
@pytest.mark.parametrize(
    "kv_sharing_fast_prefill,use_v2_model_runner",
    [
        (False, False),
        (True, False),
        (True, True),
        # (False, True) omitted: fast-prefill-off behavior is runner-generic,
        # and skipping it saves ~9 min of mostly torch.compile time in CI. The
        # no-silent-fallback assertion below is exercised by the V2 cases.
    ],
)
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_kv_sharing_fast_prefill(
    monkeypatch: pytest.MonkeyPatch,
    kv_sharing_fast_prefill: bool,
    enforce_eager: bool,
    use_v2_model_runner: bool,
):
    if not enforce_eager and current_platform.is_rocm():
        # Relevant context: https://github.com/vllm-project/vllm/pull/29244
        pytest.skip(
            "ROCm: torch.compile produces incorrect output for gemma-3n's GELU "
            "with tanh approximation. Use enforce_eager=True instead."
        )
    if not enforce_eager and not use_v2_model_runner and current_platform.is_cuda():
        pytest.skip(
            "cublasCreate fails with CUBLAS_STATUS_ALLOC_FAILED during memory "
            "profiling with torch 2.14: "
            "https://github.com/pytorch/pytorch/issues/199944"
        )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    compilation_config = CompilationConfig(
        # This allows vLLM compilation backend to handle allocating and
        # managing buffers for cudagraph
        cudagraph_copy_inputs=True,
        mode=CompilationMode.VLLM_COMPILE
        if not enforce_eager
        else CompilationMode.NONE,
    )
    batch_size = 10

    with monkeypatch.context() as m:
        # Make scheduling deterministic for reproducibility
        if current_platform.is_rocm():
            # Use spawn to prevent cuda re-initialization error
            m.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        else:
            m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        m.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2_model_runner else "0")

        prompts, answer, indices = prep_prompts(batch_size)

        llm = LLM(
            model="google/gemma-3n-E2B-it",
            enforce_eager=enforce_eager,
            compilation_config=compilation_config,
            seed=SEED,
            kv_sharing_fast_prefill=kv_sharing_fast_prefill,
            attention_backend="TRITON_ATTN",
        )
        # Guard against a silent fallback to the other model runner.
        assert llm.llm_engine.vllm_config.use_v2_model_runner == use_v2_model_runner
        responses = llm.generate(prompts, sampling_params)
        check_answers(
            indices,
            answer,
            [response.outputs[0].text for response in responses],
            accept_rate=1.0,
        )


GEMMA4_MODEL = "google/gemma-4-E2B-it"


def _tiny_gemma4(hf_config):
    """10 layers, the last 5 share the KV cache of layers 3 and 4. Layer 4, the
    last KV-producing layer, uses full attention as in the real model."""
    hf_config.get_text_config().update(
        {
            "num_hidden_layers": 10,
            "num_kv_shared_layers": 5,
            "hidden_size": 256,
            "intermediate_size": 512,
            "hidden_size_per_layer_input": 32,
        }
    )
    return hf_config


def _random_dummy_weights(seed: int):
    """Dummy weights that make the layers of a random model matter. The default
    ones are tiny, and identical for all tensors of the same shape."""
    generator = None

    @torch.no_grad()
    def init(param: torch.Tensor, *args, **kwargs) -> None:
        nonlocal generator
        if param.device.type == "meta" or not torch.is_floating_point(param):
            return
        if (value := getattr(param, "dummy_weight_value", None)) is not None:
            param.fill_(value)
            return
        if generator is None:
            generator = torch.Generator(device=param.device).manual_seed(seed)
        values = torch.randn(param.shape, generator=generator, device=param.device)
        if param.ndim < 2:  # norm weights and scalars
            param.copy_(1 + 0.1 * values)
        else:
            param.copy_(values * param.shape[-1] ** -0.5)

    return init


def _gemma4_first_token_logprobs(
    monkeypatch: pytest.MonkeyPatch,
    prompts: list[list[int]],
    *,
    kv_sharing_fast_prefill: bool,
    quantization: str | None,
) -> tuple[bool, list[dict[int, float]]]:
    """Whether the last KV-producing layer is split, and the top logprobs of
    the first generated token of each prompt."""
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.dummy_loader.initialize_single_dummy_weight",
        _random_dummy_weights(SEED),
    )
    llm = LLM(
        model=GEMMA4_MODEL,
        hf_overrides=_tiny_gemma4,
        load_format="dummy",
        language_model_only=True,
        quantization=quantization,
        kv_sharing_fast_prefill=kv_sharing_fast_prefill,
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=16,
        gpu_memory_utilization=0.3,
        seed=SEED,
    )
    is_split = any(
        llm.apply_model(
            lambda model: any(
                isinstance(m, Gemma4SelfDecoderLayers) and m.kv_only_layer is not None
                for m in model.modules()
            )
        )
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(temperature=0.0, max_tokens=1, logprobs=8),
    )
    logprobs = [
        {token: lp.logprob for token, lp in output.outputs[0].logprobs[0].items()}
        for output in outputs
    ]
    del llm
    cleanup_dist_env_and_memory()
    return is_split, logprobs


@use_fork_for_test
@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only test")
@pytest.mark.parametrize(
    "quantization,min_shared,atol",
    [
        # bf16: rounding noise of fast prefill on a random-weight model. Without
        # the K/V cache write of the split layer the top tokens share nothing.
        (None, 5, 0.25),
        # Online FP8 scales activations per tensor, i.e. by the rows of the
        # batch, which fast prefill changes: only check the outputs correlate.
        ("fp8", 1, None),
    ],
)
def test_gemma4_kv_sharing_fast_prefill_split_last_layer(
    monkeypatch: pytest.MonkeyPatch,
    quantization: str | None,
    min_shared: int,
    atol: float | None,
):
    """Under fast prefill, the last KV-producing layer only caches K/V for all
    tokens and runs the rest on the logits rows: the logprobs do not change."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    rng = random.Random(SEED)
    # More tokens than max_num_batched_tokens, so that prefills are chunked.
    prompts = [
        [rng.randrange(1000, 20000) for _ in range(rng.randrange(20, 100))]
        for _ in range(8)
    ]

    is_split, expected = _gemma4_first_token_logprobs(
        monkeypatch, prompts, kv_sharing_fast_prefill=False, quantization=quantization
    )
    assert not is_split
    is_split, actual = _gemma4_first_token_logprobs(
        monkeypatch, prompts, kv_sharing_fast_prefill=True, quantization=quantization
    )
    assert is_split, "fast prefill did not split the last KV-producing layer"

    for expected_lps, actual_lps in zip(expected, actual):
        shared = expected_lps.keys() & actual_lps.keys()
        assert len(shared) >= min_shared
        if atol is not None:
            for token in shared:
                assert actual_lps[token] == pytest.approx(expected_lps[token], abs=atol)

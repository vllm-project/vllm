# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import os

import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.attention.utils import full_cg_backend_configs as backend_configs
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.platforms import current_platform
from vllm.utils.torch_utils import is_torch_equal_or_newer
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@contextlib.contextmanager
def temporary_environ(env_vars):
    """
    Temporarily set environment variables and restore them afterward.
    We have to do this vs monkeypatch because monkeypatch doesn't work
    with "module" scoped fixtures.
    """
    original_env = {k: os.environ.get(k) for k in env_vars}
    try:
        os.environ.update(env_vars)
        yield
    finally:
        for k, v in original_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


model_backends_full_cudagraph = []

# deepseek-ai/DeepSeek-V2-Lite with MLA
MLA_backends = ["FlashMLA", "FlashAttentionMLA", "CutlassMLA"]
for mla_backend in MLA_backends:
    model_backends_full_cudagraph.append(
        ("deepseek-ai/DeepSeek-V2-Lite", backend_configs[mla_backend])
    )

# Qwen/Qwen2-1.5B-Instruct with other backends
other_backend_configs = [
    backend_configs[c] for c in backend_configs if c not in MLA_backends
]
for backend_config in other_backend_configs:
    model_backends_full_cudagraph.append(("Qwen/Qwen2-1.5B-Instruct", backend_config))


BATCH_SIZE_MAX_TOKENS_CASES = [
    (1, 10),
    (7, 10),
    (16, 10),
    (25, 10),
    (32, 10),
    (45, 10),
    (64, 10),
    (123, 10),
    (8, 5),
    (8, 30),
]


def _generate_all_cases(llm: LLM) -> dict[tuple[int, int], list[str]]:
    outputs = {}
    for batch_size, max_tokens in BATCH_SIZE_MAX_TOKENS_CASES:
        prompts = ["the quick brown fox"] * batch_size
        # Use purely greedy decoding to avoid top-p truncation sensitivity
        # that can amplify tiny numeric differences across runtimes.
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, top_p=1.0
        )
        responses = llm.generate(prompts, sampling_params)
        outputs[(batch_size, max_tokens)] = [
            response.outputs[0].text.lower() for response in responses
        ]
    return outputs


def _run_all_cases(model: str, **llm_kwargs) -> dict[tuple[int, int], list[str]]:
    """Build an LLM, generate outputs for every case, and tear it down.

    Owns the LLM's lifetime so no reference outlives the shutdown: the engine
    runs in-process (see env_vars in llm_pair), where GC alone does not
    release its GPU memory.
    """
    llm = LLM(
        model=model,
        gpu_memory_utilization=0.43,
        trust_remote_code=True,
        max_model_len=1024,
        max_num_seqs=128,
        generation_config="vllm",
        seed=42,
        **llm_kwargs,
    )
    try:
        return _generate_all_cases(llm)
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="class")
def llm_pair(request):
    model, backend_config, use_inductor_graph_partition = request.param
    backend_config.comp_config["use_inductor_graph_partition"] = (
        use_inductor_graph_partition
    )

    if use_inductor_graph_partition and not is_torch_equal_or_newer("2.9.0.dev"):
        pytest.skip("Inductor graph partition only supported in torch>=2.9")

    # Dynamically skip test if GPU capability is not met
    if (
        backend_config.specific_gpu_arch
        and backend_config.specific_gpu_arch != current_platform.get_device_capability()
    ):
        if backend_config.specific_gpu_arch == (9, 0):
            pytest.skip("Only Hopper GPUs support FA3 and FlashMLA")
        elif backend_config.specific_gpu_arch == (10, 0):
            pytest.skip("Only Blackwell GPUs support Cutlass MLA")

    # FlashInfer is not supported on ROCm
    if backend_config == AttentionBackendEnum.FLASHINFER and current_platform.is_rocm():
        pytest.skip("FlashInfer is not supported on ROCm")

    env_vars = {
        # Force native sampler to avoid potential nondeterminism in FlashInfer
        # when per-request generators are not used in V1.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        # Run the engines in-process so each generate() batches all requests
        # deterministically. With the multiprocess engine, the core busy loop
        # starts prefilling while requests are still arriving over ZMQ, so
        # requests land in timing-dependent prefill waves with different
        # padded batch shapes. The resulting (legitimate) batch-shape numeric
        # differences flip greedy near-ties, breaking the exact-match
        # comparison between the two LLMs at larger batch sizes.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    # Run the two engines one at a time, generating all cases' outputs up
    # front: in-process engines share process-global state (e.g. the workspace
    # manager singleton), so a second live engine would invalidate addresses
    # baked into the first engine's CUDA graphs.
    with temporary_environ(env_vars):
        full_outputs = _run_all_cases(
            model,
            compilation_config=CompilationConfig(**backend_config.comp_config),
        )
        piecewise_outputs = _run_all_cases(
            model,
            compilation_config=CompilationConfig(
                cudagraph_mode=CUDAGraphMode.PIECEWISE
            ),
        )

    yield full_outputs, piecewise_outputs

    wait_for_gpu_memory_to_clear(
        devices=[0],
        threshold_ratio=0.1,
    )


@pytest.mark.parametrize(
    "llm_pair",
    [
        pytest.param((model, backend_config, use_inductor_graph_partition))
        for model, backend_config in model_backends_full_cudagraph
        for use_inductor_graph_partition in [True, False]
    ],
    indirect=True,
)
# The llm_pair fixture already cleans up after its engines; the autouse
# cleanup_fixture's per-test cleanup_dist_env_and_memory() is redundant here.
@pytest.mark.skip_global_cleanup
class TestFullCUDAGraph:
    """
    Use a class such that the llm_pair outputs are computed once for all
    batch_size/max_tokens combinations and released immediately after.
    """

    @pytest.mark.parametrize(
        ("batch_size", "max_tokens"),
        BATCH_SIZE_MAX_TOKENS_CASES,
    )
    def test_full_cudagraph(self, batch_size, max_tokens, llm_pair):
        """
        Test various batch sizes and max_tokens to ensure that the
        full cudagraph compilation works for padded cases too.
        """
        full_outputs, piecewise_outputs = llm_pair
        case = (batch_size, max_tokens)

        # Check that all responses are the same
        for piecewise_text, full_text in zip(
            piecewise_outputs[case], full_outputs[case]
        ):
            assert piecewise_text == full_text

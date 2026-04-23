import copy
import logging
import re

import pytest

from tests.compile.fusions_e2e.common import AttentionBackendCase
from vllm import LLM, RequestOutput, SamplingParams
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode


def run_model(
    compile_config: CompilationConfig, model: str, **model_kwargs
) -> list[RequestOutput]:
    """Run a model with the given compilation config for E2E lowering tests."""
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)
    # Allow override from model_kwargs
    model_kwargs = {"tensor_parallel_size": 1, **model_kwargs}
    model_kwargs = {"disable_custom_all_reduce": True, **model_kwargs}

    # No cudagraphs by default
    if compile_config.cudagraph_mode is None:
        compile_config.cudagraph_mode = CUDAGraphMode.NONE
    llm = LLM(
        model=model,
        compilation_config=compile_config,
        **model_kwargs,
    )

    outputs = llm.generate(prompts, sampling_params)

    return outputs


@pytest.fixture
def run_e2e_lowering_test(monkeypatch, caplog_mp_spawn):
    def run(
        model_name: str,
        model_kwargs: dict,
        attn_backend: AttentionBackendCase,
        compilation_config: dict,
        use_deepgemm: bool = False,
        use_aiter: bool = False,
        tp_size: int = 1,
    ):
        monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1" if use_deepgemm else "0")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1" if use_aiter else "0")
        from vllm._aiter_ops import rocm_aiter_ops

        rocm_aiter_ops.refresh_env_variables()

        # Disable, compile cache to make sure custom passes run.
        monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
        monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")

        model_kwargs = {**attn_backend.model_kwargs, **model_kwargs}
        model_kwargs["attention_config"] = {"backend": attn_backend.backend.name}
        model_kwargs["tensor_parallel_size"] = tp_size

        # Always compile the full graph instead of piecewise
        compilation_config["splitting_ops"] = []

        full_compilation_config = CompilationConfig(
            cudagraph_mode=CUDAGraphMode.NONE,
            mode=CompilationMode.VLLM_COMPILE,
            inductor_compile_config={"force_disable_caches": True},
            **compilation_config,
        )

        # get output from lowered impl
        lowering_compilation_config = copy.deepcopy(full_compilation_config)
        lowering_compilation_config.ir_enable_torch_wrap = True
        lowering_compilation_config.mode = CompilationMode.VLLM_COMPILE
        with caplog_mp_spawn(logging.DEBUG) as log_holder:
            lowered_outputs = run_model(
                lowering_compilation_config, model_name, **model_kwargs
            )

        # check atleast 1 op is lowered
        matches = re.findall(
            r"lowering_pass.py:\d+] VllmIRLoweringPass "
            r"lowered (\d+) with vLLM IR Nodes",
            log_holder.text,
        )
        assert len(matches) == 1
        assert int(matches[0]) > 0

        # check no ops remaining in graph after lowering pass
        matches = re.findall(
            r"lowering_pass.py:\d+] Failed to lower vLLM IR ops",
            log_holder.text,
        )
        assert len(matches) == 0

        # get output without torch wrap
        unwrapped_compilation_config = copy.deepcopy(lowering_compilation_config)
        unwrapped_compilation_config.ir_enable_torch_wrap = False
        unwrapped_compilation_config.mode = CompilationMode.VLLM_COMPILE
        unwrapped_outputs = run_model(
            unwrapped_compilation_config, model_name, **model_kwargs
        )

        # get output without lowering backend and wrapped torch ops
        inductor_compilation_config = copy.deepcopy(full_compilation_config)
        inductor_compilation_config.mode = CompilationMode.STOCK_TORCH_COMPILE
        inductor_compilation_config.backend = "inductor"
        inductor_compilation_config.ir_enable_torch_wrap = True
        inductor_outputs = run_model(
            inductor_compilation_config, model_name, **model_kwargs
        )

        for lowered, unwrapped, inductor in zip(
            lowered_outputs, unwrapped_outputs, inductor_outputs
        ):
            assert lowered.outputs[0].text == unwrapped.outputs[0].text
            assert lowered.outputs[0].text == inductor.outputs[0].text

    return run

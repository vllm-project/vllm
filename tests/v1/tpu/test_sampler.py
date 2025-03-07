# SPDX-License-Identifier: Apache-2.0
import tempfile
from time import time

import pytest

from vllm import LLM, envs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams

if not envs.VLLM_USE_V1:
    pytest.skip(
        "Skipping V1 tests. Rerun with `VLLM_USE_V1=1` to test.",
        allow_module_level=True,
    )


@pytest.mark.parametrize("model_name", ["D4nt3/Qwen2.5-two-layers"])
@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test needs a TPU")
def test_sampler_compilation(model_name: str, monkeypatch):
    """
    Check that no recompilation happens despite changing sampling parameters.
    We can't read XLA metrics from the engine process, hence we measure time.  
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setenv("VLLM_XLA_CACHE_PATH", temp_dir)
        # Compiling model init may still take some time, enforce_eager to skip.
        llm = LLM(model_name,
                  enforce_eager=True,
                  max_num_seqs=16,
                  max_model_len=1024,
                  gpu_memory_utilization=0.5)
        prompts = [
            "A robot may not injure a human being",
            "It is only with the heart that one can see rightly;",
        ]
        # First inference should be slow
        sampling_params = SamplingParams(
            temperature=0.7,
            # top_p=0.6, # TODO too slow!
            # top_k=10,
            min_p=0.2,
            max_tokens=16)
        s = time()
        _ = llm.generate(prompts, sampling_params)
        run1 = time() - s

        # Second request with different params, but for which we
        # compiled for in previous eager iteration.
        sampling_params = SamplingParams(temperature=0.1,
                                         min_p=0.8,
                                         max_tokens=24)
        s = time()
        _ = llm.generate(prompts, sampling_params)
        run2 = time() - s
        # Much faster after compiling
        assert run1 * 0.1 > run2
        print("TIMES", run1, run2)

        # Third request with min_p set to "None". It will not trigger
        # recompilation as a default 0 value will be used.
        sampling_params = SamplingParams(max_tokens=24, temperature=0.0)
        s = time()
        _ = llm.generate(prompts, sampling_params)
        run3 = time() - s
        assert run1 * 0.1 > run3
        print("TIMES", run1, run3)


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2.5-1.5B-Instruct"])
@pytest.mark.skipif(not current_platform.is_tpu(),
                    reason="This test needs a TPU")
def test_sampler_different(model_name: str):
    """
    Test significantly different sampling params to assert the model produces 
    different results.
    """
    llm = LLM(
        model_name,
        enforce_eager=True,
        max_num_seqs=1,
        max_model_len=64,
        # TODO: setting to 0.5 or it will go OOM
        gpu_memory_utilization=0.5)
    prompts = [
        "Write a short story about a robot that dreams for the first time."
    ]
    sampling_params = SamplingParams(temperature=0.9, min_p=0.2, max_tokens=64)
    output = llm.generate(prompts, sampling_params)

    sampling_params = SamplingParams(temperature=0.1, min_p=0.8, max_tokens=64)
    output2 = llm.generate(prompts, sampling_params)
    assert output[0].outputs[0].text != output2[0].outputs[0].text

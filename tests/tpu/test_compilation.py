# SPDX-License-Identifier: Apache-2.0

import glob
import os
import tempfile

import depyf
import pytest

from vllm.config import CompilationLevel


@pytest.mark.skip(reason="Not working; needs investigation.")
def test_tpu_compilation():
    temp_dir = tempfile.mkdtemp()
    with depyf.prepare_debug(temp_dir):
        from vllm import LLM, SamplingParams

        prompts = [
            "A robot may not injure a human being",
            "It is only with the heart that one can see rightly;",
            "The greatest glory in living lies not in never falling,",
        ]
        answers = [
            " or, through inaction, allow a human being to come to harm.",
            " what is essential is invisible to the eye.",
            " but in rising every time we fall.",
        ]
        N = 1
        # Currently, top-p sampling is disabled. `top_p` should be 1.0.
        sampling_params = SamplingParams(temperature=0.7,
                                         top_p=1.0,
                                         n=N,
                                         max_tokens=16)

        # Set `enforce_eager=True` to avoid ahead-of-time compilation.
        # In real workloads, `enforace_eager` should be `False`.

        # disable custom dispatcher, let Dynamo takes over
        # all the control
        llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
                  max_model_len=512,
                  max_num_seqs=64,
                  enforce_eager=True,
                  compilation_config={"level": CompilationLevel.DYNAMO_AS_IS})
        outputs = llm.generate(prompts, sampling_params)
        for output, answer in zip(outputs, answers):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            assert generated_text.startswith(answer)

    compiled_codes = sorted(
        glob.glob(os.path.join(temp_dir, "__transformed_code*.py")))

    for i, compiled_code in enumerate(compiled_codes):
        print("{} file: {}".format(i + 1, compiled_code))

    # We should only trigger Dynamo compilation 4 times:
    # 1. forward pass (symbolic)
    # 2. compute_logits (symbolic)
    # 3. forward pass (shape 16)
    # 4. forward pass (shape 32)
    # and later calls should not trigger Dynamo compilation again.
    # NOTE: It might still trigger XLA compilation.

    # Check we have 4 compiled codes
    assert len(compiled_codes) == 4

    kv_cache_prefix = "kv_cache"
    attn_prefix = "ragged_paged_attention"

    # Check all the compilations are as expected
    compiled_fns = sorted(
        glob.glob(os.path.join(temp_dir, "__compiled_fn*Captured*.py")))

    for i, compiled_fn in enumerate(compiled_fns):
        print("{} file: {}".format(i + 1, compiled_fn))

    # The first compilation is symbolic, so it should not have any kv_caches
    with open(compiled_fns[0]) as f:
        content = f.read()
        assert kv_cache_prefix not in content

    # The second compilation is symbolic, so it should not have any kv_caches
    with open(compiled_fns[1]) as f:
        content = f.read()
        assert kv_cache_prefix not in content

    # The third compilation is shape 16, so it should have kv_caches and the
    # ragged_paged_attention
    with open(compiled_fns[2]) as f:
        content = f.read()
        assert (kv_cache_prefix in content and attn_prefix in content)

    # The forth compilation is shape 32, so it should have kv_caches and the
    # ragged_paged_attention
    with open(compiled_fns[3]) as f:
        content = f.read()
        assert (kv_cache_prefix in content and attn_prefix in content)

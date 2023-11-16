"""Containing tests that check for regressions in vLLM's behavior.

It should include tests that are reported by users and making sure they
will never happen again.

"""
from vllm import LLM, SamplingParams


def test_duplicated_ignored_sequence_group():
    """https://github.com/vllm-project/vllm/issues/1655"""

    sampling_params = SamplingParams(temperature=0.01,
                                     top_p=0.1,
                                     max_tokens=256)
    llm = LLM(model="facebook/opt-125m",
              max_num_batched_tokens=4096,
              tensor_parallel_size=1)
    prompts = ["This is a short prompt", "This is a very long prompt " * 1000]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    assert len(prompts) == len(outputs)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])

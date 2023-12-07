"""Compare the outputs of Specutiave Decoding and original vLLM

Run `pytest tests/models/test_spec_dec.py --forked`.
"""
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from vllm.config import FLAGS
import pytest

MODELS = [
    "lmsys/vicuna-7b-v1.3",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [50])
@pytest.mark.parametrize("draft_model", ["JackFram/llama-160m"])
@pytest.mark.parametrize("propose_cnt", [5])
def test_models(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    draft_model: str,
    propose_cnt: int,
) -> None:
    spec_vllm_model = vllm_runner(model,
                                  dtype=dtype,
                                  draft_model=draft_model,
                                  propose_cnt=propose_cnt)
    spec_vllm_outputs = spec_vllm_model.generate_greedy(
        example_prompts, max_tokens)
    del spec_vllm_model
    destroy_model_parallel()

    FLAGS.ENABLE_SD = False
    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)
    del vllm_model

    for i in range(len(example_prompts)):
        spec_output_ids, spec_output_str = spec_vllm_outputs[i]
        vllm_output_ids, vllm_output_str = vllm_outputs[i]
        # assert spec_output_str == vllm_output_str, (
        #     f"Test{i}:\nSpec: {len(spec_output_str)}\nvLLM: {len(vllm_output_str)}")
        print(len(spec_output_ids), len(vllm_output_ids))
        print(spec_output_ids)
        print(vllm_output_ids)
        assert spec_output_ids == vllm_output_ids, (
            f"Test{i}:\nSpec: {len(spec_output_ids)}\nvLLM: {len(vllm_output_ids)}"
        )

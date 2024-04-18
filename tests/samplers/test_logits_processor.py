import pytest
import torch

from vllm import SamplingParams

MODELS = ["facebook/opt-125m"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_logits_processor_force_generate(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    vllm_model = vllm_runner(model, dtype=dtype)
    tokenizer = vllm_model.model.get_tokenizer()
    repeat_times = 2
    enforced_answers = " vLLM"
    vllm_token_ids = tokenizer.encode(enforced_answers,
                                      add_special_tokens=False)
    max_tokens = len(vllm_token_ids) * repeat_times

    def pick_vllm(token_ids, logits):
        token_id = vllm_token_ids[len(token_ids) % len(vllm_token_ids)]
        logits[token_id] = torch.finfo(logits.dtype).max
        return logits

    params_with_logprobs = SamplingParams(
        logits_processors=[pick_vllm],
        prompt_logprobs=3,
        max_tokens=max_tokens,
    )

    # test logits_processors when prompt_logprobs is not None
    vllm_model.model._add_request(
        prompt=example_prompts[0],
        sampling_params=params_with_logprobs,
        prompt_token_ids=None,
    )

    # test prompt_logprobs is not None
    vllm_model.model._add_request(
        prompt=example_prompts[1],
        sampling_params=SamplingParams(
            prompt_logprobs=3,
            max_tokens=max_tokens,
        ),
        prompt_token_ids=None,
    )

    # test grouped requests
    vllm_model.model._add_request(
        prompt=example_prompts[2],
        sampling_params=SamplingParams(max_tokens=max_tokens),
        prompt_token_ids=None,
    )

    outputs = vllm_model.model._run_engine(False)

    assert outputs[0].outputs[0].text == enforced_answers * repeat_times

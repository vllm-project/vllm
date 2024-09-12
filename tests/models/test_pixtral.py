"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import pytest

from vllm.sampling_params import SamplingParams

pytestmark = pytest.mark.vlm

MODELS = ["mistralai/Pixtral-12B-2409"]


@pytest.mark.skip(
    reason=
    "Model is too big, test passed on A100 locally but will OOM on CI machine."
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    image_urls = [
        "https://picsum.photos/id/237/200/300",
        "https://picsum.photos/seed/picsum/200/300"
    ]
    expected = [
        "The image depicts a black dog lying on a wooden surface, looking directly at the camera with a calm expression.",  # noqa
        "The image depicts a serene landscape with a snow-covered mountain under a pastel-colored sky during sunset."  # noqa
    ]
    prompt = "Describe the image in one short sentence."

    sampling_params = SamplingParams(max_tokens=512, temperature=0.0)

    with vllm_runner(model, dtype=dtype,
                     tokenizer_mode="mistral") as vllm_model:

        for i, image_url in enumerate(image_urls):
            messages = [
                {
                    "role":
                    "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }]
                },
            ]

            outputs = vllm_model.model.chat(messages,
                                            sampling_params=sampling_params)
            assert outputs[0].outputs[0].text == expected[i]

from vllm import LLM
import pytest

@pytest.mark.parametrize("model_id", [
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
])
def test_hf_multimodal_chat(
    model_id: str,
):
    vlm = LLM(
        model=model_id,
        model_impl="transformers",
        disable_mm_preprocessor_cache=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False
    )

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "What is the content of this image?"}
            ],
        },
    ]

    # Perform inference and log output.
    outputs = vlm.chat(conversation)[0]
    generated_text = outputs.outputs[0].text
    assert isinstance(generated_text, str)

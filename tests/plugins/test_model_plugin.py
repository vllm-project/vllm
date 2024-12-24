import os

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

from ..utils import (VLLM_PATH, RemoteOpenAIServer,
                     fork_new_process_for_each_test)


@fork_new_process_for_each_test
def test_oot_registration_text_generation(dummy_opt_path):
    os.environ["VLLM_PLUGINS"] = "register_dummy_model"
    prompts = ["Hello, my name is", "The text does not matter"]
    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=dummy_opt_path, load_format="dummy")
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""


@fork_new_process_for_each_test
def test_oot_registration_embedding(dummy_gemma2_embedding_path):
    os.environ["VLLM_PLUGINS"] = "register_dummy_model"
    prompts = ["Hello, my name is", "The text does not matter"]
    llm = LLM(model=dummy_gemma2_embedding_path, load_format="dummy")
    outputs = llm.embed(prompts)

    for output in outputs:
        assert all(v == 0 for v in output.outputs.embedding)


image = ImageAsset("cherry_blossom").pil_image.convert("RGB")


@fork_new_process_for_each_test
def test_oot_registration_multimodal(dummy_llava_path):
    os.environ["VLLM_PLUGINS"] = "register_dummy_model"
    prompts = [{
        "prompt": "What's in the image?<image>",
        "multi_modal_data": {
            "image": image
        },
    }, {
        "prompt": "Describe the image<image>",
        "multi_modal_data": {
            "image": image
        },
    }]

    sampling_params = SamplingParams(temperature=0)
    llm = LLM(model=dummy_llava_path,
              load_format="dummy",
              max_num_seqs=1,
              trust_remote_code=True,
              gpu_memory_utilization=0.98,
              max_model_len=4096,
              enforce_eager=True,
              limit_mm_per_prompt={"image": 1})
    first_token = llm.get_tokenizer().decode(0)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        generated_text = output.outputs[0].text
        # make sure only the first token is generated
        rest = generated_text.replace(first_token, "")
        assert rest == ""


chatml_jinja_path = VLLM_PATH / "examples/template_chatml.jinja"
assert chatml_jinja_path.exists()


def run_and_test_dummy_opt_api_server(model, tp=1):
    # the model is registered through the plugin
    server_args = [
        "--gpu-memory-utilization",
        "0.10",
        "--dtype",
        "float32",
        "--chat-template",
        str(chatml_jinja_path),
        "--load-format",
        "dummy",
        "-tp",
        f"{tp}",
    ]
    with RemoteOpenAIServer(model, server_args) as server:
        client = server.get_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "You are a helpful assistant."
            }, {
                "role": "user",
                "content": "Hello!"
            }],
            temperature=0,
        )
        generated_text = completion.choices[0].message.content
        assert generated_text is not None
        # make sure only the first token is generated
        rest = generated_text.replace("<s>", "")
        assert rest == ""


def test_oot_registration_for_api_server(dummy_opt_path: str):
    run_and_test_dummy_opt_api_server(dummy_opt_path)

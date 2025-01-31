import os

import pytest

from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

from ..utils import fork_new_process_for_each_test


@fork_new_process_for_each_test
def test_plugin(dummy_opt_path):
    os.environ["VLLM_PLUGINS"] = ""
    with pytest.raises(Exception) as excinfo:
        LLM(model=dummy_opt_path, load_format="dummy")
    assert "are not supported for now" in str(excinfo.value)


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

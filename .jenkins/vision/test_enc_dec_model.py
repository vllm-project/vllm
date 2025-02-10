import atexit
import os
from pathlib import Path

import yaml
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE",
    ".jenkins/vision/configs/Meta-Llama-3.2-11B-Vision-Instruct.yaml")

TP_SIZE = int(os.environ.get("TP_SIZE", 1))


def fail_on_exit():
    os._exit(1)


def launch_enc_dec_model(config, question):
    model_name = config.get('model_name')
    dtype = config.get('dtype', 'bfloat16')
    max_num_seqs = config.get('max_num_seqs', 128)
    max_model_len = config.get('max_model_len', 4096)
    tensor_parallel_size = TP_SIZE
    num_scheduler_steps = config.get('num_scheduler_steps', 1)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        num_scheduler_steps=num_scheduler_steps,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": f"{question}"
        }]
    }]
    prompt = tokenizer.apply_chat_template(messages,
                                           add_generation_prompt=True,
                                           tokenize=False)
    return llm, prompt


def get_input():
    image = Image.open("data/cherry_blossom.jpg").convert("RGB")
    img_question = "What is the content of this image?"

    return {
        "image": image,
        "question": img_question,
    }


def get_current_gaudi_platform():

    #Inspired by: https://github.com/HabanaAI/Model-References/blob/a87c21f14f13b70ffc77617b9e80d1ec989a3442/PyTorch/computer_vision/classification/torchvision/utils.py#L274

    import habana_frameworks.torch.utils.experimental as htexp

    device_type = htexp._get_device_type()

    if device_type == htexp.synDeviceType.synDeviceGaudi:
        return "Gaudi1"
    elif device_type == htexp.synDeviceType.synDeviceGaudi2:
        return "Gaudi2"
    elif device_type == htexp.synDeviceType.synDeviceGaudi3:
        return "Gaudi3"
    else:
        raise ValueError(
            f"Unsupported device: the device type is {device_type}.")


def test_enc_dec_model(record_xml_attribute, record_property):
    try:
        config = yaml.safe_load(
            Path(TEST_DATA_FILE).read_text(encoding="utf-8"))
        # Record JUnitXML test name
        platform = get_current_gaudi_platform()
        testname = (f'test_{Path(TEST_DATA_FILE).stem}_{platform}_'
                    f'tp{TP_SIZE}')
        record_xml_attribute("name", testname)

        mm_input = get_input()
        image = mm_input["image"]
        question = mm_input["question"]
        llm, prompt = launch_enc_dec_model(config, question)

        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=100,
                                         stop_token_ids=None)

        num_prompts = config.get('num_prompts', 1)
        inputs = [{
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        } for _ in range(num_prompts)]

        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for o in outputs:
            generated_text = o.outputs[0].text
            assert generated_text, "Generated text is empty"
            print(generated_text)
        os._exit(0)

    except Exception as exc:
        atexit.register(fail_on_exit)
        raise exc

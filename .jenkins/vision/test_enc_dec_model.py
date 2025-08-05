# SPDX-License-Identifier: Apache-2.0
import atexit
import os
from pathlib import Path

import yaml
from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

TEST_DATA_FILE = os.environ.get(
    "TEST_DATA_FILE",
    ".jenkins/vision/configs/Meta-Llama-3.2-11B-Vision-Instruct.yaml")

TP_SIZE = int(os.environ.get("TP_SIZE", 1))


def fail_on_exit():
    os._exit(1)


def launch_enc_dec_model(config, question, images):
    model_name = config.get('model_name')
    dtype = config.get('dtype', 'bfloat16')
    max_num_seqs = config.get('max_num_seqs', 128)
    max_model_len = config.get('max_model_len', 4096)
    enforce_eager = config.get('enforce_eager', False)
    enable_expert_parallel = config.get('enable_expert_parallel', False)
    tensor_parallel_size = TP_SIZE
    num_scheduler_steps = config.get('num_scheduler_steps', 1)
    limit_mm_per_prompt_image = config.get('limit_mm_per_prompt_image', 1)
    llm = LLM(
        model=model_name,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        num_scheduler_steps=num_scheduler_steps,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enable_expert_parallel=enable_expert_parallel,
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": limit_mm_per_prompt_image},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_placeholder = [{
        "type": "image",
        "image": (i)
    } for i in range(len(images))]
    messages = [{
        "role":
        "user",
        "content":
        [*image_placeholder, {
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
        "image": [image],
        "question": img_question,
    }


def get_input_long(num_image):
    IMAGE_URLS = [
        "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/2/26/Ultramarine_Flycatcher_%28Ficedula_superciliaris%29_Naggar%2C_Himachal_Pradesh%2C_2013_%28cropped%29.JPG",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg/2560px-Anim1754_-_Flickr_-_NOAA_Photo_Library_%281%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/d/d4/Starfish%2C_Caswell_Bay_-_geograph.org.uk_-_409413.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/6/69/Grapevinesnail_01.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Texas_invasive_Musk_Thistle_1.jpg/1920px-Texas_invasive_Musk_Thistle_1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg"
    ]

    return {
        "image": [fetch_image(IMAGE_URLS[i]) for i in range(num_image)],
        "question":
        "What is the content of each image? "
        "Once done, write a story that combines them all.",
    }


def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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

        num_image = config.get('limit_mm_per_prompt_image', 1)
        mm_input = get_input() if num_image == 1 else get_input_long(num_image)
        image = mm_input["image"]
        question = mm_input["question"]
        llm, prompt = launch_enc_dec_model(config, question, image)
        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=100,
                                         stop_token_ids=None)

        num_prompts = config.get('num_prompts', 1)
        model_name = os.path.basename(config.get('model_name'))
        if 'Llama-4' in model_name:
            image_path = "data/cherry_blossom.jpg"
            base64_image = encode_image(image_path)
            messages = [
                {
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "what is in the image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                    ],
                },
            ]
            outputs = llm.chat(messages, sampling_params=sampling_params)

            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print("-----------------------------------")
                print(
                    f"Prompt: {prompt!r}\nGenerated text:\n {generated_text}\n"
                )

            return

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
        return

    except Exception as exc:
        atexit.register(fail_on_exit)
        raise exc

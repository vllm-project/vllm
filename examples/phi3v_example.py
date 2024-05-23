import os

from PIL import Image
import requests
from transformers import AutoProcessor

from vllm import LLM
from vllm.sequence import MultiModalData


# os.environ["VLLM_CPU_KVCACHE_SPACE"] = "10"

def run_phi3v():
    model_path = "/data/LLM-model/Phi-3-vision-128k-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        image_input_type="pixel_values",
        image_token_id=-1,
        image_input_shape="1008, 1344",
        image_feature_size=1024,
    )

    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end|>\n"

    # single-image prompt
    prompt = f"{user_prompt}<|image_1|>\nWhat is shown in this image?{prompt_suffix}{assistant_prompt}"
    inputs = processor(prompt, image, return_tensors="pt")
    multi_modal_data = MultiModalData(type=MultiModalData.Type.IMAGE, data=inputs["pixel_values"])

    outputs = llm.generate(prompt_token_ids=inputs["input_ids"].tolist(), multi_modal_data=multi_modal_data)
    # outputs = llm.generate(prompt)
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_phi3v()

import requests
from PIL import Image

from vllm import LLM, SamplingParams


def run_fuyu():
    llm = LLM(model="adept/fuyu-8b", max_model_len=4096)

    # single-image prompt
    prompt = "What is the highest life expectancy at of male?\n"
    url = "https://huggingface.co/adept/fuyu-8b/resolve/main/chart.png"
    image = Image.open(requests.get(url, stream=True).raw)
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_fuyu()

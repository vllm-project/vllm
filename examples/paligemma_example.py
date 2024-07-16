from vllm import LLM
from vllm.assets.image import ImageAsset


def run_paligemma():
    llm = LLM(model="google/paligemma-3b-mix-224")

    prompt = "caption es"

    image = ImageAsset("stop_sign").pil_image

    outputs = llm.generate({
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    })

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


if __name__ == "__main__":
    run_paligemma()

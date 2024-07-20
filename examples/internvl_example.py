from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def run_internvl():
    model_path = "OpenGVLab/InternVL2-4B"

    llm = LLM(
        model=model_path,
        max_model_len=4096,
        trust_remote_code=True,
        max_num_seqs=5,
    )

    image = ImageAsset("stop_sign").pil_image

    # single-image prompt
    prompt = "<image>\nWhat is the content of this image?\n"  # noqa: E501
    sampling_params = SamplingParams(temperature=0, max_tokens=128)

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
    run_internvl()